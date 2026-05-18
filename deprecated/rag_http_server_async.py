#!/usr/bin/env python3
"""CHORUS RAG HTTP Server - Async Optimized Version

Performance optimizations:
- Async HTTP server using aiohttp
- Parallel embedding + BM25 scoring
- Response caching with TTL
- Connection keep-alive
- Metrics endpoint

Features:
- Hybrid retrieval (semantic + BM25)
- LLM-based query reformulation
- Metadata filtering (document type, date, file patterns)

Run inside Docker container to expose RAG functionality via HTTP.
"""

import asyncio
import json
import os
import pickle
import re
import string
import time
import hashlib
from pathlib import Path
from dataclasses import dataclass, field, fields
from typing import Optional, Dict, List, Any
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import urllib.request

import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
from rank_bm25 import BM25Okapi

try:
    from aiohttp import web
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    print("aiohttp not available, install with: pip install aiohttp")

# Load .env file if present
from dotenv import load_dotenv
load_dotenv()

# =============================================================================
# Configuration
# =============================================================================

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ENABLE_LLM_REFORMULATION = os.environ.get("ENABLE_LLM_REFORMULATION", "true").lower() == "true"
ENABLE_RERANKING = os.environ.get("ENABLE_RERANKING", "true").lower() == "true"
RERANK_MODEL = os.environ.get("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
ENABLE_MMR = os.environ.get("ENABLE_MMR", "true").lower() == "true"
MMR_LAMBDA = float(os.environ.get("MMR_LAMBDA", "0.7"))  # Balance relevance vs diversity


@dataclass
class ChunkWithMetadata:
    text: str
    source_file: str = ""
    source_path: str = ""
    element_type: str = ""
    chunk_index: int = 0
    total_chunks: int = 0
    page_number: Optional[int] = None
    section: str = ""
    char_start: int = 0
    char_end: int = 0
    # Metadata fields for filtering
    doc_type: str = ""  # proposal, transcript, paper, cv, newsletter, meeting, other
    date_year: Optional[int] = None
    date_month: Optional[int] = None
    # Transcript-specific metadata
    timestamp_start: Optional[str] = None  # "00:05:30"
    timestamp_end: Optional[str] = None
    speakers: Optional[str] = None  # Comma-separated speaker names

    @classmethod
    def from_dict(cls, d: dict) -> "ChunkWithMetadata":
        # Handle legacy data without new fields
        known_fields = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in d.items() if k in known_fields}
        return cls(**filtered)

    def citation(self) -> str:
        parts = [self.source_file] if self.source_file else []
        if self.timestamp_start:
            parts.append(f"@{self.timestamp_start}")
        elif self.page_number:
            parts.append(f"p.{self.page_number}")
        if self.section:
            parts.append(f"§{self.section}")
        return " | ".join(parts) if parts else "Unknown"

    def matches_filters(self, filters: Dict[str, Any]) -> bool:
        """Check if chunk matches all provided filters."""
        if not filters:
            return True

        # Document type filter
        if "doc_type" in filters and filters["doc_type"]:
            if self.doc_type and self.doc_type.lower() != filters["doc_type"].lower():
                return False
            if not self.doc_type:
                inferred = infer_doc_type(self.source_file)
                if inferred.lower() != filters["doc_type"].lower():
                    return False

        # Year filter
        if "year" in filters and filters["year"]:
            target_year = int(filters["year"])
            if self.date_year and self.date_year != target_year:
                return False
            if not self.date_year:
                inferred_year = extract_year_from_filename(self.source_file)
                if inferred_year and inferred_year != target_year:
                    return False

        # File pattern filter
        if "file_pattern" in filters and filters["file_pattern"]:
            pattern = filters["file_pattern"].lower()
            if "*" in pattern:
                regex = pattern.replace(".", r"\.").replace("*", ".*")
                if not re.search(regex, self.source_file.lower()):
                    return False
            else:
                if pattern not in self.source_file.lower():
                    return False

        return True


def infer_doc_type(filename: str) -> str:
    """Infer document type from filename."""
    filename_lower = filename.lower()
    if "transcript" in filename_lower or filename_lower.endswith(".vtt"):
        return "transcript"
    elif "proposal" in filename_lower or "grant" in filename_lower or "muri" in filename_lower:
        return "proposal"
    elif "cv" in filename_lower or "biosketch" in filename_lower or "resume" in filename_lower:
        return "cv"
    elif "newsletter" in filename_lower:
        return "newsletter"
    elif "meeting" in filename_lower or "weekly" in filename_lower:
        return "meeting"
    elif "report" in filename_lower:
        return "report"
    elif filename_lower.endswith(".pdf") and "paper" in filename_lower:
        return "paper"
    return "other"


def extract_year_from_filename(filename: str) -> Optional[int]:
    """Extract year from filename patterns."""
    match = re.search(r'(20[12]\d)', filename)
    if match:
        return int(match.group(1))
    match = re.search(r'FY(\d{2})', filename)
    if match:
        year = int(match.group(1))
        return 2000 + year if year < 50 else 1900 + year
    return None


@dataclass
class SearchMetrics:
    """Track search performance metrics."""
    total_searches: int = 0
    cache_hits: int = 0
    total_latency_ms: float = 0
    embedding_latency_ms: float = 0
    bm25_latency_ms: float = 0

    def record(self, total_ms: float, embed_ms: float, bm25_ms: float, cache_hit: bool = False):
        self.total_searches += 1
        self.total_latency_ms += total_ms
        self.embedding_latency_ms += embed_ms
        self.bm25_latency_ms += bm25_ms
        if cache_hit:
            self.cache_hits += 1

    def summary(self) -> dict:
        if self.total_searches == 0:
            return {"total_searches": 0}
        return {
            "total_searches": self.total_searches,
            "cache_hit_rate": f"{self.cache_hits / self.total_searches * 100:.1f}%",
            "avg_total_ms": f"{self.total_latency_ms / self.total_searches:.1f}",
            "avg_embedding_ms": f"{self.embedding_latency_ms / self.total_searches:.1f}",
            "avg_bm25_ms": f"{self.bm25_latency_ms / self.total_searches:.1f}",
        }


# =============================================================================
# LLM Query Reformulation (Async)
# =============================================================================

class LLMQueryReformulator:
    """Uses Claude to reformulate queries for better retrieval."""

    REFORMULATION_PROMPT = """You are a query reformulation assistant for a research lab's knowledge base. Transform user queries into optimized search queries.

The knowledge base contains: grant proposals (NSF, NIH, MURI, AFOSR), meeting transcripts, CVs, newsletters, reports, research papers, lab presentations.

Transform by:
1. Expanding acronyms (MURI = Multidisciplinary University Research Initiative, NSF = National Science Foundation, etc.)
2. Converting relative time to absolute (e.g., "last month" → specific month/year)
3. Adding likely synonyms
4. Unpacking implicit knowledge (e.g., "APTO project" → "NSF APTO Accelerating the Prediction of Technological Opportunities")

Current date: {current_date}

Return 2-3 reformulated search queries, one per line. Keep them concise. Don't explain.

User query: {query}

Reformulated queries:"""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or ANTHROPIC_API_KEY
        self.enabled = bool(self.api_key) and ENABLE_LLM_REFORMULATION

    async def reformulate(self, query: str, executor) -> List[str]:
        """Reformulate a query using Claude (async)."""
        if not self.enabled:
            return [query]

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(executor, self._reformulate_sync, query)

    def _reformulate_sync(self, query: str) -> List[str]:
        """Synchronous reformulation for thread pool execution."""
        try:
            prompt = self.REFORMULATION_PROMPT.format(
                current_date=datetime.now().strftime("%B %d, %Y"),
                query=query
            )

            request_body = json.dumps({
                "model": "claude-3-5-haiku-20241022",
                "max_tokens": 200,
                "messages": [{"role": "user", "content": prompt}]
            }).encode()

            req = urllib.request.Request(
                "https://api.anthropic.com/v1/messages",
                data=request_body,
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01"
                }
            )

            with urllib.request.urlopen(req, timeout=5) as response:
                result = json.loads(response.read().decode())

            if "content" in result and result["content"]:
                text = result["content"][0].get("text", "")
                reformulated = [q.strip() for q in text.strip().split("\n") if q.strip()]
                return [query] + reformulated[:2]

        except Exception as e:
            print(f"LLM reformulation failed: {e}")

        return [query]


class QueryExpander:
    ACRONYMS = {
        "ai": "artificial intelligence", "ml": "machine learning",
        "nlp": "natural language processing", "llm": "large language model",
        "nsf": "national science foundation", "nih": "national institutes of health",
        "rag": "retrieval augmented generation", "css": "computational social science",
        "apto": "accelerating prediction technological opportunities",
        "muri": "multidisciplinary university research initiative",
        "afosr": "air force office scientific research",
    }

    def expand_query(self, query: str) -> list[str]:
        queries = [query]
        q_lower = query.lower()
        for acr, full in self.ACRONYMS.items():
            if acr in q_lower.split():
                queries.append(q_lower.replace(acr, full))
        return queries


class AsyncRAGServer:
    """Async RAG server with parallel search, caching, LLM reformulation, and filtering."""

    def __init__(self, index_path: str):
        self.index_path = Path(index_path)
        self.faiss_index = None
        self.bm25_index = None
        self.chunks: List[ChunkWithMetadata] = []
        self.tokenized_chunks: List[List[str]] = []
        self.model = None
        self.expander = QueryExpander()
        self.llm_reformulator = LLMQueryReformulator()
        self.reranker = None

        # Thread pool for CPU-bound operations (embedding, FAISS, LLM calls)
        self.executor = ThreadPoolExecutor(max_workers=4)

        # Response cache
        self.cache: Dict[str, Dict] = {}
        self.cache_ttl = 300  # 5 minutes
        self.max_cache_size = 500

        # Metrics
        self.metrics = SearchMetrics()

        # Document type index for filtering
        self.doc_type_index: Dict[str, List[int]] = {}

    def load(self):
        """Load indexes and model (called once at startup)."""
        print(f"Loading RAG index from {self.index_path}...")
        self.faiss_index = faiss.read_index(str(self.index_path / "faiss_index.faiss"))
        with open(self.index_path / "hybrid_data.pkl", "rb") as f:
            data = pickle.load(f)
            self.chunks = [ChunkWithMetadata.from_dict(c) for c in data["chunks"]]
            self.tokenized_chunks = data["tokenized_chunks"]
        self.bm25_index = BM25Okapi(self.tokenized_chunks)
        print("Loading embedding model...")
        self.model = SentenceTransformer("BAAI/bge-large-en-v1.5")

        # Load cross-encoder reranker
        if ENABLE_RERANKING:
            print(f"Loading reranker: {RERANK_MODEL}...")
            self.reranker = CrossEncoder(RERANK_MODEL)

        print(f"Loaded {self.faiss_index.ntotal} vectors, {len(self.chunks)} chunks")

        # Build doc_type index
        self._build_doc_type_index()
        print(f"LLM reformulation: {'enabled' if self.llm_reformulator.enabled else 'disabled'}")
        print(f"Reranking: {'enabled' if self.reranker else 'disabled'}")

    def _build_doc_type_index(self):
        """Build index of chunks by document type."""
        self.doc_type_index = {}
        for i, chunk in enumerate(self.chunks):
            doc_type = chunk.doc_type or infer_doc_type(chunk.source_file)
            if doc_type not in self.doc_type_index:
                self.doc_type_index[doc_type] = []
            self.doc_type_index[doc_type].append(i)
        print(f"Doc type distribution: {', '.join(f'{k}:{len(v)}' for k, v in self.doc_type_index.items())}")

    def _get_cache_key(self, query: str, top_k: int) -> str:
        """Generate cache key."""
        return hashlib.md5(f"{query.lower().strip()}:{top_k}".encode()).hexdigest()

    def _check_cache(self, query: str, top_k: int) -> Optional[list]:
        """Check cache for results."""
        key = self._get_cache_key(query, top_k)
        if key in self.cache:
            entry = self.cache[key]
            if time.time() - entry["timestamp"] < self.cache_ttl:
                return entry["results"]
            else:
                del self.cache[key]
        return None

    def _store_cache(self, query: str, top_k: int, results: list):
        """Store results in cache."""
        key = self._get_cache_key(query, top_k)
        self.cache[key] = {
            "results": results,
            "timestamp": time.time()
        }
        # LRU eviction
        if len(self.cache) > self.max_cache_size:
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k]["timestamp"])
            del self.cache[oldest_key]

    def _compute_embedding(self, query: str) -> np.ndarray:
        """Compute embedding for query (CPU-bound)."""
        emb = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(emb)
        return emb

    def _compute_bm25_scores(self, query: str) -> np.ndarray:
        """Compute BM25 scores for query (CPU-bound)."""
        tokens = [t for t in query.lower().translate(
            str.maketrans("", "", string.punctuation)
        ).split() if len(t) > 2]
        return self.bm25_index.get_scores(tokens)

    def _compute_rerank_scores(self, query: str, texts: List[str]) -> np.ndarray:
        """Compute cross-encoder rerank scores (CPU-bound)."""
        pairs = [(query, text) for text in texts]
        return self.reranker.predict(pairs)

    def _apply_mmr(
        self,
        query_embedding: np.ndarray,
        candidates: List[dict],
        top_k: int,
        lambda_param: float = 0.7
    ) -> List[dict]:
        """
        Apply Maximal Marginal Relevance to diversify results.

        MMR = λ * similarity(doc, query) - (1-λ) * max(similarity(doc, selected_docs))
        """
        if len(candidates) <= top_k:
            return candidates

        # Get embeddings for all candidates
        candidate_indices = [int(c["chunk_idx"]) for c in candidates]
        candidate_embeddings = np.array([
            self.faiss_index.reconstruct(int(idx)) for idx in candidate_indices
        ])

        # Normalize for cosine similarity
        faiss.normalize_L2(candidate_embeddings)
        query_norm = query_embedding.copy()
        faiss.normalize_L2(query_norm)

        selected = []
        selected_embeddings = []
        remaining = list(range(len(candidates)))

        for _ in range(min(top_k, len(candidates))):
            best_idx = None
            best_score = float('-inf')

            for i in remaining:
                relevance = candidates[i]["score"]

                if selected_embeddings:
                    similarities = np.dot(
                        np.array(selected_embeddings),
                        candidate_embeddings[i]
                    )
                    max_sim = float(np.max(similarities))
                else:
                    max_sim = 0.0

                mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i

            if best_idx is not None:
                selected.append(candidates[best_idx])
                selected_embeddings.append(candidate_embeddings[best_idx])
                remaining.remove(best_idx)

        return selected

    async def search(
        self,
        query: str,
        top_k: int = 5,
        filters: Dict[str, Any] = None,
        use_llm_reformulation: bool = True,
        use_reranking: bool = True,
        use_mmr: bool = True
    ) -> list[dict]:
        """
        Async search with parallel embedding + BM25, LLM reformulation, reranking, MMR, and filtering.

        Optimizations:
        - Check cache first
        - LLM query reformulation (runs in thread pool)
        - Run embedding and BM25 in parallel using thread pool
        - Metadata filtering
        - Cross-encoder reranking
        - MMR diversity
        - Combine scores and return top results
        """
        start_time = time.time()

        # Check cache (include filters in cache key)
        cache_key_suffix = json.dumps(filters, sort_keys=True) if filters else ""
        cached = self._check_cache(query + cache_key_suffix, top_k)
        if cached:
            self.metrics.record(
                total_ms=(time.time() - start_time) * 1000,
                embed_ms=0, bm25_ms=0, cache_hit=True
            )
            return cached

        # LLM query reformulation (async)
        if use_llm_reformulation and self.llm_reformulator.enabled:
            queries = await self.llm_reformulator.reformulate(query, self.executor)
            print(f"LLM reformulation: {query} → {queries}")
        else:
            queries = [query]

        # Add acronym expansion
        expanded = []
        for q in queries:
            expanded.extend(self.expander.expand_query(q))
        queries = list(dict.fromkeys(expanded))  # Deduplicate

        results = {}
        # Get more candidates if reranking, filtering, or MMR
        if use_mmr and ENABLE_MMR:
            candidate_multiplier = 8
        elif use_reranking and self.reranker:
            candidate_multiplier = 6
        elif filters:
            candidate_multiplier = 4
        else:
            candidate_multiplier = 2

        # Store query embedding for MMR
        query_embedding = None

        for i, q in enumerate(queries):
            weight = 1.0 if i == 0 else 0.5

            # Run embedding and BM25 in PARALLEL
            loop = asyncio.get_event_loop()

            embed_start = time.time()
            bm25_start = time.time()

            embed_future = loop.run_in_executor(self.executor, self._compute_embedding, q)
            bm25_future = loop.run_in_executor(self.executor, self._compute_bm25_scores, q)

            emb, kw_scores = await asyncio.gather(embed_future, bm25_future)

            # Store first query embedding for MMR
            if query_embedding is None:
                query_embedding = emb

            embed_ms = (time.time() - embed_start) * 1000
            bm25_ms = (time.time() - bm25_start) * 1000

            # FAISS search
            scores, indices = self.faiss_index.search(emb, top_k * candidate_multiplier)

            # Normalize BM25 scores
            max_kw = max(kw_scores) if max(kw_scores) > 0 else 1

            # Combine scores with filtering
            for idx, sem in zip(indices[0], scores[0]):
                if idx < 0:
                    continue

                chunk = self.chunks[idx]

                # Apply filters
                if filters and not chunk.matches_filters(filters):
                    continue

                combined = 0.7 * float(sem) + 0.3 * (kw_scores[idx] / max_kw)
                if idx not in results:
                    result = {
                        "text": chunk.text,
                        "citation": chunk.citation(),
                        "source_file": chunk.source_file,
                        "doc_type": chunk.doc_type or infer_doc_type(chunk.source_file),
                        "score": 0,
                        "chunk_idx": idx  # Store for MMR
                    }
                    # Add transcript-specific metadata if present
                    if chunk.timestamp_start:
                        result["timestamp_start"] = chunk.timestamp_start
                        result["timestamp_end"] = chunk.timestamp_end
                    if chunk.speakers:
                        result["speakers"] = chunk.speakers
                    results[idx] = result
                results[idx]["score"] += combined * weight

        # Sort by initial score
        sorted_results = sorted(results.values(), key=lambda x: x["score"], reverse=True)

        # Apply cross-encoder reranking (async)
        if use_reranking and self.reranker and len(sorted_results) > 1:
            # Take top candidates for reranking (limit to avoid slowdown)
            rerank_count = min(len(sorted_results), top_k * 3)
            candidates = sorted_results[:rerank_count]

            # Run reranking in thread pool
            loop = asyncio.get_event_loop()
            texts = [r["text"] for r in candidates]
            rerank_scores = await loop.run_in_executor(
                self.executor, self._compute_rerank_scores, query, texts
            )

            # Update scores with reranking (blend: 40% initial + 60% rerank)
            for i, score in enumerate(rerank_scores):
                # Normalize rerank score to 0-1 range (cross-encoder outputs can vary)
                normalized_score = 1 / (1 + np.exp(-score))  # sigmoid
                candidates[i]["score"] = 0.4 * candidates[i]["score"] + 0.6 * normalized_score
                candidates[i]["rerank_score"] = float(score)

            # Re-sort by blended score
            sorted_results = sorted(candidates, key=lambda x: x["score"], reverse=True)

        # Apply MMR for diversity
        if use_mmr and ENABLE_MMR and query_embedding is not None and len(sorted_results) > top_k:
            sorted_results = self._apply_mmr(
                query_embedding,
                sorted_results[:top_k * 2],
                top_k,
                lambda_param=MMR_LAMBDA
            )

        # Remove internal chunk_idx before returning
        final_results = sorted_results[:top_k]
        for r in final_results:
            r.pop("chunk_idx", None)

        # Cache results
        self._store_cache(query + cache_key_suffix, top_k, final_results)

        # Record metrics
        total_ms = (time.time() - start_time) * 1000
        self.metrics.record(total_ms=total_ms, embed_ms=embed_ms, bm25_ms=bm25_ms)

        return final_results

    def list_sources(self) -> list[dict]:
        """List all indexed sources with document types."""
        sources = {}
        for c in self.chunks:
            if c.source_file not in sources:
                sources[c.source_file] = {
                    "chunks": 0,
                    "doc_type": c.doc_type or infer_doc_type(c.source_file)
                }
            sources[c.source_file]["chunks"] += 1
        return sorted(
            [{"file": k, "chunks": v["chunks"], "doc_type": v["doc_type"]} for k, v in sources.items()],
            key=lambda x: -x["chunks"]
        )

    def list_doc_types(self) -> Dict[str, int]:
        """List document types with counts."""
        return {k: len(v) for k, v in self.doc_type_index.items()}


# Initialize RAG server
rag = AsyncRAGServer("Data/rag_indexes/hybrid")
rag.load()


def parse_filters(request) -> Dict[str, Any]:
    """Parse filter parameters from query string."""
    filters = {}
    if "doc_type" in request.query:
        filters["doc_type"] = request.query["doc_type"]
    if "year" in request.query:
        filters["year"] = request.query["year"]
    if "file_pattern" in request.query:
        filters["file_pattern"] = request.query["file_pattern"]
    return filters if filters else None


# Async HTTP handlers
async def handle_search(request):
    """Handle /search endpoint."""
    query = request.query.get("q", "")
    top_k = int(request.query.get("top_k", 5))
    use_llm = request.query.get("llm", "true").lower() == "true"
    use_rerank = request.query.get("rerank", "true").lower() == "true"
    use_mmr = request.query.get("mmr", "true").lower() == "true"

    if not query:
        return web.json_response({"error": "Missing query parameter 'q'"}, status=400)

    filters = parse_filters(request)
    results = await rag.search(query, top_k, filters=filters,
                               use_llm_reformulation=use_llm, use_reranking=use_rerank,
                               use_mmr=use_mmr)

    response = {
        "query": query,
        "results": results,
        "llm_reformulation": use_llm and rag.llm_reformulator.enabled,
        "reranking": use_rerank and rag.reranker is not None,
        "mmr_diversity": use_mmr and ENABLE_MMR
    }
    if filters:
        response["filters"] = filters

    return web.json_response(response)


async def handle_context(request):
    """Handle /context endpoint."""
    question = request.query.get("q", "")
    max_sources = int(request.query.get("max_sources", 5))
    use_llm = request.query.get("llm", "true").lower() == "true"
    use_rerank = request.query.get("rerank", "true").lower() == "true"
    use_mmr = request.query.get("mmr", "true").lower() == "true"

    if not question:
        return web.json_response({"error": "Missing query parameter 'q'"}, status=400)

    filters = parse_filters(request)
    results = await rag.search(question, max_sources, filters=filters,
                               use_llm_reformulation=use_llm, use_reranking=use_rerank,
                               use_mmr=use_mmr)
    context = "\n\n".join(f"[{r['citation']}]\n{r['text']}" for r in results)

    return web.json_response({
        "question": question,
        "context": context,
        "citations": [r["citation"] for r in results],
        "filters": filters,
        "reranking": use_rerank and rag.reranker is not None,
        "mmr_diversity": use_mmr and ENABLE_MMR
    })


async def handle_sources(request):
    """Handle /sources endpoint."""
    sources = rag.list_sources()

    # Optional doc_type filter
    doc_type = request.query.get("doc_type")
    if doc_type:
        sources = [s for s in sources if s["doc_type"].lower() == doc_type.lower()]

    return web.json_response({
        "total_sources": len(sources),
        "sources": sources[:50]
    })


async def handle_doc_types(request):
    """Handle /doc_types endpoint."""
    return web.json_response({
        "doc_types": rag.list_doc_types()
    })


async def handle_health(request):
    """Handle /health endpoint."""
    return web.json_response({
        "status": "ok",
        "chunks": len(rag.chunks),
        "cache_size": len(rag.cache),
        "llm_reformulation_enabled": rag.llm_reformulator.enabled,
        "reranking_enabled": rag.reranker is not None,
        "rerank_model": RERANK_MODEL if rag.reranker else None,
        "mmr_enabled": ENABLE_MMR,
        "mmr_lambda": MMR_LAMBDA
    })


async def handle_metrics(request):
    """Handle /metrics endpoint - performance stats."""
    return web.json_response({
        "search_metrics": rag.metrics.summary(),
        "cache_size": len(rag.cache),
        "cache_ttl_seconds": rag.cache_ttl
    })


async def handle_root(request):
    """Handle root endpoint - API documentation."""
    return web.json_response({
        "name": "CHORUS RAG Server (Async)",
        "version": "2.2",
        "features": {
            "hybrid_retrieval": "70% semantic + 30% keyword (BM25)",
            "llm_reformulation": "Query expansion using Claude",
            "cross_encoder_reranking": "Re-score top results with cross-encoder model",
            "mmr_diversity": f"Maximal Marginal Relevance (λ={MMR_LAMBDA})",
            "metadata_filtering": "Filter by doc_type, year, file_pattern"
        },
        "endpoints": {
            "/search?q=query&top_k=5": {
                "description": "Search documents",
                "params": {
                    "q": "Search query (required)",
                    "top_k": "Number of results (default: 5)",
                    "llm": "Use LLM reformulation (default: true)",
                    "rerank": "Use cross-encoder reranking (default: true)",
                    "mmr": "Use MMR diversity (default: true)",
                    "doc_type": "Filter: proposal|transcript|cv|newsletter|meeting|report|paper|other",
                    "year": "Filter: e.g., 2024",
                    "file_pattern": "Filter: e.g., *.pdf, *MURI*, transcript*"
                }
            },
            "/context?q=question&max_sources=5": "Get formatted context for question",
            "/sources": "List all sources (optional: ?doc_type=proposal)",
            "/doc_types": "List document types with counts",
            "/health": "Health check",
            "/metrics": "Performance metrics"
        }
    })


def create_app():
    """Create the aiohttp application."""
    app = web.Application()
    app.router.add_get("/", handle_root)
    app.router.add_get("/search", handle_search)
    app.router.add_get("/context", handle_context)
    app.router.add_get("/sources", handle_sources)
    app.router.add_get("/doc_types", handle_doc_types)
    app.router.add_get("/health", handle_health)
    app.router.add_get("/metrics", handle_metrics)
    return app


if __name__ == "__main__":
    if not AIOHTTP_AVAILABLE:
        print("Error: aiohttp required. Install with: pip install aiohttp")
        exit(1)

    port = 8765
    print(f"\nAsync RAG HTTP Server running on http://localhost:{port}")
    print("Endpoints:")
    print(f"  GET /search?q=your+query&top_k=5")
    print(f"  GET /context?q=your+question&max_sources=5")
    print(f"  GET /sources")
    print(f"  GET /health")
    print(f"  GET /metrics")
    print("\nPress Ctrl+C to stop\n")

    app = create_app()
    web.run_app(app, host="0.0.0.0", port=port)
