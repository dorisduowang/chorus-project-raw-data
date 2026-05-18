#!/usr/bin/env python3
"""
CHORUS RAG Core Module

Core RAG functionality extracted for use by multiple server implementations.
This module contains the shared RAG logic without HTTP-specific code.

Features:
- Hybrid retrieval (semantic + BM25)
- LLM-based query reformulation
- Cross-encoder reranking
- MMR diversity
- Metadata filtering
"""

import json
import os
import pickle
import re
import ssl
import string
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
import urllib.parse
import urllib.request

import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
from rank_bm25 import BM25Okapi


# =============================================================================
# Environment Loading
# =============================================================================

from config import load_dotenv, get_env_bool, get_env_float

load_dotenv()


# =============================================================================
# SSL Configuration
# =============================================================================

try:
    import certifi
    ssl_context = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    ssl_context = ssl.create_default_context()


# =============================================================================
# Configuration
# =============================================================================

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ENABLE_LLM_REFORMULATION = get_env_bool("ENABLE_LLM_REFORMULATION", True)
ENABLE_RERANKING = get_env_bool("ENABLE_RERANKING", True)
RERANK_MODEL = os.environ.get("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
ENABLE_MMR = get_env_bool("ENABLE_MMR", True)
MMR_LAMBDA = get_env_float("MMR_LAMBDA", 0.7)


# =============================================================================
# Optional Dependencies
# =============================================================================

# Reformulation cache integration
try:
    from cache import ReformulationCache, get_reformulation_cache
    REFORMULATION_CACHE_AVAILABLE = True
except ImportError:
    REFORMULATION_CACHE_AVAILABLE = False

# =============================================================================
# Adaptive Search Weighting
# =============================================================================

NAME_PATTERNS = re.compile(
    r'^(who is|what is|tell me about|find)\s+[A-Z][a-z]+(\s+[A-Z][a-z]+)*$|'
    r'^[A-Z][a-z]+(\s+[A-Z][a-z]+){1,3}$|'
    r'\b(h-index|publication|citations?|papers?)\s+(of|for|by)\b',
    re.IGNORECASE
)

CONCEPTUAL_PATTERNS = re.compile(
    r'\b(how|why|what is the|explain|describe|compare|difference|relationship|'
    r'concept|theory|method|approach|impact|effect|trend|overview|'
    r'research on|studies about|work on|field of)\b',
    re.IGNORECASE
)

EXACT_MATCH_INDICATORS = re.compile(
    r'^"[^"]+"|'
    r'\b(exactly|specifically|called|named|titled)\b',
    re.IGNORECASE
)


@dataclass
class SearchWeights:
    """Weights for hybrid search components."""
    semantic: float
    bm25: float
    query_type: str

    def __post_init__(self):
        total = self.semantic + self.bm25
        if abs(total - 1.0) > 0.001:
            self.semantic = self.semantic / total
            self.bm25 = self.bm25 / total


class AdaptiveWeighting:
    """Calculates adaptive weights for semantic vs BM25 search."""

    PRESETS = {
        "name_lookup": SearchWeights(semantic=0.50, bm25=0.50, query_type="name_lookup"),
        "conceptual": SearchWeights(semantic=0.85, bm25=0.15, query_type="conceptual"),
        "exact_match": SearchWeights(semantic=0.30, bm25=0.70, query_type="exact_match"),
        "mixed": SearchWeights(semantic=0.70, bm25=0.30, query_type="mixed"),
    }

    def classify_query(self, query: str) -> str:
        """Classify query type based on patterns."""
        if EXACT_MATCH_INDICATORS.search(query):
            return "exact_match"
        if NAME_PATTERNS.search(query):
            return "name_lookup"
        if CONCEPTUAL_PATTERNS.search(query):
            return "conceptual"
        return "mixed"

    def calculate_weights(self, query: str) -> SearchWeights:
        """Calculate adaptive weights based on query characteristics."""
        query_type = self.classify_query(query)
        return self.PRESETS[query_type]

    def get_weight_tuple(self, query: str) -> Tuple[float, float]:
        """Get weights as a simple tuple (semantic_weight, bm25_weight)."""
        weights = self.calculate_weights(query)
        return (weights.semantic, weights.bm25)


_adaptive_weighting = None


def get_adaptive_weighting() -> AdaptiveWeighting:
    """Get the singleton AdaptiveWeighting instance."""
    global _adaptive_weighting
    if _adaptive_weighting is None:
        _adaptive_weighting = AdaptiveWeighting()
    return _adaptive_weighting

# Registry integration
try:
    from registry import RegistryRAG, SemanticRegistryIndex, merge_with_structured
    REGISTRY_AVAILABLE = True
    SEMANTIC_REGISTRY_AVAILABLE = True
except ImportError:
    REGISTRY_AVAILABLE = False
    SEMANTIC_REGISTRY_AVAILABLE = False

# Hypergraph integration
try:
    from hypergraph import HyperGraph, load_hypergraph
    HYPERGRAPH_AVAILABLE = True
except ImportError:
    HYPERGRAPH_AVAILABLE = False


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class ChunkWithMetadata:
    """A text chunk with associated metadata for RAG retrieval."""
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
    doc_type: str = ""
    date_year: Optional[int] = None
    date_month: Optional[int] = None
    timestamp_start: Optional[str] = None
    timestamp_end: Optional[str] = None
    speakers: Optional[str] = None
    # Rich extraction metadata
    entities_people: Optional[List[str]] = None
    entities_orgs: Optional[List[str]] = None
    entities_projects: Optional[List[str]] = None
    entities_grants: Optional[List[str]] = None
    dates_mentioned: Optional[List[str]] = None
    topics: Optional[List[str]] = None
    key_terms: Optional[List[str]] = None

    @classmethod
    def from_dict(cls, d: dict) -> "ChunkWithMetadata":
        """Create from dictionary, ignoring unknown fields."""
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in known_fields}
        return cls(**filtered)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'text': self.text,
            'source_file': self.source_file,
            'source_path': self.source_path,
            'element_type': self.element_type,
            'chunk_index': self.chunk_index,
            'total_chunks': self.total_chunks,
            'page_number': self.page_number,
            'section': self.section,
            'char_start': self.char_start,
            'char_end': self.char_end,
            'doc_type': self.doc_type,
            'date_year': self.date_year,
            'date_month': self.date_month,
            'timestamp_start': self.timestamp_start,
            'timestamp_end': self.timestamp_end,
            'speakers': self.speakers,
            'entities_people': self.entities_people,
            'entities_orgs': self.entities_orgs,
            'entities_projects': self.entities_projects,
            'entities_grants': self.entities_grants,
            'dates_mentioned': self.dates_mentioned,
            'topics': self.topics,
            'key_terms': self.key_terms,
        }

    def citation(self) -> str:
        """Generate citation string."""
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

        if "doc_type" in filters and filters["doc_type"]:
            if self.doc_type and self.doc_type.lower() != filters["doc_type"].lower():
                return False
            if not self.doc_type:
                inferred = infer_doc_type(self.source_file)
                if inferred.lower() != filters["doc_type"].lower():
                    return False

        if "year" in filters and filters["year"]:
            target_year = int(filters["year"])
            if self.date_year and self.date_year != target_year:
                return False
            if not self.date_year:
                inferred_year = extract_year_from_filename(self.source_file)
                if inferred_year and inferred_year != target_year:
                    return False

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


# =============================================================================
# Utility Functions
# =============================================================================

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
    else:
        return "other"


def extract_year_from_filename(filename: str) -> Optional[int]:
    """Extract year from filename patterns like 2024, FY25, etc."""
    match = re.search(r'(20[12]\d)', filename)
    if match:
        return int(match.group(1))
    match = re.search(r'FY(\d{2})', filename)
    if match:
        year = int(match.group(1))
        return 2000 + year if year < 50 else 1900 + year
    return None


# =============================================================================
# Query Reformulation
# =============================================================================

class LLMQueryReformulator:
    """Uses Claude to reformulate queries for better retrieval."""

    REFORMULATION_PROMPT = """You are a query reformulation assistant for a research lab's knowledge base. Your job is to transform user queries into optimized search queries.

The knowledge base contains:
- Grant proposals (NSF, NIH, MURI, AFOSR)
- Meeting transcripts and recordings
- CVs and biosketches
- Newsletters and reports
- Research papers and publications
- Lab presentations and notes

Transform the user's query by:
1. Expanding acronyms (MURI = Multidisciplinary University Research Initiative, NSF = National Science Foundation, etc.)
2. Converting relative time references to absolute (e.g., "last month" → specific month/year based on current date)
3. Adding likely synonyms or related terms
4. Unpacking implicit knowledge (e.g., "the APTO project" → "NSF APTO Accelerating the Prediction of Technological Opportunities")

Current date: {current_date}

Return 2-3 reformulated search queries, one per line. Keep them concise but specific. Don't explain, just return the queries.

User query: {query}

Reformulated queries:"""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or ANTHROPIC_API_KEY
        self.enabled = bool(self.api_key) and ENABLE_LLM_REFORMULATION
        self.cache = None
        self._async_client = None  # Lazy-loaded httpx client
        if REFORMULATION_CACHE_AVAILABLE:
            try:
                self.cache = get_reformulation_cache()
            except Exception as e:
                print(f"Failed to initialize reformulation cache: {e}")

    def _get_prompt(self, query: str) -> str:
        """Generate the reformulation prompt."""
        return self.REFORMULATION_PROMPT.format(
            current_date=datetime.now().strftime("%B %d, %Y"),
            query=query
        )

    def _parse_response(self, query: str, result: dict) -> List[str]:
        """Parse LLM response and cache results."""
        if "content" in result and result["content"]:
            text = result["content"][0].get("text", "")
            reformulated = [q.strip() for q in text.strip().split("\n") if q.strip()]
            result_queries = [query] + reformulated[:2]

            if self.cache:
                self.cache.set(query, result_queries)

            return result_queries
        return [query]

    def reformulate(self, query: str) -> List[str]:
        """Reformulate a query using Claude with caching (synchronous)."""
        if not self.enabled:
            return [query]

        if self.cache:
            cached = self.cache.get(query)
            if cached:
                return cached.reformulations

        try:
            prompt = self._get_prompt(query)

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

            with urllib.request.urlopen(req, timeout=5, context=ssl_context) as response:
                result = json.loads(response.read().decode())

            return self._parse_response(query, result)

        except Exception as e:
            print(f"LLM reformulation failed: {e}")

        return [query]

    async def reformulate_async(self, query: str) -> List[str]:
        """
        Reformulate a query using Claude with caching (asynchronous).

        This method uses httpx for non-blocking HTTP requests, allowing
        FastAPI to handle other requests while waiting for the LLM response.
        Falls back to synchronous if httpx is not available.
        """
        if not self.enabled:
            return [query]

        # Check cache first
        if self.cache:
            cached = self.cache.get(query)
            if cached:
                return cached.reformulations

        # Try async with httpx
        try:
            import httpx
        except ImportError:
            # Fall back to sync if httpx not available
            return self.reformulate(query)

        try:
            prompt = self._get_prompt(query)

            # Lazy-load async client
            if self._async_client is None:
                self._async_client = httpx.AsyncClient(timeout=5.0)

            response = await self._async_client.post(
                "https://api.anthropic.com/v1/messages",
                json={
                    "model": "claude-3-5-haiku-20241022",
                    "max_tokens": 200,
                    "messages": [{"role": "user", "content": prompt}]
                },
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01"
                }
            )
            response.raise_for_status()
            result = response.json()

            return self._parse_response(query, result)

        except Exception as e:
            print(f"Async LLM reformulation failed: {e}")

        return [query]

    async def close(self):
        """Close the async HTTP client."""
        if self._async_client is not None:
            await self._async_client.aclose()
            self._async_client = None


class QueryExpander:
    """Simple acronym expansion for queries."""

    ACRONYMS = {
        "ai": "artificial intelligence",
        "ml": "machine learning",
        "nlp": "natural language processing",
        "llm": "large language model",
        "nsf": "national science foundation",
        "nih": "national institutes of health",
    }

    def expand_query(self, query: str) -> List[str]:
        """Expand acronyms in query."""
        queries = [query]
        q_lower = query.lower()
        for acr, full in self.ACRONYMS.items():
            if acr in q_lower.split():
                queries.append(q_lower.replace(acr, full))
        return queries


# =============================================================================
# RAG Server Core
# =============================================================================

class RAGServer:
    """
    Core RAG server with hybrid retrieval, reranking, and MMR.

    This class provides the core search functionality without HTTP handling.
    """

    def __init__(self, index_path: str):
        self.index_path = Path(index_path)
        self.faiss_index = None
        self.bm25_index = None
        self.chunks: List[ChunkWithMetadata] = []
        self.tokenized_chunks: List[List[str]] = []
        self.model = None
        self.reranker = None
        self.expander = QueryExpander()
        self.llm_reformulator = LLMQueryReformulator()
        self.doc_type_index: Dict[str, List[int]] = {}
        self._last_index_mtime: Optional[float] = None
        self._reload_check_interval = 30
        self._last_reload_check = 0

    def load(self):
        """Load the RAG index and models."""
        print(f"Loading RAG index from {self.index_path}...")
        self._load_index()

        print("Loading embedding model...")
        self.model = SentenceTransformer("BAAI/bge-large-en-v1.5")

        if ENABLE_RERANKING:
            print(f"Loading reranker: {RERANK_MODEL}...")
            self.reranker = CrossEncoder(RERANK_MODEL)

        print(f"Loaded {self.faiss_index.ntotal} vectors, {len(self.chunks)} chunks")
        self._build_doc_type_index()
        print(f"LLM reformulation: {'enabled' if self.llm_reformulator.enabled else 'disabled'}")
        print(f"Reranking: {'enabled' if self.reranker else 'disabled'}")

    def _load_index(self):
        """Load or reload the FAISS index and chunks."""
        faiss_path = self.index_path / "faiss_index.faiss"
        self.faiss_index = faiss.read_index(str(faiss_path))

        with open(self.index_path / "hybrid_data.pkl", "rb") as f:
            data = pickle.load(f)
            self.chunks = [ChunkWithMetadata.from_dict(c) for c in data["chunks"]]
            self.tokenized_chunks = data["tokenized_chunks"]
        self.bm25_index = BM25Okapi(self.tokenized_chunks)
        self._last_index_mtime = faiss_path.stat().st_mtime

    def reload_if_changed(self) -> bool:
        """Check if index files changed and reload if needed."""
        now = time.time()
        if now - self._last_reload_check < self._reload_check_interval:
            return False
        self._last_reload_check = now

        faiss_path = self.index_path / "faiss_index.faiss"
        if not faiss_path.exists():
            return False

        current_mtime = faiss_path.stat().st_mtime
        if current_mtime != self._last_index_mtime:
            print(f"Index changed, reloading...")
            old_count = len(self.chunks)
            self._load_index()
            self._build_doc_type_index()
            print(f"Reloaded: {old_count} → {len(self.chunks)} chunks")
            return True

        return False

    def force_reload(self) -> Dict[str, Any]:
        """Force reload the index."""
        old_count = len(self.chunks)
        self._load_index()
        self._build_doc_type_index()
        return {
            "status": "reloaded",
            "previous_chunks": old_count,
            "current_chunks": len(self.chunks),
            "timestamp": datetime.now().isoformat()
        }

    def _build_doc_type_index(self):
        """Build index of chunks by document type."""
        self.doc_type_index = {}
        for i, chunk in enumerate(self.chunks):
            doc_type = chunk.doc_type or infer_doc_type(chunk.source_file)
            if doc_type not in self.doc_type_index:
                self.doc_type_index[doc_type] = []
            self.doc_type_index[doc_type].append(i)
        print(f"Doc type distribution: {', '.join(f'{k}:{len(v)}' for k, v in self.doc_type_index.items())}")

    def _apply_mmr(
        self,
        query_embedding: np.ndarray,
        candidates: List[dict],
        top_k: int,
        lambda_param: float = 0.7
    ) -> List[dict]:
        """Apply Maximal Marginal Relevance to diversify results."""
        if len(candidates) <= top_k:
            return candidates

        candidate_indices = [int(c["chunk_idx"]) for c in candidates]
        candidate_embeddings = np.array([
            self.faiss_index.reconstruct(int(idx)) for idx in candidate_indices
        ])

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
                    similarities = np.dot(np.array(selected_embeddings), candidate_embeddings[i])
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

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: Dict[str, Any] = None,
        use_llm_reformulation: bool = True,
        use_reranking: bool = True,
        use_mmr: bool = True
    ) -> List[dict]:
        """
        Search with hybrid retrieval, LLM reformulation, reranking, MMR, and filtering.

        Args:
            query: The search query
            top_k: Number of results to return
            filters: Optional filters dict with keys: doc_type, year, file_pattern
            use_llm_reformulation: Whether to use LLM to reformulate the query
            use_reranking: Whether to use cross-encoder reranking
            use_mmr: Whether to apply MMR for diversity

        Returns:
            List of search results with text, citation, source_file, score
        """
        # Get query variants
        if use_llm_reformulation and self.llm_reformulator.enabled:
            queries = self.llm_reformulator.reformulate(query)
            print(f"LLM reformulation: {query} → {queries}")
        else:
            queries = [query]

        # Add acronym expansion
        expanded = []
        for q in queries:
            expanded.extend(self.expander.expand_query(q))
        queries = list(dict.fromkeys(expanded))

        # Use shared pipeline
        return self._search_pipeline(query, queries, top_k, filters, use_reranking, use_mmr)

    async def search_async(
        self,
        query: str,
        top_k: int = 5,
        filters: Dict[str, Any] = None,
        use_llm_reformulation: bool = True,
        use_reranking: bool = True,
        use_mmr: bool = True
    ) -> List[dict]:
        """
        Async search with hybrid retrieval, LLM reformulation, reranking, MMR.

        This method uses async LLM reformulation for better concurrency
        in FastAPI handlers. The rest of the search pipeline is the same
        as the sync version.

        Args:
            query: The search query
            top_k: Number of results to return
            filters: Optional filters dict with keys: doc_type, year, file_pattern
            use_llm_reformulation: Whether to use LLM to reformulate the query
            use_reranking: Whether to use cross-encoder reranking
            use_mmr: Whether to apply MMR for diversity

        Returns:
            List of search results with text, citation, source_file, score
        """
        # Get query variants using async reformulation
        if use_llm_reformulation and self.llm_reformulator.enabled:
            queries = await self.llm_reformulator.reformulate_async(query)
            print(f"LLM reformulation (async): {query} → {queries}")
        else:
            queries = [query]

        # Add acronym expansion
        expanded = []
        for q in queries:
            expanded.extend(self.expander.expand_query(q))
        queries = list(dict.fromkeys(expanded))

        # The rest is CPU-bound, so run sync
        return self._search_pipeline(query, queries, top_k, filters, use_reranking, use_mmr)

    def _search_pipeline(
        self,
        original_query: str,
        queries: List[str],
        top_k: int,
        filters: Dict[str, Any],
        use_reranking: bool,
        use_mmr: bool
    ) -> List[dict]:
        """
        Internal search pipeline (shared by sync and async search).

        Args:
            original_query: The original query for reranking
            queries: List of query variants to search
            top_k: Number of results to return
            filters: Optional filters
            use_reranking: Whether to use cross-encoder reranking
            use_mmr: Whether to apply MMR for diversity

        Returns:
            List of search results
        """
        results = {}

        # Calculate candidate count
        if use_mmr and ENABLE_MMR:
            candidate_multiplier = 8
        elif use_reranking and self.reranker:
            candidate_multiplier = 6
        elif filters:
            candidate_multiplier = 4
        else:
            candidate_multiplier = 2

        query_embedding = None

        for i, q in enumerate(queries):
            weight = 1.0 if i == 0 else 0.5

            emb = self.model.encode([q], convert_to_numpy=True)
            faiss.normalize_L2(emb)

            if query_embedding is None:
                query_embedding = emb

            scores, indices = self.faiss_index.search(emb, top_k * candidate_multiplier)

            tokens = [t for t in q.lower().translate(str.maketrans("", "", string.punctuation)).split() if len(t) > 2]
            kw_scores = self.bm25_index.get_scores(tokens)
            max_kw = max(kw_scores) if max(kw_scores) > 0 else 1

            for idx, sem in zip(indices[0], scores[0]):
                if idx < 0:
                    continue

                chunk = self.chunks[idx]

                if filters and not chunk.matches_filters(filters):
                    continue

                # Use adaptive weighting based on query type
                weights = get_adaptive_weighting().calculate_weights(original_query)
                combined = weights.semantic * float(sem) + weights.bm25 * (kw_scores[idx] / max_kw)
                if idx not in results:
                    result = {
                        "text": chunk.text,
                        "citation": chunk.citation(),
                        "source_file": chunk.source_file,
                        "doc_type": chunk.doc_type or infer_doc_type(chunk.source_file),
                        "score": 0,
                        "chunk_idx": idx
                    }
                    if chunk.timestamp_start:
                        result["timestamp_start"] = chunk.timestamp_start
                        result["timestamp_end"] = chunk.timestamp_end
                    if chunk.speakers:
                        result["speakers"] = chunk.speakers
                    results[idx] = result
                results[idx]["score"] += combined * weight

        sorted_results = sorted(results.values(), key=lambda x: x["score"], reverse=True)

        # Cross-encoder reranking
        if use_reranking and self.reranker and len(sorted_results) > 1:
            rerank_count = min(len(sorted_results), top_k * 3)
            candidates = sorted_results[:rerank_count]

            pairs = [(original_query, r["text"]) for r in candidates]
            rerank_scores = self.reranker.predict(pairs)

            for i, score in enumerate(rerank_scores):
                normalized_score = 1 / (1 + np.exp(-score))
                candidates[i]["score"] = 0.4 * candidates[i]["score"] + 0.6 * normalized_score
                candidates[i]["rerank_score"] = float(score)

            sorted_results = sorted(candidates, key=lambda x: x["score"], reverse=True)

        # MMR diversity
        if use_mmr and ENABLE_MMR and query_embedding is not None and len(sorted_results) > top_k:
            sorted_results = self._apply_mmr(
                query_embedding,
                sorted_results[:top_k * 2],
                top_k,
                lambda_param=MMR_LAMBDA
            )

        # Clean up results
        final_results = sorted_results[:top_k]
        for r in final_results:
            r.pop("chunk_idx", None)

        return final_results

    def search_semantic_only(
        self,
        query: str,
        top_k: int = 5,
        filters: Dict[str, Any] = None,
        use_reranking: bool = True,
        use_mmr: bool = True
    ) -> List[dict]:
        """
        Pure semantic (embedding-based) search without BM25 component.

        Args:
            query: The search query
            top_k: Number of results to return
            filters: Optional filters
            use_reranking: Whether to use cross-encoder reranking
            use_mmr: Whether to apply MMR for diversity

        Returns:
            List of search results
        """
        results = {}
        candidate_multiplier = 8 if (use_mmr and ENABLE_MMR) else 4

        emb = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(emb)
        query_embedding = emb

        scores, indices = self.faiss_index.search(emb, top_k * candidate_multiplier)

        for idx, sem in zip(indices[0], scores[0]):
            if idx < 0:
                continue

            chunk = self.chunks[idx]
            if filters and not chunk.matches_filters(filters):
                continue

            if idx not in results:
                result = {
                    "text": chunk.text,
                    "citation": chunk.citation(),
                    "source_file": chunk.source_file,
                    "doc_type": chunk.doc_type or infer_doc_type(chunk.source_file),
                    "score": float(sem),
                    "chunk_idx": idx
                }
                if chunk.timestamp_start:
                    result["timestamp_start"] = chunk.timestamp_start
                    result["timestamp_end"] = chunk.timestamp_end
                if chunk.speakers:
                    result["speakers"] = chunk.speakers
                results[idx] = result

        sorted_results = sorted(results.values(), key=lambda x: x["score"], reverse=True)

        # Cross-encoder reranking
        if use_reranking and self.reranker and len(sorted_results) > 1:
            rerank_count = min(len(sorted_results), top_k * 3)
            candidates = sorted_results[:rerank_count]
            pairs = [(query, r["text"]) for r in candidates]
            rerank_scores = self.reranker.predict(pairs)

            for i, score in enumerate(rerank_scores):
                normalized_score = 1 / (1 + np.exp(-score))
                candidates[i]["score"] = 0.4 * candidates[i]["score"] + 0.6 * normalized_score
                candidates[i]["rerank_score"] = float(score)

            sorted_results = sorted(candidates, key=lambda x: x["score"], reverse=True)

        # MMR diversity
        if use_mmr and ENABLE_MMR and query_embedding is not None and len(sorted_results) > top_k:
            sorted_results = self._apply_mmr(
                query_embedding,
                sorted_results[:top_k * 2],
                top_k,
                lambda_param=MMR_LAMBDA
            )

        final_results = sorted_results[:top_k]
        for r in final_results:
            r.pop("chunk_idx", None)

        return final_results

    def search_bm25_only(
        self,
        query: str,
        top_k: int = 5,
        filters: Dict[str, Any] = None
    ) -> List[dict]:
        """
        Pure BM25 (keyword-based) search without semantic component.

        Args:
            query: The search query
            top_k: Number of results to return
            filters: Optional filters

        Returns:
            List of search results
        """
        tokens = [t for t in query.lower().translate(
            str.maketrans("", "", string.punctuation)
        ).split() if len(t) > 2]

        if not tokens:
            return []

        kw_scores = self.bm25_index.get_scores(tokens)
        max_kw = max(kw_scores) if max(kw_scores) > 0 else 1

        # Get top indices by BM25 score
        scored_indices = [(i, kw_scores[i] / max_kw) for i in range(len(kw_scores))]
        scored_indices.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in scored_indices[:top_k * 4]:
            if score <= 0:
                continue

            chunk = self.chunks[idx]
            if filters and not chunk.matches_filters(filters):
                continue

            result = {
                "text": chunk.text,
                "citation": chunk.citation(),
                "source_file": chunk.source_file,
                "doc_type": chunk.doc_type or infer_doc_type(chunk.source_file),
                "score": float(score),
            }
            if chunk.timestamp_start:
                result["timestamp_start"] = chunk.timestamp_start
                result["timestamp_end"] = chunk.timestamp_end
            if chunk.speakers:
                result["speakers"] = chunk.speakers
            results.append(result)

            if len(results) >= top_k:
                break

        return results

    def list_sources(self) -> List[dict]:
        """List all indexed sources with their document types."""
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


# =============================================================================
# Convenience Functions
# =============================================================================

def create_rag_server(index_path: str = "Data/rag_indexes/hybrid") -> RAGServer:
    """Create and load a RAG server instance."""
    rag = RAGServer(index_path)
    rag.load()
    return rag
