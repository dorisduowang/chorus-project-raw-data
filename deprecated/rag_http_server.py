#!/usr/bin/env python3
"""CHORUS RAG HTTP Server

DEPRECATED: This module is deprecated. Use rag_server_fastapi.py instead.
Core RAG functionality has been moved to rag_core.py.

This server will be removed in a future version.

Run inside Docker container to expose RAG functionality via HTTP.

Features:
- Hybrid retrieval (semantic + BM25)
- LLM-based query reformulation
- Metadata filtering (document type, date, file patterns)
"""

import warnings
warnings.warn(
    "rag_http_server is deprecated. Use rag_server_fastapi.py with rag_core.py instead.",
    DeprecationWarning,
    stacklevel=2
)

import json
import os
import pickle
import re
import ssl
import string
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime
import urllib.parse
import urllib.request

# Load .env file if it exists
def load_dotenv():
    """Load environment variables from .env file."""
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    # Set if not already set OR if currently empty
                    if key and (key not in os.environ or not os.environ.get(key)):
                        os.environ[key] = value

load_dotenv()

# Fix SSL certificate verification on macOS
try:
    import certifi
    ssl_context = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    ssl_context = ssl.create_default_context()

import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
from rank_bm25 import BM25Okapi

# Registry integration
try:
    from registry import RegistryRAG, SemanticRegistryIndex, merge_with_structured
    REGISTRY_AVAILABLE = True
    SEMANTIC_REGISTRY_AVAILABLE = True
except ImportError:
    # Fallback to old import path for backward compatibility
    try:
        from registry_rag import RegistryRAG
        REGISTRY_AVAILABLE = True
        SEMANTIC_REGISTRY_AVAILABLE = False
    except ImportError:
        REGISTRY_AVAILABLE = False
        SEMANTIC_REGISTRY_AVAILABLE = False
        print("Warning: registry module not available, structured queries disabled")

# Hypergraph integration
try:
    from hypergraph import HyperGraph, load_hypergraph
    HYPERGRAPH_AVAILABLE = True
except ImportError:
    HYPERGRAPH_AVAILABLE = False

# Reformulation cache integration
try:
    from cache import ReformulationCache, get_reformulation_cache
    REFORMULATION_CACHE_AVAILABLE = True
except ImportError:
    REFORMULATION_CACHE_AVAILABLE = False


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
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
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
            # Infer doc type from filename if not set
            if not self.doc_type:
                inferred = infer_doc_type(self.source_file)
                if inferred.lower() != filters["doc_type"].lower():
                    return False

        # Year filter
        if "year" in filters and filters["year"]:
            target_year = int(filters["year"])
            if self.date_year and self.date_year != target_year:
                return False
            # Try to extract year from filename if not set
            if not self.date_year:
                inferred_year = extract_year_from_filename(self.source_file)
                if inferred_year and inferred_year != target_year:
                    return False

        # File pattern filter (glob-like)
        if "file_pattern" in filters and filters["file_pattern"]:
            pattern = filters["file_pattern"].lower()
            if "*" in pattern:
                # Simple glob: *.pdf, *transcript*, etc.
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
    else:
        return "other"


def extract_year_from_filename(filename: str) -> Optional[int]:
    """Extract year from filename patterns like 2024, FY25, etc."""
    # Look for 4-digit years
    match = re.search(r'(20[12]\d)', filename)
    if match:
        return int(match.group(1))
    # Look for FY patterns
    match = re.search(r'FY(\d{2})', filename)
    if match:
        year = int(match.group(1))
        return 2000 + year if year < 50 else 1900 + year
    return None


# =============================================================================
# LLM Query Reformulation
# =============================================================================

class LLMQueryReformulator:
    """
    Uses Claude to reformulate queries for better retrieval.

    Transforms natural language queries into search-optimized versions
    that unpack implicit knowledge and add relevant context.
    """

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
        # Initialize reformulation cache if available
        self.cache = None
        if REFORMULATION_CACHE_AVAILABLE:
            try:
                self.cache = get_reformulation_cache()
            except Exception as e:
                print(f"Failed to initialize reformulation cache: {e}")

    def reformulate(self, query: str) -> List[str]:
        """
        Reformulate a query using Claude with caching.

        Returns:
            List of reformulated queries (original + LLM-generated)
        """
        if not self.enabled:
            return [query]

        # Check cache first
        if self.cache:
            cached = self.cache.get(query)
            if cached:
                return cached.reformulations

        try:
            # Call Claude API directly via urllib to avoid dependency
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

            with urllib.request.urlopen(req, timeout=5, context=ssl_context) as response:
                result = json.loads(response.read().decode())

            # Parse response
            if "content" in result and result["content"]:
                text = result["content"][0].get("text", "")
                # Split into lines and filter empty
                reformulated = [q.strip() for q in text.strip().split("\n") if q.strip()]
                # Return original + reformulated
                result_queries = [query] + reformulated[:2]  # Cap at 2 additional queries

                # Cache the result
                if self.cache:
                    self.cache.set(query, result_queries)

                return result_queries

        except Exception as e:
            print(f"LLM reformulation failed: {e}")

        return [query]


class QueryExpander:
    ACRONYMS = {
        "ai": "artificial intelligence", "ml": "machine learning",
        "nlp": "natural language processing", "llm": "large language model",
        "nsf": "national science foundation", "nih": "national institutes of health",
    }

    def expand_query(self, query: str) -> list[str]:
        queries = [query]
        q_lower = query.lower()
        for acr, full in self.ACRONYMS.items():
            if acr in q_lower.split():
                queries.append(q_lower.replace(acr, full))
        return queries


class RAGServer:
    def __init__(self, index_path: str):
        self.index_path = Path(index_path)
        self.faiss_index = None
        self.bm25_index = None
        self.chunks = []
        self.tokenized_chunks = []
        self.model = None
        self.reranker = None
        self.expander = QueryExpander()
        self.llm_reformulator = LLMQueryReformulator()

        # Build index of chunk indices by doc_type for faster filtering
        self.doc_type_index: Dict[str, List[int]] = {}

        # Track index file modification time for hot-reload
        self._last_index_mtime: Optional[float] = None
        self._reload_check_interval = 30  # seconds
        self._last_reload_check = 0

    def load(self):
        print(f"Loading RAG index from {self.index_path}...")
        self._load_index()

        print("Loading embedding model...")
        self.model = SentenceTransformer("BAAI/bge-large-en-v1.5")

        # Load cross-encoder reranker
        if ENABLE_RERANKING:
            print(f"Loading reranker: {RERANK_MODEL}...")
            self.reranker = CrossEncoder(RERANK_MODEL)

        print(f"Loaded {self.faiss_index.ntotal} vectors, {len(self.chunks)} chunks")

        # Build doc_type index for filtering
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

        # Track file modification time
        self._last_index_mtime = faiss_path.stat().st_mtime

    def reload_if_changed(self) -> bool:
        """
        Check if index files changed and reload if needed.

        Returns True if index was reloaded.
        """
        import time
        now = time.time()

        # Only check every N seconds
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
        """Force reload the index. Returns reload status."""
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
        """Build index of chunks by document type for efficient filtering."""
        self.doc_type_index = {}
        for i, chunk in enumerate(self.chunks):
            # Use stored doc_type or infer from filename
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
        """
        Apply Maximal Marginal Relevance to diversify results.

        MMR = λ * similarity(doc, query) - (1-λ) * max(similarity(doc, selected_docs))

        Args:
            query_embedding: The query embedding vector
            candidates: List of candidate results with 'chunk_idx' key
            top_k: Number of results to select
            lambda_param: Balance between relevance (1.0) and diversity (0.0)

        Returns:
            Diversified list of results
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
                # Relevance: similarity to query (use the existing score)
                relevance = candidates[i]["score"]

                # Diversity: max similarity to already selected docs
                if selected_embeddings:
                    similarities = np.dot(
                        np.array(selected_embeddings),
                        candidate_embeddings[i]
                    )
                    max_sim = float(np.max(similarities))
                else:
                    max_sim = 0.0

                # MMR score
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
    ) -> list[dict]:
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
        # Get all query variants
        if use_llm_reformulation and self.llm_reformulator.enabled:
            queries = self.llm_reformulator.reformulate(query)
            print(f"LLM reformulation: {query} → {queries}")
        else:
            queries = [query]

        # Add acronym expansion
        expanded = []
        for q in queries:
            expanded.extend(self.expander.expand_query(q))
        queries = list(dict.fromkeys(expanded))  # Deduplicate while preserving order

        results = {}

        # Calculate how many candidates we need
        # More candidates if filtering, reranking, or MMR
        if use_mmr and ENABLE_MMR:
            candidate_multiplier = 8  # Need more for MMR diversity selection
        elif use_reranking and self.reranker:
            candidate_multiplier = 6  # Get more candidates for reranking
        elif filters:
            candidate_multiplier = 4
        else:
            candidate_multiplier = 2

        # Store query embedding for MMR
        query_embedding = None

        for i, q in enumerate(queries):
            # Original query weighted higher than reformulations
            weight = 1.0 if i == 0 else 0.5

            # Semantic search
            emb = self.model.encode([q], convert_to_numpy=True)
            faiss.normalize_L2(emb)

            # Store first query embedding for MMR
            if query_embedding is None:
                query_embedding = emb

            scores, indices = self.faiss_index.search(emb, top_k * candidate_multiplier)

            # Keyword search
            tokens = [t for t in q.lower().translate(str.maketrans("", "", string.punctuation)).split() if len(t) > 2]
            kw_scores = self.bm25_index.get_scores(tokens)
            max_kw = max(kw_scores) if max(kw_scores) > 0 else 1

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

        # Apply cross-encoder reranking
        if use_reranking and self.reranker and len(sorted_results) > 1:
            # Take top candidates for reranking (limit to avoid slowdown)
            rerank_count = min(len(sorted_results), top_k * 3)
            candidates = sorted_results[:rerank_count]

            # Score query-document pairs with cross-encoder
            pairs = [(query, r["text"]) for r in candidates]
            rerank_scores = self.reranker.predict(pairs)

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
                sorted_results[:top_k * 2],  # Consider top 2x candidates for MMR
                top_k,
                lambda_param=MMR_LAMBDA
            )

        # Remove internal chunk_idx before returning
        final_results = sorted_results[:top_k]
        for r in final_results:
            r.pop("chunk_idx", None)

        return final_results

    def list_sources(self) -> list[dict]:
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


# Initialize RAG
rag = RAGServer("Data/rag_indexes/hybrid")
rag.load()

# Initialize Registry RAG (if available)
registry_rag = None
if REGISTRY_AVAILABLE:
    try:
        registry_rag = RegistryRAG()
        print(f"Registry loaded: {registry_rag.get_registry_stats()}")
    except Exception as e:
        print(f"Failed to load registry: {e}")

# Initialize Semantic Registry Index (if available)
semantic_registry = None
if SEMANTIC_REGISTRY_AVAILABLE:
    try:
        semantic_registry_path = Path("Data/indexes/semantic_registry.pkl")
        if semantic_registry_path.exists():
            semantic_registry = SemanticRegistryIndex()
            semantic_registry.load_index(str(semantic_registry_path))
            stats = semantic_registry.stats()
            print(f"Semantic registry loaded: {stats['total_entities']} entities")
        else:
            # Build index on first run
            print("Building semantic registry index (first run)...")
            semantic_registry = SemanticRegistryIndex()
            semantic_registry.build_index()
            semantic_registry.save_index(str(semantic_registry_path))
            print(f"Semantic registry built and saved")
    except Exception as e:
        print(f"Failed to load semantic registry: {e}")
        semantic_registry = None

# Initialize Hypergraph (if available)
hypergraph = None
if HYPERGRAPH_AVAILABLE:
    try:
        hypergraph_path = Path("Data/hypergraph.json")
        if hypergraph_path.exists():
            hypergraph = load_hypergraph(str(hypergraph_path))
            stats = hypergraph.stats()
            print(f"Hypergraph loaded: {stats['total_nodes']} nodes, {stats['total_edges']} edges")
        else:
            print("Warning: hypergraph.json not found, run 'python -m hypergraph.build' first")
    except Exception as e:
        print(f"Failed to load hypergraph: {e}")


class RAGHandler(BaseHTTPRequestHandler):
    def _send_response(self, data: dict, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2).encode())

    def _parse_filters(self, params: dict) -> Dict[str, Any]:
        """Parse filter parameters from query string."""
        filters = {}

        # doc_type: proposal, transcript, cv, newsletter, meeting, report, paper, other
        if "doc_type" in params:
            filters["doc_type"] = params["doc_type"][0]

        # year: 2024, 2025, etc.
        if "year" in params:
            filters["year"] = params["year"][0]

        # file_pattern: *.pdf, *transcript*, MURI*, etc.
        if "file_pattern" in params:
            filters["file_pattern"] = params["file_pattern"][0]

        return filters if filters else None

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)

        if parsed.path == "/search":
            # Check for index updates (auto-reload every 30s)
            rag.reload_if_changed()

            query = params.get("q", [""])[0]
            top_k = int(params.get("top_k", [5])[0])
            use_llm = params.get("llm", ["true"])[0].lower() == "true"
            use_rerank = params.get("rerank", ["true"])[0].lower() == "true"
            use_mmr = params.get("mmr", ["true"])[0].lower() == "true"

            if not query:
                self._send_response({"error": "Missing query parameter 'q'"}, 400)
                return

            filters = self._parse_filters(params)
            results = rag.search(query, top_k, filters=filters,
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

            self._send_response(response)

        elif parsed.path == "/context":
            question = params.get("q", [""])[0]
            max_sources = int(params.get("max_sources", [5])[0])
            use_llm = params.get("llm", ["true"])[0].lower() == "true"
            use_rerank = params.get("rerank", ["true"])[0].lower() == "true"
            use_mmr = params.get("mmr", ["true"])[0].lower() == "true"

            if not question:
                self._send_response({"error": "Missing query parameter 'q'"}, 400)
                return

            filters = self._parse_filters(params)
            results = rag.search(question, max_sources, filters=filters,
                               use_llm_reformulation=use_llm, use_reranking=use_rerank,
                               use_mmr=use_mmr)
            context = "\n\n".join(f"[{r['citation']}]\n{r['text']}" for r in results)

            self._send_response({
                "question": question,
                "context": context,
                "citations": [r["citation"] for r in results],
                "filters": filters,
                "reranking": use_rerank and rag.reranker is not None,
                "mmr_diversity": use_mmr and ENABLE_MMR
            })

        elif parsed.path == "/sources":
            sources = rag.list_sources()

            # Optional doc_type filter
            doc_type = params.get("doc_type", [None])[0]
            if doc_type:
                sources = [s for s in sources if s["doc_type"].lower() == doc_type.lower()]

            self._send_response({
                "total_sources": len(sources),
                "sources": sources[:50]
            })

        elif parsed.path == "/doc_types":
            """List available document types with counts."""
            self._send_response({
                "doc_types": rag.list_doc_types()
            })

        elif parsed.path == "/health":
            self._send_response({
                "status": "ok",
                "chunks": len(rag.chunks),
                "llm_reformulation_enabled": rag.llm_reformulator.enabled,
                "reranking_enabled": rag.reranker is not None,
                "rerank_model": RERANK_MODEL if rag.reranker else None,
                "mmr_enabled": ENABLE_MMR,
                "mmr_lambda": MMR_LAMBDA
            })

        elif parsed.path == "/reload":
            """Force reload the index (for incremental updates)."""
            result = rag.force_reload()
            self._send_response(result)

        elif parsed.path == "/registry":
            """Query the structured registry."""
            if not registry_rag:
                self._send_response({"error": "Registry not available"}, 503)
                return

            # Check for registry updates (hot-reload)
            registry_rag.reload_if_changed()

            query = params.get("q", [""])[0]
            if not query:
                self._send_response({"error": "Missing query parameter 'q'"}, 400)
                return

            result = registry_rag.query(query)
            self._send_response(result)

        elif parsed.path == "/registry/reload":
            """Force reload the registry."""
            if not registry_rag:
                self._send_response({"error": "Registry not available"}, 503)
                return

            result = registry_rag.force_reload()
            self._send_response(result)

        elif parsed.path == "/registry/stats":
            """Get registry statistics."""
            if not registry_rag:
                self._send_response({"error": "Registry not available"}, 503)
                return

            self._send_response({
                "stats": registry_rag.get_registry_stats(),
                "available": True
            })

        elif parsed.path == "/registry/topics":
            """List all unique research topics with counts."""
            if not registry_rag:
                self._send_response({"error": "Registry not available"}, 503)
                return

            # Check for registry updates (hot-reload)
            registry_rag.reload_if_changed()

            # Get all topics with counts
            topic_counts = registry_rag.lookup.get_all_topics()

            # Sort by count descending, then alphabetically
            sorted_topics = sorted(
                [{"topic": t, "count": c} for t, c in topic_counts.items()],
                key=lambda x: (-x["count"], x["topic"])
            )

            # Optional search filter
            search = params.get("search", [""])[0].lower()
            if search:
                sorted_topics = [
                    t for t in sorted_topics
                    if search in t["topic"].lower()
                ]

            self._send_response({
                "total_topics": len(sorted_topics),
                "topics": sorted_topics,
                "people_with_topics": sum(
                    1 for m in registry_rag.lookup.registry.get("people", {}).get("members", [])
                    if m.get("openalex", {}).get("topics")
                )
            })

        elif parsed.path == "/registry/topic-search":
            """Search for people by research topic."""
            if not registry_rag:
                self._send_response({"error": "Registry not available"}, 503)
                return

            # Check for registry updates (hot-reload)
            registry_rag.reload_if_changed()

            topic = params.get("q", params.get("topic", [""]))[0]
            if not topic:
                self._send_response({"error": "Missing topic parameter 'q' or 'topic'"}, 400)
                return

            matches = registry_rag.lookup.find_people_by_topic(topic)

            self._send_response({
                "query": topic,
                "found": len(matches),
                "results": [
                    {
                        "name": m["person"].get("name"),
                        "role": m["person"].get("role"),
                        "institution": m["person"].get("institution"),
                        "matched_topics": m["matched_topics"],
                        "all_topics": m["person"].get("openalex", {}).get("topics", [])
                    }
                    for m in matches
                ]
            })

        elif parsed.path == "/registry/person":
            """Lookup a specific person."""
            if not registry_rag:
                self._send_response({"error": "Registry not available"}, 503)
                return

            # Check for registry updates (hot-reload)
            registry_rag.reload_if_changed()

            name = params.get("name", [""])[0]
            if not name:
                self._send_response({"error": "Missing name parameter"}, 400)
                return

            person = registry_rag.lookup.find_person(name)
            if person:
                self._send_response({"found": True, "person": person})
            else:
                self._send_response({"found": False, "query": name})

        elif parsed.path == "/registry/publications":
            """Search publications by title, author, venue, or topic."""
            if not registry_rag:
                self._send_response({"error": "Registry not available"}, 503)
                return

            # Check for registry updates (hot-reload)
            registry_rag.reload_if_changed()

            query = params.get("q", [""])[0]
            limit = int(params.get("limit", [20])[0])

            if query:
                # Search for specific publications
                publications = registry_rag.lookup.find_publications(query)
                self._send_response({
                    "query": query,
                    "found": len(publications),
                    "results": publications[:limit]
                })
            else:
                # No query - return all publications
                all_pubs = registry_rag.lookup.get_all_publications()
                self._send_response({
                    "total": len(all_pubs),
                    "results": all_pubs[:limit],
                    "note": "Add ?q=search_term to search publications"
                })

        elif parsed.path == "/registry/semantic":
            """Semantic search across registry entities (people, datasets, projects)."""
            if not semantic_registry:
                self._send_response({"error": "Semantic registry not available"}, 503)
                return

            query = params.get("q", [""])[0]
            if not query:
                self._send_response({"error": "Missing query parameter 'q'"}, 400)
                return

            top_k = int(params.get("top_k", [10])[0])
            entity_type = params.get("type", [None])[0]  # person, dataset, project, funding
            merge_structured = params.get("merge", ["false"])[0].lower() == "true"

            # Semantic search
            semantic_results = semantic_registry.search(
                query=query,
                top_k=top_k,
                entity_type=entity_type,
            )

            response = {
                "query": query,
                "entity_type": entity_type,
                "results": semantic_results,
                "total": len(semantic_results),
            }

            # Optionally merge with structured results
            if merge_structured and registry_rag:
                registry_rag.reload_if_changed()
                structured_result = registry_rag.query(query)
                structured_results = structured_result.get("structured_results", [])

                merged = merge_with_structured(semantic_results, structured_results)
                response["merged_results"] = merged
                response["structured_count"] = len(structured_results)

            self._send_response(response)

        elif parsed.path == "/registry/semantic/similar":
            """Find entities similar to a given entity."""
            if not semantic_registry:
                self._send_response({"error": "Semantic registry not available"}, 503)
                return

            entity_id = params.get("id", [""])[0]
            if not entity_id:
                self._send_response({"error": "Missing entity ID parameter 'id'"}, 400)
                return

            top_k = int(params.get("top_k", [5])[0])
            same_type = params.get("same_type", ["false"])[0].lower() == "true"

            # Get the source entity
            entity = semantic_registry.get_entity(entity_id)
            if not entity:
                self._send_response({"error": f"Entity '{entity_id}' not found"}, 404)
                return

            # Find similar entities
            similar = semantic_registry.find_similar(
                entity_id=entity_id,
                top_k=top_k,
                same_type_only=same_type,
            )

            self._send_response({
                "entity": entity,
                "similar": similar,
                "same_type_only": same_type,
            })

        elif parsed.path == "/registry/semantic/stats":
            """Get semantic registry index statistics."""
            if not semantic_registry:
                self._send_response({"error": "Semantic registry not available"}, 503)
                return

            self._send_response(semantic_registry.stats())

        elif parsed.path == "/registry/semantic/rebuild":
            """Rebuild the semantic registry index."""
            if not semantic_registry:
                self._send_response({"error": "Semantic registry not available"}, 503)
                return

            try:
                semantic_registry.build_index()
                semantic_registry.save_index("Data/indexes/semantic_registry.pkl")
                self._send_response({
                    "status": "rebuilt",
                    "stats": semantic_registry.stats(),
                })
            except Exception as e:
                self._send_response({"error": str(e)}, 500)

        elif parsed.path == "/hybrid":
            """Hybrid search: combines registry + semantic RAG."""
            if not registry_rag:
                # Fall back to regular search
                self._send_response({"error": "Registry not available, use /search instead"}, 503)
                return

            query = params.get("q", [""])[0]
            top_k = int(params.get("top_k", [5])[0])

            if not query:
                self._send_response({"error": "Missing query parameter 'q'"}, 400)
                return

            # Check for index and registry updates (hot-reload)
            rag.reload_if_changed()
            registry_rag.reload_if_changed()

            # Get structured results from registry
            registry_result = registry_rag.query(query)

            # Get semantic results from RAG
            filters = self._parse_filters(params)
            semantic_results = rag.search(query, top_k, filters=filters)

            # Combine results
            response = {
                "query": query,
                "classification": registry_result["classification"],
                "structured_results": registry_result["structured_results"],
                "structured_answer": registry_result.get("answer"),
                "semantic_results": semantic_results,
                "hybrid": True
            }

            self._send_response(response)

        # =====================================================================
        # Hypergraph Endpoints
        # =====================================================================
        elif parsed.path == "/hypergraph/stats":
            """Get hypergraph statistics."""
            if not hypergraph:
                self._send_response({"error": "Hypergraph not available"}, 503)
                return
            stats = hypergraph.stats()
            self._send_response(stats)

        elif parsed.path == "/hypergraph/collaborators":
            """Find collaborators of a person."""
            if not hypergraph:
                self._send_response({"error": "Hypergraph not available"}, 503)
                return
            person_id = params.get("person", [""])[0]
            if not person_id:
                self._send_response({"error": "Missing 'person' parameter"}, 400)
                return
            # Try to find the person node
            person_node = hypergraph.get_node(person_id) or hypergraph.get_node(f"person-{person_id}")
            if not person_node:
                self._send_response({"error": f"Person '{person_id}' not found", "suggestion": "Try using the full ID like 'person-james-evans'"}, 404)
                return
            collabs = hypergraph.collaborators_of(person_node.id)
            self._send_response({
                "person": {"id": person_node.id, "name": person_node.name},
                "collaborators": [{"id": c.id, "name": c.name} for c in collabs]
            })

        elif parsed.path == "/hypergraph/outputs":
            """Find what a person has produced."""
            if not hypergraph:
                self._send_response({"error": "Hypergraph not available"}, 503)
                return
            person_id = params.get("person", [""])[0]
            if not person_id:
                self._send_response({"error": "Missing 'person' parameter"}, 400)
                return
            person_node = hypergraph.get_node(person_id) or hypergraph.get_node(f"person-{person_id}")
            if not person_node:
                self._send_response({"error": f"Person '{person_id}' not found"}, 404)
                return
            outputs = hypergraph.what_produced(person_node.id)
            self._send_response({
                "person": {"id": person_node.id, "name": person_node.name},
                "outputs": [{"id": o.id, "name": o.name, "type": o.type.value if hasattr(o.type, 'value') else str(o.type)} for o in outputs]
            })

        elif parsed.path == "/hypergraph/terminal":
            """Find terminal nodes (dead ends - outputs never reused)."""
            if not hypergraph:
                self._send_response({"error": "Hypergraph not available"}, 503)
                return
            terminal = hypergraph.terminal_nodes()
            # Group by type
            by_type = {}
            for node in terminal:
                t = node.type.value if hasattr(node.type, 'value') else str(node.type)
                if t not in by_type:
                    by_type[t] = []
                by_type[t].append({"id": node.id, "name": node.name})
            self._send_response({
                "total": len(terminal),
                "by_type": {k: {"count": len(v), "examples": v[:5]} for k, v in by_type.items()}
            })

        elif parsed.path == "/hypergraph/search":
            """Search nodes by name."""
            if not hypergraph:
                self._send_response({"error": "Hypergraph not available"}, 503)
                return
            query = params.get("q", [""])[0].lower()
            if not query:
                self._send_response({"error": "Missing 'q' parameter"}, 400)
                return
            # Search all nodes
            matches = []
            for node in hypergraph.nodes.values():
                if query in node.name.lower() or query in node.id.lower():
                    matches.append({
                        "id": node.id,
                        "name": node.name,
                        "type": node.type.value if hasattr(node.type, 'value') else str(node.type)
                    })
            self._send_response({"query": query, "matches": matches[:20]})

        else:
            self._send_response({
                "name": "CHORUS RAG Server",
                "version": "2.4",
                "features": {
                    "hybrid_retrieval": "70% semantic + 30% keyword (BM25)",
                    "llm_reformulation": "Query expansion using Claude",
                    "cross_encoder_reranking": "Re-score top results with cross-encoder model",
                    "mmr_diversity": f"Maximal Marginal Relevance (λ={MMR_LAMBDA})",
                    "metadata_filtering": "Filter by doc_type, year, file_pattern",
                    "registry_integration": "Structured lookup of people, projects, funding" if registry_rag else "Not available",
                    "semantic_registry": "Embedding-based semantic search for registry entities" if semantic_registry else "Not available"
                },
                "endpoints": {
                    "/search?q=query&top_k=5": {
                        "description": "Semantic search documents",
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
                    "/hybrid?q=query": {
                        "description": "Hybrid search: structured registry + semantic RAG",
                        "params": {
                            "q": "Query (e.g., 'Who is Jake?', 'List PhD students', 'What is APTO?')",
                            "top_k": "Number of semantic results (default: 5)"
                        }
                    },
                    "/registry?q=query": "Structured registry query only (supports topic and publication queries)",
                    "/registry/stats": "Registry statistics",
                    "/registry/topics": "List all unique research topics with counts (optional: ?search=network)",
                    "/registry/topic-search?q=topic": "Search for people by research topic",
                    "/registry/person?name=Jake": "Lookup a specific person",
                    "/registry/publications?q=search": "Search publications by title, author, venue, or topic (optional: ?limit=20)",
                    "/registry/reload": "Force reload registry (hot-reload)",
                    "/registry/semantic?q=query": {
                        "description": "Semantic search across registry entities (people, datasets, projects, funding)",
                        "params": {
                            "q": "Natural language query (e.g., 'researchers working on network science')",
                            "top_k": "Number of results (default: 10)",
                            "type": "Filter by entity type: person|dataset|project|funding (optional)",
                            "merge": "Merge with structured results (default: false)"
                        }
                    },
                    "/registry/semantic/similar?id=entity-id": {
                        "description": "Find entities similar to a given entity",
                        "params": {
                            "id": "Entity ID (e.g., 'person-james-evans', 'dataset-mag-dec-2021')",
                            "top_k": "Number of similar entities (default: 5)",
                            "same_type": "Only return same entity type (default: false)"
                        }
                    },
                    "/registry/semantic/stats": "Get semantic registry index statistics",
                    "/registry/semantic/rebuild": "Rebuild the semantic registry index",
                    "/context?q=question&max_sources=5": "Get formatted context for question",
                    "/sources": "List all sources (optional: ?doc_type=proposal)",
                    "/doc_types": "List document types with counts",
                    "/health": "Health check",
                    "/reload": "Force reload index (after incremental updates)"
                }
            })

    def log_message(self, format, *args):
        print(f"[{self.log_date_time_string()}] {args[0]}")


if __name__ == "__main__":
    port = 8765
    server = HTTPServer(("0.0.0.0", port), RAGHandler)
    print(f"\nRAG HTTP Server running on http://localhost:{port}")
    print("Endpoints:")
    print(f"  GET /search?q=your+query&top_k=5")
    print(f"  GET /context?q=your+question&max_sources=5")
    print(f"  GET /sources")
    print(f"  GET /health")
    print("\nPress Ctrl+C to stop\n")
    server.serve_forever()
