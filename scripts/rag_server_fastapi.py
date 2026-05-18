#!/usr/bin/env python3
"""
CHORUS RAG Server - FastAPI Edition

Production-ready ASGI server with:
- Concurrent request handling via async
- Auto-generated OpenAPI docs at /docs
- Request validation via Pydantic
- CORS support

Usage:
    python rag_server_fastapi.py                    # Development mode (auto-reload)
    CHORUS_ENV=production python rag_server_fastapi.py  # Production mode (4 workers)

Or with uvicorn directly:
    uvicorn rag_server_fastapi:app --reload --port 8765
"""

import os
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, Query, HTTPException, Path as PathParam
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# =============================================================================
# Load environment and existing RAG components
# =============================================================================

from config import load_dotenv

load_dotenv()

# Import RAG components from core module
from rag_core import (
    RAGServer,
    ChunkWithMetadata,
    infer_doc_type,
    ENABLE_MMR,
    MMR_LAMBDA,
    RERANK_MODEL,
    REGISTRY_AVAILABLE,
    SEMANTIC_REGISTRY_AVAILABLE,
    HYPERGRAPH_AVAILABLE,
)

# Optional imports
if REGISTRY_AVAILABLE:
    from registry import RegistryRAG
if SEMANTIC_REGISTRY_AVAILABLE:
    from registry import SemanticRegistryIndex, merge_with_structured
if HYPERGRAPH_AVAILABLE:
    from hypergraph import load_hypergraph

# Query caching
try:
    from cache import QueryCache, get_cache
    from cache.query_cache import compute_source_hash
    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False
    print("Warning: cache module not available, query caching disabled")

    # Stub for compute_source_hash when cache is not available
    def compute_source_hash(path: str) -> str:
        return ""

# Reformulation caching
try:
    from cache import ReformulationCache, get_reformulation_cache
    REFORMULATION_CACHE_AVAILABLE = True
except ImportError:
    REFORMULATION_CACHE_AVAILABLE = False
    print("Warning: reformulation cache not available")

# Query routing
try:
    from registry.query_router import QueryRouter, RoutingDecision, get_router
    QUERY_ROUTER_AVAILABLE = True
except ImportError:
    QUERY_ROUTER_AVAILABLE = False
    print("Warning: query router not available")

# Latency instrumentation
try:
    from utils.instrumentation import (
        Timer, LatencyTracker, RequestContext,
        get_tracker, reset_tracker, get_stats
    )
    INSTRUMENTATION_AVAILABLE = True
except ImportError:
    INSTRUMENTATION_AVAILABLE = False
    print("Warning: instrumentation not available")

# Confidence scoring
try:
    from agents.confidence import (
        ConfidenceCalculator,
        extract_sources_from_rag_result,
        SourceType,
    )
    CONFIDENCE_AVAILABLE = True
except ImportError:
    CONFIDENCE_AVAILABLE = False
    print("Warning: confidence module not available, confidence scoring disabled")

# =============================================================================
# JSON Sanitization (handle NaN values)
# =============================================================================

import math

def sanitize_for_json(obj: Any) -> Any:
    """
    Recursively sanitize an object for JSON serialization.

    Replaces NaN and Infinity float values with None, as these are not
    valid JSON values and will cause serialization errors.

    Args:
        obj: Any Python object (dict, list, primitive, etc.)

    Returns:
        Sanitized object safe for JSON serialization
    """
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(item) for item in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    else:
        return obj

# =============================================================================
# Pydantic Response Models
# =============================================================================

class SearchResult(BaseModel):
    """A single search result."""
    text: str
    citation: str
    source_file: str
    doc_type: str
    score: float
    timestamp_start: Optional[str] = None
    timestamp_end: Optional[str] = None
    speakers: Optional[str] = None
    rerank_score: Optional[float] = None

class SearchResponse(BaseModel):
    """Response from /search endpoint."""
    query: str
    results: List[SearchResult]
    llm_reformulation: bool
    reranking: bool
    mmr_diversity: bool
    filters: Optional[Dict[str, Any]] = None

class ContextResponse(BaseModel):
    """Response from /context endpoint."""
    question: str
    context: str
    citations: List[str]
    filters: Optional[Dict[str, Any]] = None
    reranking: bool
    mmr_diversity: bool

class HealthResponse(BaseModel):
    """Response from /health endpoint."""
    status: str
    chunks: int
    llm_reformulation_enabled: bool
    reranking_enabled: bool
    rerank_model: Optional[str]
    mmr_enabled: bool
    mmr_lambda: float

class ReloadResponse(BaseModel):
    """Response from /reload endpoint."""
    status: str
    previous_chunks: int
    current_chunks: int
    timestamp: str

class SourceInfo(BaseModel):
    """Information about an indexed source."""
    file: str
    chunks: int
    doc_type: str

class SourcesResponse(BaseModel):
    """Response from /sources endpoint."""
    total_sources: int
    sources: List[SourceInfo]

class DocTypesResponse(BaseModel):
    """Response from /doc_types endpoint."""
    doc_types: Dict[str, int]


class SourceModel(BaseModel):
    """A single information source."""
    type: str  # "registry", "document", "web", "inference"
    name: str
    citation: str
    relevance: float = 1.0
    url: Optional[str] = None


class ConfidenceModel(BaseModel):
    """Confidence scoring information."""
    level: str  # "high", "medium", "low", "uncertain"
    score: float  # 0.0 to 1.0
    reasoning: str
    sources: List[SourceModel] = []
    factors: Dict[str, float] = {}


class CacheStatsResponse(BaseModel):
    """Response from /cache/stats endpoint."""
    hits: int
    misses: int
    evictions: int
    invalidations: int
    size: int
    max_size: int
    hit_rate_percent: float
    ttl_seconds: int
    backend: str
    enabled: bool
    uptime_seconds: float


class CacheClearResponse(BaseModel):
    """Response from /cache/clear endpoint."""
    status: str
    entries_cleared: int
    timestamp: str


class SynthesisSource(BaseModel):
    """A source used in answer synthesis."""
    text: str
    citation: str
    source_file: str
    doc_type: str
    score: float


class SynthesisResponse(BaseModel):
    """Response from /synthesize endpoint."""
    query: str
    answer: str
    sources: List[SynthesisSource]
    model: str
    latency_ms: float
    search_latency_ms: float
    synthesis_latency_ms: float


class SynthesisRequest(BaseModel):
    """Request body for /synthesize endpoint."""
    query: str = Field(..., description="The question to answer")
    top_k: int = Field(5, ge=1, le=20, description="Number of sources to use")
    doc_type: Optional[str] = Field(None, description="Filter by document type")
    year: Optional[int] = Field(None, description="Filter by year")
    max_tokens: int = Field(1024, ge=100, le=4096, description="Max tokens in answer")


# =============================================================================
# Global State (initialized at startup)
# =============================================================================

rag: Optional[RAGServer] = None
registry_rag = None
semantic_registry = None
hypergraph = None
query_cache: Optional["QueryCache"] = None
confidence_calculator: Optional["ConfidenceCalculator"] = None
query_router = None
reformulation_cache = None
latency_tracker = None

# =============================================================================
# FastAPI Lifespan (startup/shutdown)
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize RAG components on startup."""
    global rag, registry_rag, semantic_registry, hypergraph, query_cache, confidence_calculator
    global query_router, reformulation_cache, latency_tracker

    print("\n" + "=" * 60)
    print("CHORUS RAG Server v2.6 (FastAPI) - Optimized")
    print("=" * 60)

    # Load RAG
    print("\nLoading RAG index...")
    rag = RAGServer("Data/rag_indexes/hybrid")
    rag.load()

    # Initialize query cache
    if CACHE_AVAILABLE:
        try:
            query_cache = get_cache()
            # Set initial source hash for cache invalidation
            source_hash = compute_source_hash("Data/rag_indexes/hybrid")
            query_cache.set_source_hash(source_hash)
            stats = query_cache.stats()
            print(f"Query cache initialized: backend={stats['backend']}, max_size={stats['max_size']}, ttl={stats['ttl_seconds']}s")
        except Exception as e:
            print(f"Failed to initialize query cache: {e}")
            query_cache = None

    # Load Registry (if available)
    if REGISTRY_AVAILABLE:
        try:
            registry_rag = RegistryRAG()
            print(f"Registry loaded: {registry_rag.get_registry_stats()}")
        except Exception as e:
            print(f"Failed to load registry: {e}")

    # Load Semantic Registry (if available)
    if SEMANTIC_REGISTRY_AVAILABLE:
        try:
            semantic_registry_path = Path("Data/indexes/semantic_registry.pkl")
            if semantic_registry_path.exists():
                semantic_registry = SemanticRegistryIndex()
                semantic_registry.load_index(str(semantic_registry_path))
                stats = semantic_registry.stats()
                print(f"Semantic registry loaded: {stats['total_entities']} entities")
            else:
                print("Building semantic registry index (first run)...")
                semantic_registry = SemanticRegistryIndex()
                semantic_registry.build_index()
                semantic_registry.save_index(str(semantic_registry_path))
                print("Semantic registry built and saved")
        except Exception as e:
            print(f"Failed to load semantic registry: {e}")

    # Load Hypergraph (if available)
    if HYPERGRAPH_AVAILABLE:
        try:
            hypergraph_path = Path("Data/hypergraph.json")
            if hypergraph_path.exists():
                hypergraph = load_hypergraph(str(hypergraph_path))
                stats = hypergraph.stats()
                print(f"Hypergraph loaded: {stats['total_nodes']} nodes, {stats['total_edges']} edges")
        except Exception as e:
            print(f"Failed to load hypergraph: {e}")

    # Initialize confidence calculator
    if CONFIDENCE_AVAILABLE:
        try:
            confidence_calculator = ConfidenceCalculator(enable_llm_validation=False)
            print("Confidence scoring enabled")
        except Exception as e:
            print(f"Failed to initialize confidence calculator: {e}")

    # Initialize query router (for fast-path routing)
    if QUERY_ROUTER_AVAILABLE:
        try:
            query_router = get_router()
            print("Query router enabled (fast-path for simple queries)")
        except Exception as e:
            print(f"Failed to initialize query router: {e}")

    # Initialize reformulation cache
    if REFORMULATION_CACHE_AVAILABLE:
        try:
            reformulation_cache = get_reformulation_cache()
            stats = reformulation_cache.stats()
            print(f"Reformulation cache enabled: max_size={stats['max_size']}, ttl={stats['ttl_seconds']}s")
        except Exception as e:
            print(f"Failed to initialize reformulation cache: {e}")

    # Initialize latency tracker
    if INSTRUMENTATION_AVAILABLE:
        try:
            latency_tracker = get_tracker()
            print("Latency instrumentation enabled")
        except Exception as e:
            print(f"Failed to initialize latency tracker: {e}")

    print("\n" + "=" * 60)
    print("Server ready!")
    print("=" * 60 + "\n")

    yield

    # Cleanup on shutdown (if needed)
    print("Shutting down...")

# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="CHORUS RAG Server",
    version="2.5",
    description="Hybrid semantic + BM25 retrieval with registry integration",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# Helper Functions
# =============================================================================

def build_filters(
    doc_type: Optional[str],
    year: Optional[int],
    file_pattern: Optional[str],
) -> Optional[Dict[str, Any]]:
    """Build filters dict from query parameters."""
    filters = {}
    if doc_type:
        filters["doc_type"] = doc_type
    if year:
        filters["year"] = year
    if file_pattern:
        filters["file_pattern"] = file_pattern
    return filters if filters else None

# =============================================================================
# Core Endpoints
# =============================================================================

@app.get("/", include_in_schema=False)
async def root():
    """Return API information (also available at /docs)."""
    return {
        "name": "CHORUS RAG Server",
        "version": "2.5",
        "docs": "/docs",
        "features": {
            "hybrid_retrieval": "70% semantic + 30% keyword (BM25)",
            "llm_reformulation": "Query expansion using Claude",
            "cross_encoder_reranking": f"Re-score with {RERANK_MODEL}",
            "mmr_diversity": f"Maximal Marginal Relevance (λ={MMR_LAMBDA})",
            "registry_integration": "Structured lookup" if REGISTRY_AVAILABLE else "Not available",
            "semantic_registry": "Embedding-based search" if semantic_registry else "Not available",
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health():
    """Check server health and configuration."""
    return HealthResponse(
        status="ok",
        chunks=len(rag.chunks) if rag else 0,
        llm_reformulation_enabled=rag.llm_reformulator.enabled if rag else False,
        reranking_enabled=rag.reranker is not None if rag else False,
        rerank_model=RERANK_MODEL if (rag and rag.reranker) else None,
        mmr_enabled=ENABLE_MMR,
        mmr_lambda=MMR_LAMBDA,
    )

@app.post("/reload", response_model=ReloadResponse)
async def reload_index():
    """Force reload the RAG index and invalidate cache."""
    if not rag:
        raise HTTPException(status_code=503, detail="RAG not initialized")

    result = rag.force_reload()

    # Invalidate cache on index reload
    if query_cache:
        new_source_hash = compute_source_hash("Data/rag_indexes/hybrid") if CACHE_AVAILABLE else None
        query_cache.invalidate(new_source_hash)

    return ReloadResponse(**result)

@app.get("/search")
async def search(
    q: str = Query(..., description="Search query"),
    top_k: int = Query(5, ge=1, le=50, description="Number of results"),
    llm: bool = Query(True, description="Use LLM reformulation"),
    rerank: bool = Query(True, description="Use cross-encoder reranking"),
    mmr: bool = Query(True, description="Use MMR diversity"),
    doc_type: Optional[str] = Query(None, description="Filter by doc type"),
    year: Optional[int] = Query(None, description="Filter by year"),
    file_pattern: Optional[str] = Query(None, description="Filter by file pattern"),
    cache: bool = Query(True, description="Use query cache"),
):
    """
    Semantic search with hybrid retrieval (semantic + BM25).

    Supports LLM-based query reformulation, cross-encoder reranking,
    MMR diversity selection, and query caching for improved latency.
    """
    if not rag:
        raise HTTPException(status_code=503, detail="RAG not initialized")

    # Check for index updates
    if rag.reload_if_changed() and query_cache:
        # Index was reloaded, invalidate cache
        new_source_hash = compute_source_hash("Data/rag_indexes/hybrid") if CACHE_AVAILABLE else None
        query_cache.invalidate(new_source_hash)

    # Build filters
    filters = build_filters(doc_type, year, file_pattern)

    # Build cache key from query parameters
    cache_params = {
        "top_k": top_k,
        "llm": llm,
        "rerank": rerank,
        "mmr": mmr,
        "doc_type": doc_type,
        "year": year,
        "file_pattern": file_pattern,
    }

    # Try to get cached result
    cache_key = None
    if cache and query_cache:
        cache_key = query_cache.generate_key("/search", q, cache_params)
        cached = query_cache.get(cache_key)
        if cached:
            # Return cached response with cache_hit indicator
            response = cached.result.copy()
            response["cache_hit"] = True
            return response

    # Execute search
    results = rag.search(
        query=q,
        top_k=top_k,
        filters=filters,
        use_llm_reformulation=llm,
        use_reranking=rerank,
        use_mmr=mmr,
    )

    response = {
        "query": q,
        "results": results,
        "llm_reformulation": llm and rag.llm_reformulator.enabled,
        "reranking": rerank and rag.reranker is not None,
        "mmr_diversity": mmr and ENABLE_MMR,
        "filters": filters,
        "cache_hit": False,
    }

    # Cache the result
    if cache and query_cache and cache_key:
        query_cache.set(cache_key, q, response, endpoint="/search")

    return response


@app.get("/semantic")
async def semantic_search(
    q: str = Query(..., description="Search query"),
    top_k: int = Query(5, ge=1, le=50, description="Number of results"),
    rerank: bool = Query(True, description="Use cross-encoder reranking"),
    mmr: bool = Query(True, description="Use MMR diversity"),
    doc_type: Optional[str] = Query(None, description="Filter by doc type"),
    year: Optional[int] = Query(None, description="Filter by year"),
):
    """
    Pure semantic (embedding-based) search without BM25 component.

    Use this when you want results based purely on semantic similarity,
    without keyword matching influence.
    """
    if not rag:
        raise HTTPException(status_code=503, detail="RAG not initialized")

    filters = build_filters(doc_type, year, None)

    results = rag.search_semantic_only(
        query=q,
        top_k=top_k,
        filters=filters,
        use_reranking=rerank,
        use_mmr=mmr,
    )

    return {
        "query": q,
        "results": results,
        "search_type": "semantic",
        "reranking": rerank and rag.reranker is not None,
        "mmr_diversity": mmr and ENABLE_MMR,
        "filters": filters,
    }


@app.get("/bm25")
async def bm25_search(
    q: str = Query(..., description="Search query"),
    top_k: int = Query(5, ge=1, le=50, description="Number of results"),
    doc_type: Optional[str] = Query(None, description="Filter by doc type"),
    year: Optional[int] = Query(None, description="Filter by year"),
):
    """
    Pure BM25 (keyword-based) search without semantic component.

    Use this when you want results based purely on keyword matching,
    such as for exact name lookups or when you need precise term matches.
    """
    if not rag:
        raise HTTPException(status_code=503, detail="RAG not initialized")

    filters = build_filters(doc_type, year, None)

    results = rag.search_bm25_only(
        query=q,
        top_k=top_k,
        filters=filters,
    )

    return {
        "query": q,
        "results": results,
        "search_type": "bm25",
        "filters": filters,
    }


@app.post("/synthesize", response_model=SynthesisResponse)
async def synthesize_answer(request: SynthesisRequest):
    """
    Search for relevant sources and synthesize an answer using Claude.

    This endpoint combines RAG retrieval with LLM synthesis to provide
    a complete answer with citations. Unlike /search which returns raw
    chunks, this endpoint generates a coherent answer from the sources.

    The answer includes inline citations and a list of sources used.
    """
    import time
    import ssl
    import urllib.request
    import json as json_module

    if not rag:
        raise HTTPException(status_code=503, detail="RAG not initialized")

    start_time = time.time()

    # Build filters
    filters = build_filters(request.doc_type, request.year, None)

    # Search for relevant sources
    search_start = time.time()
    results = rag.search(
        query=request.query,
        top_k=request.top_k,
        filters=filters,
        use_llm_reformulation=True,
        use_reranking=True,
        use_mmr=True,
    )
    search_time = (time.time() - search_start) * 1000

    if not results:
        raise HTTPException(
            status_code=404,
            detail="No relevant sources found for this query"
        )

    # Format sources for the prompt
    sources_text = ""
    for i, r in enumerate(results, 1):
        sources_text += f"\n[Source {i}: {r['citation']}]\n{r['text']}\n"

    # Build synthesis prompt
    synthesis_prompt = f"""You are a knowledgeable research assistant. Answer the following question using ONLY the information provided in the sources below. If the sources don't contain enough information to fully answer the question, say so.

Question: {request.query}

Sources:
{sources_text}

Instructions:
1. Provide a clear, accurate answer based on the sources
2. Cite sources using [Source N] format inline
3. If sources contain conflicting information, acknowledge it
4. If the sources don't fully answer the question, indicate what's missing
5. Keep your answer concise but complete

Answer:"""

    # Call Claude for synthesis
    synthesis_start = time.time()
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")

    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="ANTHROPIC_API_KEY not configured for synthesis"
        )

    try:
        # Try async with httpx first
        try:
            import httpx

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    json={
                        "model": "claude-3-5-sonnet-20241022",
                        "max_tokens": request.max_tokens,
                        "messages": [{"role": "user", "content": synthesis_prompt}]
                    },
                    headers={
                        "Content-Type": "application/json",
                        "x-api-key": api_key,
                        "anthropic-version": "2023-06-01"
                    }
                )
                response.raise_for_status()
                result = response.json()

        except ImportError:
            # Fall back to sync urllib
            try:
                import certifi
                ssl_ctx = ssl.create_default_context(cafile=certifi.where())
            except ImportError:
                ssl_ctx = ssl.create_default_context()

            req_body = json_module.dumps({
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": request.max_tokens,
                "messages": [{"role": "user", "content": synthesis_prompt}]
            }).encode()

            req = urllib.request.Request(
                "https://api.anthropic.com/v1/messages",
                data=req_body,
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01"
                }
            )

            with urllib.request.urlopen(req, timeout=30, context=ssl_ctx) as resp:
                result = json_module.loads(resp.read().decode())

        # Extract answer
        if "content" in result and result["content"]:
            answer = result["content"][0].get("text", "")
        else:
            raise HTTPException(
                status_code=500,
                detail="Failed to get answer from Claude"
            )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Synthesis failed: {str(e)}"
        )

    synthesis_time = (time.time() - synthesis_start) * 1000
    total_time = (time.time() - start_time) * 1000

    # Format sources for response
    sources = [
        SynthesisSource(
            text=r["text"][:500] + "..." if len(r["text"]) > 500 else r["text"],
            citation=r["citation"],
            source_file=r["source_file"],
            doc_type=r["doc_type"],
            score=r["score"]
        )
        for r in results
    ]

    return SynthesisResponse(
        query=request.query,
        answer=answer,
        sources=sources,
        model="claude-3-5-sonnet-20241022",
        latency_ms=round(total_time, 2),
        search_latency_ms=round(search_time, 2),
        synthesis_latency_ms=round(synthesis_time, 2)
    )


@app.get("/context", response_model=ContextResponse)
async def get_context(
    q: str = Query(..., description="Question to find context for"),
    max_sources: int = Query(5, ge=1, le=20, description="Maximum sources"),
    llm: bool = Query(True, description="Use LLM reformulation"),
    rerank: bool = Query(True, description="Use cross-encoder reranking"),
    mmr: bool = Query(True, description="Use MMR diversity"),
    doc_type: Optional[str] = Query(None, description="Filter by doc type"),
    year: Optional[int] = Query(None, description="Filter by year"),
    file_pattern: Optional[str] = Query(None, description="Filter by file pattern"),
):
    """Get formatted context for a question."""
    if not rag:
        raise HTTPException(status_code=503, detail="RAG not initialized")

    filters = build_filters(doc_type, year, file_pattern)
    results = rag.search(
        query=q,
        top_k=max_sources,
        filters=filters,
        use_llm_reformulation=llm,
        use_reranking=rerank,
        use_mmr=mmr,
    )

    context = "\n\n".join(f"[{r['citation']}]\n{r['text']}" for r in results)

    return ContextResponse(
        question=q,
        context=context,
        citations=[r["citation"] for r in results],
        filters=filters,
        reranking=rerank and rag.reranker is not None,
        mmr_diversity=mmr and ENABLE_MMR,
    )

@app.get("/sources", response_model=SourcesResponse)
async def list_sources(
    doc_type: Optional[str] = Query(None, description="Filter by doc type"),
):
    """List all indexed sources."""
    if not rag:
        raise HTTPException(status_code=503, detail="RAG not initialized")

    sources = rag.list_sources()

    if doc_type:
        sources = [s for s in sources if s["doc_type"].lower() == doc_type.lower()]

    return SourcesResponse(
        total_sources=len(sources),
        sources=[SourceInfo(**s) for s in sources[:50]],
    )

@app.get("/doc_types", response_model=DocTypesResponse)
async def list_doc_types():
    """List document types with counts."""
    if not rag:
        raise HTTPException(status_code=503, detail="RAG not initialized")

    return DocTypesResponse(doc_types=rag.list_doc_types())

# =============================================================================
# Registry Endpoints
# =============================================================================

@app.get("/registry")
async def query_registry(
    q: str = Query(..., description="Registry query"),
    cache: bool = Query(True, description="Use query cache"),
):
    """Query the structured registry with caching."""
    if not registry_rag:
        raise HTTPException(status_code=503, detail="Registry not available")

    # Check for registry updates with cache invalidation
    if registry_rag.reload_if_changed() and query_cache:
        query_cache.invalidate()

    # Try to get cached result
    cache_key = None
    if cache and query_cache:
        cache_key = query_cache.generate_key("/registry", q, {})
        cached = query_cache.get(cache_key)
        if cached:
            response = cached.result.copy()
            response["cache_hit"] = True
            return response

    result = registry_rag.query(q)
    result["cache_hit"] = False

    # Cache the result
    if cache and query_cache and cache_key:
        query_cache.set(cache_key, q, result, endpoint="/registry")

    return result

@app.post("/registry/reload")
async def reload_registry():
    """Force reload the registry and invalidate cache."""
    if not registry_rag:
        raise HTTPException(status_code=503, detail="Registry not available")

    result = registry_rag.force_reload()

    # Invalidate cache on registry reload
    if query_cache:
        query_cache.invalidate()

    return result

@app.get("/registry/stats")
async def registry_stats():
    """Get registry statistics."""
    if not registry_rag:
        raise HTTPException(status_code=503, detail="Registry not available")

    return {
        "stats": registry_rag.get_registry_stats(),
        "available": True,
    }

@app.get("/registry/topics")
async def list_topics(
    search: Optional[str] = Query(None, description="Search filter"),
):
    """List all unique research topics with counts."""
    if not registry_rag:
        raise HTTPException(status_code=503, detail="Registry not available")

    registry_rag.reload_if_changed()
    topic_counts = registry_rag.lookup.get_all_topics()

    sorted_topics = sorted(
        [{"topic": t, "count": c} for t, c in topic_counts.items()],
        key=lambda x: (-x["count"], x["topic"]),
    )

    if search:
        search_lower = search.lower()
        sorted_topics = [t for t in sorted_topics if search_lower in t["topic"].lower()]

    return {
        "total_topics": len(sorted_topics),
        "topics": sorted_topics,
        "people_with_topics": sum(
            1 for m in registry_rag.lookup.registry.get("people", {}).get("members", [])
            if m.get("openalex", {}).get("topics")
        ),
    }

@app.get("/registry/topic-search")
async def search_by_topic(
    q: str = Query(None, description="Topic to search for"),
    topic: str = Query(None, description="Topic to search for (alias)"),
):
    """Search for people by research topic."""
    if not registry_rag:
        raise HTTPException(status_code=503, detail="Registry not available")

    search_topic = q or topic
    if not search_topic:
        raise HTTPException(status_code=400, detail="Missing topic parameter 'q' or 'topic'")

    registry_rag.reload_if_changed()
    matches = registry_rag.lookup.find_people_by_topic(search_topic)

    return {
        "query": search_topic,
        "found": len(matches),
        "results": [
            {
                "name": m["person"].get("name"),
                "role": m["person"].get("role"),
                "institution": m["person"].get("institution"),
                "matched_topics": m["matched_topics"],
                "all_topics": m["person"].get("openalex", {}).get("topics", []),
            }
            for m in matches
        ],
    }

@app.get("/registry/person")
async def lookup_person(
    name: str = Query(..., description="Person name to lookup"),
):
    """Lookup a specific person."""
    if not registry_rag:
        raise HTTPException(status_code=503, detail="Registry not available")

    registry_rag.reload_if_changed()
    person = registry_rag.lookup.find_person(name)

    if person:
        return {"found": True, "person": person}
    else:
        return {"found": False, "query": name}

@app.get("/registry/publications")
async def search_publications(
    q: Optional[str] = Query(None, description="Search query"),
    limit: int = Query(20, ge=1, le=100, description="Result limit"),
):
    """Search publications by title, author, venue, or topic."""
    if not registry_rag:
        raise HTTPException(status_code=503, detail="Registry not available")

    registry_rag.reload_if_changed()

    if q:
        publications = registry_rag.lookup.find_publications(q)
        return {
            "query": q,
            "found": len(publications),
            "results": publications[:limit],
        }
    else:
        all_pubs = registry_rag.lookup.get_all_publications()
        return {
            "total": len(all_pubs),
            "results": all_pubs[:limit],
            "note": "Add ?q=search_term to search publications",
        }

# =============================================================================
# Semantic Registry Endpoints
# =============================================================================

@app.get("/registry/semantic")
async def semantic_search(
    q: str = Query(..., description="Natural language query"),
    top_k: int = Query(10, ge=1, le=50, description="Number of results"),
    type: Optional[str] = Query(None, description="Entity type filter"),
    merge: bool = Query(False, description="Merge with structured results"),
):
    """Semantic search across registry entities."""
    if not semantic_registry:
        raise HTTPException(status_code=503, detail="Semantic registry not available")

    results = semantic_registry.search(query=q, top_k=top_k, entity_type=type)

    response = {
        "query": q,
        "entity_type": type,
        "results": results,
        "total": len(results),
    }

    if merge and registry_rag:
        registry_rag.reload_if_changed()
        structured_result = registry_rag.query(q)
        structured_results = structured_result.get("structured_results", [])
        merged = merge_with_structured(results, structured_results)
        response["merged_results"] = merged
        response["structured_count"] = len(structured_results)

    return response

@app.get("/registry/semantic/similar")
async def find_similar(
    id: str = Query(..., description="Entity ID"),
    top_k: int = Query(5, ge=1, le=20, description="Number of similar entities"),
    same_type: bool = Query(False, description="Only same entity type"),
):
    """Find entities similar to a given entity."""
    if not semantic_registry:
        raise HTTPException(status_code=503, detail="Semantic registry not available")

    entity = semantic_registry.get_entity(id)
    if not entity:
        raise HTTPException(status_code=404, detail=f"Entity '{id}' not found")

    similar = semantic_registry.find_similar(entity_id=id, top_k=top_k, same_type_only=same_type)

    return {
        "entity": entity,
        "similar": similar,
        "same_type_only": same_type,
    }

@app.get("/registry/semantic/stats")
async def semantic_stats():
    """Get semantic registry statistics."""
    if not semantic_registry:
        raise HTTPException(status_code=503, detail="Semantic registry not available")

    return semantic_registry.stats()

@app.post("/registry/semantic/rebuild")
async def rebuild_semantic():
    """Rebuild the semantic registry index."""
    if not semantic_registry:
        raise HTTPException(status_code=503, detail="Semantic registry not available")

    try:
        semantic_registry.build_index()
        semantic_registry.save_index("Data/indexes/semantic_registry.pkl")
        return {
            "status": "rebuilt",
            "stats": semantic_registry.stats(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# Hybrid Endpoint
# =============================================================================

@app.get("/hybrid")
async def hybrid_search(
    q: str = Query(..., description="Query"),
    top_k: int = Query(5, ge=1, le=20, description="Semantic results count"),
    doc_type: Optional[str] = Query(None, description="Filter by doc type"),
    year: Optional[int] = Query(None, description="Filter by year"),
    file_pattern: Optional[str] = Query(None, description="Filter by file pattern"),
    cache: bool = Query(True, description="Use query cache"),
    include_confidence: bool = Query(True, description="Include confidence scoring"),
    fast_path: bool = Query(True, description="Use fast-path routing for simple queries"),
):
    """Hybrid search: structured registry + semantic RAG with caching, routing, and confidence scoring."""
    # Start request timing
    ctx = None
    if INSTRUMENTATION_AVAILABLE and latency_tracker:
        ctx = RequestContext()
        ctx.start("total")

    if not registry_rag:
        raise HTTPException(status_code=503, detail="Registry not available, use /search instead")

    if not rag:
        raise HTTPException(status_code=503, detail="RAG not initialized")

    # Hot-reload checks with cache invalidation
    index_reloaded = rag.reload_if_changed()
    registry_reloaded = registry_rag.reload_if_changed()

    if (index_reloaded or registry_reloaded) and query_cache:
        new_source_hash = compute_source_hash("Data/rag_indexes/hybrid") if CACHE_AVAILABLE else None
        query_cache.invalidate(new_source_hash)

    # Build cache key
    cache_params = {
        "top_k": top_k,
        "doc_type": doc_type,
        "year": year,
        "file_pattern": file_pattern,
        "include_confidence": include_confidence,
    }

    # Try to get cached result
    cache_key = None
    if cache and query_cache:
        cache_key = query_cache.generate_key("/hybrid", q, cache_params)
        cached = query_cache.get(cache_key)
        if cached:
            response = cached.result.copy()
            response["cache_hit"] = True
            if ctx:
                ctx.end("total")
                latency_tracker.record("total", ctx.get_elapsed("total"))
            return response

    # Query routing - determine if we can skip RAG for simple queries
    routing_decision = None
    routing_info = {}
    if fast_path and QUERY_ROUTER_AVAILABLE and query_router:
        if ctx:
            ctx.start("query_routing")
        routing_result = query_router.route(q)
        routing_decision = routing_result.decision
        routing_info = {
            "decision": routing_result.decision.value,
            "confidence": routing_result.confidence,
            "reasoning": routing_result.reasoning,
            "expected_latency_ms": routing_result.expected_latency_ms,
        }
        if ctx:
            ctx.end("query_routing")
            latency_tracker.record("query_routing", ctx.get_elapsed("query_routing"))

    # Get structured results (always needed)
    if ctx:
        ctx.start("registry_lookup")
    registry_result = registry_rag.query(q)
    if ctx:
        ctx.end("registry_lookup")
        latency_tracker.record("registry_lookup", ctx.get_elapsed("registry_lookup"))

    # Fast-path: If routing says registry_only, skip RAG entirely
    semantic_results = []
    if routing_decision == RoutingDecision.REGISTRY_ONLY:
        # Skip RAG - use registry results only
        pass
    else:
        # Get semantic results (full RAG path)
        if ctx:
            ctx.start("rag_search")
        filters = build_filters(doc_type, year, file_pattern)
        semantic_results = rag.search(q, top_k, filters=filters)
        if ctx:
            ctx.end("rag_search")
            latency_tracker.record("rag_search", ctx.get_elapsed("rag_search"))

    response = {
        "query": q,
        "classification": registry_result["classification"],
        "structured_results": registry_result["structured_results"],
        "structured_answer": registry_result.get("answer"),
        "semantic_results": semantic_results,
        "hybrid": True,
        "cache_hit": False,
        "routing": routing_info if routing_info else None,
    }

    # Calculate confidence score if enabled
    if include_confidence and CONFIDENCE_AVAILABLE and confidence_calculator:
        try:
            # Build combined result for source extraction
            combined_result = {
                "structured_results": registry_result["structured_results"],
                "semantic_results": semantic_results,
            }

            # Extract sources
            sources = extract_sources_from_rag_result(combined_result)

            # Extract similarity scores from semantic results
            similarity_scores = [r.get("score", 0.5) for r in semantic_results]

            # Check if we have structured data
            has_structured = bool(registry_result["structured_results"])

            # Get classification confidence
            classification_confidence = registry_result["classification"].get("confidence", 0.9)

            # Calculate confidence
            confidence = confidence_calculator.calculate(
                sources=sources,
                query=q,
                response_text=registry_result.get("answer", ""),
                classification_confidence=classification_confidence,
                has_structured_data=has_structured,
                similarity_scores=similarity_scores,
            )

            # Add confidence to response
            response["confidence"] = {
                "level": confidence.level,
                "score": confidence.score,
                "reasoning": confidence.reasoning,
                "sources": [
                    {
                        "type": s.type.value,
                        "name": s.name,
                        "citation": s.citation,
                        "relevance": s.relevance,
                        "url": s.url,
                    }
                    for s in confidence.sources
                ],
                "factors": confidence.factors,
            }
        except Exception as e:
            # On error, add minimal confidence info
            response["confidence"] = {
                "level": "uncertain",
                "score": 0.5,
                "reasoning": f"Confidence calculation error: {str(e)}",
                "sources": [],
                "factors": {},
            }

    # End timing and add latency breakdown
    if ctx:
        ctx.end("total")
        latency_tracker.record("total", ctx.get_elapsed("total"))
        response["latency_ms"] = ctx.get_breakdown()

    # Sanitize response to handle NaN values (e.g., in email fields)
    response = sanitize_for_json(response)

    # Cache the result
    if cache and query_cache and cache_key:
        query_cache.set(cache_key, q, response, endpoint="/hybrid")

    return response

# =============================================================================
# Statistics Endpoint
# =============================================================================

@app.get("/stats")
async def get_latency_stats():
    """Get latency statistics for all RAG pipeline components."""
    if not INSTRUMENTATION_AVAILABLE or not latency_tracker:
        raise HTTPException(status_code=503, detail="Instrumentation not available")

    stats = get_stats()

    # Add cache stats if available
    if query_cache:
        stats["cache"] = query_cache.stats()

    # Add reformulation cache stats if available
    if reformulation_cache:
        stats["reformulation_cache"] = reformulation_cache.stats()

    return stats

@app.post("/stats/reset")
async def reset_latency_stats():
    """Reset latency statistics."""
    if not INSTRUMENTATION_AVAILABLE:
        raise HTTPException(status_code=503, detail="Instrumentation not available")

    reset_tracker()
    return {"status": "reset", "timestamp": datetime.now().isoformat()}

# =============================================================================
# Hypergraph Endpoints
# =============================================================================

@app.get("/hypergraph/stats")
async def hypergraph_stats():
    """Get hypergraph statistics."""
    if not hypergraph:
        raise HTTPException(status_code=503, detail="Hypergraph not available")
    return hypergraph.stats()

@app.get("/hypergraph/collaborators")
async def get_collaborators(
    person: str = Query(..., description="Person ID"),
):
    """Find collaborators of a person."""
    if not hypergraph:
        raise HTTPException(status_code=503, detail="Hypergraph not available")

    person_node = hypergraph.get_node(person) or hypergraph.get_node(f"person-{person}")
    if not person_node:
        raise HTTPException(
            status_code=404,
            detail=f"Person '{person}' not found. Try using full ID like 'person-james-evans'",
        )

    collabs = hypergraph.collaborators_of(person_node.id)
    return {
        "person": {"id": person_node.id, "name": person_node.name},
        "collaborators": [{"id": c.id, "name": c.name} for c in collabs],
    }

@app.get("/hypergraph/outputs")
async def get_outputs(
    person: str = Query(..., description="Person ID"),
):
    """Find what a person has produced."""
    if not hypergraph:
        raise HTTPException(status_code=503, detail="Hypergraph not available")

    person_node = hypergraph.get_node(person) or hypergraph.get_node(f"person-{person}")
    if not person_node:
        raise HTTPException(status_code=404, detail=f"Person '{person}' not found")

    outputs = hypergraph.what_produced(person_node.id)
    return {
        "person": {"id": person_node.id, "name": person_node.name},
        "outputs": [
            {
                "id": o.id,
                "name": o.name,
                "type": o.type.value if hasattr(o.type, 'value') else str(o.type),
            }
            for o in outputs
        ],
    }

@app.get("/hypergraph/terminal")
async def get_terminal_nodes():
    """Find terminal nodes (dead ends - outputs never reused)."""
    if not hypergraph:
        raise HTTPException(status_code=503, detail="Hypergraph not available")

    terminal = hypergraph.terminal_nodes()
    by_type = {}
    for node in terminal:
        t = node.type.value if hasattr(node.type, 'value') else str(node.type)
        if t not in by_type:
            by_type[t] = []
        by_type[t].append({"id": node.id, "name": node.name})

    return {
        "total": len(terminal),
        "by_type": {k: {"count": len(v), "examples": v[:5]} for k, v in by_type.items()},
    }

@app.get("/hypergraph/search")
async def search_hypergraph(
    q: str = Query(..., description="Search query"),
):
    """Search nodes by name."""
    if not hypergraph:
        raise HTTPException(status_code=503, detail="Hypergraph not available")

    query_lower = q.lower()
    matches = []
    for node in hypergraph.nodes.values():
        if query_lower in node.name.lower() or query_lower in node.id.lower():
            matches.append({
                "id": node.id,
                "name": node.name,
                "type": node.type.value if hasattr(node.type, 'value') else str(node.type),
            })

    return {"query": q, "matches": matches[:20]}

# =============================================================================
# Cache Endpoints
# =============================================================================

@app.get("/cache/stats", response_model=CacheStatsResponse)
async def cache_stats():
    """
    Get query cache statistics.

    Returns hit rate, cache size, and other performance metrics.
    """
    if not query_cache:
        raise HTTPException(status_code=503, detail="Query cache not available")

    stats = query_cache.stats()
    return CacheStatsResponse(**stats)


@app.post("/cache/clear", response_model=CacheClearResponse)
async def cache_clear():
    """
    Clear all cached query results.

    Use this to force fresh results for all queries.
    """
    if not query_cache:
        raise HTTPException(status_code=503, detail="Query cache not available")

    count = query_cache.clear()
    return CacheClearResponse(
        status="cleared",
        entries_cleared=count,
        timestamp=datetime.now().isoformat(),
    )


@app.get("/cache/keys")
async def cache_keys():
    """
    List all cache keys (for debugging).

    Returns the list of current cache keys and cache status.
    """
    if not query_cache:
        raise HTTPException(status_code=503, detail="Query cache not available")

    return {
        "keys": query_cache.keys,
        "count": query_cache.size,
        "enabled": query_cache.enabled,
    }


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    env = os.environ.get("CHORUS_ENV", "development")
    port = int(os.environ.get("RAG_PORT", "8765"))

    config = {
        "app": "rag_server_fastapi:app",
        "host": "0.0.0.0",
        "port": port,
        "log_level": "info",
    }

    if env == "development":
        config.update({
            "reload": True,
            "workers": 1,
        })
    elif env == "production":
        config.update({
            "workers": 4,
            "reload": False,
        })

    print(f"\nStarting CHORUS RAG Server (FastAPI)")
    print(f"Environment: {env}")
    print(f"Port: {port}")
    print(f"Workers: {config.get('workers', 1)}")
    print(f"API Docs: http://localhost:{port}/docs")
    print()

    uvicorn.run(**config)
