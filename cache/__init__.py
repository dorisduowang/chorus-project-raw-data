"""
CHORUS Query Caching Module

Provides a sophisticated caching layer for the CHORUS RAG system to reduce
latency for frequent queries. Supports both in-memory LRU cache and optional
Redis backend for distributed caching.

Features:
- In-memory LRU cache with configurable max size
- Optional Redis backend for distributed deployments
- TTL-based expiration (configurable, default 5 minutes)
- Cache statistics tracking (hits, misses, hit rate)
- Automatic cache key generation from normalized query + parameters
- Cache invalidation on index/registry reload
- Reformulation cache for LLM query reformulations (reduces API costs)
- Semantic similarity matching for reformulation cache hits

Configuration via environment variables:
- CHORUS_CACHE_ENABLED=true|false (default: true)
- CHORUS_CACHE_TTL=300 (seconds, default: 5 minutes)
- CHORUS_CACHE_MAX_SIZE=1000 (entries for in-memory cache)
- CHORUS_CACHE_BACKEND=memory|redis (default: memory)
- REDIS_URL=redis://localhost:6379 (for redis backend)
- REFORMULATION_CACHE_ENABLED=true|false (default: true)
- REFORMULATION_CACHE_TTL=86400 (seconds, default: 24 hours)
- REFORMULATION_CACHE_MAX_SIZE=5000 (entries)
- REFORMULATION_SEMANTIC_THRESHOLD=0.95 (similarity threshold)

Usage:
    from cache import QueryCache, CachedResult

    cache = QueryCache()

    # Check cache first
    key = cache.generate_key(query, params)
    cached = cache.get(key)
    if cached:
        return cached.result

    # Compute result and cache it
    result = expensive_search(query)
    cache.set(key, query, result, source_hash)

    # Get statistics
    stats = cache.stats()

Reformulation Cache Usage:
    from cache import ReformulationCache, get_reformulation_cache

    reformulation_cache = get_reformulation_cache()

    # Check cache first
    cached = reformulation_cache.get(query)
    if cached:
        reformulations = cached.reformulations
    else:
        reformulations = call_llm_reformulation(query)
        reformulation_cache.set(query, reformulations)
"""

# Base utilities
from .base import (
    LRUMemoryBackend,
    BaseCacheEntry,
    BaseCacheStats,
    get_env_bool,
    get_env_int,
    get_env_float,
)

from .query_cache import (
    QueryCache,
    CachedResult,
    CacheBackend,
    MemoryBackend,
    RedisBackend,
    get_cache,
)

from .reformulation_cache import (
    ReformulationCache,
    CachedReformulation,
    ReformulationCacheStats,
    get_reformulation_cache,
    reset_reformulation_cache,
    normalize_query,
)

__all__ = [
    # Base utilities
    "LRUMemoryBackend",
    "BaseCacheEntry",
    "BaseCacheStats",
    "get_env_bool",
    "get_env_int",
    "get_env_float",
    # Query cache
    "QueryCache",
    "CachedResult",
    "CacheBackend",
    "MemoryBackend",
    "RedisBackend",
    "get_cache",
    # Reformulation cache
    "ReformulationCache",
    "CachedReformulation",
    "ReformulationCacheStats",
    "get_reformulation_cache",
    "reset_reformulation_cache",
    "normalize_query",
]
