"""
CHORUS Query Cache Implementation

Provides a sophisticated caching layer for the RAG system with support for
both in-memory LRU cache and optional Redis backend.

Performance targets:
- Cache hit latency: < 50ms (vs ~2000ms uncached)
- Cache hit rate target: > 30% for typical usage
"""

import hashlib
import json
import os
import time
import threading
from collections import OrderedDict
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

# Import base utilities
from .base import (
    get_env_bool,
    get_env_int,
    CacheBackend,
    LRUMemoryBackend,
    BaseCacheStats,
)


# =============================================================================
# Configuration
# =============================================================================

# Cache configuration from environment
CACHE_ENABLED = get_env_bool("CHORUS_CACHE_ENABLED", True)
CACHE_TTL = get_env_int("CHORUS_CACHE_TTL", 300)  # 5 minutes default
CACHE_MAX_SIZE = get_env_int("CHORUS_CACHE_MAX_SIZE", 1000)
CACHE_BACKEND = os.environ.get("CHORUS_CACHE_BACKEND", "memory").lower()
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class CachedResult:
    """
    Represents a cached query result.

    Attributes:
        query: The original query string
        result: The cached result data (search results, etc.)
        timestamp: Unix timestamp when the result was cached
        hit_count: Number of times this cached result has been retrieved
        source_hash: Hash of source data for invalidation detection
        ttl: Time-to-live in seconds for this entry
        endpoint: The endpoint this cache entry is for (/search, /hybrid, etc.)
        params_hash: Hash of query parameters for cache key uniqueness
    """
    query: str
    result: Dict[str, Any]
    timestamp: float
    hit_count: int = 0
    source_hash: str = ""
    ttl: int = CACHE_TTL
    endpoint: str = ""
    params_hash: str = ""

    def is_expired(self) -> bool:
        """Check if this cached result has expired based on TTL."""
        return time.time() > self.timestamp + self.ttl

    def increment_hit(self) -> None:
        """Increment the hit counter."""
        self.hit_count += 1

    def age_seconds(self) -> float:
        """Return the age of this cache entry in seconds."""
        return time.time() - self.timestamp

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CachedResult":
        """Create CachedResult from dictionary."""
        return cls(**data)


@dataclass
class CacheStats:
    """
    Cache statistics for monitoring and debugging.

    Attributes:
        hits: Total number of cache hits
        misses: Total number of cache misses
        evictions: Total number of entries evicted (LRU or expired)
        size: Current number of entries in cache
        max_size: Maximum cache size
        ttl: Default TTL for cache entries
        backend: Cache backend type (memory or redis)
        enabled: Whether caching is enabled
        start_time: Unix timestamp when cache was initialized
        invalidations: Number of manual cache invalidations
    """
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size: int = 0
    max_size: int = CACHE_MAX_SIZE
    ttl: int = CACHE_TTL
    backend: str = CACHE_BACKEND
    enabled: bool = CACHE_ENABLED
    start_time: float = field(default_factory=time.time)
    invalidations: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate the cache hit rate as a percentage."""
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return (self.hits / total) * 100

    @property
    def uptime_seconds(self) -> float:
        """Return cache uptime in seconds."""
        return time.time() - self.start_time

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON response."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "invalidations": self.invalidations,
            "size": self.size,
            "max_size": self.max_size,
            "hit_rate_percent": round(self.hit_rate, 2),
            "ttl_seconds": self.ttl,
            "backend": self.backend,
            "enabled": self.enabled,
            "uptime_seconds": round(self.uptime_seconds, 2),
        }


# =============================================================================
# Redis Cache Backend
# =============================================================================

# Alias for backward compatibility
MemoryBackend = LRUMemoryBackend

class RedisBackend(CacheBackend):
    """
    Redis cache backend for distributed caching.

    Requires the redis package: pip install redis
    Uses JSON serialization for CachedResult objects.
    """

    def __init__(self, redis_url: str = REDIS_URL, key_prefix: str = "chorus:cache:"):
        """
        Initialize the Redis cache backend.

        Args:
            redis_url: Redis connection URL
            key_prefix: Prefix for all cache keys in Redis
        """
        self.redis_url = redis_url
        self.key_prefix = key_prefix
        self._client = None
        self._eviction_count = 0

    @property
    def client(self):
        """Lazy initialization of Redis client."""
        if self._client is None:
            try:
                import redis
                self._client = redis.from_url(self.redis_url, decode_responses=True)
                # Test connection
                self._client.ping()
            except ImportError:
                raise ImportError(
                    "Redis backend requires the 'redis' package. "
                    "Install with: pip install redis"
                )
            except Exception as e:
                raise ConnectionError(f"Failed to connect to Redis at {self.redis_url}: {e}")
        return self._client

    def _make_key(self, key: str) -> str:
        """Create a prefixed key for Redis."""
        return f"{self.key_prefix}{key}"

    def get(self, key: str) -> Optional[CachedResult]:
        """Retrieve a cached result from Redis."""
        try:
            redis_key = self._make_key(key)
            data = self.client.get(redis_key)
            if data is None:
                return None

            result = CachedResult.from_dict(json.loads(data))

            # Check if expired (Redis TTL handles this, but double-check)
            if result.is_expired():
                self.client.delete(redis_key)
                self._eviction_count += 1
                return None

            return result
        except Exception as e:
            print(f"Redis get error: {e}")
            return None

    def set(self, key: str, result: CachedResult) -> None:
        """Store a cached result in Redis with TTL."""
        try:
            redis_key = self._make_key(key)
            data = json.dumps(result.to_dict())
            # Set with TTL
            self.client.setex(redis_key, result.ttl, data)
        except Exception as e:
            print(f"Redis set error: {e}")

    def delete(self, key: str) -> bool:
        """Delete a specific cache entry from Redis."""
        try:
            redis_key = self._make_key(key)
            return bool(self.client.delete(redis_key))
        except Exception as e:
            print(f"Redis delete error: {e}")
            return False

    def clear(self) -> int:
        """Clear all CHORUS cache entries from Redis."""
        try:
            pattern = f"{self.key_prefix}*"
            keys = list(self.client.scan_iter(match=pattern))
            if keys:
                return self.client.delete(*keys)
            return 0
        except Exception as e:
            print(f"Redis clear error: {e}")
            return 0

    def size(self) -> int:
        """Get current number of cached entries."""
        try:
            pattern = f"{self.key_prefix}*"
            return len(list(self.client.scan_iter(match=pattern)))
        except Exception as e:
            print(f"Redis size error: {e}")
            return 0

    def keys(self) -> List[str]:
        """Get all cache keys (without prefix)."""
        try:
            pattern = f"{self.key_prefix}*"
            prefix_len = len(self.key_prefix)
            return [key[prefix_len:] for key in self.client.scan_iter(match=pattern)]
        except Exception as e:
            print(f"Redis keys error: {e}")
            return []

    @property
    def eviction_count(self) -> int:
        """Get eviction count (note: Redis handles TTL expiration internally)."""
        return self._eviction_count


# =============================================================================
# Main Query Cache Class
# =============================================================================

class QueryCache:
    """
    Main query cache implementation with statistics tracking.

    Supports both in-memory LRU and Redis backends.
    Provides cache key generation from normalized query and parameters.

    Example usage:
        cache = QueryCache()

        # Generate cache key
        key = cache.generate_key("/search", "machine learning", {"top_k": 5})

        # Try to get cached result
        cached = cache.get(key)
        if cached:
            return cached.result

        # Compute and cache result
        result = expensive_search(query)
        cache.set(key, query, result, source_hash="abc123")

        # Get statistics
        stats = cache.stats()
    """

    def __init__(
        self,
        enabled: bool = CACHE_ENABLED,
        ttl: int = CACHE_TTL,
        max_size: int = CACHE_MAX_SIZE,
        backend: str = CACHE_BACKEND,
        redis_url: str = REDIS_URL,
    ):
        """
        Initialize the query cache.

        Args:
            enabled: Whether caching is enabled
            ttl: Default TTL in seconds
            max_size: Maximum cache size (for memory backend)
            backend: Backend type ("memory" or "redis")
            redis_url: Redis URL (for redis backend)
        """
        self.enabled = enabled
        self.ttl = ttl
        self.max_size = max_size
        self.backend_type = backend

        # Initialize statistics
        self._stats = CacheStats(
            max_size=max_size,
            ttl=ttl,
            backend=backend,
            enabled=enabled,
        )

        # Initialize backend
        if backend == "redis":
            self._backend: CacheBackend = RedisBackend(redis_url=redis_url)
        else:
            self._backend = MemoryBackend(max_size=max_size)

        # Track current source hash for invalidation
        self._current_source_hash: Optional[str] = None
        self._lock = threading.RLock()

    def generate_key(
        self,
        endpoint: str,
        query: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Generate a unique cache key from endpoint, query, and parameters.

        The key is generated by:
        1. Normalizing the query (lowercase, strip whitespace)
        2. Sorting and serializing parameters
        3. Creating a SHA256 hash of the combined data

        Args:
            endpoint: The API endpoint (e.g., "/search", "/hybrid")
            query: The search query string
            params: Optional query parameters

        Returns:
            A unique cache key string
        """
        # Normalize query
        normalized_query = query.lower().strip()

        # Sort and serialize parameters
        if params:
            # Filter out None values and sort
            filtered_params = {k: v for k, v in sorted(params.items()) if v is not None}
            params_str = json.dumps(filtered_params, sort_keys=True)
        else:
            params_str = ""

        # Create hash input
        hash_input = f"{endpoint}:{normalized_query}:{params_str}"
        hash_bytes = hashlib.sha256(hash_input.encode()).hexdigest()[:16]

        return hash_bytes

    def get(self, key: str) -> Optional[CachedResult]:
        """
        Retrieve a cached result.

        Automatically tracks hit/miss statistics and validates source hash.

        Args:
            key: The cache key

        Returns:
            CachedResult if found, valid, and not expired; None otherwise
        """
        if not self.enabled:
            return None

        with self._lock:
            result = self._backend.get(key)

            if result is None:
                self._stats.misses += 1
                return None

            # Check source hash for invalidation
            if self._current_source_hash and result.source_hash:
                if result.source_hash != self._current_source_hash:
                    # Source data has changed, invalidate this entry
                    self._backend.delete(key)
                    self._stats.misses += 1
                    return None

            # Update stats and hit count
            result.increment_hit()
            self._stats.hits += 1
            self._stats.size = self._backend.size()

            return result

    def set(
        self,
        key: str,
        query: str,
        result: Dict[str, Any],
        source_hash: str = "",
        endpoint: str = "",
        ttl: Optional[int] = None,
    ) -> None:
        """
        Store a result in the cache.

        Args:
            key: The cache key
            query: The original query string
            result: The result data to cache
            source_hash: Hash of source data for invalidation
            endpoint: The API endpoint
            ttl: Optional TTL override
        """
        if not self.enabled:
            return

        with self._lock:
            cached_result = CachedResult(
                query=query,
                result=result,
                timestamp=time.time(),
                hit_count=0,
                source_hash=source_hash or self._current_source_hash or "",
                ttl=ttl or self.ttl,
                endpoint=endpoint,
                params_hash=key,
            )

            old_size = self._backend.size()
            self._backend.set(key, cached_result)
            new_size = self._backend.size()

            # Track evictions (if size didn't increase, something was evicted)
            if new_size <= old_size and old_size >= self.max_size:
                self._stats.evictions += 1

            self._stats.size = new_size

    def delete(self, key: str) -> bool:
        """
        Delete a specific cache entry.

        Args:
            key: The cache key

        Returns:
            True if entry was deleted
        """
        if not self.enabled:
            return False

        with self._lock:
            result = self._backend.delete(key)
            self._stats.size = self._backend.size()
            return result

    def clear(self) -> int:
        """
        Clear all cache entries.

        Returns:
            Number of entries cleared
        """
        with self._lock:
            count = self._backend.clear()
            self._stats.size = 0
            self._stats.invalidations += 1
            return count

    def invalidate(self, new_source_hash: Optional[str] = None) -> int:
        """
        Invalidate the cache due to source data change.

        This clears all cache entries and optionally updates the source hash.

        Args:
            new_source_hash: New source hash to track (optional)

        Returns:
            Number of entries invalidated
        """
        with self._lock:
            count = self._backend.clear()
            self._stats.size = 0
            self._stats.invalidations += 1

            if new_source_hash:
                self._current_source_hash = new_source_hash

            return count

    def set_source_hash(self, source_hash: str) -> None:
        """
        Update the current source hash for invalidation tracking.

        Args:
            source_hash: New source hash
        """
        with self._lock:
            self._current_source_hash = source_hash

    def stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        with self._lock:
            self._stats.size = self._backend.size()

            # Get eviction count from backend if available
            if hasattr(self._backend, "eviction_count"):
                self._stats.evictions = self._backend.eviction_count

            return self._stats.to_dict()

    def cleanup_expired(self) -> int:
        """
        Remove expired entries from the cache.

        Note: Only effective for memory backend. Redis handles TTL automatically.

        Returns:
            Number of entries removed
        """
        with self._lock:
            if isinstance(self._backend, MemoryBackend):
                count = self._backend.cleanup_expired()
                self._stats.evictions += count
                self._stats.size = self._backend.size()
                return count
            return 0

    @property
    def size(self) -> int:
        """Get current cache size."""
        return self._backend.size()

    @property
    def keys(self) -> List[str]:
        """Get all cache keys."""
        return self._backend.keys()


# =============================================================================
# Module-Level Singleton and Factory
# =============================================================================

_default_cache: Optional[QueryCache] = None
_cache_lock = threading.Lock()


def get_cache(
    enabled: bool = CACHE_ENABLED,
    ttl: int = CACHE_TTL,
    max_size: int = CACHE_MAX_SIZE,
    backend: str = CACHE_BACKEND,
    redis_url: str = REDIS_URL,
) -> QueryCache:
    """
    Get or create the default QueryCache singleton.

    Thread-safe factory function that returns a single cache instance.

    Args:
        enabled: Whether caching is enabled
        ttl: Default TTL in seconds
        max_size: Maximum cache size
        backend: Backend type ("memory" or "redis")
        redis_url: Redis URL for redis backend

    Returns:
        The default QueryCache instance
    """
    global _default_cache

    with _cache_lock:
        if _default_cache is None:
            _default_cache = QueryCache(
                enabled=enabled,
                ttl=ttl,
                max_size=max_size,
                backend=backend,
                redis_url=redis_url,
            )
        return _default_cache


def reset_cache() -> None:
    """
    Reset the default cache singleton.

    Useful for testing or when configuration changes.
    """
    global _default_cache

    with _cache_lock:
        if _default_cache is not None:
            _default_cache.clear()
        _default_cache = None


# =============================================================================
# Helper Functions
# =============================================================================

def compute_source_hash(index_path: str) -> str:
    """
    Compute a hash representing the current state of source data.

    Uses the modification time of the index files.

    Args:
        index_path: Path to the index directory

    Returns:
        Hash string representing the current data state
    """
    import os
    from pathlib import Path

    index_dir = Path(index_path)
    mtimes = []

    # Check key index files
    for filename in ["faiss_index.faiss", "hybrid_data.pkl"]:
        filepath = index_dir / filename
        if filepath.exists():
            mtimes.append(filepath.stat().st_mtime)

    if not mtimes:
        return ""

    # Create hash from modification times
    hash_input = ":".join(str(m) for m in sorted(mtimes))
    return hashlib.md5(hash_input.encode()).hexdigest()[:12]
