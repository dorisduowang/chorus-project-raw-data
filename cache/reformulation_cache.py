"""
CHORUS Reformulation Cache Implementation

Caches LLM query reformulations to reduce Claude API costs.
Supports both exact match and semantic similarity matching.

Performance targets:
- Exact match: < 1ms lookup
- Semantic match: < 50ms lookup
- Expected hit rate: 40-60% for typical usage patterns

Configuration via environment variables:
- REFORMULATION_CACHE_ENABLED=true|false (default: true)
- REFORMULATION_CACHE_TTL=86400 (seconds, default: 24 hours)
- REFORMULATION_CACHE_MAX_SIZE=5000 (entries)
- REFORMULATION_SEMANTIC_THRESHOLD=0.95 (similarity threshold)
"""

import hashlib
import os
import re
import time
import threading
from collections import OrderedDict
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

# Import base utilities
from .base import (
    get_env_bool,
    get_env_int,
    get_env_float,
    normalize_query as base_normalize_query,
    generate_cache_key as base_generate_key,
)

# Make numpy optional (only needed for semantic matching)
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None


# =============================================================================
# Configuration
# =============================================================================

# Reformulation cache configuration
REFORMULATION_CACHE_ENABLED = get_env_bool("REFORMULATION_CACHE_ENABLED", True)
REFORMULATION_CACHE_TTL = get_env_int("REFORMULATION_CACHE_TTL", 86400)  # 24 hours
REFORMULATION_CACHE_MAX_SIZE = get_env_int("REFORMULATION_CACHE_MAX_SIZE", 5000)
REFORMULATION_SEMANTIC_THRESHOLD = get_env_float("REFORMULATION_SEMANTIC_THRESHOLD", 0.95)


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class CachedReformulation:
    """
    Represents a cached query reformulation.

    Attributes:
        original_query: The original query string (normalized)
        reformulations: List of reformulated queries
        timestamp: Unix timestamp when cached
        hit_count: Number of times this entry has been retrieved
        embedding: Optional embedding vector for semantic matching
        ttl: Time-to-live in seconds
    """
    original_query: str
    reformulations: List[str]
    timestamp: float
    hit_count: int = 0
    embedding: Optional[List[float]] = None
    ttl: int = REFORMULATION_CACHE_TTL

    def is_expired(self) -> bool:
        """Check if this cached entry has expired based on TTL."""
        return time.time() > self.timestamp + self.ttl

    def increment_hit(self) -> None:
        """Increment the hit counter."""
        self.hit_count += 1

    def age_seconds(self) -> float:
        """Return the age of this cache entry in seconds."""
        return time.time() - self.timestamp

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        # Convert embedding to list for JSON serialization
        if self.embedding is not None:
            if NUMPY_AVAILABLE and isinstance(self.embedding, np.ndarray):
                data["embedding"] = list(self.embedding)
            else:
                data["embedding"] = list(self.embedding) if hasattr(self.embedding, '__iter__') else self.embedding
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CachedReformulation":
        """Create CachedReformulation from dictionary."""
        # Handle embedding conversion
        if data.get("embedding") is not None:
            data["embedding"] = list(data["embedding"])
        return cls(**data)


@dataclass
class ReformulationCacheStats:
    """
    Statistics for reformulation cache monitoring.

    Attributes:
        hits: Total exact match cache hits
        misses: Total cache misses
        semantic_hits: Hits from semantic similarity matching
        expirations: Number of entries expired
        size: Current number of entries
        max_size: Maximum cache size
        ttl: Default TTL for entries
        enabled: Whether caching is enabled
        semantic_enabled: Whether semantic matching is enabled
        semantic_threshold: Similarity threshold for semantic matching
        start_time: Unix timestamp when cache was initialized
    """
    hits: int = 0
    misses: int = 0
    semantic_hits: int = 0
    expirations: int = 0
    size: int = 0
    max_size: int = REFORMULATION_CACHE_MAX_SIZE
    ttl: int = REFORMULATION_CACHE_TTL
    enabled: bool = REFORMULATION_CACHE_ENABLED
    semantic_enabled: bool = False
    semantic_threshold: float = REFORMULATION_SEMANTIC_THRESHOLD
    start_time: float = field(default_factory=time.time)

    @property
    def total_hits(self) -> int:
        """Total hits (exact + semantic)."""
        return self.hits + self.semantic_hits

    @property
    def hit_rate(self) -> float:
        """Calculate the overall hit rate as a percentage."""
        total = self.total_hits + self.misses
        if total == 0:
            return 0.0
        return (self.total_hits / total) * 100

    @property
    def exact_hit_rate(self) -> float:
        """Calculate the exact match hit rate as a percentage."""
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return (self.hits / total) * 100

    @property
    def semantic_hit_rate(self) -> float:
        """Calculate semantic hit rate (semantic hits / total lookups)."""
        total = self.total_hits + self.misses
        if total == 0:
            return 0.0
        return (self.semantic_hits / total) * 100

    @property
    def uptime_seconds(self) -> float:
        """Return cache uptime in seconds."""
        return time.time() - self.start_time

    @property
    def estimated_savings(self) -> Dict[str, float]:
        """Estimate cost savings from cache hits."""
        # Estimate based on typical Claude API costs
        cost_per_call = 0.001  # ~$0.001 per reformulation call
        time_per_call = 0.350  # ~350ms per reformulation call

        return {
            "api_calls_saved": self.total_hits,
            "estimated_cost_saved_usd": self.total_hits * cost_per_call,
            "estimated_time_saved_ms": self.total_hits * time_per_call * 1000,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON response."""
        savings = self.estimated_savings
        return {
            "hits": self.hits,
            "misses": self.misses,
            "semantic_hits": self.semantic_hits,
            "total_hits": self.total_hits,
            "expirations": self.expirations,
            "size": self.size,
            "max_size": self.max_size,
            "hit_rate_percent": round(self.hit_rate, 2),
            "exact_hit_rate_percent": round(self.exact_hit_rate, 2),
            "semantic_hit_rate_percent": round(self.semantic_hit_rate, 2),
            "ttl_seconds": self.ttl,
            "enabled": self.enabled,
            "semantic_enabled": self.semantic_enabled,
            "semantic_threshold": self.semantic_threshold,
            "uptime_seconds": round(self.uptime_seconds, 2),
            "api_calls_saved": savings["api_calls_saved"],
            "estimated_cost_saved_usd": round(savings["estimated_cost_saved_usd"], 4),
            "estimated_time_saved_ms": round(savings["estimated_time_saved_ms"], 2),
        }


# =============================================================================
# Query Normalization (extends base with punctuation stripping)
# =============================================================================

def normalize_query(query: str) -> str:
    """
    Normalize a query string for cache key generation.

    Uses base normalization plus removes leading/trailing punctuation.

    Args:
        query: The raw query string

    Returns:
        Normalized query string
    """
    # Use base normalization (lowercase, strip, collapse whitespace)
    normalized = base_normalize_query(query)

    # Additionally remove leading/trailing punctuation (keep internal)
    if normalized:
        normalized = normalized.strip('.,!?;:')

    return normalized


def generate_cache_key(query: str) -> str:
    """
    Generate a unique cache key from a query.

    Args:
        query: The query string (will be normalized)

    Returns:
        SHA256 hash of normalized query (first 16 chars)
    """
    normalized = normalize_query(query)
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]


# =============================================================================
# Reformulation Cache Class
# =============================================================================

class ReformulationCache:
    """
    Cache for LLM query reformulations with semantic similarity support.

    Provides two levels of matching:
    1. Exact match: Normalized query string matches exactly
    2. Semantic match: Query embedding similarity exceeds threshold

    Thread-safe implementation using RLock.

    Example usage:
        cache = ReformulationCache()

        # Check cache first
        cached = cache.get("machine learning papers")
        if cached:
            reformulations = cached.reformulations
        else:
            reformulations = call_llm_reformulation(query)
            cache.set(query, reformulations)

        # Get statistics
        stats = cache.stats()
    """

    def __init__(
        self,
        enabled: bool = REFORMULATION_CACHE_ENABLED,
        ttl: int = REFORMULATION_CACHE_TTL,
        max_size: int = REFORMULATION_CACHE_MAX_SIZE,
        semantic_threshold: float = REFORMULATION_SEMANTIC_THRESHOLD,
        embedding_model: Optional[Any] = None,
    ):
        """
        Initialize the reformulation cache.

        Args:
            enabled: Whether caching is enabled
            ttl: Default TTL in seconds (24 hours default)
            max_size: Maximum number of entries to cache
            semantic_threshold: Similarity threshold for semantic matching (0.95 default)
            embedding_model: Optional SentenceTransformer model for semantic matching
        """
        self.enabled = enabled
        self.ttl = ttl
        self.max_size = max_size
        self.semantic_threshold = semantic_threshold

        # Cache storage: key -> CachedReformulation
        self._cache: OrderedDict[str, CachedReformulation] = OrderedDict()

        # Semantic search structures
        self._embedding_model = embedding_model
        self._embeddings_matrix: Optional[Any] = None  # np.ndarray if available
        self._embedding_keys: List[str] = []  # Maps matrix row -> cache key

        # Thread safety
        self._lock = threading.RLock()

        # Statistics
        self._stats = ReformulationCacheStats(
            max_size=max_size,
            ttl=ttl,
            enabled=enabled,
            semantic_enabled=embedding_model is not None,
            semantic_threshold=semantic_threshold,
        )

    def _compute_embedding(self, query: str) -> Optional[Any]:
        """
        Compute embedding for a query using the embedding model.

        Args:
            query: The query string

        Returns:
            Embedding vector as numpy array, or None if model not available
        """
        if self._embedding_model is None:
            return None

        try:
            embedding = self._embedding_model.encode(
                normalize_query(query),
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            return embedding
        except Exception:
            return None

    def _rebuild_embeddings_matrix(self) -> None:
        """Rebuild the embeddings matrix from cached entries."""
        if self._embedding_model is None or not NUMPY_AVAILABLE:
            return

        self._embedding_keys = []
        embeddings = []

        for key, entry in self._cache.items():
            if entry.embedding is not None and not entry.is_expired():
                self._embedding_keys.append(key)
                embeddings.append(entry.embedding)

        if embeddings:
            self._embeddings_matrix = np.array(embeddings, dtype=np.float32)
        else:
            self._embeddings_matrix = None

    def _find_semantic_match(self, query: str) -> Optional[CachedReformulation]:
        """
        Find a semantically similar cached query.

        Args:
            query: The query to match

        Returns:
            CachedReformulation if similarity exceeds threshold, None otherwise
        """
        if not NUMPY_AVAILABLE:
            return None

        if self._embedding_model is None or self._embeddings_matrix is None:
            return None

        if len(self._embedding_keys) == 0:
            return None

        # Compute query embedding
        query_embedding = self._compute_embedding(query)
        if query_embedding is None:
            return None

        # Compute similarities
        query_embedding = query_embedding.reshape(1, -1)
        similarities = np.dot(self._embeddings_matrix, query_embedding.T).flatten()

        # Find best match
        best_idx = np.argmax(similarities)
        best_similarity = similarities[best_idx]

        if best_similarity >= self.semantic_threshold:
            cache_key = self._embedding_keys[best_idx]
            entry = self._cache.get(cache_key)

            if entry and not entry.is_expired():
                return entry

        return None

    def get(self, query: str) -> Optional[CachedReformulation]:
        """
        Retrieve cached reformulations for a query.

        Tries exact match first, then falls back to semantic matching.

        Args:
            query: The query string

        Returns:
            CachedReformulation if found, None otherwise
        """
        if not self.enabled:
            return None

        with self._lock:
            normalized = normalize_query(query)
            cache_key = generate_cache_key(query)

            # Try exact match first
            if cache_key in self._cache:
                entry = self._cache[cache_key]

                if entry.is_expired():
                    # Remove expired entry
                    del self._cache[cache_key]
                    self._stats.expirations += 1
                    self._stats.size = len(self._cache)
                else:
                    # Cache hit - move to end (most recently used)
                    self._cache.move_to_end(cache_key)
                    entry.increment_hit()
                    self._stats.hits += 1
                    return entry

            # Try semantic match
            semantic_match = self._find_semantic_match(query)
            if semantic_match:
                semantic_match.increment_hit()
                self._stats.semantic_hits += 1
                return semantic_match

            # Cache miss
            self._stats.misses += 1
            return None

    def set(
        self,
        query: str,
        reformulations: List[str],
        ttl: Optional[int] = None,
    ) -> None:
        """
        Store reformulations in the cache.

        Args:
            query: The original query string
            reformulations: List of reformulated queries
            ttl: Optional TTL override
        """
        if not self.enabled:
            return

        with self._lock:
            cache_key = generate_cache_key(query)

            # Compute embedding for semantic matching
            embedding = self._compute_embedding(query)
            embedding_list = embedding.tolist() if embedding is not None else None

            # Create cache entry
            entry = CachedReformulation(
                original_query=normalize_query(query),
                reformulations=reformulations,
                timestamp=time.time(),
                hit_count=0,
                embedding=embedding_list,
                ttl=ttl or self.ttl,
            )

            # Remove existing entry if present
            if cache_key in self._cache:
                del self._cache[cache_key]

            # Evict LRU entries if at capacity
            while len(self._cache) >= self.max_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                self._stats.expirations += 1

            # Add new entry
            self._cache[cache_key] = entry
            self._stats.size = len(self._cache)

            # Rebuild embeddings matrix if semantic matching enabled
            if embedding is not None:
                self._rebuild_embeddings_matrix()

    def delete(self, query: str) -> bool:
        """
        Delete a specific cache entry.

        Args:
            query: The query to delete

        Returns:
            True if entry was deleted, False if not found
        """
        if not self.enabled:
            return False

        with self._lock:
            cache_key = generate_cache_key(query)

            if cache_key in self._cache:
                del self._cache[cache_key]
                self._stats.size = len(self._cache)
                self._rebuild_embeddings_matrix()
                return True

            return False

    def clear(self) -> int:
        """
        Clear all cache entries.

        Returns:
            Number of entries cleared
        """
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._embeddings_matrix = None
            self._embedding_keys = []
            self._stats.size = 0
            return count

    def cleanup_expired(self) -> int:
        """
        Remove all expired entries from the cache.

        Returns:
            Number of entries removed
        """
        with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.is_expired()
            ]

            for key in expired_keys:
                del self._cache[key]

            self._stats.expirations += len(expired_keys)
            self._stats.size = len(self._cache)

            if expired_keys:
                self._rebuild_embeddings_matrix()

            return len(expired_keys)

    def stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        with self._lock:
            self._stats.size = len(self._cache)
            self._stats.semantic_enabled = self._embedding_model is not None
            return self._stats.to_dict()

    def set_embedding_model(self, model: Any) -> None:
        """
        Set or update the embedding model for semantic matching.

        Args:
            model: SentenceTransformer model instance
        """
        with self._lock:
            self._embedding_model = model
            self._stats.semantic_enabled = model is not None

            # Recompute embeddings for existing entries
            if model is not None:
                for key, entry in self._cache.items():
                    if not entry.is_expired():
                        embedding = self._compute_embedding(entry.original_query)
                        if embedding is not None:
                            entry.embedding = embedding.tolist()

                self._rebuild_embeddings_matrix()

    @property
    def size(self) -> int:
        """Get current cache size."""
        with self._lock:
            return len(self._cache)

    @property
    def keys(self) -> List[str]:
        """Get all cache keys."""
        with self._lock:
            return list(self._cache.keys())

    @property
    def queries(self) -> List[str]:
        """Get all cached original queries."""
        with self._lock:
            return [entry.original_query for entry in self._cache.values()]


# =============================================================================
# Module-Level Singleton and Factory
# =============================================================================

_default_cache: Optional[ReformulationCache] = None
_cache_lock = threading.Lock()


def get_reformulation_cache(
    enabled: bool = REFORMULATION_CACHE_ENABLED,
    ttl: int = REFORMULATION_CACHE_TTL,
    max_size: int = REFORMULATION_CACHE_MAX_SIZE,
    semantic_threshold: float = REFORMULATION_SEMANTIC_THRESHOLD,
    embedding_model: Optional[Any] = None,
) -> ReformulationCache:
    """
    Get or create the default ReformulationCache singleton.

    Thread-safe factory function that returns a single cache instance.

    Args:
        enabled: Whether caching is enabled
        ttl: Default TTL in seconds
        max_size: Maximum cache size
        semantic_threshold: Similarity threshold for semantic matching
        embedding_model: Optional SentenceTransformer model

    Returns:
        The default ReformulationCache instance
    """
    global _default_cache

    with _cache_lock:
        if _default_cache is None:
            _default_cache = ReformulationCache(
                enabled=enabled,
                ttl=ttl,
                max_size=max_size,
                semantic_threshold=semantic_threshold,
                embedding_model=embedding_model,
            )
        return _default_cache


def reset_reformulation_cache() -> None:
    """
    Reset the default cache singleton.

    Useful for testing or when configuration changes.
    """
    global _default_cache

    with _cache_lock:
        if _default_cache is not None:
            _default_cache.clear()
        _default_cache = None
