"""
CHORUS Cache Base Module

Common cache utilities and base classes for the caching system.
Provides thread-safe LRU cache backends and configuration helpers.

This module consolidates shared functionality between:
- QueryCache (search result caching)
- ReformulationCache (LLM query reformulation caching)
"""

import hashlib
import json
import os
import time
import threading
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Generic, List, Optional, Protocol, TypeVar

# =============================================================================
# Environment Configuration Helpers (imported from centralized config)
# =============================================================================

from config import get_env_bool, get_env_int, get_env_float


# =============================================================================
# Cache Entry Protocol
# =============================================================================

class CacheEntryProtocol(Protocol):
    """Protocol that cache entries must implement."""

    timestamp: float
    ttl: int
    hit_count: int

    def is_expired(self) -> bool:
        """Check if entry has expired."""
        ...

    def increment_hit(self) -> None:
        """Increment hit count."""
        ...

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        ...

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CacheEntryProtocol":
        """Create from dictionary."""
        ...


# Type variable for cache entries
T = TypeVar("T")


# =============================================================================
# Base Cache Entry
# =============================================================================

@dataclass
class BaseCacheEntry:
    """
    Base class for cache entries with common TTL and hit tracking.

    Subclasses should add their specific data fields.
    """
    timestamp: float = field(default_factory=time.time)
    hit_count: int = 0
    ttl: int = 300  # Default 5 minutes

    def is_expired(self) -> bool:
        """Check if this cache entry has expired based on TTL."""
        return time.time() > self.timestamp + self.ttl

    def increment_hit(self) -> None:
        """Increment the hit counter."""
        self.hit_count += 1

    def age_seconds(self) -> float:
        """Return the age of this cache entry in seconds."""
        return time.time() - self.timestamp


# =============================================================================
# Base Cache Stats
# =============================================================================

@dataclass
class BaseCacheStats:
    """
    Base statistics for cache monitoring.

    Subclasses can add additional metrics.
    """
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size: int = 0
    max_size: int = 1000
    ttl: int = 300
    enabled: bool = True
    start_time: float = field(default_factory=time.time)

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
            "size": self.size,
            "max_size": self.max_size,
            "hit_rate_percent": round(self.hit_rate, 2),
            "ttl_seconds": self.ttl,
            "enabled": self.enabled,
            "uptime_seconds": round(self.uptime_seconds, 2),
        }


# =============================================================================
# Cache Backend Abstract Base Class
# =============================================================================

class CacheBackend(ABC, Generic[T]):
    """
    Abstract base class for cache backends.

    Implementations must provide thread-safe get/set/delete operations.
    Generic over the type of cache entry stored.
    """

    @abstractmethod
    def get(self, key: str) -> Optional[T]:
        """
        Retrieve a cached entry by key.

        Args:
            key: The cache key

        Returns:
            Cache entry if found and not expired, None otherwise
        """
        pass

    @abstractmethod
    def set(self, key: str, entry: T) -> None:
        """
        Store a cache entry.

        Args:
            key: The cache key
            entry: The entry to store
        """
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """
        Delete a cached entry.

        Args:
            key: The cache key

        Returns:
            True if entry was deleted, False if not found
        """
        pass

    @abstractmethod
    def clear(self) -> int:
        """
        Clear all cached entries.

        Returns:
            Number of entries cleared
        """
        pass

    @abstractmethod
    def size(self) -> int:
        """
        Get the current number of entries in the cache.

        Returns:
            Number of cached entries
        """
        pass

    @abstractmethod
    def keys(self) -> List[str]:
        """
        Get all cache keys.

        Returns:
            List of cache keys
        """
        pass


# =============================================================================
# In-Memory LRU Cache Backend
# =============================================================================

class LRUMemoryBackend(CacheBackend[T]):
    """
    In-memory LRU cache backend using OrderedDict.

    Thread-safe implementation using a lock for all operations.
    Automatically evicts least recently used entries when max_size is reached.

    Works with any cache entry type that has is_expired() method.
    """

    def __init__(self, max_size: int = 1000):
        """
        Initialize the memory cache backend.

        Args:
            max_size: Maximum number of entries to store
        """
        self.max_size = max_size
        self._cache: OrderedDict[str, T] = OrderedDict()
        self._lock = threading.RLock()
        self._eviction_count = 0

    def get(self, key: str) -> Optional[T]:
        """
        Retrieve a cached entry, moving it to the end (most recent).

        Automatically removes expired entries.
        """
        with self._lock:
            if key not in self._cache:
                return None

            entry = self._cache[key]

            # Check if expired (entry must have is_expired method)
            if hasattr(entry, 'is_expired') and entry.is_expired():
                del self._cache[key]
                self._eviction_count += 1
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            return entry

    def set(self, key: str, entry: T) -> None:
        """
        Store a cache entry, evicting LRU entries if necessary.
        """
        with self._lock:
            # If key exists, remove it first (to update position)
            if key in self._cache:
                del self._cache[key]

            # Evict LRU entries if at capacity
            while len(self._cache) >= self.max_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                self._eviction_count += 1

            # Add new entry at end (most recent)
            self._cache[key] = entry

    def delete(self, key: str) -> bool:
        """Delete a specific cache entry."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def clear(self) -> int:
        """Clear all cache entries."""
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            return count

    def size(self) -> int:
        """Get current cache size."""
        with self._lock:
            return len(self._cache)

    def keys(self) -> List[str]:
        """Get all cache keys."""
        with self._lock:
            return list(self._cache.keys())

    @property
    def eviction_count(self) -> int:
        """Get total number of evictions."""
        return self._eviction_count

    def cleanup_expired(self) -> int:
        """
        Remove all expired entries from the cache.

        Returns:
            Number of entries removed
        """
        with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if hasattr(entry, 'is_expired') and entry.is_expired()
            ]
            for key in expired_keys:
                del self._cache[key]
            self._eviction_count += len(expired_keys)
            return len(expired_keys)

    def items(self) -> List[tuple]:
        """Get all key-value pairs (thread-safe copy)."""
        with self._lock:
            return list(self._cache.items())


# =============================================================================
# Key Generation Utilities
# =============================================================================

def generate_cache_key(*parts: str, hash_length: int = 16) -> str:
    """
    Generate a unique cache key from multiple string parts.

    Args:
        *parts: String parts to combine into key
        hash_length: Length of hash to return (default 16)

    Returns:
        SHA256 hash of combined parts (first hash_length chars)
    """
    combined = ":".join(str(p) for p in parts if p)
    return hashlib.sha256(combined.encode()).hexdigest()[:hash_length]


def normalize_query(query: str) -> str:
    """
    Normalize a query string for cache key generation.

    Normalization steps:
    1. Convert to lowercase
    2. Strip leading/trailing whitespace
    3. Collapse multiple whitespaces to single space

    Args:
        query: The raw query string

    Returns:
        Normalized query string
    """
    if not query:
        return ""
    import re
    normalized = query.lower().strip()
    normalized = re.sub(r'\s+', ' ', normalized)
    return normalized


__all__ = [
    # Environment helpers
    "get_env_bool",
    "get_env_int",
    "get_env_float",
    # Base classes
    "BaseCacheEntry",
    "BaseCacheStats",
    "CacheBackend",
    "LRUMemoryBackend",
    # Utilities
    "generate_cache_key",
    "normalize_query",
]
