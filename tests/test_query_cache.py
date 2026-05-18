"""
Unit tests for the CHORUS Query Cache module.

Tests cover:
- CachedResult dataclass functionality
- MemoryBackend LRU cache operations
- QueryCache main class operations
- Cache key generation
- TTL expiration
- Cache statistics
- Cache invalidation
- Thread safety (basic)
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cache.query_cache import (
    CachedResult,
    CacheStats,
    MemoryBackend,
    QueryCache,
    compute_source_hash,
    get_cache,
    reset_cache,
    CACHE_TTL,
    CACHE_MAX_SIZE,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_result():
    """Create a sample cached result."""
    return {
        "query": "machine learning",
        "results": [
            {"text": "Test result", "score": 0.95}
        ],
        "llm_reformulation": True,
        "reranking": True,
        "mmr_diversity": True,
    }


@pytest.fixture
def cached_result(sample_result):
    """Create a CachedResult instance."""
    return CachedResult(
        query="machine learning",
        result=sample_result,
        timestamp=time.time(),
        hit_count=0,
        source_hash="abc123",
        ttl=300,
        endpoint="/search",
        params_hash="hash123",
    )


@pytest.fixture
def memory_backend():
    """Create a MemoryBackend instance."""
    return MemoryBackend(max_size=10)


@pytest.fixture
def query_cache():
    """Create a QueryCache instance for testing."""
    reset_cache()  # Ensure clean state
    return QueryCache(enabled=True, ttl=300, max_size=100, backend="memory")


# =============================================================================
# CachedResult Tests
# =============================================================================

class TestCachedResult:
    """Tests for CachedResult dataclass."""

    def test_cached_result_creation(self, sample_result):
        """Test CachedResult can be created with all fields."""
        result = CachedResult(
            query="test query",
            result=sample_result,
            timestamp=time.time(),
        )
        assert result.query == "test query"
        assert result.result == sample_result
        assert result.hit_count == 0

    def test_is_expired_false_when_fresh(self, cached_result):
        """Test is_expired returns False for fresh entries."""
        assert not cached_result.is_expired()

    def test_is_expired_true_when_old(self):
        """Test is_expired returns True for old entries."""
        result = CachedResult(
            query="test",
            result={},
            timestamp=time.time() - 400,  # 400 seconds ago
            ttl=300,  # 5 minute TTL
        )
        assert result.is_expired()

    def test_increment_hit(self, cached_result):
        """Test hit count increments correctly."""
        assert cached_result.hit_count == 0
        cached_result.increment_hit()
        assert cached_result.hit_count == 1
        cached_result.increment_hit()
        assert cached_result.hit_count == 2

    def test_age_seconds(self, cached_result):
        """Test age_seconds returns correct value."""
        age = cached_result.age_seconds()
        assert 0 <= age < 1  # Should be very recent

    def test_to_dict_and_from_dict(self, cached_result):
        """Test serialization round-trip."""
        data = cached_result.to_dict()
        restored = CachedResult.from_dict(data)

        assert restored.query == cached_result.query
        assert restored.result == cached_result.result
        assert restored.hit_count == cached_result.hit_count
        assert restored.source_hash == cached_result.source_hash


# =============================================================================
# CacheStats Tests
# =============================================================================

class TestCacheStats:
    """Tests for CacheStats dataclass."""

    def test_hit_rate_zero_when_empty(self):
        """Test hit rate is 0 when no operations."""
        stats = CacheStats()
        assert stats.hit_rate == 0.0

    def test_hit_rate_calculation(self):
        """Test hit rate calculates correctly."""
        stats = CacheStats(hits=30, misses=70)
        assert stats.hit_rate == 30.0

    def test_hit_rate_perfect(self):
        """Test hit rate with all hits."""
        stats = CacheStats(hits=100, misses=0)
        assert stats.hit_rate == 100.0

    def test_to_dict(self):
        """Test stats can be converted to dict."""
        stats = CacheStats(hits=10, misses=5, size=50)
        data = stats.to_dict()

        assert data["hits"] == 10
        assert data["misses"] == 5
        assert data["size"] == 50
        assert "hit_rate_percent" in data
        assert "uptime_seconds" in data


# =============================================================================
# MemoryBackend Tests
# =============================================================================

class TestMemoryBackend:
    """Tests for MemoryBackend LRU cache."""

    def test_set_and_get(self, memory_backend, cached_result):
        """Test basic set and get operations."""
        memory_backend.set("key1", cached_result)
        retrieved = memory_backend.get("key1")

        assert retrieved is not None
        assert retrieved.query == cached_result.query

    def test_get_nonexistent_key(self, memory_backend):
        """Test get returns None for missing keys."""
        result = memory_backend.get("nonexistent")
        assert result is None

    def test_delete(self, memory_backend, cached_result):
        """Test delete removes entry."""
        memory_backend.set("key1", cached_result)
        assert memory_backend.get("key1") is not None

        deleted = memory_backend.delete("key1")
        assert deleted is True
        assert memory_backend.get("key1") is None

    def test_delete_nonexistent(self, memory_backend):
        """Test delete returns False for missing keys."""
        assert memory_backend.delete("nonexistent") is False

    def test_clear(self, memory_backend, cached_result):
        """Test clear removes all entries."""
        memory_backend.set("key1", cached_result)
        memory_backend.set("key2", cached_result)

        count = memory_backend.clear()
        assert count == 2
        assert memory_backend.size() == 0

    def test_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        backend = MemoryBackend(max_size=3)

        for i in range(5):
            result = CachedResult(
                query=f"query{i}",
                result={},
                timestamp=time.time(),
            )
            backend.set(f"key{i}", result)

        # Only last 3 should remain
        assert backend.size() == 3
        assert backend.get("key0") is None  # Evicted
        assert backend.get("key1") is None  # Evicted
        assert backend.get("key2") is not None
        assert backend.get("key3") is not None
        assert backend.get("key4") is not None

    def test_get_moves_to_end(self):
        """Test that get moves item to end (most recent)."""
        backend = MemoryBackend(max_size=3)

        for i in range(3):
            result = CachedResult(
                query=f"query{i}",
                result={},
                timestamp=time.time(),
            )
            backend.set(f"key{i}", result)

        # Access key0 to make it most recent
        backend.get("key0")

        # Add new item, should evict key1 (now oldest)
        new_result = CachedResult(query="new", result={}, timestamp=time.time())
        backend.set("key_new", new_result)

        assert backend.get("key0") is not None  # Should still exist
        assert backend.get("key1") is None  # Should be evicted
        assert backend.get("key2") is not None
        assert backend.get("key_new") is not None

    def test_expired_entry_returns_none(self, memory_backend):
        """Test that expired entries are not returned."""
        expired_result = CachedResult(
            query="test",
            result={},
            timestamp=time.time() - 400,  # 400 seconds ago
            ttl=300,  # 5 minute TTL
        )
        memory_backend.set("expired_key", expired_result)

        result = memory_backend.get("expired_key")
        assert result is None

    def test_keys_list(self, memory_backend, cached_result):
        """Test keys() returns all keys."""
        memory_backend.set("key1", cached_result)
        memory_backend.set("key2", cached_result)

        keys = memory_backend.keys()
        assert len(keys) == 2
        assert "key1" in keys
        assert "key2" in keys

    def test_cleanup_expired(self, memory_backend):
        """Test cleanup_expired removes old entries."""
        fresh = CachedResult(query="fresh", result={}, timestamp=time.time(), ttl=300)
        expired = CachedResult(query="expired", result={}, timestamp=time.time() - 400, ttl=300)

        memory_backend.set("fresh_key", fresh)
        memory_backend.set("expired_key", expired)

        count = memory_backend.cleanup_expired()
        assert count == 1
        assert memory_backend.size() == 1
        assert memory_backend.get("fresh_key") is not None


# =============================================================================
# QueryCache Tests
# =============================================================================

class TestQueryCache:
    """Tests for QueryCache main class."""

    def test_generate_key_deterministic(self, query_cache):
        """Test key generation is deterministic."""
        key1 = query_cache.generate_key("/search", "machine learning", {"top_k": 5})
        key2 = query_cache.generate_key("/search", "machine learning", {"top_k": 5})
        assert key1 == key2

    def test_generate_key_different_queries(self, query_cache):
        """Test different queries produce different keys."""
        key1 = query_cache.generate_key("/search", "machine learning", {})
        key2 = query_cache.generate_key("/search", "deep learning", {})
        assert key1 != key2

    def test_generate_key_different_params(self, query_cache):
        """Test different params produce different keys."""
        key1 = query_cache.generate_key("/search", "test", {"top_k": 5})
        key2 = query_cache.generate_key("/search", "test", {"top_k": 10})
        assert key1 != key2

    def test_generate_key_case_insensitive(self, query_cache):
        """Test key generation is case insensitive."""
        key1 = query_cache.generate_key("/search", "Machine Learning", {})
        key2 = query_cache.generate_key("/search", "machine learning", {})
        assert key1 == key2

    def test_generate_key_trims_whitespace(self, query_cache):
        """Test key generation trims whitespace."""
        key1 = query_cache.generate_key("/search", "  machine learning  ", {})
        key2 = query_cache.generate_key("/search", "machine learning", {})
        assert key1 == key2

    def test_set_and_get(self, query_cache, sample_result):
        """Test basic set and get operations."""
        key = query_cache.generate_key("/search", "test", {})
        query_cache.set(key, "test", sample_result)

        cached = query_cache.get(key)
        assert cached is not None
        assert cached.result == sample_result

    def test_get_miss_returns_none(self, query_cache):
        """Test get returns None for cache miss."""
        result = query_cache.get("nonexistent_key")
        assert result is None

    def test_stats_tracking(self, query_cache, sample_result):
        """Test statistics are tracked correctly."""
        key = query_cache.generate_key("/search", "test", {})

        # Initial stats
        stats = query_cache.stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0

        # Miss
        query_cache.get("nonexistent")
        stats = query_cache.stats()
        assert stats["misses"] == 1

        # Set and hit
        query_cache.set(key, "test", sample_result)
        query_cache.get(key)
        stats = query_cache.stats()
        assert stats["hits"] == 1

    def test_clear(self, query_cache, sample_result):
        """Test clear removes all entries."""
        key1 = query_cache.generate_key("/search", "test1", {})
        key2 = query_cache.generate_key("/search", "test2", {})

        query_cache.set(key1, "test1", sample_result)
        query_cache.set(key2, "test2", sample_result)

        assert query_cache.size == 2

        count = query_cache.clear()
        assert count == 2
        assert query_cache.size == 0

    def test_invalidate(self, query_cache, sample_result):
        """Test invalidate clears cache and updates source hash."""
        key = query_cache.generate_key("/search", "test", {})
        query_cache.set(key, "test", sample_result)

        query_cache.invalidate(new_source_hash="new_hash_123")

        assert query_cache.size == 0
        stats = query_cache.stats()
        assert stats["invalidations"] == 1

    def test_source_hash_invalidation(self, query_cache, sample_result):
        """Test entries are invalidated when source hash changes."""
        key = query_cache.generate_key("/search", "test", {})

        # Set with source hash
        query_cache.set_source_hash("hash1")
        query_cache.set(key, "test", sample_result)

        # First get should work
        assert query_cache.get(key) is not None

        # Change source hash
        query_cache.set_source_hash("hash2")

        # Now get should return None (stale entry)
        assert query_cache.get(key) is None

    def test_disabled_cache(self, sample_result):
        """Test disabled cache doesn't store or retrieve."""
        cache = QueryCache(enabled=False)

        key = cache.generate_key("/search", "test", {})
        cache.set(key, "test", sample_result)

        result = cache.get(key)
        assert result is None

    def test_delete(self, query_cache, sample_result):
        """Test delete removes specific entry."""
        key = query_cache.generate_key("/search", "test", {})
        query_cache.set(key, "test", sample_result)

        assert query_cache.get(key) is not None

        query_cache.delete(key)
        assert query_cache.get(key) is None

    def test_keys_property(self, query_cache, sample_result):
        """Test keys property returns all cache keys."""
        key1 = query_cache.generate_key("/search", "test1", {})
        key2 = query_cache.generate_key("/search", "test2", {})

        query_cache.set(key1, "test1", sample_result)
        query_cache.set(key2, "test2", sample_result)

        keys = query_cache.keys
        assert len(keys) == 2
        assert key1 in keys
        assert key2 in keys


# =============================================================================
# Thread Safety Tests
# =============================================================================

class TestThreadSafety:
    """Basic thread safety tests."""

    def test_concurrent_writes(self, query_cache, sample_result):
        """Test concurrent writes don't corrupt cache."""
        errors = []

        def write_cache(i):
            try:
                key = query_cache.generate_key("/search", f"query{i}", {})
                query_cache.set(key, f"query{i}", sample_result)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=write_cache, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert query_cache.size <= 20  # May have some due to timing

    def test_concurrent_reads(self, query_cache, sample_result):
        """Test concurrent reads work correctly."""
        key = query_cache.generate_key("/search", "test", {})
        query_cache.set(key, "test", sample_result)

        results = []
        errors = []

        def read_cache():
            try:
                result = query_cache.get(key)
                results.append(result is not None)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=read_cache) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert all(results)  # All reads should succeed


# =============================================================================
# Helper Function Tests
# =============================================================================

class TestHelperFunctions:
    """Tests for helper functions."""

    def test_compute_source_hash_nonexistent_path(self):
        """Test compute_source_hash with non-existent path."""
        result = compute_source_hash("/nonexistent/path")
        assert result == ""

    def test_get_cache_singleton(self):
        """Test get_cache returns singleton."""
        reset_cache()
        cache1 = get_cache()
        cache2 = get_cache()
        assert cache1 is cache2

    def test_reset_cache(self):
        """Test reset_cache clears singleton."""
        reset_cache()
        cache1 = get_cache()
        cache1.set("test_key", "test", {"data": "value"})

        reset_cache()
        cache2 = get_cache()

        # Should be a fresh cache
        assert cache2.get("test_key") is None


# =============================================================================
# Integration Tests
# =============================================================================

class TestCacheIntegration:
    """Integration tests for cache with realistic usage."""

    def test_search_caching_pattern(self, query_cache):
        """Test typical search endpoint caching pattern."""
        # Simulate search endpoint
        query = "machine learning"
        params = {"top_k": 5, "llm": True, "rerank": True}

        # Generate key
        cache_key = query_cache.generate_key("/search", query, params)

        # First request - miss
        cached = query_cache.get(cache_key)
        assert cached is None

        # Simulate search execution
        result = {
            "query": query,
            "results": [{"text": "Test result", "score": 0.95}],
            "cache_hit": False,
        }

        # Cache the result
        query_cache.set(cache_key, query, result, endpoint="/search")

        # Second request - hit
        cached = query_cache.get(cache_key)
        assert cached is not None
        assert cached.result["query"] == query

        # Check stats
        stats = query_cache.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1

    def test_ttl_expiration(self):
        """Test TTL expiration works correctly."""
        cache = QueryCache(enabled=True, ttl=1, max_size=100, backend="memory")

        key = cache.generate_key("/search", "test", {})
        cache.set(key, "test", {"data": "value"})

        # Should exist immediately
        assert cache.get(key) is not None

        # Wait for expiration
        time.sleep(1.5)

        # Should be expired
        assert cache.get(key) is None


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
