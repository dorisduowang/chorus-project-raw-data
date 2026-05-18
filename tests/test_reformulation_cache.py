"""
Unit tests for the CHORUS Reformulation Cache module.

Tests cover:
- CachedReformulation dataclass functionality
- ReformulationCacheStats statistics
- Query normalization
- Cache key generation
- Exact match caching
- TTL expiration
- Cache size limits (LRU eviction)
- Statistics tracking (hits, misses, semantic_hits, expirations)
- Thread safety
- Semantic matching (with mocked embedding model)
"""

import pytest
import time
import threading
from unittest.mock import Mock, MagicMock, patch
import sys
from pathlib import Path

# Make numpy optional (only needed for semantic matching tests)
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cache.reformulation_cache import (
    CachedReformulation,
    ReformulationCacheStats,
    ReformulationCache,
    normalize_query,
    generate_cache_key,
    get_reformulation_cache,
    reset_reformulation_cache,
    REFORMULATION_CACHE_TTL,
    REFORMULATION_CACHE_MAX_SIZE,
    REFORMULATION_SEMANTIC_THRESHOLD,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_reformulations():
    """Create sample reformulations list."""
    return [
        "machine learning research papers 2024",
        "ML academic publications machine learning",
    ]


@pytest.fixture
def cached_reformulation(sample_reformulations):
    """Create a CachedReformulation instance."""
    return CachedReformulation(
        original_query="machine learning papers",
        reformulations=sample_reformulations,
        timestamp=time.time(),
        hit_count=0,
        embedding=None,
        ttl=300,
    )


@pytest.fixture
def reformulation_cache():
    """Create a ReformulationCache instance for testing."""
    reset_reformulation_cache()  # Ensure clean state
    return ReformulationCache(enabled=True, ttl=300, max_size=100)


@pytest.fixture
def mock_embedding_model():
    """Create a mock embedding model."""
    if not NUMPY_AVAILABLE:
        pytest.skip("numpy not available")

    model = MagicMock()

    def mock_encode(text, convert_to_numpy=True, normalize_embeddings=True):
        # Generate deterministic embeddings based on text
        np.random.seed(hash(text) % 2**32)
        embedding = np.random.randn(384).astype(np.float32)
        if normalize_embeddings:
            embedding = embedding / np.linalg.norm(embedding)
        return embedding

    model.encode = mock_encode
    return model


# =============================================================================
# CachedReformulation Tests
# =============================================================================

class TestCachedReformulation:
    """Tests for CachedReformulation dataclass."""

    def test_cached_reformulation_creation(self, sample_reformulations):
        """Test CachedReformulation can be created with all fields."""
        entry = CachedReformulation(
            original_query="test query",
            reformulations=sample_reformulations,
            timestamp=time.time(),
        )
        assert entry.original_query == "test query"
        assert entry.reformulations == sample_reformulations
        assert entry.hit_count == 0

    def test_is_expired_false_when_fresh(self, cached_reformulation):
        """Test is_expired returns False for fresh entries."""
        assert not cached_reformulation.is_expired()

    def test_is_expired_true_when_old(self, sample_reformulations):
        """Test is_expired returns True for old entries."""
        entry = CachedReformulation(
            original_query="test",
            reformulations=sample_reformulations,
            timestamp=time.time() - 400,  # 400 seconds ago
            ttl=300,  # 5 minute TTL
        )
        assert entry.is_expired()

    def test_increment_hit(self, cached_reformulation):
        """Test hit count increments correctly."""
        assert cached_reformulation.hit_count == 0
        cached_reformulation.increment_hit()
        assert cached_reformulation.hit_count == 1
        cached_reformulation.increment_hit()
        assert cached_reformulation.hit_count == 2

    def test_age_seconds(self, cached_reformulation):
        """Test age_seconds returns correct value."""
        age = cached_reformulation.age_seconds()
        assert 0 <= age < 1  # Should be very recent

    def test_to_dict_and_from_dict(self, cached_reformulation):
        """Test serialization round-trip."""
        data = cached_reformulation.to_dict()
        restored = CachedReformulation.from_dict(data)

        assert restored.original_query == cached_reformulation.original_query
        assert restored.reformulations == cached_reformulation.reformulations
        assert restored.hit_count == cached_reformulation.hit_count

    def test_to_dict_with_embedding(self, sample_reformulations):
        """Test serialization with embedding."""
        embedding = [0.1, 0.2, 0.3, 0.4]
        entry = CachedReformulation(
            original_query="test",
            reformulations=sample_reformulations,
            timestamp=time.time(),
            embedding=embedding,
        )

        data = entry.to_dict()
        assert data["embedding"] == embedding

        restored = CachedReformulation.from_dict(data)
        assert restored.embedding == embedding


# =============================================================================
# ReformulationCacheStats Tests
# =============================================================================

class TestReformulationCacheStats:
    """Tests for ReformulationCacheStats dataclass."""

    def test_hit_rate_zero_when_empty(self):
        """Test hit rate is 0 when no operations."""
        stats = ReformulationCacheStats()
        assert stats.hit_rate == 0.0

    def test_hit_rate_calculation(self):
        """Test hit rate calculates correctly."""
        stats = ReformulationCacheStats(hits=20, misses=80, semantic_hits=10)
        # Total hits = 30, total = 110
        expected_rate = (30 / 110) * 100
        assert abs(stats.hit_rate - expected_rate) < 0.01

    def test_exact_hit_rate(self):
        """Test exact hit rate calculation."""
        stats = ReformulationCacheStats(hits=30, misses=70)
        assert stats.exact_hit_rate == 30.0

    def test_semantic_hit_rate(self):
        """Test semantic hit rate calculation."""
        stats = ReformulationCacheStats(hits=20, misses=60, semantic_hits=20)
        # Total = 100, semantic = 20
        assert stats.semantic_hit_rate == 20.0

    def test_total_hits(self):
        """Test total hits combines exact and semantic."""
        stats = ReformulationCacheStats(hits=50, semantic_hits=25)
        assert stats.total_hits == 75

    def test_estimated_savings(self):
        """Test estimated savings calculation."""
        stats = ReformulationCacheStats(hits=100, semantic_hits=50)
        savings = stats.estimated_savings

        assert savings["api_calls_saved"] == 150
        assert savings["estimated_cost_saved_usd"] == 0.15  # 150 * $0.001
        assert savings["estimated_time_saved_ms"] > 0

    def test_to_dict(self):
        """Test stats can be converted to dict."""
        stats = ReformulationCacheStats(
            hits=10,
            misses=5,
            semantic_hits=3,
            size=50,
            expirations=2,
        )
        data = stats.to_dict()

        assert data["hits"] == 10
        assert data["misses"] == 5
        assert data["semantic_hits"] == 3
        assert data["total_hits"] == 13
        assert data["expirations"] == 2
        assert data["size"] == 50
        assert "hit_rate_percent" in data
        assert "uptime_seconds" in data
        assert "api_calls_saved" in data
        assert "estimated_cost_saved_usd" in data


# =============================================================================
# Query Normalization Tests
# =============================================================================

class TestQueryNormalization:
    """Tests for query normalization."""

    def test_normalize_lowercase(self):
        """Test normalization converts to lowercase."""
        assert normalize_query("Machine Learning") == "machine learning"

    def test_normalize_strips_whitespace(self):
        """Test normalization strips whitespace."""
        assert normalize_query("  machine learning  ") == "machine learning"

    def test_normalize_collapses_whitespace(self):
        """Test normalization collapses multiple whitespaces."""
        assert normalize_query("machine   learning") == "machine learning"

    def test_normalize_strips_punctuation(self):
        """Test normalization strips leading/trailing punctuation."""
        assert normalize_query("machine learning?") == "machine learning"
        assert normalize_query("...machine learning...") == "machine learning"

    def test_normalize_keeps_internal_punctuation(self):
        """Test normalization keeps internal punctuation."""
        assert normalize_query("what's machine learning?") == "what's machine learning"

    def test_normalize_empty_string(self):
        """Test normalization handles empty string."""
        assert normalize_query("") == ""
        assert normalize_query("   ") == ""

    def test_normalize_combined(self):
        """Test normalization with combined transformations."""
        assert normalize_query("  What is  Machine   Learning?  ") == "what is machine learning"


class TestCacheKeyGeneration:
    """Tests for cache key generation."""

    def test_generate_key_deterministic(self):
        """Test key generation is deterministic."""
        key1 = generate_cache_key("machine learning")
        key2 = generate_cache_key("machine learning")
        assert key1 == key2

    def test_generate_key_different_queries(self):
        """Test different queries produce different keys."""
        key1 = generate_cache_key("machine learning")
        key2 = generate_cache_key("deep learning")
        assert key1 != key2

    def test_generate_key_case_insensitive(self):
        """Test key generation is case insensitive."""
        key1 = generate_cache_key("Machine Learning")
        key2 = generate_cache_key("machine learning")
        assert key1 == key2

    def test_generate_key_whitespace_insensitive(self):
        """Test key generation is whitespace insensitive."""
        key1 = generate_cache_key("  machine   learning  ")
        key2 = generate_cache_key("machine learning")
        assert key1 == key2

    def test_generate_key_length(self):
        """Test generated key has expected length."""
        key = generate_cache_key("test query")
        assert len(key) == 16  # First 16 chars of SHA256


# =============================================================================
# ReformulationCache Basic Tests
# =============================================================================

class TestReformulationCacheBasic:
    """Basic tests for ReformulationCache."""

    def test_set_and_get(self, reformulation_cache, sample_reformulations):
        """Test basic set and get operations."""
        reformulation_cache.set("machine learning", sample_reformulations)
        cached = reformulation_cache.get("machine learning")

        assert cached is not None
        assert cached.reformulations == sample_reformulations

    def test_get_miss_returns_none(self, reformulation_cache):
        """Test get returns None for cache miss."""
        result = reformulation_cache.get("nonexistent query")
        assert result is None

    def test_get_exact_match_case_insensitive(self, reformulation_cache, sample_reformulations):
        """Test exact match is case insensitive."""
        reformulation_cache.set("Machine Learning", sample_reformulations)
        cached = reformulation_cache.get("machine learning")

        assert cached is not None
        assert cached.reformulations == sample_reformulations

    def test_get_exact_match_whitespace_insensitive(self, reformulation_cache, sample_reformulations):
        """Test exact match is whitespace insensitive."""
        reformulation_cache.set("machine learning", sample_reformulations)
        cached = reformulation_cache.get("  machine   learning  ")

        assert cached is not None
        assert cached.reformulations == sample_reformulations

    def test_delete(self, reformulation_cache, sample_reformulations):
        """Test delete removes specific entry."""
        reformulation_cache.set("test query", sample_reformulations)
        assert reformulation_cache.get("test query") is not None

        deleted = reformulation_cache.delete("test query")
        assert deleted is True
        assert reformulation_cache.get("test query") is None

    def test_delete_nonexistent(self, reformulation_cache):
        """Test delete returns False for missing entries."""
        assert reformulation_cache.delete("nonexistent") is False

    def test_clear(self, reformulation_cache, sample_reformulations):
        """Test clear removes all entries."""
        reformulation_cache.set("query1", sample_reformulations)
        reformulation_cache.set("query2", sample_reformulations)

        assert reformulation_cache.size == 2

        count = reformulation_cache.clear()
        assert count == 2
        assert reformulation_cache.size == 0

    def test_disabled_cache(self, sample_reformulations):
        """Test disabled cache doesn't store or retrieve."""
        cache = ReformulationCache(enabled=False)
        cache.set("test", sample_reformulations)

        result = cache.get("test")
        assert result is None

    def test_size_property(self, reformulation_cache, sample_reformulations):
        """Test size property returns correct count."""
        assert reformulation_cache.size == 0

        reformulation_cache.set("query1", sample_reformulations)
        assert reformulation_cache.size == 1

        reformulation_cache.set("query2", sample_reformulations)
        assert reformulation_cache.size == 2

    def test_keys_property(self, reformulation_cache, sample_reformulations):
        """Test keys property returns all cache keys."""
        reformulation_cache.set("query1", sample_reformulations)
        reformulation_cache.set("query2", sample_reformulations)

        keys = reformulation_cache.keys
        assert len(keys) == 2

    def test_queries_property(self, reformulation_cache, sample_reformulations):
        """Test queries property returns original queries."""
        reformulation_cache.set("Query One", sample_reformulations)
        reformulation_cache.set("Query Two", sample_reformulations)

        queries = reformulation_cache.queries
        assert len(queries) == 2
        assert "query one" in queries  # Normalized
        assert "query two" in queries


# =============================================================================
# TTL Expiration Tests
# =============================================================================

class TestTTLExpiration:
    """Tests for TTL expiration functionality."""

    def test_expired_entry_returns_none(self, sample_reformulations):
        """Test that expired entries are not returned."""
        cache = ReformulationCache(enabled=True, ttl=1, max_size=100)
        cache.set("test query", sample_reformulations)

        # Should exist immediately
        assert cache.get("test query") is not None

        # Wait for expiration
        time.sleep(1.5)

        # Should be expired
        assert cache.get("test query") is None

    def test_cleanup_expired(self, sample_reformulations):
        """Test cleanup_expired removes old entries."""
        cache = ReformulationCache(enabled=True, ttl=1, max_size=100)
        cache.set("query1", sample_reformulations)

        # Wait for expiration
        time.sleep(1.5)

        # Add a fresh entry
        cache.set("query2", sample_reformulations)

        # Cleanup should remove expired entry
        count = cache.cleanup_expired()
        assert count == 1
        assert cache.size == 1
        assert cache.get("query2") is not None

    def test_custom_ttl_per_entry(self, sample_reformulations):
        """Test custom TTL can be set per entry."""
        cache = ReformulationCache(enabled=True, ttl=300, max_size=100)

        # Set with short TTL
        cache.set("short_ttl", sample_reformulations, ttl=1)

        # Should exist immediately
        assert cache.get("short_ttl") is not None

        # Wait for expiration
        time.sleep(1.5)

        # Should be expired
        assert cache.get("short_ttl") is None

    def test_expiration_updates_stats(self, sample_reformulations):
        """Test that expiration updates statistics."""
        cache = ReformulationCache(enabled=True, ttl=1, max_size=100)
        cache.set("test", sample_reformulations)

        time.sleep(1.5)

        # Access expired entry
        cache.get("test")

        stats = cache.stats()
        assert stats["expirations"] >= 1


# =============================================================================
# Cache Size Limit Tests
# =============================================================================

class TestCacheSizeLimits:
    """Tests for cache size limits and LRU eviction."""

    def test_lru_eviction(self, sample_reformulations):
        """Test LRU eviction when cache is full."""
        cache = ReformulationCache(enabled=True, ttl=300, max_size=3)

        # Fill cache
        for i in range(5):
            cache.set(f"query{i}", sample_reformulations)

        # Only last 3 should remain
        assert cache.size == 3
        assert cache.get("query0") is None  # Evicted
        assert cache.get("query1") is None  # Evicted
        assert cache.get("query2") is not None
        assert cache.get("query3") is not None
        assert cache.get("query4") is not None

    def test_lru_access_updates_order(self, sample_reformulations):
        """Test that get() updates LRU order."""
        cache = ReformulationCache(enabled=True, ttl=300, max_size=3)

        # Fill cache
        cache.set("query0", sample_reformulations)
        cache.set("query1", sample_reformulations)
        cache.set("query2", sample_reformulations)

        # Access query0 to make it most recent
        cache.get("query0")

        # Add new item, should evict query1 (now oldest)
        cache.set("query3", sample_reformulations)

        assert cache.get("query0") is not None  # Should still exist
        assert cache.get("query1") is None  # Should be evicted
        assert cache.get("query2") is not None
        assert cache.get("query3") is not None


# =============================================================================
# Statistics Tracking Tests
# =============================================================================

class TestStatisticsTracking:
    """Tests for statistics tracking accuracy."""

    def test_hit_tracking(self, reformulation_cache, sample_reformulations):
        """Test hit statistics are tracked correctly."""
        reformulation_cache.set("test", sample_reformulations)

        # First get - hit
        reformulation_cache.get("test")

        stats = reformulation_cache.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 0

    def test_miss_tracking(self, reformulation_cache):
        """Test miss statistics are tracked correctly."""
        # Miss
        reformulation_cache.get("nonexistent")

        stats = reformulation_cache.stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 1

    def test_combined_stats(self, reformulation_cache, sample_reformulations):
        """Test combined hit/miss statistics."""
        reformulation_cache.set("test1", sample_reformulations)

        # 2 hits
        reformulation_cache.get("test1")
        reformulation_cache.get("test1")

        # 2 misses
        reformulation_cache.get("nonexistent1")
        reformulation_cache.get("nonexistent2")

        stats = reformulation_cache.stats()
        assert stats["hits"] == 2
        assert stats["misses"] == 2
        assert stats["hit_rate_percent"] == 50.0

    def test_hit_count_on_entry(self, reformulation_cache, sample_reformulations):
        """Test hit count is tracked on individual entries."""
        reformulation_cache.set("test", sample_reformulations)

        # Multiple accesses
        for _ in range(5):
            cached = reformulation_cache.get("test")

        assert cached.hit_count == 5

    def test_expiration_stats(self, sample_reformulations):
        """Test expiration statistics."""
        cache = ReformulationCache(enabled=True, ttl=1, max_size=100)
        cache.set("test", sample_reformulations)

        time.sleep(1.5)
        cache.get("test")  # Triggers expiration detection

        stats = cache.stats()
        assert stats["expirations"] >= 1

    def test_stats_include_savings(self, reformulation_cache, sample_reformulations):
        """Test stats include estimated savings."""
        reformulation_cache.set("test", sample_reformulations)

        for _ in range(10):
            reformulation_cache.get("test")

        stats = reformulation_cache.stats()
        assert stats["api_calls_saved"] == 10
        assert stats["estimated_cost_saved_usd"] == 0.01  # 10 * $0.001


# =============================================================================
# Semantic Matching Tests
# =============================================================================

class TestSemanticMatching:
    """Tests for semantic similarity matching."""

    def test_semantic_match_with_model(self, mock_embedding_model, sample_reformulations):
        """Test semantic matching finds similar queries."""
        cache = ReformulationCache(
            enabled=True,
            ttl=300,
            max_size=100,
            semantic_threshold=0.5,  # Lower threshold for test
            embedding_model=mock_embedding_model,
        )

        # Store with embedding
        cache.set("machine learning research", sample_reformulations)

        # Similar query should match (same seed produces same embedding)
        cached = cache.get("machine learning research")
        assert cached is not None

    def test_semantic_stats_tracking(self, mock_embedding_model, sample_reformulations):
        """Test semantic hit statistics are tracked."""
        cache = ReformulationCache(
            enabled=True,
            ttl=300,
            max_size=100,
            semantic_threshold=0.5,
            embedding_model=mock_embedding_model,
        )

        cache.set("test query", sample_reformulations)

        stats = cache.stats()
        assert stats["semantic_enabled"] is True
        assert stats["semantic_threshold"] == 0.5

    def test_set_embedding_model(self, mock_embedding_model, sample_reformulations):
        """Test setting embedding model after creation."""
        cache = ReformulationCache(enabled=True, ttl=300, max_size=100)

        # Initially no semantic matching
        stats = cache.stats()
        assert stats["semantic_enabled"] is False

        # Add some entries
        cache.set("query1", sample_reformulations)
        cache.set("query2", sample_reformulations)

        # Set model
        cache.set_embedding_model(mock_embedding_model)

        stats = cache.stats()
        assert stats["semantic_enabled"] is True

    def test_cache_without_model_skips_semantic(self, sample_reformulations):
        """Test cache without model skips semantic matching."""
        cache = ReformulationCache(enabled=True, ttl=300, max_size=100)

        cache.set("exact query", sample_reformulations)

        # Should only do exact match
        assert cache.get("exact query") is not None
        assert cache.get("similar but not exact") is None


# =============================================================================
# Thread Safety Tests
# =============================================================================

class TestThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_writes(self, reformulation_cache, sample_reformulations):
        """Test concurrent writes don't corrupt cache."""
        errors = []

        def write_cache(i):
            try:
                reformulation_cache.set(f"query{i}", sample_reformulations)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=write_cache, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert reformulation_cache.size <= 20

    def test_concurrent_reads(self, reformulation_cache, sample_reformulations):
        """Test concurrent reads work correctly."""
        reformulation_cache.set("test query", sample_reformulations)

        results = []
        errors = []

        def read_cache():
            try:
                result = reformulation_cache.get("test query")
                results.append(result is not None)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=read_cache) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert all(results)

    def test_concurrent_read_write(self, reformulation_cache, sample_reformulations):
        """Test concurrent reads and writes work correctly."""
        errors = []

        def writer():
            try:
                for i in range(10):
                    reformulation_cache.set(f"query{i}", sample_reformulations)
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for i in range(10):
                    reformulation_cache.get(f"query{i}")
            except Exception as e:
                errors.append(e)

        write_threads = [threading.Thread(target=writer) for _ in range(3)]
        read_threads = [threading.Thread(target=reader) for _ in range(3)]

        all_threads = write_threads + read_threads
        for t in all_threads:
            t.start()
        for t in all_threads:
            t.join()

        assert len(errors) == 0


# =============================================================================
# Singleton and Factory Tests
# =============================================================================

class TestSingletonFactory:
    """Tests for singleton factory functions."""

    def test_get_reformulation_cache_singleton(self):
        """Test get_reformulation_cache returns singleton."""
        reset_reformulation_cache()
        cache1 = get_reformulation_cache()
        cache2 = get_reformulation_cache()
        assert cache1 is cache2

    def test_reset_reformulation_cache(self, sample_reformulations):
        """Test reset_reformulation_cache clears singleton."""
        reset_reformulation_cache()
        cache1 = get_reformulation_cache()
        cache1.set("test", sample_reformulations)

        reset_reformulation_cache()
        cache2 = get_reformulation_cache()

        # Should be a fresh cache
        assert cache2.get("test") is None

    def test_factory_with_custom_config(self):
        """Test factory respects custom configuration on first call."""
        reset_reformulation_cache()
        cache = get_reformulation_cache(
            ttl=600,
            max_size=200,
            semantic_threshold=0.9,
        )

        stats = cache.stats()
        assert stats["ttl_seconds"] == 600
        assert stats["max_size"] == 200
        assert stats["semantic_threshold"] == 0.9


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for realistic usage patterns."""

    def test_reformulation_caching_pattern(self, reformulation_cache):
        """Test typical reformulation caching pattern."""
        # Simulate reformulation workflow
        query = "what grants did james evans get?"

        # First request - miss
        cached = reformulation_cache.get(query)
        assert cached is None

        # Simulate LLM reformulation
        reformulations = [
            "James Evans grant funding awards",
            "grants received by James Evans NSF NIH MURI",
        ]

        # Cache the result
        reformulation_cache.set(query, reformulations)

        # Second request - hit
        cached = reformulation_cache.get(query)
        assert cached is not None
        assert cached.reformulations == reformulations

        # Verify stats
        stats = reformulation_cache.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1

    def test_cache_with_variations(self, reformulation_cache):
        """Test cache handles query variations correctly."""
        reformulations = ["test reformulation"]

        # Store with one variation
        reformulation_cache.set("Machine Learning", reformulations)

        # Should match different variations (due to normalization)
        assert reformulation_cache.get("machine learning") is not None
        assert reformulation_cache.get("MACHINE LEARNING") is not None
        assert reformulation_cache.get("  machine learning  ") is not None

        # Should not match different queries
        assert reformulation_cache.get("deep learning") is None

    def test_cache_eviction_preserves_recent(self, sample_reformulations):
        """Test eviction preserves most recently used entries."""
        cache = ReformulationCache(enabled=True, ttl=300, max_size=5)

        # Fill cache
        for i in range(5):
            cache.set(f"query{i}", sample_reformulations)

        # Access first entries to make them recent
        cache.get("query0")
        cache.get("query1")

        # Add more entries, triggering eviction
        for i in range(5, 8):
            cache.set(f"query{i}", sample_reformulations)

        # Recently accessed should still exist
        assert cache.get("query0") is not None
        assert cache.get("query1") is not None

        # Middle entries should be evicted
        assert cache.get("query2") is None
        assert cache.get("query3") is None


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
