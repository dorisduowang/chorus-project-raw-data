"""
Unit tests for the CHORUS instrumentation module.

Tests cover:
- Timer context manager functionality
- LatencyTracker percentile calculations and thread safety
- RequestContext per-request timing
- Statistics export format
- Thread safety (basic)

Run with:
    pytest tests/test_instrumentation.py -v
"""

import pytest
import time
import threading
import json
import sys
from pathlib import Path
from unittest.mock import patch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.instrumentation import (
    Timer,
    ComponentStats,
    LatencyTracker,
    RequestContext,
    TimingEntry,
    get_tracker,
    reset_tracker,
    timed,
    get_stats,
)


# =============================================================================
# Timer Tests
# =============================================================================

class TestTimer:
    """Tests for Timer context manager."""

    def test_timer_basic_usage(self):
        """Test basic Timer context manager usage."""
        with Timer("test_op") as t:
            time.sleep(0.01)  # 10ms

        # Should measure approximately 10ms
        assert t.elapsed_ms >= 10
        assert t.elapsed_ms < 50  # Allow some variance

    def test_timer_name(self):
        """Test Timer stores name correctly."""
        with Timer("my_operation") as t:
            pass
        assert t.name == "my_operation"

    def test_timer_empty_name(self):
        """Test Timer with empty name."""
        with Timer() as t:
            pass
        assert t.name == ""

    def test_timer_elapsed_properties(self):
        """Test elapsed_ns, elapsed_ms, elapsed_s consistency."""
        with Timer("test") as t:
            time.sleep(0.01)

        # Check consistency between units
        assert abs(t.elapsed_ms - t.elapsed_ns / 1_000_000) < 0.001
        assert abs(t.elapsed_s - t.elapsed_ns / 1_000_000_000) < 0.000001

    def test_timer_not_started(self):
        """Test Timer before being used as context manager."""
        t = Timer("test")
        assert t.elapsed_ns == 0.0
        assert t.elapsed_ms == 0.0
        assert t.elapsed_s == 0.0

    def test_timer_repr(self):
        """Test Timer string representation."""
        # Before starting
        t = Timer("test")
        assert "not started" in repr(t)

        # After running
        with Timer("test") as t:
            pass
        assert "elapsed_ms=" in repr(t)
        assert "test" in repr(t)

    def test_timer_during_execution(self):
        """Test accessing elapsed during execution."""
        with Timer("test") as t:
            time.sleep(0.005)
            # Should be able to read elapsed during execution
            intermediate = t.elapsed_ms
            time.sleep(0.005)

        # Final should be greater than intermediate
        assert t.elapsed_ms > intermediate

    def test_timer_zero_time_operation(self):
        """Test Timer with very fast operation."""
        with Timer("fast") as t:
            pass  # No-op

        # Should still have positive elapsed time (non-zero)
        assert t.elapsed_ns >= 0

    def test_timer_exception_handling(self):
        """Test Timer still records time when exception occurs."""
        elapsed = 0
        try:
            with Timer("exception_test") as t:
                time.sleep(0.01)
                raise ValueError("Test exception")
        except ValueError:
            elapsed = t.elapsed_ms

        # Time should still be recorded
        assert elapsed >= 10


# =============================================================================
# ComponentStats Tests
# =============================================================================

class TestComponentStats:
    """Tests for ComponentStats dataclass."""

    def test_default_stats(self):
        """Test default ComponentStats values."""
        stats = ComponentStats()
        assert stats.count == 0
        assert stats.min_ms == 0.0
        assert stats.max_ms == 0.0
        assert stats.p50 == 0.0
        assert stats.p95 == 0.0
        assert stats.p99 == 0.0

    def test_to_dict(self):
        """Test ComponentStats to_dict serialization."""
        stats = ComponentStats(
            count=100,
            min_ms=10.123,
            max_ms=100.567,
            mean_ms=45.789,
            stddev_ms=12.345,
            p50=40.111,
            p95=80.222,
            p99=95.333,
            total_ms=4578.9,
        )

        data = stats.to_dict()

        assert data["count"] == 100
        assert data["min_ms"] == 10.123
        assert data["max_ms"] == 100.567
        assert data["mean_ms"] == 45.789
        assert "p50" in data
        assert "p95" in data
        assert "p99" in data

    def test_to_dict_rounding(self):
        """Test to_dict rounds to 3 decimal places."""
        stats = ComponentStats(
            min_ms=10.123456789,
            max_ms=100.987654321,
        )
        data = stats.to_dict()

        # Should be rounded to 3 decimals
        assert data["min_ms"] == 10.123
        assert data["max_ms"] == 100.988


# =============================================================================
# LatencyTracker Tests
# =============================================================================

class TestLatencyTracker:
    """Tests for LatencyTracker class."""

    @pytest.fixture
    def tracker(self):
        """Create a fresh tracker for each test."""
        return LatencyTracker(window_size=100)

    def test_record_single_measurement(self, tracker):
        """Test recording a single measurement."""
        tracker.record("embedding", 45.5)
        stats = tracker.get_stats("embedding")

        assert stats.count == 1
        assert stats.min_ms == 45.5
        assert stats.max_ms == 45.5
        assert stats.mean_ms == 45.5

    def test_record_multiple_measurements(self, tracker):
        """Test recording multiple measurements."""
        measurements = [10.0, 20.0, 30.0, 40.0, 50.0]
        for m in measurements:
            tracker.record("search", m)

        stats = tracker.get_stats("search")

        assert stats.count == 5
        assert stats.min_ms == 10.0
        assert stats.max_ms == 50.0
        assert stats.mean_ms == 30.0

    def test_record_timer(self, tracker):
        """Test recording from Timer instance."""
        with Timer("faiss_search") as t:
            time.sleep(0.01)

        tracker.record_timer(t)
        stats = tracker.get_stats("faiss_search")

        assert stats.count == 1
        assert stats.min_ms >= 10  # At least 10ms

    def test_record_timer_no_name(self, tracker):
        """Test record_timer does nothing for unnamed Timer."""
        with Timer() as t:  # No name
            pass

        # Should not raise, just do nothing
        tracker.record_timer(t)
        assert len(tracker.get_components()) == 0

    def test_percentile_calculation_odd_count(self, tracker):
        """Test percentile calculation with odd number of measurements."""
        # 1 to 99 (99 values)
        for i in range(1, 100):
            tracker.record("test", float(i))

        stats = tracker.get_stats("test")

        # P50 should be around 50
        assert 48 <= stats.p50 <= 52
        # P95 should be around 95
        assert 93 <= stats.p95 <= 97
        # P99 should be around 99
        assert 97 <= stats.p99 <= 99

    def test_percentile_calculation_even_count(self, tracker):
        """Test percentile calculation with even number of measurements."""
        for i in range(1, 101):  # 100 values
            tracker.record("test", float(i))

        stats = tracker.get_stats("test")

        # Check reasonable percentile values
        assert 48 <= stats.p50 <= 52
        assert 93 <= stats.p95 <= 97

    def test_percentile_single_value(self, tracker):
        """Test percentile with single value."""
        tracker.record("single", 50.0)
        stats = tracker.get_stats("single")

        # All percentiles should equal the single value
        assert stats.p50 == 50.0
        assert stats.p95 == 50.0
        assert stats.p99 == 50.0

    def test_stats_nonexistent_component(self, tracker):
        """Test get_stats for non-existent component."""
        stats = tracker.get_stats("nonexistent")

        assert stats.count == 0
        assert stats.p50 == 0.0

    def test_stddev_calculation(self, tracker):
        """Test standard deviation calculation."""
        # Values with known stddev
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        for v in values:
            tracker.record("stddev_test", v)

        stats = tracker.get_stats("stddev_test")

        # Python statistics.stdev for [10,20,30,40,50] is ~15.81
        assert 15 < stats.stddev_ms < 17

    def test_stddev_single_value(self, tracker):
        """Test stddev is 0 for single value."""
        tracker.record("single", 50.0)
        stats = tracker.get_stats("single")
        assert stats.stddev_ms == 0.0

    def test_window_size_limit(self):
        """Test window size limiting (rolling window)."""
        tracker = LatencyTracker(window_size=5)

        # Record 10 measurements
        for i in range(10):
            tracker.record("test", float(i))

        stats = tracker.get_stats("test")

        # Should only keep last 5 (5, 6, 7, 8, 9)
        assert stats.count == 5
        assert stats.min_ms == 5.0
        assert stats.max_ms == 9.0

    def test_multiple_components(self, tracker):
        """Test tracking multiple components independently."""
        tracker.record("embedding", 50.0)
        tracker.record("embedding", 60.0)
        tracker.record("search", 20.0)
        tracker.record("search", 30.0)
        tracker.record("reranking", 100.0)

        assert tracker.get_stats("embedding").count == 2
        assert tracker.get_stats("search").count == 2
        assert tracker.get_stats("reranking").count == 1

        assert tracker.get_stats("embedding").mean_ms == 55.0
        assert tracker.get_stats("search").mean_ms == 25.0

    def test_get_all_stats(self, tracker):
        """Test getting stats for all components."""
        tracker.record("a", 10.0)
        tracker.record("b", 20.0)
        tracker.record("c", 30.0)

        all_stats = tracker.get_all_stats()

        assert len(all_stats) == 3
        assert "a" in all_stats
        assert "b" in all_stats
        assert "c" in all_stats

    def test_export_stats(self, tracker):
        """Test export_stats returns JSON-serializable dict."""
        tracker.record("test", 50.0)
        exported = tracker.export_stats()

        # Should be JSON serializable
        json_str = json.dumps(exported)
        assert "test" in json_str
        assert "p50" in json_str

    def test_export_json(self, tracker):
        """Test export_json returns valid JSON string."""
        tracker.record("test", 50.0)
        json_str = tracker.export_json()

        # Should be valid JSON
        data = json.loads(json_str)
        assert "test" in data

    def test_reset_single_component(self, tracker):
        """Test resetting a single component."""
        tracker.record("keep", 10.0)
        tracker.record("reset_me", 20.0)

        tracker.reset("reset_me")

        assert tracker.get_stats("keep").count == 1
        assert tracker.get_stats("reset_me").count == 0

    def test_reset_all(self, tracker):
        """Test resetting all components."""
        tracker.record("a", 10.0)
        tracker.record("b", 20.0)

        tracker.reset()

        assert tracker.get_stats("a").count == 0
        assert tracker.get_stats("b").count == 0

    def test_get_components(self, tracker):
        """Test get_components returns tracked components."""
        tracker.record("embedding", 10.0)
        tracker.record("search", 20.0)

        components = tracker.get_components()

        assert "embedding" in components
        assert "search" in components
        assert len(components) == 2

    def test_summary(self, tracker):
        """Test summary method."""
        tracker.record("a", 10.0)
        tracker.record("a", 20.0)
        tracker.record("b", 30.0)

        summary = tracker.summary()

        assert summary["components_tracked"] == 2
        assert summary["total_measurements"] == 3
        assert "uptime_seconds" in summary
        assert "a" in summary["components"]
        assert "b" in summary["components"]

    def test_uptime_seconds(self, tracker):
        """Test uptime_seconds property."""
        time.sleep(0.01)
        uptime = tracker.uptime_seconds
        assert uptime >= 0.01

    def test_standard_components_defined(self):
        """Test STANDARD_COMPONENTS is defined."""
        assert "embedding" in LatencyTracker.STANDARD_COMPONENTS
        assert "faiss_search" in LatencyTracker.STANDARD_COMPONENTS
        assert "reranking" in LatencyTracker.STANDARD_COMPONENTS
        assert "total" in LatencyTracker.STANDARD_COMPONENTS


# =============================================================================
# RequestContext Tests
# =============================================================================

class TestRequestContext:
    """Tests for RequestContext class."""

    def test_create_with_request_id(self):
        """Test creating RequestContext with custom ID."""
        ctx = RequestContext(request_id="abc123")
        assert ctx.request_id == "abc123"

    def test_create_auto_request_id(self):
        """Test creating RequestContext with auto-generated ID."""
        ctx = RequestContext()
        assert ctx.request_id is not None
        assert len(ctx.request_id) > 0

    def test_start_and_end(self):
        """Test basic start and end timing."""
        ctx = RequestContext()

        ctx.start("embedding")
        time.sleep(0.01)
        elapsed = ctx.end("embedding")

        assert elapsed >= 10  # At least 10ms

    def test_get_elapsed(self):
        """Test get_elapsed returns correct value."""
        ctx = RequestContext()

        ctx.start("test")
        time.sleep(0.01)
        ctx.end("test")

        elapsed = ctx.get_elapsed("test")
        assert elapsed is not None
        assert elapsed >= 10

    def test_get_elapsed_nonexistent(self):
        """Test get_elapsed for non-existent component."""
        ctx = RequestContext()
        assert ctx.get_elapsed("nonexistent") is None

    def test_get_elapsed_active(self):
        """Test get_elapsed for active (not ended) component."""
        ctx = RequestContext()
        ctx.start("active")

        # Still active, elapsed_ms should be None
        assert ctx.get_elapsed("active") is None

        ctx.end("active")  # Cleanup

    def test_multiple_components(self):
        """Test timing multiple components."""
        ctx = RequestContext()

        ctx.start("total")
        ctx.start("embedding")
        time.sleep(0.01)
        ctx.end("embedding")

        ctx.start("search")
        time.sleep(0.01)
        ctx.end("search")
        ctx.end("total")

        breakdown = ctx.get_breakdown()

        assert "total" in breakdown
        assert "embedding" in breakdown
        assert "search" in breakdown
        assert breakdown["embedding"]["elapsed_ms"] >= 10
        assert breakdown["search"]["elapsed_ms"] >= 10

    def test_nested_timing(self):
        """Test nested component timing (parent contains children)."""
        ctx = RequestContext()

        ctx.start("total")
        ctx.start("step1")
        time.sleep(0.005)
        ctx.end("step1")

        ctx.start("step2")
        time.sleep(0.005)
        ctx.end("step2")
        ctx.end("total")

        # Total should be at least as long as step1 + step2
        total = ctx.get_elapsed("total")
        step1 = ctx.get_elapsed("step1")
        step2 = ctx.get_elapsed("step2")

        assert total >= step1 + step2

    def test_time_context_manager(self):
        """Test time() context manager helper."""
        ctx = RequestContext()

        with ctx.time("embedding"):
            time.sleep(0.01)

        elapsed = ctx.get_elapsed("embedding")
        assert elapsed is not None
        assert elapsed >= 10

    def test_time_context_manager_exception(self):
        """Test time() context manager handles exceptions."""
        ctx = RequestContext()

        try:
            with ctx.time("failing"):
                time.sleep(0.01)
                raise ValueError("Test error")
        except ValueError:
            pass

        # Timing should still be recorded
        elapsed = ctx.get_elapsed("failing")
        assert elapsed is not None
        assert elapsed >= 10

    def test_start_already_started_raises(self):
        """Test starting already active component raises error."""
        ctx = RequestContext()
        ctx.start("test")

        with pytest.raises(ValueError, match="already being timed"):
            ctx.start("test")

        ctx.end("test")  # Cleanup

    def test_end_not_started_raises(self):
        """Test ending non-started component raises error."""
        ctx = RequestContext()

        with pytest.raises(ValueError, match="was not started"):
            ctx.end("not_started")

    def test_is_active(self):
        """Test is_active method."""
        ctx = RequestContext()

        assert not ctx.is_active("test")

        ctx.start("test")
        assert ctx.is_active("test")

        ctx.end("test")
        assert not ctx.is_active("test")

    def test_active_components(self):
        """Test active_components method."""
        ctx = RequestContext()

        ctx.start("a")
        ctx.start("b")

        active = ctx.active_components()
        assert "a" in active
        assert "b" in active

        ctx.end("a")

        active = ctx.active_components()
        assert "a" not in active
        assert "b" in active

        ctx.end("b")

    def test_completed_components(self):
        """Test completed_components method."""
        ctx = RequestContext()

        ctx.start("a")
        ctx.start("b")

        assert len(ctx.completed_components()) == 0

        ctx.end("a")
        completed = ctx.completed_components()
        assert "a" in completed
        assert "b" not in completed

        ctx.end("b")
        completed = ctx.completed_components()
        assert "a" in completed
        assert "b" in completed

    def test_get_breakdown(self):
        """Test get_breakdown returns correct structure."""
        ctx = RequestContext()

        ctx.start("embedding")
        time.sleep(0.005)
        ctx.end("embedding")

        ctx.start("search")  # Still active

        breakdown = ctx.get_breakdown()

        assert "embedding" in breakdown
        assert "search" in breakdown
        assert breakdown["embedding"]["active"] is False
        assert breakdown["embedding"]["elapsed_ms"] is not None
        assert breakdown["search"]["active"] is True
        assert breakdown["search"]["elapsed_ms"] is None

        ctx.end("search")  # Cleanup

    def test_get_total_ms(self):
        """Test get_total_ms sums completed components."""
        ctx = RequestContext()

        ctx.start("a")
        time.sleep(0.01)
        ctx.end("a")

        ctx.start("b")
        time.sleep(0.01)
        ctx.end("b")

        total = ctx.get_total_ms()
        assert total >= 20  # At least 20ms total

    def test_get_total_ms_excludes_active(self):
        """Test get_total_ms excludes active components."""
        ctx = RequestContext()

        ctx.start("completed")
        time.sleep(0.01)
        ctx.end("completed")

        ctx.start("active")  # Still running

        # Total should only include completed
        total = ctx.get_total_ms()
        elapsed_completed = ctx.get_elapsed("completed")
        assert abs(total - elapsed_completed) < 0.1

        ctx.end("active")  # Cleanup

    def test_to_dict(self):
        """Test to_dict serialization."""
        ctx = RequestContext(request_id="test123")

        ctx.start("embedding")
        time.sleep(0.005)
        ctx.end("embedding")

        data = ctx.to_dict()

        assert data["request_id"] == "test123"
        assert "breakdown" in data
        assert "total_ms" in data
        assert "active_components" in data
        assert "completed_components" in data
        assert "embedding" in data["completed_components"]

    def test_to_json(self):
        """Test to_json returns valid JSON."""
        ctx = RequestContext(request_id="json_test")

        ctx.start("test")
        ctx.end("test")

        json_str = ctx.to_json()
        data = json.loads(json_str)

        assert data["request_id"] == "json_test"

    def test_repr(self):
        """Test string representation."""
        ctx = RequestContext(request_id="repr_test")

        ctx.start("a")
        ctx.end("a")
        ctx.start("b")

        repr_str = repr(ctx)

        assert "repr_test" in repr_str
        assert "completed=1" in repr_str
        assert "active=1" in repr_str

        ctx.end("b")  # Cleanup

    def test_order_preserved(self):
        """Test component order is preserved in breakdown."""
        ctx = RequestContext()

        ctx.start("first")
        ctx.end("first")
        ctx.start("second")
        ctx.end("second")
        ctx.start("third")
        ctx.end("third")

        breakdown = ctx.get_breakdown()
        keys = list(breakdown.keys())

        assert keys == ["first", "second", "third"]


# =============================================================================
# Thread Safety Tests
# =============================================================================

class TestThreadSafety:
    """Basic thread safety tests."""

    def test_tracker_concurrent_writes(self):
        """Test concurrent writes to LatencyTracker."""
        tracker = LatencyTracker(window_size=1000)
        errors = []

        def write_measurements(component, count):
            try:
                for i in range(count):
                    tracker.record(component, float(i))
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(10):
            t = threading.Thread(
                target=write_measurements,
                args=(f"component_{i}", 100)
            )
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

        # Should have data for all 10 components
        components = tracker.get_components()
        assert len(components) == 10

    def test_tracker_concurrent_reads_and_writes(self):
        """Test concurrent reads and writes."""
        tracker = LatencyTracker(window_size=1000)
        errors = []
        results = []

        def writer():
            try:
                for i in range(100):
                    tracker.record("test", float(i))
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for _ in range(50):
                    stats = tracker.get_stats("test")
                    results.append(stats.count)
                    time.sleep(0.002)
            except Exception as e:
                errors.append(e)

        write_thread = threading.Thread(target=writer)
        read_threads = [threading.Thread(target=reader) for _ in range(3)]

        write_thread.start()
        for t in read_threads:
            t.start()

        write_thread.join()
        for t in read_threads:
            t.join()

        assert len(errors) == 0
        assert len(results) > 0

    def test_tracker_concurrent_reset(self):
        """Test reset during concurrent operations."""
        tracker = LatencyTracker(window_size=1000)
        errors = []

        def writer():
            try:
                for i in range(100):
                    tracker.record("test", float(i))
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        def resetter():
            try:
                for _ in range(5):
                    time.sleep(0.02)
                    tracker.reset()
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=resetter),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


# =============================================================================
# Global Tracker Tests
# =============================================================================

class TestGlobalTracker:
    """Tests for global tracker functions."""

    def setup_method(self):
        """Reset global tracker before each test."""
        reset_tracker()

    def test_get_tracker_singleton(self):
        """Test get_tracker returns same instance."""
        tracker1 = get_tracker()
        tracker2 = get_tracker()
        assert tracker1 is tracker2

    def test_reset_tracker(self):
        """Test reset_tracker creates new instance."""
        tracker1 = get_tracker()
        tracker1.record("test", 10.0)

        reset_tracker()

        tracker2 = get_tracker()
        assert tracker1 is not tracker2
        assert tracker2.get_stats("test").count == 0

    def test_get_tracker_window_size(self):
        """Test window_size parameter for new tracker."""
        reset_tracker()
        tracker = get_tracker(window_size=50)
        assert tracker.window_size == 50


# =============================================================================
# Decorator Tests
# =============================================================================

class TestTimedDecorator:
    """Tests for @timed decorator."""

    def setup_method(self):
        """Reset global tracker before each test."""
        reset_tracker()

    def test_timed_decorator_basic(self):
        """Test @timed decorator records timing."""
        @timed("test_function")
        def slow_function():
            time.sleep(0.01)
            return "result"

        result = slow_function()

        assert result == "result"

        tracker = get_tracker()
        stats = tracker.get_stats("test_function")
        assert stats.count == 1
        assert stats.min_ms >= 10

    def test_timed_decorator_with_args(self):
        """Test @timed decorator with function arguments."""
        @timed("add_numbers")
        def add(a, b):
            return a + b

        result = add(2, 3)
        assert result == 5

        tracker = get_tracker()
        assert tracker.get_stats("add_numbers").count == 1

    def test_timed_decorator_custom_tracker(self):
        """Test @timed decorator with custom tracker."""
        custom_tracker = LatencyTracker()

        @timed("custom_test", tracker=custom_tracker)
        def test_func():
            return "done"

        test_func()

        assert custom_tracker.get_stats("custom_test").count == 1
        # Global tracker should not have this
        global_tracker = get_tracker()
        assert global_tracker.get_stats("custom_test").count == 0

    def test_timed_decorator_preserves_metadata(self):
        """Test @timed decorator preserves function metadata."""
        @timed("metadata_test")
        def documented_function():
            """This is documentation."""
            pass

        assert documented_function.__name__ == "documented_function"
        assert documented_function.__doc__ == "This is documentation."


# =============================================================================
# Statistics Endpoint Helper Tests
# =============================================================================

class TestGetStats:
    """Tests for get_stats helper function."""

    def setup_method(self):
        """Reset global tracker before each test."""
        reset_tracker()

    def test_get_stats_format(self):
        """Test get_stats returns expected format."""
        tracker = get_tracker()
        tracker.record("faiss_search", 25.0)
        tracker.record("faiss_search", 30.0)
        tracker.record("reranking", 100.0)

        stats = get_stats()

        assert "faiss_search" in stats
        assert "reranking" in stats
        assert "p50" in stats["faiss_search"]
        assert "p95" in stats["faiss_search"]
        assert "p99" in stats["faiss_search"]
        assert "count" in stats["faiss_search"]

    def test_get_stats_with_custom_tracker(self):
        """Test get_stats with custom tracker."""
        custom_tracker = LatencyTracker()
        custom_tracker.record("custom", 50.0)

        stats = get_stats(tracker=custom_tracker)

        assert "custom" in stats
        assert stats["custom"]["count"] == 1

    def test_get_stats_json_serializable(self):
        """Test get_stats result is JSON serializable."""
        tracker = get_tracker()
        tracker.record("test", 50.0)

        stats = get_stats()
        json_str = json.dumps(stats)

        assert "test" in json_str


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for typical usage patterns."""

    def setup_method(self):
        """Reset global tracker before each test."""
        reset_tracker()

    def test_rag_pipeline_simulation(self):
        """Simulate RAG pipeline timing pattern."""
        tracker = LatencyTracker()
        ctx = RequestContext(request_id="req_001")

        # Simulate RAG pipeline
        ctx.start("total")

        # Query classification
        ctx.start("query_classification")
        time.sleep(0.002)
        ctx.end("query_classification")
        tracker.record("query_classification", ctx.get_elapsed("query_classification"))

        # Embedding
        ctx.start("embedding")
        time.sleep(0.01)
        ctx.end("embedding")
        tracker.record("embedding", ctx.get_elapsed("embedding"))

        # FAISS search
        ctx.start("faiss_search")
        time.sleep(0.005)
        ctx.end("faiss_search")
        tracker.record("faiss_search", ctx.get_elapsed("faiss_search"))

        # Reranking
        ctx.start("reranking")
        time.sleep(0.02)
        ctx.end("reranking")
        tracker.record("reranking", ctx.get_elapsed("reranking"))

        ctx.end("total")
        tracker.record("total", ctx.get_elapsed("total"))

        # Verify breakdown
        breakdown = ctx.get_breakdown()
        assert len(breakdown) == 5
        assert all(b["elapsed_ms"] is not None for b in breakdown.values())
        assert all(not b["active"] for b in breakdown.values())

        # Verify tracker stats
        stats = tracker.export_stats()
        assert "embedding" in stats
        assert "faiss_search" in stats
        assert "reranking" in stats
        assert "total" in stats

    def test_multiple_requests(self):
        """Test tracking across multiple requests."""
        tracker = LatencyTracker()

        # Simulate 10 requests
        for i in range(10):
            ctx = RequestContext(request_id=f"req_{i}")

            ctx.start("total")
            ctx.start("embedding")
            time.sleep(0.002 + (i * 0.001))  # Varying latency
            ctx.end("embedding")
            ctx.end("total")

            tracker.record("embedding", ctx.get_elapsed("embedding"))
            tracker.record("total", ctx.get_elapsed("total"))

        stats = tracker.get_stats("embedding")
        assert stats.count == 10
        assert stats.min_ms < stats.max_ms  # Should have variance
        assert stats.p50 > 0
        assert stats.p95 > 0

    def test_export_for_monitoring(self):
        """Test exporting stats for monitoring endpoint."""
        tracker = LatencyTracker()

        # Add some measurements
        for i in range(100):
            tracker.record("embedding", 40 + (i % 20))
            tracker.record("faiss_search", 20 + (i % 10))
            tracker.record("reranking", 80 + (i % 40))

        # Export for monitoring
        stats = tracker.export_stats()

        # Verify format matches expected endpoint response
        expected_keys = {"count", "min_ms", "max_ms", "mean_ms", "stddev_ms", "p50", "p95", "p99", "total_ms"}

        for component, component_stats in stats.items():
            assert set(component_stats.keys()) == expected_keys

        # Verify JSON export
        json_str = tracker.export_json()
        parsed = json.loads(json_str)
        assert "embedding" in parsed
        assert "faiss_search" in parsed
        assert "reranking" in parsed


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
