"""
Latency instrumentation for CHORUS RAG pipeline.

Provides timing and performance measurement utilities for:
- Individual code blocks via Timer context manager
- Aggregate statistics via LatencyTracker (P50/P95/P99)
- Per-request timing breakdown via RequestContext

Usage:
    # Measure a single code block
    with Timer("faiss_search") as t:
        results = index.search(query_embedding, k=10)
    print(f"Search took {t.elapsed_ms:.2f}ms")

    # Track statistics across requests
    tracker = LatencyTracker()
    tracker.record("embedding", 45.2)
    tracker.record("embedding", 52.1)
    stats = tracker.get_stats("embedding")  # P50, P95, P99, etc.

    # Track per-request timing breakdown
    ctx = RequestContext(request_id="abc123")
    ctx.start("total")
    ctx.start("embedding")
    # ... do embedding ...
    ctx.end("embedding")
    ctx.end("total")
    breakdown = ctx.get_breakdown()
"""

import time
import threading
import statistics
import uuid
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from collections import deque
from contextlib import contextmanager


# =============================================================================
# Timer Context Manager
# =============================================================================

class Timer:
    """
    Context manager for timing code blocks.

    Usage:
        with Timer("faiss_search") as t:
            results = index.search(...)
        print(f"Took {t.elapsed_ms:.2f}ms")

    Attributes:
        name: Optional name for this timing (for logging/identification)
        elapsed_ns: Elapsed time in nanoseconds (available after exit)
        elapsed_ms: Elapsed time in milliseconds (available after exit)
        elapsed_s: Elapsed time in seconds (available after exit)
    """

    def __init__(self, name: str = ""):
        self.name = name
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None

    def __enter__(self) -> "Timer":
        self._start_time = time.perf_counter_ns()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._end_time = time.perf_counter_ns()

    @property
    def elapsed_ns(self) -> float:
        """Elapsed time in nanoseconds."""
        if self._start_time is None:
            return 0.0
        end = self._end_time if self._end_time is not None else time.perf_counter_ns()
        return float(end - self._start_time)

    @property
    def elapsed_ms(self) -> float:
        """Elapsed time in milliseconds."""
        return self.elapsed_ns / 1_000_000

    @property
    def elapsed_s(self) -> float:
        """Elapsed time in seconds."""
        return self.elapsed_ns / 1_000_000_000

    def __repr__(self) -> str:
        if self._start_time is None:
            return f"Timer(name={self.name!r}, not started)"
        return f"Timer(name={self.name!r}, elapsed_ms={self.elapsed_ms:.3f})"


# =============================================================================
# Component Statistics
# =============================================================================

@dataclass
class ComponentStats:
    """
    Statistics for a single component/operation.

    Contains count, min, max, mean, stddev, and percentiles.
    """
    count: int = 0
    min_ms: float = 0.0
    max_ms: float = 0.0
    mean_ms: float = 0.0
    stddev_ms: float = 0.0
    p50: float = 0.0
    p95: float = 0.0
    p99: float = 0.0
    total_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "count": self.count,
            "min_ms": round(self.min_ms, 3),
            "max_ms": round(self.max_ms, 3),
            "mean_ms": round(self.mean_ms, 3),
            "stddev_ms": round(self.stddev_ms, 3),
            "p50": round(self.p50, 3),
            "p95": round(self.p95, 3),
            "p99": round(self.p99, 3),
            "total_ms": round(self.total_ms, 3),
        }


# =============================================================================
# Latency Tracker
# =============================================================================

class LatencyTracker:
    """
    Thread-safe latency tracker with configurable rolling window.

    Collects timing measurements by component name and calculates
    P50, P95, P99 percentiles along with other statistics.

    Usage:
        tracker = LatencyTracker(window_size=1000)
        tracker.record("faiss_search", 25.3)
        tracker.record("faiss_search", 28.1)
        stats = tracker.get_stats("faiss_search")
        print(f"P95: {stats.p95}ms")

    Thread-safe for concurrent request handling.
    """

    # Standard components in the RAG pipeline
    STANDARD_COMPONENTS = [
        "query_classification",
        "query_reformulation",
        "embedding",
        "faiss_search",
        "bm25_search",
        "score_combination",
        "reranking",
        "mmr_selection",
        "registry_lookup",
        "total",
    ]

    def __init__(self, window_size: int = 1000):
        """
        Initialize the latency tracker.

        Args:
            window_size: Maximum number of measurements to keep per component.
                         Older measurements are dropped (rolling window).
        """
        self.window_size = window_size
        self._measurements: Dict[str, deque] = {}
        self._lock = threading.RLock()
        self._created_at = time.time()

    def record(self, component: str, latency_ms: float) -> None:
        """
        Record a latency measurement for a component.

        Args:
            component: Component name (e.g., "faiss_search", "embedding")
            latency_ms: Latency in milliseconds
        """
        with self._lock:
            if component not in self._measurements:
                self._measurements[component] = deque(maxlen=self.window_size)
            self._measurements[component].append(latency_ms)

    def record_timer(self, timer: Timer) -> None:
        """
        Record a timing from a Timer context manager.

        Args:
            timer: Timer instance (must have a name set)
        """
        if timer.name:
            self.record(timer.name, timer.elapsed_ms)

    def get_stats(self, component: str) -> ComponentStats:
        """
        Get statistics for a component.

        Args:
            component: Component name

        Returns:
            ComponentStats with count, min, max, mean, stddev, percentiles
        """
        with self._lock:
            measurements = self._measurements.get(component, deque())
            if not measurements:
                return ComponentStats()

            data = list(measurements)
            count = len(data)
            total = sum(data)
            mean = total / count
            min_val = min(data)
            max_val = max(data)

            # Standard deviation
            if count > 1:
                stddev = statistics.stdev(data)
            else:
                stddev = 0.0

            # Percentiles
            sorted_data = sorted(data)
            p50 = self._percentile(sorted_data, 50)
            p95 = self._percentile(sorted_data, 95)
            p99 = self._percentile(sorted_data, 99)

            return ComponentStats(
                count=count,
                min_ms=min_val,
                max_ms=max_val,
                mean_ms=mean,
                stddev_ms=stddev,
                p50=p50,
                p95=p95,
                p99=p99,
                total_ms=total,
            )

    @staticmethod
    def _percentile(sorted_data: List[float], percentile: float) -> float:
        """
        Calculate percentile from sorted data.

        Uses linear interpolation method (similar to numpy's default).
        """
        if not sorted_data:
            return 0.0

        n = len(sorted_data)
        if n == 1:
            return sorted_data[0]

        # Calculate the index (0-based)
        k = (n - 1) * (percentile / 100.0)
        f = int(k)
        c = f + 1 if f + 1 < n else f

        # Linear interpolation
        d = k - f
        return sorted_data[f] * (1 - d) + sorted_data[c] * d

    def get_all_stats(self) -> Dict[str, ComponentStats]:
        """
        Get statistics for all tracked components.

        Returns:
            Dictionary mapping component names to ComponentStats
        """
        with self._lock:
            components = list(self._measurements.keys())

        return {component: self.get_stats(component) for component in components}

    def export_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Export all statistics as a JSON-serializable dictionary.

        Returns:
            Dictionary ready for JSON serialization
        """
        all_stats = self.get_all_stats()
        return {name: stats.to_dict() for name, stats in all_stats.items()}

    def export_json(self) -> str:
        """
        Export all statistics as a JSON string.

        Returns:
            JSON string with all component statistics
        """
        return json.dumps(self.export_stats(), indent=2)

    def reset(self, component: Optional[str] = None) -> None:
        """
        Reset measurements for a component or all components.

        Args:
            component: If provided, reset only this component.
                       If None, reset all components.
        """
        with self._lock:
            if component:
                if component in self._measurements:
                    self._measurements[component].clear()
            else:
                self._measurements.clear()

    def get_components(self) -> List[str]:
        """Get list of tracked component names."""
        with self._lock:
            return list(self._measurements.keys())

    @property
    def uptime_seconds(self) -> float:
        """Get tracker uptime in seconds."""
        return time.time() - self._created_at

    def summary(self) -> Dict[str, Any]:
        """
        Get a summary of tracking state.

        Returns:
            Dictionary with uptime, component count, and total measurements
        """
        with self._lock:
            total_measurements = sum(len(m) for m in self._measurements.values())
            return {
                "uptime_seconds": round(self.uptime_seconds, 2),
                "window_size": self.window_size,
                "components_tracked": len(self._measurements),
                "total_measurements": total_measurements,
                "components": list(self._measurements.keys()),
            }


# =============================================================================
# Request Context
# =============================================================================

@dataclass
class TimingEntry:
    """A single timing entry within a request."""
    component: str
    start_time: float
    end_time: Optional[float] = None
    elapsed_ms: Optional[float] = None


class RequestContext:
    """
    Per-request timing context for tracking component latencies.

    Provides nested timing support for tracking multiple components
    within a single request.

    Usage:
        ctx = RequestContext(request_id="abc123")
        ctx.start("total")
        ctx.start("embedding")
        # ... do embedding ...
        ctx.end("embedding")
        ctx.start("faiss_search")
        # ... do search ...
        ctx.end("faiss_search")
        ctx.end("total")

        breakdown = ctx.get_breakdown()
        # {
        #     "embedding": {"elapsed_ms": 45.2},
        #     "faiss_search": {"elapsed_ms": 23.1},
        #     "total": {"elapsed_ms": 68.3}
        # }
    """

    def __init__(self, request_id: Optional[str] = None):
        """
        Initialize a request context.

        Args:
            request_id: Optional unique identifier for this request.
                        If not provided, a UUID will be generated.
        """
        self.request_id = request_id or str(uuid.uuid4())
        self._created_at = time.time()
        self._timings: Dict[str, TimingEntry] = {}
        self._active: Dict[str, float] = {}  # Component -> start time
        self._order: List[str] = []  # Track order of components

    def start(self, component: str) -> None:
        """
        Start timing a component.

        Args:
            component: Component name (e.g., "embedding", "faiss_search")

        Raises:
            ValueError: If component is already being timed
        """
        if component in self._active:
            raise ValueError(f"Component '{component}' is already being timed")

        start_time = time.perf_counter_ns()
        self._active[component] = start_time
        self._timings[component] = TimingEntry(
            component=component,
            start_time=start_time,
        )
        if component not in self._order:
            self._order.append(component)

    def end(self, component: str) -> float:
        """
        End timing a component.

        Args:
            component: Component name

        Returns:
            Elapsed time in milliseconds

        Raises:
            ValueError: If component was not started
        """
        if component not in self._active:
            raise ValueError(f"Component '{component}' was not started")

        end_time = time.perf_counter_ns()
        start_time = self._active.pop(component)
        elapsed_ns = end_time - start_time
        elapsed_ms = elapsed_ns / 1_000_000

        entry = self._timings[component]
        entry.end_time = end_time
        entry.elapsed_ms = elapsed_ms

        return elapsed_ms

    @contextmanager
    def time(self, component: str):
        """
        Context manager for timing a component.

        Usage:
            with ctx.time("embedding"):
                embedding = model.encode(query)

        Args:
            component: Component name

        Yields:
            None
        """
        self.start(component)
        try:
            yield
        finally:
            self.end(component)

    def get_elapsed(self, component: str) -> Optional[float]:
        """
        Get elapsed time for a component.

        Args:
            component: Component name

        Returns:
            Elapsed time in milliseconds, or None if not recorded
        """
        entry = self._timings.get(component)
        if entry and entry.elapsed_ms is not None:
            return entry.elapsed_ms
        return None

    def get_breakdown(self) -> Dict[str, Dict[str, Any]]:
        """
        Get timing breakdown for all components.

        Returns:
            Dictionary mapping component names to timing info.
            Each entry contains:
                - elapsed_ms: Time in milliseconds (None if still active)
                - active: Whether the component is still being timed
        """
        breakdown = {}
        for component in self._order:
            entry = self._timings[component]
            breakdown[component] = {
                "elapsed_ms": round(entry.elapsed_ms, 3) if entry.elapsed_ms is not None else None,
                "active": component in self._active,
            }
        return breakdown

    def get_total_ms(self) -> float:
        """
        Get total elapsed time across all completed components.

        Returns:
            Total time in milliseconds
        """
        return sum(
            entry.elapsed_ms
            for entry in self._timings.values()
            if entry.elapsed_ms is not None
        )

    def is_active(self, component: str) -> bool:
        """Check if a component is currently being timed."""
        return component in self._active

    def active_components(self) -> List[str]:
        """Get list of components currently being timed."""
        return list(self._active.keys())

    def completed_components(self) -> List[str]:
        """Get list of completed (ended) components."""
        return [
            comp for comp in self._order
            if comp not in self._active and self._timings[comp].elapsed_ms is not None
        ]

    def to_dict(self) -> Dict[str, Any]:
        """
        Export request context as a dictionary.

        Returns:
            Dictionary with request_id, breakdown, and metadata
        """
        return {
            "request_id": self.request_id,
            "created_at": self._created_at,
            "breakdown": self.get_breakdown(),
            "total_ms": round(self.get_total_ms(), 3),
            "active_components": self.active_components(),
            "completed_components": self.completed_components(),
        }

    def to_json(self) -> str:
        """Export request context as JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    def __repr__(self) -> str:
        completed = len(self.completed_components())
        active = len(self.active_components())
        return (
            f"RequestContext(request_id={self.request_id!r}, "
            f"completed={completed}, active={active}, "
            f"total_ms={self.get_total_ms():.2f})"
        )


# =============================================================================
# Global Tracker Instance
# =============================================================================

# Global tracker for application-wide statistics
_global_tracker: Optional[LatencyTracker] = None
_tracker_lock = threading.Lock()


def get_tracker(window_size: int = 1000) -> LatencyTracker:
    """
    Get or create the global latency tracker.

    Args:
        window_size: Window size for new tracker (ignored if already created)

    Returns:
        Global LatencyTracker instance
    """
    global _global_tracker
    with _tracker_lock:
        if _global_tracker is None:
            _global_tracker = LatencyTracker(window_size=window_size)
        return _global_tracker


def reset_tracker() -> None:
    """Reset the global tracker to a fresh instance."""
    global _global_tracker
    with _tracker_lock:
        _global_tracker = None


# =============================================================================
# Convenience Decorators
# =============================================================================

def timed(component: str, tracker: Optional[LatencyTracker] = None):
    """
    Decorator to time a function and record to tracker.

    Usage:
        @timed("embedding")
        def generate_embedding(text):
            return model.encode(text)

    Args:
        component: Component name for recording
        tracker: LatencyTracker to use (defaults to global tracker)
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            with Timer(component) as t:
                result = func(*args, **kwargs)
            target_tracker = tracker or get_tracker()
            target_tracker.record_timer(t)
            return result
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper
    return decorator


# =============================================================================
# Statistics Endpoint Helper
# =============================================================================

def get_stats(tracker: Optional[LatencyTracker] = None) -> Dict[str, Dict[str, Any]]:
    """
    Get statistics in the format expected by the stats endpoint.

    Returns format:
    {
        "faiss_search": {"p50": 25, "p95": 45, "p99": 80, "count": 1000, ...},
        "reranking": {"p50": 100, "p95": 180, "p99": 250, "count": 800, ...},
        ...
    }

    Args:
        tracker: LatencyTracker to use (defaults to global tracker)

    Returns:
        Dictionary of component statistics
    """
    target_tracker = tracker or get_tracker()
    return target_tracker.export_stats()
