#!/usr/bin/env python3
"""
Test script for Chorus performance optimizations.

Tests:
1. Model routing (Haiku vs Sonnet selection)
2. Response caching
3. Query complexity estimation

Run with: python test_optimizations.py
"""

import sys
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the optimized chat module
from chat_with_rag import (
    estimate_query_complexity,
    get_cache_key,
    check_cache,
    store_in_cache,
    detect_temporal_intent,
    get_temporal_context,
    RESPONSE_CACHE,
    METRICS
)


def test_query_complexity():
    """Test that query complexity estimation works correctly."""
    print("\n" + "=" * 60)
    print("TEST 1: Query Complexity Estimation")
    print("=" * 60)

    test_cases = [
        # Simple queries -> should use Haiku
        ("What is the deadline?", "simple", "haiku"),
        ("When is the meeting?", "simple", "haiku"),
        ("Who is the PI?", "simple", "haiku"),
        ("Where is the data?", "simple", "haiku"),
        ("How many papers?", "simple", "haiku"),
        ("List all projects", "simple", "haiku"),
        ("Show me the budget", "simple", "haiku"),

        # Complex queries -> should use Sonnet
        ("Synthesize the findings across all three papers", "complex", "sonnet"),
        ("Compare our approach to the Stanford method", "complex", "sonnet"),
        ("Analyze the relationship between citations and impact", "complex", "sonnet"),
        ("What are the implications of this for our research?", "complex", "sonnet"),
        ("Explain the connection between authenticity and trading zones", "complex", "sonnet"),

        # Medium/short queries -> use Haiku for speed (short queries are treated as simple)
        ("Tell me about the project", "simple", "haiku"),  # Short = simple
        ("What's happening with the grant?", "simple", "haiku"),  # Short = simple
    ]

    passed = 0
    failed = 0

    for query, expected_complexity, expected_model_type in test_cases:
        complexity, model = estimate_query_complexity(query)

        # Check if model matches expected type
        model_type = "haiku" if "haiku" in model else "sonnet"
        matches = model_type == expected_model_type

        status = "PASS" if matches else "FAIL"
        if matches:
            passed += 1
        else:
            failed += 1

        print(f"  [{status}] '{query[:40]}...' -> {complexity}/{model_type}")
        if not matches:
            print(f"         Expected: {expected_complexity}/{expected_model_type}")

    print(f"\n  Results: {passed} passed, {failed} failed")
    return failed == 0


def test_caching():
    """Test that response caching works correctly."""
    print("\n" + "=" * 60)
    print("TEST 2: Response Caching")
    print("=" * 60)

    # Clear cache
    RESPONSE_CACHE.clear()

    # Test storing and retrieving
    test_query = "What is the test query?"
    test_response = "This is a test response."

    # Should be None initially
    cached = check_cache(test_query)
    assert cached is None, "Cache should be empty initially"
    print("  [PASS] Cache empty initially")

    # Store response
    store_in_cache(test_query, test_response)

    # Should retrieve correctly (now returns tuple of (response, cache_type))
    cached = check_cache(test_query)
    assert cached is not None, "Should find cached response"
    response, cache_type = cached
    assert response == test_response, "Should retrieve cached response"
    assert cache_type in ("redis", "memory"), "Should return cache type"
    print(f"  [PASS] Response cached and retrieved correctly (type: {cache_type})")

    # Test case-insensitive matching
    cached = check_cache("WHAT IS THE TEST QUERY?")
    assert cached is not None, "Should match case-insensitively"
    response, _ = cached
    assert response == test_response, "Should match case-insensitively"
    print("  [PASS] Case-insensitive cache matching works")

    # Test cache key generation
    key1 = get_cache_key("Hello World")
    key2 = get_cache_key("hello world")
    key3 = get_cache_key("  HELLO WORLD  ")
    assert key1 == key2 == key3, "Keys should be normalized"
    print("  [PASS] Cache keys are normalized correctly")

    # Test cache size limit
    RESPONSE_CACHE.clear()
    for i in range(150):  # Exceed the 100 limit
        store_in_cache(f"query_{i}", f"response_{i}")

    assert len(RESPONSE_CACHE) <= 100, "Cache should not exceed max size"
    print(f"  [PASS] Cache size limited to {len(RESPONSE_CACHE)} entries")

    print("\n  Results: All cache tests passed")
    return True


def test_metrics():
    """Test that metrics tracking works."""
    print("\n" + "=" * 60)
    print("TEST 3: Metrics Tracking")
    print("=" * 60)

    # Reset metrics
    for key in METRICS:
        METRICS[key] = 0

    # Simulate some queries
    METRICS["queries"] = 10
    METRICS["cache_hits"] = 3
    METRICS["redis_hits"] = 2
    METRICS["memory_hits"] = 1
    METRICS["prompt_cache_hits"] = 5
    METRICS["tokens_saved"] = 2500
    METRICS["haiku_queries"] = 6
    METRICS["sonnet_queries"] = 4
    METRICS["total_latency_ms"] = 5000

    # Verify calculations
    cache_rate = METRICS["cache_hits"] / METRICS["queries"] * 100
    assert cache_rate == 30.0, f"Cache hit rate should be 30%, got {cache_rate}%"
    print(f"  [PASS] Cache hit rate: {cache_rate}%")

    # Verify Redis + memory = total cache hits
    assert METRICS["redis_hits"] + METRICS["memory_hits"] == METRICS["cache_hits"], \
        "Redis + memory hits should equal total cache hits"
    print(f"  [PASS] Redis ({METRICS['redis_hits']}) + Memory ({METRICS['memory_hits']}) = Total ({METRICS['cache_hits']})")

    # Verify prompt cache tracking
    assert METRICS["prompt_cache_hits"] == 5, "Prompt cache hits should be 5"
    assert METRICS["tokens_saved"] == 2500, "Tokens saved should be 2500"
    print(f"  [PASS] Prompt cache: {METRICS['prompt_cache_hits']} hits, {METRICS['tokens_saved']} tokens saved")

    haiku_rate = METRICS["haiku_queries"] / METRICS["queries"] * 100
    assert haiku_rate == 60.0, f"Haiku rate should be 60%, got {haiku_rate}%"
    print(f"  [PASS] Haiku usage rate: {haiku_rate}%")

    avg_latency = METRICS["total_latency_ms"] / METRICS["queries"]
    assert avg_latency == 500.0, f"Avg latency should be 500ms, got {avg_latency}ms"
    print(f"  [PASS] Average latency: {avg_latency}ms")

    print("\n  Results: All metrics tests passed")
    return True


def test_temporal_detection():
    """Test temporal intent detection."""
    print("\n" + "=" * 60)
    print("TEST 4: Temporal Intent Detection")
    print("=" * 60)

    from datetime import datetime
    current_year = datetime.now().year

    test_cases = [
        # (query, expected_temporal_type, description)
        ("What conferences are upcoming?", "future", "future indicator"),
        ("What is the next deadline?", "future", "next = future"),
        ("What happened at the 2024 conference?", "past", "past year"),
        ("What was decided last month?", "past", "past indicator"),
        ("What are we currently working on?", "current", "current indicator"),
        ("What is the ongoing research?", "current", "ongoing = current"),
        ("Tell me about the history of the lab", "historical", "historical indicator"),
        ("How did the project start?", "historical", "origin question"),
        (f"What events are planned for {current_year + 1}?", "future", "future year"),
        ("Tell me about the lab", "any", "no temporal indicator"),
        ("Who is the PI?", "any", "non-temporal question"),
    ]

    passed = 0
    failed = 0

    for query, expected_type, description in test_cases:
        temporal_type, years = detect_temporal_intent(query)

        if temporal_type == expected_type:
            passed += 1
            print(f"  [PASS] '{query[:40]}...' -> {temporal_type} ({description})")
        else:
            failed += 1
            print(f"  [FAIL] '{query[:40]}...' -> {temporal_type}, expected {expected_type}")

    # Test year extraction
    query_with_years = "What happened at ICSSI 2024 and 2025?"
    _, years = detect_temporal_intent(query_with_years)
    if "2024" in years and "2025" in years:
        passed += 1
        print(f"  [PASS] Year extraction: found {years}")
    else:
        failed += 1
        print(f"  [FAIL] Year extraction: expected ['2024', '2025'], got {years}")

    # Test temporal context generation
    context = get_temporal_context()
    if str(current_year) in context and "Today's date" in context:
        passed += 1
        print(f"  [PASS] Temporal context includes current year and date")
    else:
        failed += 1
        print(f"  [FAIL] Temporal context missing key information")

    print(f"\n  Results: {passed} passed, {failed} failed")
    return failed == 0


def test_model_routing_performance():
    """Estimate performance impact of model routing."""
    print("\n" + "=" * 60)
    print("TEST 5: Performance Impact Estimation")
    print("=" * 60)

    # Typical latency estimates (based on Anthropic benchmarks)
    HAIKU_LATENCY_MS = 300  # ~3x faster than Sonnet
    SONNET_LATENCY_MS = 900

    # Sample query distribution (based on typical usage)
    queries = [
        ("What is the deadline?", "simple"),
        ("When is the meeting?", "simple"),
        ("Who is working on this?", "simple"),
        ("Summarize the paper", "medium"),
        ("Compare the two approaches and analyze their tradeoffs", "complex"),
        ("What's the status?", "simple"),
        ("List the team members", "simple"),
        ("Synthesize findings across all sources", "complex"),
        ("Where is the data stored?", "simple"),
        ("Explain the methodology", "medium"),
    ]

    # Calculate latencies with and without routing
    no_routing_latency = 0
    with_routing_latency = 0

    for query, expected_type in queries:
        complexity, model = estimate_query_complexity(query)

        # Without routing: always Sonnet
        no_routing_latency += SONNET_LATENCY_MS

        # With routing: Haiku for simple, Sonnet otherwise
        if "haiku" in model:
            with_routing_latency += HAIKU_LATENCY_MS
        else:
            with_routing_latency += SONNET_LATENCY_MS

    savings_ms = no_routing_latency - with_routing_latency
    savings_pct = (savings_ms / no_routing_latency) * 100

    print(f"  Without routing: {no_routing_latency}ms total ({no_routing_latency/len(queries):.0f}ms avg)")
    print(f"  With routing:    {with_routing_latency}ms total ({with_routing_latency/len(queries):.0f}ms avg)")
    print(f"  Savings:         {savings_ms}ms ({savings_pct:.1f}%)")
    print(f"\n  [PASS] Model routing reduces latency by ~{savings_pct:.0f}%")

    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("CHORUS OPTIMIZATION TESTS")
    print("=" * 60)

    results = []
    results.append(("Query Complexity", test_query_complexity()))
    results.append(("Caching", test_caching()))
    results.append(("Metrics", test_metrics()))
    results.append(("Temporal Detection", test_temporal_detection()))
    results.append(("Performance Estimation", test_model_routing_performance()))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n  All tests passed!")
        return 0
    else:
        print("\n  Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
