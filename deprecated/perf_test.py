#!/usr/bin/env python3
"""
Performance testing for CHORUS RAG + Registry system.
Tests query latency, accuracy, and throughput.
"""

import time
import json
import urllib.request
import urllib.parse
import statistics
from typing import Dict, List, Any, Tuple

RAG_SERVER = "http://localhost:8765"

# Test queries organized by category
TEST_QUERIES = {
    "person_lookup": [
        ("Who is James Evans?", ["James", "Evans", "director"]),
        ("Tell me about Julia Koschinsky", ["Julia", "Koschinsky"]),
        ("Who is Bhargav Srinivasa Desikan?", ["Bhargav", "Desikan"]),
    ],
    "role_query": [
        ("Who are the research scientists?", ["research scientist"]),
        ("List the PhD students", ["student", "phd"]),
        ("Who are the postdocs?", ["postdoc"]),
    ],
    "topic_search": [
        ("Who works on natural language processing?", ["nlp", "natural language", "text"]),
        ("Researchers in network analysis", ["network"]),
        ("Who studies machine learning?", ["machine learning", "ml"]),
    ],
    "topic_alias": [
        ("Who works on NLP?", ["natural language"]),  # Should expand alias
        ("Researchers in ML", ["machine learning"]),
        ("Who studies CSS?", ["computational social science", "social"]),
    ],
    "publication_search": [
        ("Papers by James Evans", ["publication", "paper", "Evans"]),
        ("Recent work on deep learning", ["deep learning", "neural"]),
        ("Publications about social networks", ["network", "social"]),
    ],
    "compute_tools": [
        ("What compute resources are available?", ["midway", "gpu", "cluster", "compute"]),
        ("What tools does the lab use?", ["slurm", "jupyter", "tool"]),
        ("Tell me about the GPUs", ["gpu", "1080", "a100"]),
    ],
    "compound_query": [
        ("UChicago researchers in network science", ["chicago", "network"]),
        ("Students working on machine learning", ["student", "machine learning"]),
        ("Faculty studying computational social science", ["faculty", "social"]),
    ],
    "project_funding": [
        ("Tell me about the APTO project", ["apto"]),
        ("What grants does the lab have?", ["grant", "nsf", "funding"]),
        ("What is C3S2?", ["c3s2"]),
    ],
}


def query_rag(question: str, endpoint: str = "hybrid") -> Tuple[Dict, float]:
    """Query RAG server and return result with timing."""
    start = time.perf_counter()
    try:
        url = f"{RAG_SERVER}/{endpoint}?q={urllib.parse.quote(question)}&top_k=5"
        with urllib.request.urlopen(url, timeout=30) as response:
            result = json.loads(response.read().decode())
        elapsed = time.perf_counter() - start
        return result, elapsed
    except Exception as e:
        elapsed = time.perf_counter() - start
        return {"error": str(e)}, elapsed


def check_result_quality(result: Dict, expected_keywords: List[str]) -> Tuple[bool, str]:
    """Check if result contains expected keywords."""
    # Convert result to searchable text
    text_parts = []

    # Check structured results
    for sr in result.get("structured_results", []):
        data = sr.get("data", {})
        text_parts.append(json.dumps(data).lower())

    # Check structured answer
    if result.get("structured_answer"):
        text_parts.append(result["structured_answer"].lower())

    # Check semantic results
    for sem in result.get("semantic_results", []):
        text_parts.append(sem.get("text", "").lower())

    full_text = " ".join(text_parts)

    # Check for expected keywords
    found = []
    missing = []
    for kw in expected_keywords:
        if kw.lower() in full_text:
            found.append(kw)
        else:
            missing.append(kw)

    success = len(found) >= len(expected_keywords) / 2  # At least half must match
    detail = f"Found: {found}" if found else f"Missing: {missing}"
    return success, detail


def run_category_tests(category: str, queries: List[Tuple[str, List[str]]]) -> Dict[str, Any]:
    """Run tests for a category and return stats."""
    times = []
    successes = 0
    failures = []

    for query, expected in queries:
        result, elapsed = query_rag(query)
        times.append(elapsed)

        if "error" in result:
            failures.append((query, f"Error: {result['error']}"))
        else:
            success, detail = check_result_quality(result, expected)
            if success:
                successes += 1
            else:
                failures.append((query, detail))

    return {
        "category": category,
        "total": len(queries),
        "passed": successes,
        "failed": len(failures),
        "failures": failures,
        "avg_time_ms": statistics.mean(times) * 1000,
        "min_time_ms": min(times) * 1000,
        "max_time_ms": max(times) * 1000,
        "p95_time_ms": sorted(times)[int(len(times) * 0.95)] * 1000 if len(times) > 1 else times[0] * 1000,
    }


def test_throughput(duration_seconds: float = 5.0) -> Dict[str, Any]:
    """Test query throughput over a fixed duration."""
    queries = [q for qs in TEST_QUERIES.values() for q, _ in qs]

    start = time.perf_counter()
    count = 0
    times = []

    while time.perf_counter() - start < duration_seconds:
        query = queries[count % len(queries)]
        _, elapsed = query_rag(query)
        times.append(elapsed)
        count += 1

    total_time = time.perf_counter() - start

    return {
        "duration_seconds": total_time,
        "queries_completed": count,
        "queries_per_second": count / total_time,
        "avg_latency_ms": statistics.mean(times) * 1000,
        "p50_latency_ms": statistics.median(times) * 1000,
        "p95_latency_ms": sorted(times)[int(len(times) * 0.95)] * 1000,
        "p99_latency_ms": sorted(times)[int(len(times) * 0.99)] * 1000 if len(times) > 100 else max(times) * 1000,
    }


def test_endpoint_comparison() -> Dict[str, Any]:
    """Compare different RAG endpoints."""
    test_query = "Who works on machine learning?"
    endpoints = ["hybrid", "semantic", "bm25"]
    results = {}

    for endpoint in endpoints:
        times = []
        for _ in range(3):  # 3 runs each
            _, elapsed = query_rag(test_query, endpoint)
            times.append(elapsed)

        results[endpoint] = {
            "avg_ms": statistics.mean(times) * 1000,
            "min_ms": min(times) * 1000,
            "max_ms": max(times) * 1000,
        }

    return results


def main():
    print("=" * 60)
    print("CHORUS RAG + Registry Performance Test")
    print("=" * 60)

    # Check server health
    try:
        with urllib.request.urlopen(f"{RAG_SERVER}/health", timeout=5) as r:
            health = json.loads(r.read().decode())
            print(f"\nServer: {health['chunks']} chunks indexed")
            print(f"Reranking: {health.get('reranking_enabled', False)}")
            print(f"MMR: {health.get('mmr_enabled', False)} (λ={health.get('mmr_lambda', 'N/A')})")
    except Exception as e:
        print(f"\nError: Cannot connect to RAG server: {e}")
        return

    # Run category tests
    print("\n" + "-" * 60)
    print("Query Category Tests")
    print("-" * 60)

    all_results = []
    total_passed = 0
    total_failed = 0

    for category, queries in TEST_QUERIES.items():
        result = run_category_tests(category, queries)
        all_results.append(result)
        total_passed += result["passed"]
        total_failed += result["failed"]

        status = "✓" if result["failed"] == 0 else "✗"
        print(f"\n{status} {category}: {result['passed']}/{result['total']} passed")
        print(f"  Latency: avg={result['avg_time_ms']:.0f}ms, min={result['min_time_ms']:.0f}ms, max={result['max_time_ms']:.0f}ms")

        if result["failures"]:
            for query, detail in result["failures"]:
                print(f"  ✗ '{query[:40]}...' - {detail}")

    # Summary
    print("\n" + "-" * 60)
    print("Summary")
    print("-" * 60)
    total = total_passed + total_failed
    print(f"Overall: {total_passed}/{total} queries passed ({100*total_passed/total:.1f}%)")

    all_times = [r["avg_time_ms"] for r in all_results]
    print(f"Average latency across categories: {statistics.mean(all_times):.0f}ms")

    # Throughput test
    print("\n" + "-" * 60)
    print("Throughput Test (5 seconds)")
    print("-" * 60)

    throughput = test_throughput(5.0)
    print(f"Queries completed: {throughput['queries_completed']}")
    print(f"Throughput: {throughput['queries_per_second']:.1f} queries/sec")
    print(f"Latency: p50={throughput['p50_latency_ms']:.0f}ms, p95={throughput['p95_latency_ms']:.0f}ms, p99={throughput['p99_latency_ms']:.0f}ms")

    # Endpoint comparison
    print("\n" + "-" * 60)
    print("Endpoint Comparison")
    print("-" * 60)

    endpoints = test_endpoint_comparison()
    for name, stats in endpoints.items():
        print(f"  {name:10s}: avg={stats['avg_ms']:.0f}ms, min={stats['min_ms']:.0f}ms, max={stats['max_ms']:.0f}ms")

    print("\n" + "=" * 60)
    print("Test Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
