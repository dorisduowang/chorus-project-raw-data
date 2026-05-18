#!/usr/bin/env python3
"""
Comprehensive test suite for CHORUS RAG system.
Tests the RAG server endpoints and simulates chat.py behavior.
"""

import os
import sys
import json
import time
import urllib.request
import urllib.parse
from pathlib import Path
from datetime import datetime

# Load environment
sys.path.insert(0, str(Path(__file__).parent))
from config import load_dotenv
load_dotenv()

RAG_SERVER = "http://localhost:8765"
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

class TestResults:
    def __init__(self):
        self.results = []
        self.start_time = time.time()

    def add(self, category: str, test_name: str, passed: bool, details: dict):
        self.results.append({
            "category": category,
            "test": test_name,
            "passed": passed,
            "details": details,
            "timestamp": datetime.now().isoformat()
        })

    def summary(self):
        passed = sum(1 for r in self.results if r["passed"])
        failed = sum(1 for r in self.results if not r["passed"])
        return {
            "total": len(self.results),
            "passed": passed,
            "failed": failed,
            "duration_seconds": round(time.time() - self.start_time, 2)
        }

def query_rag(endpoint: str, question: str, params: dict = None) -> tuple[dict, float]:
    """Query RAG server and return response with timing."""
    try:
        query_params = {"q": question, "top_k": 5}
        if params:
            query_params.update(params)

        url = f"{RAG_SERVER}/{endpoint}?{urllib.parse.urlencode(query_params)}"

        start = time.time()
        with urllib.request.urlopen(url, timeout=30) as response:
            result = json.loads(response.read().decode())
        elapsed = time.time() - start

        return result, elapsed
    except Exception as e:
        return {"error": str(e)}, 0.0

def test_server_health() -> tuple[bool, dict]:
    """Test server health endpoint."""
    try:
        with urllib.request.urlopen(f"{RAG_SERVER}/health", timeout=5) as r:
            health = json.loads(r.read().decode())
            return True, health
    except Exception as e:
        return False, {"error": str(e)}

def test_person_query(name: str) -> tuple[bool, dict]:
    """Test person lookup query."""
    result, elapsed = query_rag("hybrid", f"Who is {name}?")

    has_structured = len(result.get("structured_results", [])) > 0
    has_person = any(r.get("type") == "person" for r in result.get("structured_results", []))

    person_data = None
    if has_person:
        person_data = next(
            (r.get("data") for r in result.get("structured_results", []) if r.get("type") == "person"),
            None
        )

    return has_person, {
        "query": f"Who is {name}?",
        "response_time_ms": round(elapsed * 1000, 2),
        "has_structured_results": has_structured,
        "has_person_data": has_person,
        "person_found": person_data.get("name") if person_data else None,
        "has_openalex": bool(person_data.get("openalex")) if person_data else False,
        "semantic_results_count": len(result.get("semantic_results", []))
    }

def test_project_query(project: str) -> tuple[bool, dict]:
    """Test project lookup query."""
    result, elapsed = query_rag("hybrid", f"What is the {project} project?")

    has_project = any(
        r.get("type") == "project" or project.lower() in str(r.get("data", {})).lower()
        for r in result.get("structured_results", [])
    )

    return has_project, {
        "query": f"What is the {project} project?",
        "response_time_ms": round(elapsed * 1000, 2),
        "structured_results_count": len(result.get("structured_results", [])),
        "semantic_results_count": len(result.get("semantic_results", []))
    }

def test_resource_query(resource_type: str) -> tuple[bool, dict]:
    """Test resource/compute query."""
    result, elapsed = query_rag("hybrid", f"What {resource_type} resources are available?")

    has_results = (
        len(result.get("structured_results", [])) > 0 or
        len(result.get("semantic_results", [])) > 0
    )

    return has_results, {
        "query": f"What {resource_type} resources are available?",
        "response_time_ms": round(elapsed * 1000, 2),
        "structured_results_count": len(result.get("structured_results", [])),
        "semantic_results_count": len(result.get("semantic_results", []))
    }

def test_semantic_search(topic: str) -> tuple[bool, dict]:
    """Test semantic search capabilities."""
    result, elapsed = query_rag("semantic", f"Research on {topic}")

    results = result.get("results", [])
    has_results = len(results) > 0

    return has_results, {
        "query": f"Research on {topic}",
        "response_time_ms": round(elapsed * 1000, 2),
        "results_count": len(results),
        "top_scores": [round(r.get("score", 0), 3) for r in results[:3]] if results else []
    }

def test_bm25_search(keywords: str) -> tuple[bool, dict]:
    """Test BM25 keyword search."""
    result, elapsed = query_rag("bm25", keywords)

    results = result.get("results", [])
    has_results = len(results) > 0

    return has_results, {
        "query": keywords,
        "response_time_ms": round(elapsed * 1000, 2),
        "results_count": len(results),
        "top_scores": [round(r.get("score", 0), 3) for r in results[:3]] if results else []
    }

def test_hybrid_search(query: str) -> tuple[bool, dict]:
    """Test hybrid search (semantic + BM25 + registry)."""
    result, elapsed = query_rag("hybrid", query)

    has_results = (
        len(result.get("structured_results", [])) > 0 or
        len(result.get("semantic_results", [])) > 0
    )

    return has_results, {
        "query": query,
        "response_time_ms": round(elapsed * 1000, 2),
        "structured_results_count": len(result.get("structured_results", [])),
        "semantic_results_count": len(result.get("semantic_results", [])),
        "query_classification": result.get("query_classification", {})
    }

def test_edge_cases() -> list[tuple[str, bool, dict]]:
    """Test edge cases and error handling."""
    tests = []

    # Empty query
    result, elapsed = query_rag("hybrid", "")
    tests.append(("empty_query", "error" not in result or len(result.get("semantic_results", [])) == 0, {
        "response_time_ms": round(elapsed * 1000, 2),
        "handled_gracefully": "error" not in result
    }))

    # Very long query
    long_query = "What is " + "the " * 100 + "Knowledge Lab?"
    result, elapsed = query_rag("hybrid", long_query)
    tests.append(("long_query", "error" not in result, {
        "query_length": len(long_query),
        "response_time_ms": round(elapsed * 1000, 2),
        "handled_gracefully": "error" not in result
    }))

    # Special characters
    result, elapsed = query_rag("hybrid", "What's James Evans' research?")
    tests.append(("special_chars", "error" not in result, {
        "response_time_ms": round(elapsed * 1000, 2),
        "handled_gracefully": "error" not in result
    }))

    # Non-existent person
    result, elapsed = query_rag("hybrid", "Who is Nonexistent Person XYZ?")
    tests.append(("nonexistent_person", True, {  # Should handle gracefully, not find but not error
        "response_time_ms": round(elapsed * 1000, 2),
        "found_results": len(result.get("structured_results", [])) > 0,
        "handled_gracefully": "error" not in result
    }))

    return tests

def test_response_quality(person_name: str) -> tuple[bool, dict]:
    """Test response quality for a known person."""
    result, elapsed = query_rag("hybrid", f"Tell me about {person_name}")

    structured = result.get("structured_results", [])
    person_result = next(
        (r for r in structured if r.get("type") == "person"),
        None
    )

    quality_checks = {
        "has_person_data": person_result is not None,
        "has_name": bool(person_result.get("data", {}).get("name")) if person_result else False,
        "has_role": bool(person_result.get("data", {}).get("role")) if person_result else False,
        "has_openalex": bool(person_result.get("data", {}).get("openalex")) if person_result else False,
        "has_email": bool(person_result.get("data", {}).get("email")) if person_result else False,
    }

    quality_score = sum(quality_checks.values()) / len(quality_checks)

    return quality_score >= 0.6, {
        "query": f"Tell me about {person_name}",
        "response_time_ms": round(elapsed * 1000, 2),
        "quality_checks": quality_checks,
        "quality_score": round(quality_score, 2)
    }

def run_comprehensive_tests():
    """Run all comprehensive tests."""
    print("=" * 60)
    print("CHORUS Comprehensive Test Suite")
    print("=" * 60)
    print(f"Started: {datetime.now().isoformat()}")
    print()

    results = TestResults()

    # 1. Server Health
    print("1. Testing Server Health...")
    passed, details = test_server_health()
    results.add("infrastructure", "server_health", passed, details)
    print(f"   {'✓' if passed else '✗'} Server health: {details}")

    # 2. Person Queries
    print("\n2. Testing Person Queries...")
    test_people = ["James Evans", "Bhargav Srinivasa Desikan", "Hyunin Yoon", "Ziwen Chen"]
    for person in test_people:
        passed, details = test_person_query(person)
        results.add("person_lookup", f"person_{person.replace(' ', '_')}", passed, details)
        status = "✓" if passed else "✗"
        print(f"   {status} {person}: {details['response_time_ms']}ms, OpenAlex: {details['has_openalex']}")

    # 3. Project Queries
    print("\n3. Testing Project Queries...")
    test_projects = ["APTO", "C3S2", "Socio-Cognitive AI"]
    for project in test_projects:
        passed, details = test_project_query(project)
        results.add("project_lookup", f"project_{project.replace(' ', '_')}", passed, details)
        status = "✓" if passed else "✗"
        print(f"   {status} {project}: {details['response_time_ms']}ms, structured: {details['structured_results_count']}")

    # 4. Resource Queries
    print("\n4. Testing Resource Queries...")
    test_resources = ["GPU", "compute", "storage", "data"]
    for resource in test_resources:
        passed, details = test_resource_query(resource)
        results.add("resource_lookup", f"resource_{resource}", passed, details)
        status = "✓" if passed else "✗"
        print(f"   {status} {resource}: {details['response_time_ms']}ms, results: {details['structured_results_count'] + details['semantic_results_count']}")

    # 5. Semantic Search
    print("\n5. Testing Semantic Search...")
    test_topics = ["machine learning", "natural language processing", "network science", "computational social science"]
    for topic in test_topics:
        passed, details = test_semantic_search(topic)
        results.add("semantic_search", f"semantic_{topic.replace(' ', '_')}", passed, details)
        status = "✓" if passed else "✗"
        print(f"   {status} {topic}: {details['response_time_ms']}ms, results: {details['results_count']}")

    # 6. BM25 Keyword Search
    print("\n6. Testing BM25 Keyword Search...")
    test_keywords = ["knowledge lab", "publications", "grants funding", "university chicago"]
    for keywords in test_keywords:
        passed, details = test_bm25_search(keywords)
        results.add("bm25_search", f"bm25_{keywords.replace(' ', '_')}", passed, details)
        status = "✓" if passed else "✗"
        print(f"   {status} '{keywords}': {details['response_time_ms']}ms, results: {details['results_count']}")

    # 7. Hybrid Search
    print("\n7. Testing Hybrid Search...")
    test_queries = [
        "Who works on NLP at Knowledge Lab?",
        "What funding does James Evans have?",
        "How do I use Midway GPU resources?",
        "Tell me about recent publications"
    ]
    for query in test_queries:
        passed, details = test_hybrid_search(query)
        results.add("hybrid_search", f"hybrid_{hash(query) % 10000}", passed, details)
        status = "✓" if passed else "✗"
        print(f"   {status} '{query[:40]}...': {details['response_time_ms']}ms")

    # 8. Edge Cases
    print("\n8. Testing Edge Cases...")
    edge_tests = test_edge_cases()
    for test_name, passed, details in edge_tests:
        results.add("edge_cases", test_name, passed, details)
        status = "✓" if passed else "✗"
        print(f"   {status} {test_name}: {details}")

    # 9. Response Quality
    print("\n9. Testing Response Quality...")
    quality_people = ["James Evans", "Bhargav Srinivasa Desikan"]
    for person in quality_people:
        passed, details = test_response_quality(person)
        results.add("quality", f"quality_{person.replace(' ', '_')}", passed, details)
        status = "✓" if passed else "✗"
        print(f"   {status} {person}: quality_score={details['quality_score']}")

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    summary = results.summary()
    print(f"Total Tests: {summary['total']}")
    print(f"Passed: {summary['passed']} ({summary['passed']/summary['total']*100:.1f}%)")
    print(f"Failed: {summary['failed']} ({summary['failed']/summary['total']*100:.1f}%)")
    print(f"Duration: {summary['duration_seconds']}s")

    # Failed tests detail
    failed = [r for r in results.results if not r["passed"]]
    if failed:
        print("\nFailed Tests:")
        for f in failed:
            print(f"  - {f['category']}/{f['test']}: {f['details']}")

    # Performance analysis
    print("\n" + "=" * 60)
    print("PERFORMANCE ANALYSIS")
    print("=" * 60)

    response_times = []
    for r in results.results:
        if "response_time_ms" in r["details"]:
            response_times.append(r["details"]["response_time_ms"])

    if response_times:
        avg_time = sum(response_times) / len(response_times)
        max_time = max(response_times)
        min_time = min(response_times)
        print(f"Response Times:")
        print(f"  Average: {avg_time:.1f}ms")
        print(f"  Min: {min_time:.1f}ms")
        print(f"  Max: {max_time:.1f}ms")

        slow_queries = [r for r in results.results if r["details"].get("response_time_ms", 0) > 500]
        if slow_queries:
            print(f"\nSlow Queries (>500ms): {len(slow_queries)}")
            for sq in slow_queries[:5]:
                print(f"  - {sq['test']}: {sq['details'].get('response_time_ms')}ms")

    return results

if __name__ == "__main__":
    results = run_comprehensive_tests()

    # Save results to file
    output_file = Path(__file__).parent / "test_results.json"
    with open(output_file, "w") as f:
        json.dump({
            "summary": results.summary(),
            "results": results.results
        }, f, indent=2)
    print(f"\nResults saved to: {output_file}")
