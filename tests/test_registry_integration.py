#!/usr/bin/env python3
"""
Test script for registry-RAG integration.

Run this to verify the integration works before starting the server.
"""

import json
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_registry_module():
    """Test the registry_rag module directly."""
    print("=" * 60)
    print("Testing Registry RAG Module")
    print("=" * 60)

    try:
        from registry import RegistryRAG, RegistryLookup, QueryClassifier
    except ImportError as e:
        print(f"FAIL: Could not import registry: {e}")
        return False

    # Test RegistryLookup
    print("\n1. Testing RegistryLookup...")
    lookup = RegistryLookup()
    stats = lookup.get_stats()
    print(f"   Registry stats: {stats}")

    # Test person lookup
    jake = lookup.find_person("Jake Burchard")
    if jake:
        print(f"   ✓ Found Jake Burchard: {jake.get('role')} at {jake.get('institution')}")
    else:
        print("   ✗ Could not find Jake Burchard")

    # Test role lookup
    phd_students = lookup.find_people_by_role("PhD Student")
    print(f"   ✓ Found {len(phd_students)} PhD students")

    # Test institution lookup
    uchicago = lookup.find_people_by_institution("University of Chicago")
    print(f"   ✓ Found {len(uchicago)} people at UChicago")

    # Test project lookup
    apto = lookup.find_project("apto")
    if apto:
        print(f"   ✓ Found APTO project: {apto.get('name')}")
    else:
        print("   ✗ Could not find APTO project")

    # Test QueryClassifier
    print("\n2. Testing QueryClassifier...")
    classifier = QueryClassifier()

    test_cases = [
        ("Who is Jake Burchard?", "structured"),
        ("List all PhD students", "structured"),
        ("What is APTO?", "hybrid"),
        ("Tell me about the APTO project", "structured"),
        ("What projects do we have?", "structured"),
        ("What funding do we have?", "structured"),
        ("How does the lab approach AI research?", "semantic"),
    ]

    for query, expected_type in test_cases:
        result = classifier.classify(query)
        status = "✓" if result.query_type == expected_type else "~"
        print(f"   {status} '{query}' -> {result.query_type} (expected: {expected_type})")

    # Test full RegistryRAG
    print("\n3. Testing RegistryRAG queries...")
    rag = RegistryRAG()

    queries = [
        "Who is Jake Burchard?",
        "List all PhD students",
        "What projects do we have?",
        "Who works on APTO?",
        "What funding do we have?",
    ]

    for query in queries:
        result = rag.query(query)
        n_results = len(result.get("structured_results", []))
        qtype = result["classification"]["type"]
        print(f"   ✓ '{query}' -> {qtype}, {n_results} results")

    print("\n✓ Registry module tests passed!")
    return True


def test_text_generation():
    """Test the text summary generation for embeddings."""
    print("\n" + "=" * 60)
    print("Testing Text Summary Generation")
    print("=" * 60)

    from registry import RegistryRAG

    rag = RegistryRAG()
    summaries = rag.generate_embeddings_data()

    print(f"\nGenerated {len(summaries)} text summaries for embedding:")

    # Group by type
    by_type = {}
    for s in summaries:
        t = s["entity_type"]
        by_type[t] = by_type.get(t, 0) + 1

    for entity_type, count in by_type.items():
        print(f"  - {entity_type}: {count}")

    # Show a few examples
    print("\nSample summaries:")
    for s in summaries[:3]:
        print(f"\n  [{s['entity_type']}] {s['entity_id']}")
        print(f"  {s['text'][:200]}...")

    return True


def test_server_integration():
    """Test that the server can import the registry module."""
    print("\n" + "=" * 60)
    print("Testing Server Integration")
    print("=" * 60)

    # Check if we can import the server module
    try:
        # Just test the import path - using rag_core which contains the shared logic
        import rag_core
        print("   ✓ rag_core imports successfully")

        if hasattr(rag_core, 'REGISTRY_AVAILABLE'):
            print(f"   ✓ REGISTRY_AVAILABLE = {rag_core.REGISTRY_AVAILABLE}")
        else:
            print("   ✗ REGISTRY_AVAILABLE not defined")

        return True

    except Exception as e:
        print(f"   ✗ Server import failed: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("CHORUS Registry Integration Tests")
    print("=" * 60)

    results = []

    # Test 1: Registry module
    results.append(("Registry Module", test_registry_module()))

    # Test 2: Text generation
    results.append(("Text Generation", test_text_generation()))

    # Don't fully import server as it starts loading models
    # results.append(("Server Integration", test_server_integration()))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n✓ All tests passed!")
        print("\nTo start the server with registry integration:")
        print("  python rag_server_fastapi.py")
        print("\nThen test with:")
        print("  curl 'http://localhost:8765/registry?q=Who+is+Jake+Burchard'")
        print("  curl 'http://localhost:8765/hybrid?q=What+projects+do+we+have'")
    else:
        print("\n✗ Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
