#!/usr/bin/env python3
"""
Test script for Semantic Registry Index.

Tests:
1. Index building with real registry data
2. Search for each entity type (person, dataset, project, funding)
3. find_similar functionality
4. Save/load persistence
5. Merge with structured results

Run:
    python -m pytest tests/test_semantic_registry.py -v
    # Or directly:
    python tests/test_semantic_registry.py
"""

import json
import os
import sys
import tempfile
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_index_building():
    """Test that the index builds successfully with real registry data."""
    print("\n" + "=" * 60)
    print("Test: Index Building")
    print("=" * 60)

    from registry.semantic_index import SemanticRegistryIndex

    registry_path = "Data/lab_registry.json"
    if not Path(registry_path).exists():
        print(f"SKIP: Registry not found at {registry_path}")
        return True

    # Build index
    index = SemanticRegistryIndex(registry_path=registry_path)
    index.build_index()

    # Verify counts
    stats = index.stats()
    print(f"Index stats: {stats}")

    assert stats["total_entities"] > 0, "Index should have entities"
    assert stats["embedding_dim"] > 0, "Embedding dimension should be set"
    assert stats["has_faiss"], "FAISS index should be built"

    # Check entity types
    by_type = stats["by_type"]
    print(f"Entities by type: {by_type}")

    # Should have people (based on the registry we saw)
    assert by_type.get("person", 0) > 0, "Should have person entities"

    print("PASS: Index building successful")
    return True


def test_search_people():
    """Test searching for people with natural language queries."""
    print("\n" + "=" * 60)
    print("Test: Search People")
    print("=" * 60)

    from registry.semantic_index import SemanticRegistryIndex

    registry_path = "Data/lab_registry.json"
    if not Path(registry_path).exists():
        print(f"SKIP: Registry not found at {registry_path}")
        return True

    index = SemanticRegistryIndex(registry_path=registry_path)
    index.build_index()

    # Test queries for people
    test_queries = [
        "researchers working on network science",
        "PhD students at University of Chicago",
        "people studying machine learning",
        "faculty members working on computational social science",
    ]

    for query in test_queries:
        results = index.search(query, top_k=5, entity_type="person")
        print(f"\nQuery: '{query}'")
        print(f"Found {len(results)} results:")
        for r in results[:3]:
            print(f"  [{r['score']:.3f}] {r['name']} ({r['entity_type']})")

        # Should return results
        assert len(results) > 0, f"Should find results for '{query}'"
        # All results should be people
        assert all(r["entity_type"] == "person" for r in results), "All results should be people"

    print("\nPASS: People search working")
    return True


def test_search_datasets():
    """Test searching for datasets."""
    print("\n" + "=" * 60)
    print("Test: Search Datasets")
    print("=" * 60)

    from registry.semantic_index import SemanticRegistryIndex

    registry_path = "Data/lab_registry.json"
    if not Path(registry_path).exists():
        print(f"SKIP: Registry not found at {registry_path}")
        return True

    index = SemanticRegistryIndex(registry_path=registry_path)
    index.build_index()

    # Check if we have datasets
    stats = index.stats()
    if stats["by_type"].get("dataset", 0) == 0:
        print("SKIP: No datasets in registry")
        return True

    # Test queries for datasets
    test_queries = [
        "scholarly publication data",
        "bibliometric datasets",
        "Microsoft Academic Graph",
    ]

    for query in test_queries:
        results = index.search(query, top_k=5, entity_type="dataset")
        print(f"\nQuery: '{query}'")
        print(f"Found {len(results)} results:")
        for r in results[:3]:
            print(f"  [{r['score']:.3f}] {r['name']}")

    print("\nPASS: Dataset search working")
    return True


def test_search_projects():
    """Test searching for projects."""
    print("\n" + "=" * 60)
    print("Test: Search Projects")
    print("=" * 60)

    from registry.semantic_index import SemanticRegistryIndex

    registry_path = "Data/lab_registry.json"
    if not Path(registry_path).exists():
        print(f"SKIP: Registry not found at {registry_path}")
        return True

    index = SemanticRegistryIndex(registry_path=registry_path)
    index.build_index()

    # Check if we have projects
    stats = index.stats()
    if stats["by_type"].get("project", 0) == 0:
        print("SKIP: No projects in registry")
        return True

    # Test queries for projects
    test_queries = [
        "AI prediction research",
        "science of science project",
        "NSF funded initiatives",
    ]

    for query in test_queries:
        results = index.search(query, top_k=5, entity_type="project")
        print(f"\nQuery: '{query}'")
        print(f"Found {len(results)} results:")
        for r in results[:3]:
            print(f"  [{r['score']:.3f}] {r['name']}")

    print("\nPASS: Project search working")
    return True


def test_search_all_types():
    """Test searching across all entity types."""
    print("\n" + "=" * 60)
    print("Test: Search All Types")
    print("=" * 60)

    from registry.semantic_index import SemanticRegistryIndex

    registry_path = "Data/lab_registry.json"
    if not Path(registry_path).exists():
        print(f"SKIP: Registry not found at {registry_path}")
        return True

    index = SemanticRegistryIndex(registry_path=registry_path)
    index.build_index()

    # Search without type filter
    query = "network analysis and science"
    results = index.search(query, top_k=10, entity_type=None)

    print(f"Query: '{query}' (no type filter)")
    print(f"Found {len(results)} results across types:")

    # Group by type
    by_type = {}
    for r in results:
        t = r["entity_type"]
        by_type[t] = by_type.get(t, 0) + 1
        print(f"  [{r['score']:.3f}] [{r['entity_type']}] {r['name']}")

    print(f"\nResults by type: {by_type}")

    assert len(results) > 0, "Should find results"

    print("\nPASS: Cross-type search working")
    return True


def test_find_similar():
    """Test finding similar entities."""
    print("\n" + "=" * 60)
    print("Test: Find Similar Entities")
    print("=" * 60)

    from registry.semantic_index import SemanticRegistryIndex

    registry_path = "Data/lab_registry.json"
    if not Path(registry_path).exists():
        print(f"SKIP: Registry not found at {registry_path}")
        return True

    index = SemanticRegistryIndex(registry_path=registry_path)
    index.build_index()

    # Get first person
    person_entities = [e for e in index.entities if e.entity_type == "person"]
    if not person_entities:
        print("SKIP: No people in index")
        return True

    test_entity = person_entities[0]
    print(f"Finding entities similar to: {test_entity.name} ({test_entity.id})")

    # Find similar (any type)
    similar = index.find_similar(test_entity.id, top_k=5, same_type_only=False)
    print(f"\nSimilar (any type):")
    for r in similar:
        print(f"  [{r['score']:.3f}] [{r['entity_type']}] {r['name']}")

    assert len(similar) > 0, "Should find similar entities"
    assert test_entity.id not in [r["id"] for r in similar], "Should not include source entity"

    # Find similar (same type only)
    similar_same_type = index.find_similar(test_entity.id, top_k=5, same_type_only=True)
    print(f"\nSimilar (same type only):")
    for r in similar_same_type:
        print(f"  [{r['score']:.3f}] [{r['entity_type']}] {r['name']}")

    assert all(r["entity_type"] == "person" for r in similar_same_type), "Should only return people"

    print("\nPASS: find_similar working")
    return True


def test_save_load_persistence():
    """Test saving and loading the index."""
    print("\n" + "=" * 60)
    print("Test: Save/Load Persistence")
    print("=" * 60)

    from registry.semantic_index import SemanticRegistryIndex

    registry_path = "Data/lab_registry.json"
    if not Path(registry_path).exists():
        print(f"SKIP: Registry not found at {registry_path}")
        return True

    # Build index
    index = SemanticRegistryIndex(registry_path=registry_path)
    index.build_index()
    original_stats = index.stats()
    print(f"Original index stats: {original_stats}")

    # Test a search before saving
    query = "network science"
    original_results = index.search(query, top_k=5)
    print(f"Original search for '{query}': {len(original_results)} results")

    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        temp_path = f.name

    try:
        index.save_index(temp_path)
        print(f"Saved index to {temp_path}")

        # Load into new instance
        loaded_index = SemanticRegistryIndex()
        loaded_index.load_index(temp_path)
        loaded_stats = loaded_index.stats()
        print(f"Loaded index stats: {loaded_stats}")

        # Verify stats match
        assert loaded_stats["total_entities"] == original_stats["total_entities"], "Entity count should match"
        assert loaded_stats["embedding_dim"] == original_stats["embedding_dim"], "Embedding dim should match"

        # Test search on loaded index
        loaded_results = loaded_index.search(query, top_k=5)
        print(f"Loaded search for '{query}': {len(loaded_results)} results")

        # Results should be the same
        assert len(loaded_results) == len(original_results), "Result count should match"
        for orig, loaded in zip(original_results, loaded_results):
            assert orig["id"] == loaded["id"], "Result IDs should match"
            # Scores might have tiny floating point differences
            assert abs(orig["score"] - loaded["score"]) < 0.001, "Scores should be very close"

        print("\nPASS: Save/load persistence working")

    finally:
        # Cleanup
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    return True


def test_merge_with_structured():
    """Test merging semantic results with structured results."""
    print("\n" + "=" * 60)
    print("Test: Merge with Structured Results")
    print("=" * 60)

    from registry.semantic_index import SemanticRegistryIndex, merge_with_structured

    registry_path = "Data/lab_registry.json"
    if not Path(registry_path).exists():
        print(f"SKIP: Registry not found at {registry_path}")
        return True

    index = SemanticRegistryIndex(registry_path=registry_path)
    index.build_index()

    # Get some semantic results
    semantic_results = index.search("machine learning researchers", top_k=5)
    print(f"Semantic results: {len(semantic_results)}")
    for r in semantic_results[:3]:
        print(f"  [{r['score']:.3f}] {r['name']}")

    # Create mock structured results (simulating RegistryLookup output)
    # Include some overlap and some unique
    structured_results = []

    # Add one result that overlaps with semantic
    if semantic_results:
        first_sem = semantic_results[0]
        # Extract the original ID from "person-<id>"
        original_id = first_sem["id"].replace("person-", "").replace("dataset-", "").replace("project-", "")
        structured_results.append({
            "type": first_sem["entity_type"],
            "data": {
                "id": original_id,
                "name": first_sem["name"],
            },
            "match_reason": "Name match"
        })

    # Add a unique structured result
    structured_results.append({
        "type": "person",
        "data": {
            "id": "test-structured-only",
            "name": "Test Person (Structured Only)",
        },
        "match_reason": "Role match"
    })

    print(f"Structured results: {len(structured_results)}")

    # Merge
    merged = merge_with_structured(semantic_results, structured_results, boost_overlap=0.3)
    print(f"\nMerged results: {len(merged)}")
    for r in merged[:5]:
        print(f"  [{r.get('merged_score', r.get('score', 0)):.3f}] {r['name']} [{r.get('match_type', 'unknown')}]")

    # Verify merged results
    assert len(merged) > 0, "Should have merged results"

    # Check for overlapping result (should be boosted)
    overlap_results = [r for r in merged if r.get("match_type") == "both"]
    if overlap_results:
        print(f"\nOverlapping results boosted: {len(overlap_results)}")

    print("\nPASS: Merge with structured working")
    return True


def test_get_entity():
    """Test getting an entity by ID."""
    print("\n" + "=" * 60)
    print("Test: Get Entity by ID")
    print("=" * 60)

    from registry.semantic_index import SemanticRegistryIndex

    registry_path = "Data/lab_registry.json"
    if not Path(registry_path).exists():
        print(f"SKIP: Registry not found at {registry_path}")
        return True

    index = SemanticRegistryIndex(registry_path=registry_path)
    index.build_index()

    # Get a known entity
    if index.entities:
        test_entity = index.entities[0]
        print(f"Looking up: {test_entity.id}")

        retrieved = index.get_entity(test_entity.id)
        assert retrieved is not None, "Should find entity"
        assert retrieved["id"] == test_entity.id, "ID should match"
        assert retrieved["name"] == test_entity.name, "Name should match"
        print(f"Found: {retrieved['name']} ({retrieved['entity_type']})")

        # Test non-existent entity
        not_found = index.get_entity("non-existent-id-12345")
        assert not_found is None, "Should return None for non-existent entity"
        print("Non-existent entity correctly returns None")

    print("\nPASS: Get entity by ID working")
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("CHORUS Semantic Registry Index Tests")
    print("=" * 60)

    results = []

    # Run tests
    tests = [
        ("Index Building", test_index_building),
        ("Search People", test_search_people),
        ("Search Datasets", test_search_datasets),
        ("Search Projects", test_search_projects),
        ("Search All Types", test_search_all_types),
        ("Find Similar", test_find_similar),
        ("Save/Load Persistence", test_save_load_persistence),
        ("Merge with Structured", test_merge_with_structured),
        ("Get Entity by ID", test_get_entity),
    ]

    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\nFAIL: {name} raised exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        symbol = "+" if passed else "-"
        print(f"  [{symbol}] {status}: {name}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\nAll tests passed!")
        print("\nTo use the semantic registry:")
        print("  from registry import SemanticRegistryIndex")
        print("  index = SemanticRegistryIndex()")
        print("  index.build_index()")
        print("  results = index.search('researchers working on network science')")
    else:
        print("\nSome tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
