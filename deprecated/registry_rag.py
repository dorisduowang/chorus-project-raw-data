#!/usr/bin/env python3
"""
Registry-RAG Integration Module (Backward Compatibility Wrapper)

This module provides backward compatibility for code that imports from registry_rag.
The actual implementation has been moved to the registry/ package.

Usage:
    from registry_rag import RegistryRAG

    rag = RegistryRAG()
    results = rag.query("Who works on APTO?")

For new code, prefer importing from the registry package:
    from registry import RegistryRAG
"""

# Re-export everything from the registry package for backward compatibility
from registry import (
    RegistryRAG,
    RegistryLookup,
    QueryClassifier,
    RegistryTextGenerator,
    RegistryMatch,
    QueryClassification,
    TOPIC_ALIASES,
    expand_topic_aliases,
)

__all__ = [
    "RegistryRAG",
    "RegistryLookup",
    "QueryClassifier",
    "RegistryTextGenerator",
    "RegistryMatch",
    "QueryClassification",
    "TOPIC_ALIASES",
    "expand_topic_aliases",
]


# =============================================================================
# CLI for testing
# =============================================================================

def main():
    """Test the registry RAG integration."""
    rag = RegistryRAG()

    print("Registry Statistics:")
    stats = rag.get_registry_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")

    print("\n" + "="*60)
    print("Testing queries...")
    print("="*60)

    test_queries = [
        # Basic person lookup
        "Who is Jake Burchard?",
        "Tell me about James Evans",

        # Role-based queries
        "List all PhD students",
        "Show me the faculty",

        # Research interest queries
        "What does Jake research?",
        "Jake's research interests",
        "What areas does James Evans focus on?",

        # Contact queries
        "How to contact Jake Burchard?",
        "What's Evans' email?",
        "Jake's email address",

        # Collaboration queries
        "Who collaborates with James Evans?",
        "Evans' co-authors",
        "Find collaborators of Jake",

        # Comparison queries
        "Compare Jake and James' research",
        "Difference between Evans and Burchard's work",

        # Stats queries
        "How many publications does Jake have?",
        "What's Evans' h-index?",
        "Jake's citation count",

        # Topic-based queries
        "Who works on network analysis?",
        "Find researchers studying social influence",
        "Experts in machine learning",

        # Publication queries
        "What has Jake published?",
        "Papers by Evans",
        "Recent publications on embeddings",

        # Project queries
        "Who works on APTO?",
        "What projects do we have?",

        # Institution queries
        "Who is at University of Chicago?",
        "Researchers from Stanford",

        # Ambiguous/natural language queries
        "Tell me about network analysis research in the lab",
        "What do people study here?",

        # Topic alias queries - test abbreviation expansion
        "Who works on NLP?",  # Should find people with "Natural Language Processing" topics
        "Who works on ML?",  # Should find people with "Machine Learning" topics
        "Who works on AI?",  # Should find people with "Artificial Intelligence" topics
        "Who works on XAI?",  # Should find people with "Explainable AI" topics
        "Who works on bibliometrics?",  # Should find same people as "scientometrics"
    ]

    for query in test_queries:
        print(f"\n>>> {query}")
        result = rag.query(query)
        print(f"Type: {result['classification']['type']} (confidence: {result['classification']['confidence']:.2f})")
        print(f"Entity types: {result['classification']['entity_types']}")
        print(f"Results: {len(result['structured_results'])}")
        if result["answer"]:
            print(f"\nAnswer:\n{result['answer'][:500]}...")
        print("-" * 40)


if __name__ == "__main__":
    main()
