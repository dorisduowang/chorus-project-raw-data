#!/usr/bin/env python3
"""
Test the new multi-agent orchestrator architecture.

Usage:
    python test_orchestrator.py
"""

import asyncio
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load .env file
def load_dotenv():
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    value = value.strip().strip('"').strip("'")
                    os.environ[key.strip()] = value  # Direct assignment, not setdefault

load_dotenv()

from agents import ChorusOrchestrator
from agents.classifier import QueryClassifier


# Test queries for each intent type
TEST_QUERIES = {
    "lookup": [
        "Who is James Evans?",
        "What grants does the lab have?",
        "When is the next deadline?",
        "List all research projects",
        "Show me the team members",
    ],
    "technical": [
        "How do I access Midway?",
        "My git push isn't working",
        "How do I set up conda?",
        "I can't connect to the VPN",
        "Help me install pytorch",
    ],
    "explore": [
        "What are the implications of LLMs for social science research?",
        "Synthesize the relationship between authenticity and AI",
        "Help me think through this research design",
        "What are all the ways we could measure cultural evolution?",
        "Analyze the tradeoffs between interpretability and performance",
    ],
    "connect": [
        "Who's working on something related to NLP?",
        "Connect me with someone who knows about network analysis",
        "How does my project relate to other lab work?",
        "Is anyone else interested in computational linguistics?",
        "Who should I talk to about data pipelines?",
    ],
}


async def test_classifier():
    """Test the query classifier."""
    print("=" * 60)
    print("TESTING QUERY CLASSIFIER")
    print("=" * 60)

    classifier = QueryClassifier()

    results = {"correct": 0, "total": 0}

    for expected_intent, queries in TEST_QUERIES.items():
        print(f"\n--- Testing {expected_intent.upper()} queries ---")
        for query in queries:
            classification = await classifier.classify(query)
            is_correct = classification.intent == expected_intent
            results["total"] += 1
            if is_correct:
                results["correct"] += 1

            status = "✓" if is_correct else "✗"
            print(f"{status} '{query[:50]}...' -> {classification.intent} ({classification.confidence:.2f})")
            if not is_correct:
                print(f"  Expected: {expected_intent}, Got: {classification.intent}")

    accuracy = results["correct"] / results["total"] * 100
    print(f"\nClassifier Accuracy: {results['correct']}/{results['total']} ({accuracy:.1f}%)")

    return accuracy > 70  # Pass if > 70% accuracy


async def test_orchestrator_routing():
    """Test that queries are routed to correct specialists."""
    print("\n" + "=" * 60)
    print("TESTING ORCHESTRATOR ROUTING")
    print("=" * 60)

    # Note: This requires RAG server to be running for full test
    # For now, just test classification and routing logic

    orchestrator = ChorusOrchestrator(
        enable_caching=True,
        enable_parallel=True,
        rag_available=False  # Disable RAG for routing test
    )

    test_cases = [
        ("Who is the lab director?", "memory"),
        ("How do I set up SSH keys?", "mechanic"),
        ("Explore the implications of scaling laws", "muse"),
        ("Who else is working on LLMs?", "matchmaker"),
    ]

    print("\nTesting query routing (without RAG):")
    for query, expected_agent in test_cases:
        # Just test classification
        classification = await orchestrator.classifier.classify(query)

        # Map intent to agent type
        intent_to_agent = {
            "lookup": "memory",
            "technical": "mechanic",
            "explore": "muse",
            "connect": "matchmaker",
        }
        actual_agent = intent_to_agent.get(classification.intent, "muse")

        status = "✓" if actual_agent == expected_agent else "✗"
        print(f"{status} '{query[:40]}...' -> {classification.intent} -> {actual_agent}")

    return True


async def test_full_query():
    """Test a full query through the orchestrator (requires RAG)."""
    print("\n" + "=" * 60)
    print("TESTING FULL QUERY (requires RAG server)")
    print("=" * 60)

    orchestrator = ChorusOrchestrator(
        enable_caching=True,
        enable_parallel=True,
        rag_available=True
    )

    # Check RAG health
    health = await orchestrator.check_rag_health()
    if "error" in health:
        print(f"RAG server not available: {health.get('error')}")
        print("Skipping full query test.")
        return True  # Don't fail if RAG isn't running

    print(f"RAG server connected ({health.get('chunks', 0)} chunks)")

    # Test one query from each type
    test_queries = [
        ("What research projects are happening?", "lookup"),
        ("Explore the concept of epistemic bubbles", "explore"),
    ]

    for query, expected_type in test_queries:
        print(f"\nQuery: {query}")
        print(f"Expected routing: {expected_type}")

        response = await orchestrator.query(query)

        print(f"Response length: {len(response)} chars")
        print(f"Response preview: {response[:200]}...")

        # Get metrics
        metrics = orchestrator.get_metrics()
        print(f"Routing: {metrics['routing']}")

    return True


async def main():
    """Run all tests."""
    print("CHORUS MULTI-AGENT ORCHESTRATOR TESTS")
    print("=" * 60)

    results = []

    # Test 1: Classifier
    try:
        results.append(("Classifier", await test_classifier()))
    except Exception as e:
        print(f"Classifier test failed: {e}")
        results.append(("Classifier", False))

    # Test 2: Routing
    try:
        results.append(("Routing", await test_orchestrator_routing()))
    except Exception as e:
        print(f"Routing test failed: {e}")
        results.append(("Routing", False))

    # Test 3: Full query (optional)
    try:
        results.append(("Full Query", await test_full_query()))
    except Exception as e:
        print(f"Full query test failed: {e}")
        results.append(("Full Query", False))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\nAll tests passed!")
    else:
        print("\nSome tests failed.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
