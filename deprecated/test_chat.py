#!/usr/bin/env python3
"""
Test the chat interface end-to-end with Claude.
"""
import os
import sys
import time
import json
import urllib.request
import urllib.parse

# Load .env
from pathlib import Path
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and (key not in os.environ or not os.environ.get(key)):
                    os.environ[key] = value

# Import from chat.py
sys.path.insert(0, str(Path(__file__).parent))
from chat import query_rag, call_claude, format_context

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

TEST_QUESTIONS = [
    ("Who is James Evans?", ["director", "professor", "sociology", "publications"]),
    ("Who works on NLP?", ["natural language", "text", "researcher"]),
    ("What compute resources are available?", ["gpu", "midway", "cluster"]),
    ("Papers by James Evans", ["publication", "paper", "citation"]),
    ("UChicago researchers in network science", ["network", "chicago"]),
]

def test_rag_only():
    """Test RAG pipeline without Claude."""
    print("\n" + "=" * 60)
    print("RAG-Only Tests (no Claude)")
    print("=" * 60)

    for question, keywords in TEST_QUESTIONS:
        start = time.perf_counter()
        result = query_rag(question)
        elapsed = time.perf_counter() - start

        context = format_context(result)
        context_lower = context.lower()

        found = [k for k in keywords if k.lower() in context_lower]
        status = "✓" if len(found) >= 1 else "✗"

        print(f"\n{status} {question}")
        print(f"  Time: {elapsed*1000:.0f}ms")
        print(f"  Context length: {len(context)} chars")
        print(f"  Keywords found: {found}")

        if "error" in result:
            print(f"  Error: {result['error']}")


def test_full_pipeline():
    """Test full RAG + Claude pipeline."""
    print("\n" + "=" * 60)
    print("Full Pipeline Tests (RAG + Claude)")
    print("=" * 60)

    if not ANTHROPIC_API_KEY:
        print("Warning: ANTHROPIC_API_KEY not set, skipping Claude tests")
        return

    system_prompt = """You are CHORUS, the Knowledge Lab's AI assistant.
Answer questions about the lab using the provided context. Be concise."""

    for question, keywords in TEST_QUESTIONS[:3]:  # Limit to 3 to save API calls
        print(f"\nQ: {question}")

        # RAG phase
        start_rag = time.perf_counter()
        rag_result = query_rag(question)
        context = format_context(rag_result)
        rag_time = time.perf_counter() - start_rag

        if "error" in rag_result:
            print(f"  RAG Error: {rag_result['error']}")
            continue

        # Claude phase
        augmented_query = f"Question: {question}\n\n---\nContext from knowledge base:\n{context}"
        messages = [{"role": "user", "content": augmented_query}]

        start_claude = time.perf_counter()
        try:
            response = call_claude(messages, system_prompt)
            claude_time = time.perf_counter() - start_claude

            print(f"  RAG time: {rag_time*1000:.0f}ms, Claude time: {claude_time*1000:.0f}ms")
            print(f"  Total: {(rag_time + claude_time)*1000:.0f}ms")
            print(f"\nA: {response[:500]}...")

            # Check for keywords in response
            response_lower = response.lower()
            found = [k for k in keywords if k.lower() in response_lower]
            print(f"\n  Keywords in response: {found}")

        except Exception as e:
            print(f"  Claude Error: {e}")


def main():
    # Check RAG server
    try:
        with urllib.request.urlopen("http://localhost:8765/health", timeout=5) as r:
            health = json.loads(r.read().decode())
            print(f"RAG Server: {health['chunks']} chunks indexed")
    except Exception as e:
        print(f"Error: RAG server not available - {e}")
        print("Start it with: python rag_http_server.py")
        return

    test_rag_only()
    test_full_pipeline()

    print("\n" + "=" * 60)
    print("Tests Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
