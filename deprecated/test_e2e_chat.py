#!/usr/bin/env python3
"""
End-to-end test of chat.py with Claude integration.
Tests the full RAG + LLM pipeline.
"""

import os
import sys
import json
import time
import ssl
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import load_dotenv
load_dotenv()

# Fix SSL
try:
    import certifi
    ssl_context = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    ssl_context = ssl.create_default_context()

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
RAG_SERVER = "http://localhost:8765"

def query_rag(question: str) -> dict:
    """Query RAG hybrid endpoint."""
    try:
        import urllib.parse
        url = f"{RAG_SERVER}/hybrid?q={urllib.parse.quote(question)}&top_k=5"
        with urllib.request.urlopen(url, timeout=30) as response:
            return json.loads(response.read().decode())
    except Exception as e:
        return {"error": str(e)}

def call_claude(messages: list, system: str = None) -> tuple[str, float]:
    """Call Claude API and return response with timing."""
    request_body = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 1024,
        "messages": messages
    }
    if system:
        request_body["system"] = system

    headers = {
        "Content-Type": "application/json",
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01"
    }

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(request_body).encode(),
        headers=headers
    )

    start = time.time()
    with urllib.request.urlopen(req, timeout=60, context=ssl_context) as response:
        result = json.loads(response.read().decode())
    elapsed = time.time() - start

    text_parts = []
    if "content" in result and result["content"]:
        for block in result["content"]:
            if block.get("type") == "text":
                text_parts.append(block.get("text", ""))

    return "\n".join(text_parts), elapsed

def format_context(rag_result: dict) -> str:
    """Format RAG results for Claude (simplified)."""
    parts = []

    structured = rag_result.get("structured_results", [])
    if structured:
        parts.append("## Registry Data")
        for r in structured[:3]:
            data = r.get("data", {})
            rtype = r.get("type", "")
            if rtype == "person":
                parts.append(f"### {data.get('name', 'Unknown')}")
                parts.append(f"Role: {data.get('role', 'N/A')}")
                if data.get("openalex"):
                    oa = data["openalex"]
                    parts.append(f"Works: {oa.get('works_count', 0)}, Citations: {oa.get('cited_by_count', 0)}, h-index: {oa.get('h_index', 0)}")
            else:
                parts.append(f"### {data.get('name', 'Unknown')} ({rtype})")
                if data.get("description"):
                    parts.append(data["description"][:300])

    semantic = rag_result.get("semantic_results", [])
    if semantic:
        parts.append("\n## Related Documents")
        for i, r in enumerate(semantic[:2], 1):
            parts.append(f"Source {i}: {r.get('text', '')[:300]}")

    return "\n".join(parts)

SYSTEM_PROMPT = """You are CHORUS, the Knowledge Lab's AI assistant. Answer questions about lab members, projects, funding, and documents. Use the provided context to give accurate, specific answers. Be concise."""

def run_e2e_test():
    """Run end-to-end tests."""
    print("=" * 60)
    print("CHORUS End-to-End Chat Test")
    print("=" * 60)

    test_queries = [
        ("Person query", "Who is James Evans and what does he research?"),
        ("Project query", "What is the APTO project?"),
        ("Resource query", "What GPU resources are available on Midway?"),
        ("Research topic", "Who works on natural language processing?"),
        ("General info", "What are the main research areas at Knowledge Lab?"),
    ]

    results = []

    for test_name, query in test_queries:
        print(f"\n{'='*60}")
        print(f"Test: {test_name}")
        print(f"Query: {query}")
        print("-" * 60)

        # Step 1: RAG retrieval
        rag_start = time.time()
        rag_result = query_rag(query)
        rag_time = time.time() - rag_start

        if "error" in rag_result:
            print(f"RAG Error: {rag_result['error']}")
            results.append({"test": test_name, "success": False, "error": "RAG error"})
            continue

        context = format_context(rag_result)
        structured_count = len(rag_result.get("structured_results", []))
        semantic_count = len(rag_result.get("semantic_results", []))

        print(f"RAG: {rag_time*1000:.0f}ms | Structured: {structured_count} | Semantic: {semantic_count}")

        # Step 2: Claude generation
        augmented_query = f"Question: {query}\n\n---\nContext:\n{context}"
        messages = [{"role": "user", "content": augmented_query}]

        try:
            response, claude_time = call_claude(messages, SYSTEM_PROMPT)
            print(f"Claude: {claude_time*1000:.0f}ms | Response length: {len(response)} chars")
            print("-" * 60)
            print(f"Response: {response[:500]}...")

            results.append({
                "test": test_name,
                "success": True,
                "rag_time_ms": round(rag_time * 1000, 2),
                "claude_time_ms": round(claude_time * 1000, 2),
                "total_time_ms": round((rag_time + claude_time) * 1000, 2),
                "response_length": len(response),
                "structured_results": structured_count,
                "semantic_results": semantic_count
            })
        except Exception as e:
            print(f"Claude Error: {e}")
            results.append({"test": test_name, "success": False, "error": str(e)})

    # Summary
    print("\n" + "=" * 60)
    print("E2E TEST SUMMARY")
    print("=" * 60)

    successful = [r for r in results if r.get("success")]
    failed = [r for r in results if not r.get("success")]

    print(f"Total: {len(results)} | Passed: {len(successful)} | Failed: {len(failed)}")

    if successful:
        avg_rag = sum(r["rag_time_ms"] for r in successful) / len(successful)
        avg_claude = sum(r["claude_time_ms"] for r in successful) / len(successful)
        avg_total = sum(r["total_time_ms"] for r in successful) / len(successful)

        print(f"\nPerformance (avg):")
        print(f"  RAG retrieval: {avg_rag:.0f}ms")
        print(f"  Claude generation: {avg_claude:.0f}ms")
        print(f"  Total end-to-end: {avg_total:.0f}ms")

    if failed:
        print(f"\nFailed tests:")
        for f in failed:
            print(f"  - {f['test']}: {f.get('error', 'Unknown')}")

    return results

if __name__ == "__main__":
    results = run_e2e_test()

    # Save results
    output_file = Path(__file__).parent / "test_e2e_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")
