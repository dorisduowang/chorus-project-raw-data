#!/usr/bin/env python3
"""Test multi-turn conversation handling."""

import os
import sys
import json
import ssl
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import load_dotenv
load_dotenv()

try:
    import certifi
    ssl_context = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    ssl_context = ssl.create_default_context()

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
RAG_SERVER = "http://localhost:8765"

def query_rag(question: str) -> dict:
    try:
        import urllib.parse
        url = f"{RAG_SERVER}/hybrid?q={urllib.parse.quote(question)}&top_k=5"
        with urllib.request.urlopen(url, timeout=30) as response:
            return json.loads(response.read().decode())
    except Exception as e:
        return {"error": str(e)}

def call_claude(messages: list, system: str = None) -> str:
    request_body = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 512,
        "messages": messages
    }
    if system:
        request_body["system"] = system

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(request_body).encode(),
        headers={
            "Content-Type": "application/json",
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01"
        }
    )

    with urllib.request.urlopen(req, timeout=60, context=ssl_context) as response:
        result = json.loads(response.read().decode())

    return "".join(b.get("text", "") for b in result.get("content", []) if b.get("type") == "text")

def format_context(rag_result: dict) -> str:
    parts = []
    for r in rag_result.get("structured_results", [])[:2]:
        data = r.get("data", {})
        if r.get("type") == "person":
            parts.append(f"Person: {data.get('name')} - {data.get('role')}")
            if data.get("openalex"):
                oa = data["openalex"]
                parts.append(f"  Works: {oa.get('works_count')}, Citations: {oa.get('cited_by_count')}")
        else:
            parts.append(f"{r.get('type')}: {data.get('name')}")
    return "\n".join(parts)

SYSTEM = "You are CHORUS, Knowledge Lab's assistant. Be concise. Remember prior conversation context."

def test_multiturn():
    print("=" * 60)
    print("Multi-Turn Conversation Test")
    print("=" * 60)

    conversation = []

    turns = [
        "Who is James Evans?",
        "What's his h-index?",  # Should remember James Evans
        "What projects does the lab work on?",
        "Tell me more about the first one"  # Should remember APTO
    ]

    for i, query in enumerate(turns, 1):
        print(f"\n[Turn {i}] User: {query}")

        # Get RAG context
        rag_result = query_rag(query)
        context = format_context(rag_result)

        # Build message
        if context:
            augmented = f"Question: {query}\n\nContext:\n{context}"
        else:
            augmented = query

        conversation.append({"role": "user", "content": augmented})

        try:
            response = call_claude(conversation, SYSTEM)
            print(f"CHORUS: {response[:300]}...")
            conversation.append({"role": "assistant", "content": response})
        except Exception as e:
            print(f"Error: {e}")
            break

    print("\n" + "=" * 60)
    print(f"Conversation completed: {len(conversation)//2} turns")

if __name__ == "__main__":
    test_multiturn()
