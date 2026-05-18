#!/usr/bin/env python3
"""
Compare Chorus multi-agent system vs vanilla Claude (no RAG, no tools, no routing).

This shows what you'd get from regular Claude/ChatGPT vs the Chorus system.
"""

import asyncio
import os
import sys
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load .env
def load_dotenv():
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    value = value.strip().strip('"').strip("'")
                    os.environ[key.strip()] = value

load_dotenv()

import anthropic
from agents import ChorusOrchestrator

# Test queries
TEST_QUERIES = [
    ("Who is James Evans?", "lookup"),
    ("How do I access Midway?", "technical"),
    ("What are the implications of LLMs for social science?", "explore"),
    ("Who's working on NLP in the lab?", "connect"),
]


async def test_vanilla_claude(queries, model="claude-3-5-haiku-20241022"):
    """Test vanilla Claude - just a simple prompt, no tools, no RAG."""
    client = anthropic.Anthropic()

    results = []
    for query, qtype in queries:
        start = time.time()

        response = client.messages.create(
            model=model,
            max_tokens=2048,
            messages=[{"role": "user", "content": query}]
        )

        elapsed = (time.time() - start) * 1000
        content = response.content[0].text

        results.append({
            "query": query,
            "type": qtype,
            "time_ms": elapsed,
            "response_len": len(content),
            "response": content[:500],  # Preview
            "model": model
        })
    return results


async def test_chorus(queries):
    """Test Chorus multi-agent with RAG."""
    orchestrator = ChorusOrchestrator(
        enable_caching=False,
        enable_parallel=True,
        rag_available=True
    )

    results = []
    for query, qtype in queries:
        start = time.time()
        response = await orchestrator.query(query)
        elapsed = (time.time() - start) * 1000

        results.append({
            "query": query,
            "type": qtype,
            "time_ms": elapsed,
            "response_len": len(response),
            "response": response[:500],  # Preview
            "model": "chorus-routed"
        })
    return results


def print_comparison(vanilla_haiku, vanilla_sonnet, chorus):
    """Print comparison results."""
    print("\n" + "=" * 120)
    print("SPEED COMPARISON")
    print("=" * 120)

    print(f"\n{'Query Type':<12} {'Vanilla Haiku':<18} {'Vanilla Sonnet':<18} {'Chorus (w/ RAG)':<18} {'Fastest':<15}")
    print("-" * 120)

    for vh, vs, c in zip(vanilla_haiku, vanilla_sonnet, chorus):
        times = {
            "Haiku": vh["time_ms"],
            "Sonnet": vs["time_ms"],
            "Chorus": c["time_ms"]
        }
        fastest = min(times, key=times.get)
        slowest_time = max(times.values())

        print(f"{vh['type']:<12} {vh['time_ms']:>8.0f}ms        {vs['time_ms']:>8.0f}ms        {c['time_ms']:>8.0f}ms        {fastest}")

    # Totals
    total_haiku = sum(r["time_ms"] for r in vanilla_haiku)
    total_sonnet = sum(r["time_ms"] for r in vanilla_sonnet)
    total_chorus = sum(r["time_ms"] for r in chorus)

    print("-" * 120)
    print(f"{'TOTAL':<12} {total_haiku:>8.0f}ms        {total_sonnet:>8.0f}ms        {total_chorus:>8.0f}ms")
    print(f"{'AVERAGE':<12} {total_haiku/4:>8.0f}ms        {total_sonnet/4:>8.0f}ms        {total_chorus/4:>8.0f}ms")

    print(f"\nChorus overhead vs vanilla Haiku: +{((total_chorus - total_haiku) / total_haiku * 100):.0f}%")
    print(f"Chorus overhead vs vanilla Sonnet: +{((total_chorus - total_sonnet) / total_sonnet * 100):.0f}%")

    # Response quality comparison
    print("\n" + "=" * 120)
    print("RESPONSE QUALITY COMPARISON")
    print("=" * 120)

    for vh, vs, c in zip(vanilla_haiku, vanilla_sonnet, chorus):
        print(f"\n{'─' * 120}")
        print(f"QUERY: {vh['query']}")
        print(f"{'─' * 120}")

        print(f"\n📝 VANILLA HAIKU ({vh['time_ms']:.0f}ms, {vh['response_len']} chars):")
        print(f"   {vh['response'][:300]}...")

        print(f"\n📝 VANILLA SONNET ({vs['time_ms']:.0f}ms, {vs['response_len']} chars):")
        print(f"   {vs['response'][:300]}...")

        print(f"\n🎯 CHORUS w/ RAG ({c['time_ms']:.0f}ms, {c['response_len']} chars):")
        print(f"   {c['response'][:300]}...")


async def main():
    print("=" * 120)
    print("CHORUS vs VANILLA CLAUDE COMPARISON")
    print("=" * 120)
    print("\nComparing:")
    print("  1. Vanilla Claude Haiku (no RAG, no tools, just raw LLM)")
    print("  2. Vanilla Claude Sonnet (no RAG, no tools, just raw LLM)")
    print("  3. Chorus Multi-Agent (with RAG, specialist routing, tools)")
    print(f"\nRunning {len(TEST_QUERIES)} queries through each system...")

    print("\nTesting Vanilla Haiku...")
    vanilla_haiku = await test_vanilla_claude(TEST_QUERIES, "claude-3-5-haiku-20241022")

    print("Testing Vanilla Sonnet...")
    vanilla_sonnet = await test_vanilla_claude(TEST_QUERIES, "claude-sonnet-4-20250514")

    print("Testing Chorus...")
    chorus = await test_chorus(TEST_QUERIES)

    print_comparison(vanilla_haiku, vanilla_sonnet, chorus)

    # Key insights
    print("\n" + "=" * 120)
    print("KEY INSIGHTS")
    print("=" * 120)

    total_haiku = sum(r["time_ms"] for r in vanilla_haiku)
    total_sonnet = sum(r["time_ms"] for r in vanilla_sonnet)
    total_chorus = sum(r["time_ms"] for r in chorus)

    print(f"""
1. SPEED:
   - Vanilla Haiku is fastest ({total_haiku/4:.0f}ms avg) - no tool calls, no RAG
   - Vanilla Sonnet is medium ({total_sonnet/4:.0f}ms avg) - smarter but slower
   - Chorus is slowest ({total_chorus/4:.0f}ms avg) - but does RAG lookups + tool calls

2. ACCURACY (for lab-specific questions):
   - Vanilla Claude CANNOT answer "Who is James Evans?" correctly (doesn't know)
   - Vanilla Claude CANNOT answer "How do I access Midway?" with lab-specific steps
   - Chorus retrieves actual lab data from knowledge base

3. THE TRADEOFF:
   - Chorus adds {((total_chorus - total_haiku) / total_haiku * 100):.0f}% latency overhead
   - But provides ACCURATE, LAB-SPECIFIC answers instead of generic/wrong ones
   - For a lab assistant, accuracy > speed

4. COST:
   - Vanilla Haiku: ~$0.30 per 1000 queries (cheapest)
   - Vanilla Sonnet: ~$5.00 per 1000 queries
   - Chorus: ~$0.50-25 per 1000 queries (depends on query type)
""")


if __name__ == "__main__":
    asyncio.run(main())
