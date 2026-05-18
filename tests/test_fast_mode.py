#!/usr/bin/env python3
"""
Test the fast_mode optimization for MemoryAgent.

Compares:
- Tool loop mode (2 API calls): Claude decides to search
- Fast mode (1 API call): Pre-fetch RAG and inject into prompt
"""

import asyncio
import os
import sys
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_dotenv():
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key.strip()] = value.strip().strip('"').strip("'")

load_dotenv()

from agents.specialists.memory import MemoryAgent


TEST_QUERIES = [
    "Who is James Evans?",
    "What grants does the lab have?",
    "When is the next deadline?",
    "List all research projects",
]


async def test_mode(fast_mode: bool, queries: list) -> list:
    """Test MemoryAgent in specified mode."""
    agent = MemoryAgent(fast_mode=fast_mode)

    results = []
    for query in queries:
        start = time.time()
        response = await agent.query(query)
        elapsed = (time.time() - start) * 1000

        results.append({
            "query": query,
            "time_ms": elapsed,
            "response_len": len(response.content),
            "response_preview": response.content[:200],
        })

    return results


async def main():
    print("=" * 90)
    print("FAST MODE vs TOOL LOOP COMPARISON")
    print("=" * 90)
    print("\nModes:")
    print("  Tool Loop: Claude decides to call search tool (2 API calls)")
    print("  Fast Mode: Pre-fetch RAG and inject into prompt (1 API call)")
    print()

    # Test tool loop mode
    print("Testing Tool Loop mode...")
    tool_results = await test_mode(fast_mode=False, queries=TEST_QUERIES)

    # Test fast mode
    print("Testing Fast mode...")
    fast_results = await test_mode(fast_mode=True, queries=TEST_QUERIES)

    # Results table
    print("\n" + "=" * 90)
    print("RESULTS")
    print("=" * 90)
    print(f"\n{'Query':<35} {'Tool Loop':<15} {'Fast Mode':<15} {'Speedup':<12}")
    print("-" * 90)

    total_tool = 0
    total_fast = 0

    for tool, fast in zip(tool_results, fast_results):
        speedup = tool["time_ms"] / fast["time_ms"] if fast["time_ms"] > 0 else 0
        print(f"{tool['query'][:33]:<35} {tool['time_ms']:>8.0f}ms     {fast['time_ms']:>8.0f}ms     {speedup:>5.2f}x")
        total_tool += tool["time_ms"]
        total_fast += fast["time_ms"]

    print("-" * 90)
    avg_speedup = total_tool / total_fast if total_fast > 0 else 0
    print(f"{'TOTAL':<35} {total_tool:>8.0f}ms     {total_fast:>8.0f}ms     {avg_speedup:>5.2f}x")
    print(f"{'AVERAGE':<35} {total_tool/4:>8.0f}ms     {total_fast/4:>8.0f}ms")

    # Time saved
    time_saved = total_tool - total_fast
    pct_saved = (time_saved / total_tool) * 100 if total_tool > 0 else 0

    print(f"\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print(f"""
Time saved per query: {time_saved/4:.0f}ms ({pct_saved:.0f}% faster)
Total time saved (4 queries): {time_saved:.0f}ms

Fast mode eliminates:
- 1st API call where Claude decides to use tool
- Tool definition overhead in API payload

Fast mode keeps:
- RAG search (~50ms)
- Single API call with context injection
""")

    # Response quality comparison
    print("=" * 90)
    print("RESPONSE QUALITY CHECK")
    print("=" * 90)

    for tool, fast in zip(tool_results, fast_results):
        print(f"\nQuery: {tool['query']}")
        print(f"Tool Loop ({tool['response_len']} chars): {tool['response_preview']}...")
        print(f"Fast Mode ({fast['response_len']} chars): {fast['response_preview']}...")


if __name__ == "__main__":
    asyncio.run(main())
