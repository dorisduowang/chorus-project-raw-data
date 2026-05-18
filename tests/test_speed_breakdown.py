#!/usr/bin/env python3
"""
Speed breakdown: Vanilla Claude vs Chorus (single agent) vs Chorus (multi-agent)
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
                    value = value.strip().strip('"').strip("'")
                    os.environ[key.strip()] = value

load_dotenv()

import anthropic
from agents import ChorusOrchestrator
from agents.legacy import ChorusAgent

TEST_QUERIES = [
    ("Who is James Evans?", "lookup"),
    ("How do I access Midway?", "technical"),
    ("What are the implications of LLMs for social science?", "explore"),
    ("Who's working on NLP in the lab?", "connect"),
]


async def test_vanilla(queries):
    """Vanilla Claude - no RAG, no tools."""
    client = anthropic.Anthropic()
    results = []
    for query, qtype in queries:
        start = time.time()
        response = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=2048,
            messages=[{"role": "user", "content": query}]
        )
        elapsed = (time.time() - start) * 1000
        results.append({"type": qtype, "time_ms": elapsed, "len": len(response.content[0].text)})
    return results


async def test_single_agent(queries):
    """Chorus single agent - RAG + tools, no specialist routing."""
    agent = ChorusAgent(
        enable_caching=False,
        enable_routing=True,
        enable_prompt_caching=True,
        rag_available=True
    )
    results = []
    for query, qtype in queries:
        start = time.time()
        response = await agent.query(query)
        elapsed = (time.time() - start) * 1000
        results.append({"type": qtype, "time_ms": elapsed, "len": len(response)})
    return results


async def test_multi_agent(queries):
    """Chorus multi-agent - RAG + tools + specialist routing."""
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
        results.append({"type": qtype, "time_ms": elapsed, "len": len(response)})
    return results


async def main():
    print("=" * 90)
    print("SPEED BREAKDOWN: Where does the time go?")
    print("=" * 90)
    print("\nSystems:")
    print("  1. Vanilla Haiku    - Raw LLM, no RAG, no tools")
    print("  2. Chorus (single)  - RAG + tools, single agent with routing")
    print("  3. Chorus (multi)   - RAG + tools + specialist subagents")
    print()

    print("Testing Vanilla Haiku...")
    vanilla = await test_vanilla(TEST_QUERIES)

    print("Testing Chorus Single-Agent...")
    single = await test_single_agent(TEST_QUERIES)

    print("Testing Chorus Multi-Agent...")
    multi = await test_multi_agent(TEST_QUERIES)

    # Results table
    print("\n" + "=" * 90)
    print("RESULTS (milliseconds)")
    print("=" * 90)
    print(f"\n{'Query Type':<12} {'Vanilla':<12} {'Single-Agent':<15} {'Multi-Agent':<15} {'RAG Overhead':<15}")
    print("-" * 90)

    for v, s, m in zip(vanilla, single, multi):
        rag_overhead = s["time_ms"] - v["time_ms"]
        multi_overhead = m["time_ms"] - s["time_ms"]
        print(f"{v['type']:<12} {v['time_ms']:>8.0f}ms   {s['time_ms']:>10.0f}ms    {m['time_ms']:>10.0f}ms    +{rag_overhead:.0f}ms (RAG)")

    # Totals
    total_v = sum(r["time_ms"] for r in vanilla)
    total_s = sum(r["time_ms"] for r in single)
    total_m = sum(r["time_ms"] for r in multi)

    print("-" * 90)
    print(f"{'TOTAL':<12} {total_v:>8.0f}ms   {total_s:>10.0f}ms    {total_m:>10.0f}ms")
    print(f"{'AVERAGE':<12} {total_v/4:>8.0f}ms   {total_s/4:>10.0f}ms    {total_m/4:>10.0f}ms")

    # Breakdown
    print("\n" + "=" * 90)
    print("TIME BREAKDOWN")
    print("=" * 90)

    avg_vanilla = total_v / 4
    avg_single = total_s / 4
    avg_multi = total_m / 4

    rag_cost = avg_single - avg_vanilla
    subagent_cost = avg_multi - avg_single

    print(f"""
Average response times:
  Vanilla Claude (baseline):     {avg_vanilla:>6.0f}ms
  + RAG/Tool overhead:           {rag_cost:>+6.0f}ms
  ─────────────────────────────────────
  Chorus Single-Agent:           {avg_single:>6.0f}ms
  + Subagent routing overhead:   {subagent_cost:>+6.0f}ms
  ─────────────────────────────────────
  Chorus Multi-Agent:            {avg_multi:>6.0f}ms

Cost breakdown:
  • Raw LLM call:        {avg_vanilla/avg_multi*100:>5.1f}% of total time
  • RAG + tool calls:    {rag_cost/avg_multi*100:>5.1f}% of total time
  • Subagent routing:    {subagent_cost/avg_multi*100:>5.1f}% of total time
""")

    # Recommendations
    print("=" * 90)
    print("RECOMMENDATIONS")
    print("=" * 90)
    print(f"""
If speed is critical:
  → Use Single-Agent Chorus ({avg_single:.0f}ms avg)
  → Saves {subagent_cost:.0f}ms per query vs Multi-Agent
  → Still has RAG for accurate answers

If quality/specialization matters:
  → Use Multi-Agent Chorus ({avg_multi:.0f}ms avg)
  → Better responses for explore/connect queries
  → Specialist prompts tuned per query type

The RAG overhead ({rag_cost:.0f}ms) is unavoidable for accurate lab-specific answers.
The subagent overhead ({subagent_cost:.0f}ms) is the cost of specialized routing.
""")


if __name__ == "__main__":
    asyncio.run(main())
