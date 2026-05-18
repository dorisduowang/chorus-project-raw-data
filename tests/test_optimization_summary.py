#!/usr/bin/env python3
"""
Summary comparison: Before vs After fast_mode optimization.
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

import anthropic
from agents.specialists.memory import MemoryAgent


async def main():
    print("=" * 80)
    print("OPTIMIZATION SUMMARY: Fast Mode for Lookup Queries")
    print("=" * 80)

    query = "Who is James Evans?"

    # Vanilla Claude (baseline)
    print("\nMeasuring baseline...")
    client = anthropic.Anthropic()
    start = time.time()
    response = client.messages.create(
        model="claude-3-5-haiku-20241022",
        max_tokens=1024,
        messages=[{"role": "user", "content": query}]
    )
    vanilla_time = (time.time() - start) * 1000

    # Tool loop mode (before optimization)
    print("Measuring tool loop mode...")
    agent_slow = MemoryAgent(fast_mode=False)
    start = time.time()
    response_slow = await agent_slow.query(query)
    tool_loop_time = (time.time() - start) * 1000

    # Fast mode (after optimization)
    print("Measuring fast mode...")
    agent_fast = MemoryAgent(fast_mode=True)
    start = time.time()
    response_fast = await agent_fast.query(query)
    fast_mode_time = (time.time() - start) * 1000

    # Results
    print("\n" + "=" * 80)
    print("RESULTS FOR: \"Who is James Evans?\"")
    print("=" * 80)

    print(f"""
┌────────────────────────────────────────────────────────────────────────────┐
│                         RESPONSE TIME COMPARISON                           │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  Vanilla Claude (no RAG):           {vanilla_time:>6.0f}ms  ████                      │
│                                                                            │
│  Chorus Tool Loop (before):         {tool_loop_time:>6.0f}ms  {"█" * min(int(tool_loop_time/200), 40):<40} │
│                                                                            │
│  Chorus Fast Mode (after):          {fast_mode_time:>6.0f}ms  {"█" * min(int(fast_mode_time/200), 40):<40} │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  Speedup from optimization:         {tool_loop_time/fast_mode_time:>5.1f}x faster                          │
│  Time saved per query:              {tool_loop_time - fast_mode_time:>5.0f}ms                              │
│  Percentage improvement:            {((tool_loop_time - fast_mode_time)/tool_loop_time)*100:>5.0f}%                              │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
""")

    # Overhead breakdown
    rag_overhead = fast_mode_time - vanilla_time
    tool_overhead = tool_loop_time - fast_mode_time

    print("WHERE THE TIME GOES:")
    print("-" * 50)
    print(f"  Raw LLM inference:      {vanilla_time:>6.0f}ms")
    print(f"  RAG search + context:   {rag_overhead:>+6.0f}ms")
    print(f"  Tool loop overhead:     {tool_overhead:>+6.0f}ms  ← ELIMINATED by fast mode")
    print(f"                         ────────")
    print(f"  Total (fast mode):      {fast_mode_time:>6.0f}ms")

    print("\n" + "=" * 80)
    print("WHAT THE OPTIMIZATION DOES")
    print("=" * 80)
    print("""
BEFORE (Tool Loop - 2 API calls):
  1. Send query + tool definitions to Claude
  2. Claude responds: "I'll search the knowledge base"
  3. Execute RAG search
  4. Send results back to Claude
  5. Claude synthesizes final answer

AFTER (Fast Mode - 1 API call):
  1. Pre-fetch RAG results (parallel with classification)
  2. Inject results into system prompt
  3. Send query to Claude with context already included
  4. Claude synthesizes final answer

This eliminates:
  - API call #1 (Claude deciding to search)
  - Tool definition overhead
  - Back-and-forth latency
""")


if __name__ == "__main__":
    asyncio.run(main())
