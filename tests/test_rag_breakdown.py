#!/usr/bin/env python3
"""
Detailed timing breakdown of where RAG query time goes.
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

import aiohttp
import anthropic


async def measure_rag_only():
    """Just the RAG HTTP call."""
    async with aiohttp.ClientSession() as session:
        start = time.time()
        async with session.get('http://localhost:8765/search?q=James+Evans&top_k=5') as r:
            result = await r.json()
        return (time.time() - start) * 1000


def measure_claude_only(query: str, model: str):
    """Just a Claude API call, no tools."""
    client = anthropic.Anthropic()
    start = time.time()
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": query}]
    )
    return (time.time() - start) * 1000, len(response.content[0].text)


def measure_claude_with_tool_definition(query: str, model: str):
    """Claude API call with tool definitions (but may not call tool)."""
    client = anthropic.Anthropic()

    tools = [{
        "name": "search_knowledge_base",
        "description": "Search the lab's knowledge base",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"}
            },
            "required": ["query"]
        }
    }]

    start = time.time()
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": query}],
        tools=tools
    )
    elapsed = (time.time() - start) * 1000

    # Check if tool was called
    tool_called = response.stop_reason == "tool_use"

    return elapsed, tool_called


async def measure_full_tool_loop(query: str, model: str):
    """Full tool use loop: Claude -> tool call -> Claude."""
    client = anthropic.Anthropic()

    tools = [{
        "name": "search_knowledge_base",
        "description": "Search the lab's knowledge base for information.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"}
            },
            "required": ["query"]
        }
    }]

    system = "You are a helpful assistant. Use the search tool to find information."
    messages = [{"role": "user", "content": query}]

    timings = {
        "first_api_call": 0,
        "tool_execution": 0,
        "second_api_call": 0,
        "total": 0
    }

    total_start = time.time()

    # First API call
    start = time.time()
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=system,
        messages=messages,
        tools=tools
    )
    timings["first_api_call"] = (time.time() - start) * 1000

    if response.stop_reason == "tool_use":
        # Execute tool
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                start = time.time()
                async with aiohttp.ClientSession() as session:
                    async with session.get(f'http://localhost:8765/search?q={block.input["query"]}&top_k=5') as r:
                        result = await r.json()
                timings["tool_execution"] = (time.time() - start) * 1000

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": str(result)
                })

        # Second API call with tool results
        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})

        start = time.time()
        response = client.messages.create(
            model=model,
            max_tokens=1024,
            system=system,
            messages=messages,
            tools=tools
        )
        timings["second_api_call"] = (time.time() - start) * 1000

    timings["total"] = (time.time() - total_start) * 1000

    return timings


async def main():
    print("=" * 80)
    print("RAG LATENCY BREAKDOWN: Where does the time actually go?")
    print("=" * 80)

    # Test 1: Raw RAG
    print("\n1. RAG HTTP Server (just the search):")
    print("-" * 50)
    times = []
    for _ in range(3):
        t = await measure_rag_only()
        times.append(t)
    print(f"   Average: {sum(times)/len(times):.0f}ms")

    # Test 2: Claude without tools
    print("\n2. Claude API (no tools):")
    print("-" * 50)
    for model in ["claude-3-5-haiku-20241022", "claude-sonnet-4-20250514"]:
        t, length = measure_claude_only("Who is James Evans?", model)
        model_short = "Haiku" if "haiku" in model else "Sonnet"
        print(f"   {model_short}: {t:.0f}ms ({length} chars)")

    # Test 3: Claude with tool definitions
    print("\n3. Claude API (with tool definitions, no call):")
    print("-" * 50)
    for model in ["claude-3-5-haiku-20241022", "claude-sonnet-4-20250514"]:
        t, called = measure_claude_with_tool_definition("What is 2+2?", model)
        model_short = "Haiku" if "haiku" in model else "Sonnet"
        print(f"   {model_short}: {t:.0f}ms (tool called: {called})")

    # Test 4: Full tool loop
    print("\n4. Full Tool Loop (Claude -> RAG -> Claude):")
    print("-" * 50)

    query = "Who is James Evans?"

    for model in ["claude-3-5-haiku-20241022", "claude-sonnet-4-20250514"]:
        timings = await measure_full_tool_loop(query, model)
        model_short = "Haiku" if "haiku" in model else "Sonnet"
        print(f"\n   {model_short}:")
        print(f"      1st API call (decide to use tool): {timings['first_api_call']:>6.0f}ms")
        print(f"      Tool execution (RAG search):       {timings['tool_execution']:>6.0f}ms")
        print(f"      2nd API call (synthesize):         {timings['second_api_call']:>6.0f}ms")
        print(f"      ─────────────────────────────────────────")
        print(f"      Total:                             {timings['total']:>6.0f}ms")

    # Summary
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)

    haiku_timings = await measure_full_tool_loop(query, "claude-3-5-haiku-20241022")

    print(f"""
The ~10-15 second "RAG overhead" we measured earlier is actually:

  1. First Claude API call:     {haiku_timings['first_api_call']:>5.0f}ms  (Claude decides to use tool)
  2. RAG search:                {haiku_timings['tool_execution']:>5.0f}ms  (Actually very fast!)
  3. Second Claude API call:    {haiku_timings['second_api_call']:>5.0f}ms  (Claude synthesizes answer)
                               ─────────
  Total:                       {haiku_timings['total']:>5.0f}ms

The bottleneck is NOT the RAG search - it's the TWO Claude API calls!

Each API call has:
  - Network latency to Anthropic servers
  - Model inference time
  - Response streaming time

OPTIMIZATION OPTIONS:
""")

    print("""
┌─────────────────────────────────────────────────────────────────────────────┐
│ OPTION                          │ SAVINGS      │ TRADEOFF                   │
├─────────────────────────────────┼──────────────┼────────────────────────────┤
│ 1. Pre-fetch RAG context        │ ~20-50ms     │ Minor - RAG already fast   │
│                                 │              │                            │
│ 2. Skip tool loop for lookups   │ ~3-5 sec     │ Must inject context into   │
│    (inject RAG into prompt)     │              │ system prompt              │
│                                 │              │                            │
│ 3. Use streaming responses      │ Perceived    │ First token arrives faster │
│                                 │ improvement  │ but total time same        │
│                                 │              │                            │
│ 4. Cache common queries         │ 100%         │ Only helps repeated queries│
│                                 │              │                            │
│ 5. Parallel RAG + classification│ ~500ms       │ Speculative fetch          │
│                                 │              │                            │
│ 6. Use smaller context (top_k=3)│ ~100ms       │ May miss relevant info     │
│                                 │              │                            │
│ 7. Batch API calls (if multiple │ 30-50%       │ Requires architecture      │
│    queries in flight)           │              │ changes                    │
└─────────────────────────────────┴──────────────┴────────────────────────────┘
""")


if __name__ == "__main__":
    asyncio.run(main())
