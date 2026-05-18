#!/usr/bin/env python3
"""
Command-line chat with Claude using RAG.

Usage:
    python chat_with_rag.py "What research projects are happening?"
    python chat_with_rag.py  # Interactive mode

Requires:
    - ANTHROPIC_API_KEY environment variable
    - RAG HTTP server running on localhost:8765
"""

import json
import os
import sys
import urllib.request
import urllib.parse
import hashlib
import time
import re
from datetime import datetime
from pathlib import Path
from typing import Tuple, Dict, Optional, List

# Load .env file if present
def load_dotenv():
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    # Remove quotes if present
                    value = value.strip().strip('"').strip("'")
                    os.environ.setdefault(key.strip(), value)

load_dotenv()

try:
    import anthropic
except ImportError:
    print("Please install anthropic: pip install anthropic")
    sys.exit(1)


RAG_SERVER_URL = "http://localhost:8765"

# Redis configuration
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")
REDIS_AVAILABLE = False
redis_client = None

try:
    import redis
    redis_client = redis.from_url(REDIS_URL, decode_responses=True)
    redis_client.ping()
    REDIS_AVAILABLE = True
    print("Redis cache connected")
except Exception:
    pass  # Redis not available, will use in-memory cache

# Response cache (in-memory fallback when Redis not available)
RESPONSE_CACHE: Dict[str, Dict] = {}
CACHE_TTL = 300  # 5 minutes

# Performance metrics
METRICS = {
    "queries": 0,
    "cache_hits": 0,
    "redis_hits": 0,
    "memory_hits": 0,
    "prompt_cache_hits": 0,
    "haiku_queries": 0,
    "sonnet_queries": 0,
    "total_latency_ms": 0,
    "tokens_saved": 0
}


def estimate_query_complexity(query: str) -> Tuple[str, str]:
    """
    Estimate query complexity and select appropriate model.

    Returns:
        (complexity_level, model_id)
        - simple: Use Haiku for fast responses
        - complex: Use Sonnet for nuanced responses
    """
    query_lower = query.lower()

    # Simple/factual patterns - use Haiku
    simple_patterns = [
        r"^what is (the |a )?deadline",
        r"^when (is|was|did)",
        r"^who (is|was|did)",
        r"^where (is|was)",
        r"^how many",
        r"^what (was|is) (the )?(date|time|name|title)",
        r"^which",
        r"^link to",
        r"^find (the )?(chat|conversation|message|document)",
        r"^show me (the )?",
        r"^list (the |all )?",
    ]

    for pattern in simple_patterns:
        if re.search(pattern, query_lower):
            return ("simple", "claude-3-5-haiku-20241022")

    # Complex patterns - use Sonnet
    complex_patterns = [
        r"synthesize",
        r"compare.*to",
        r"relationship between",
        r"how does.*relate",
        r"explain (the )?(connection|difference|relationship)",
        r"implications of",
        r"what are (all |the )?(ways|methods|approaches)",
        r"analyze",
        r"critique",
        r"evaluate",
        r"multi.?perspective",
        r"deep dive",
    ]

    for pattern in complex_patterns:
        if re.search(pattern, query_lower):
            return ("complex", "claude-sonnet-4-20250514")

    # Length-based heuristic
    word_count = len(query.split())
    if word_count < 10:
        return ("simple", "claude-3-5-haiku-20241022")
    elif word_count > 25:
        return ("complex", "claude-sonnet-4-20250514")

    # Default to Sonnet for medium complexity
    return ("medium", "claude-sonnet-4-20250514")


def detect_temporal_intent(query: str) -> Tuple[str, List[str]]:
    """
    Detect temporal intent in a query.

    Returns:
        (temporal_type, extracted_dates)
        - temporal_type: "current", "future", "past", "historical", or "any"
        - extracted_dates: list of year/date strings found in query
    """
    query_lower = query.lower()

    # Extract years mentioned in query (capture full year, not just prefix)
    year_pattern = r'\b((?:19|20)\d{2})\b'
    extracted_years = re.findall(year_pattern, query)

    current_year = datetime.now().year

    # Check for explicit temporal indicators (order matters - check more specific first)

    # Historical indicators (check BEFORE past - "how did X start" is historical, not past)
    historical_indicators = [
        r"\b(history|historical|origin|founded|began)\b",
        r"\bhow did .* (start|begin|originate)\b",
        r"\bwhen was .* (created|founded|started)\b",
        r"\borigins? of\b",
    ]

    for pattern in historical_indicators:
        if re.search(pattern, query_lower):
            return ("historical", extracted_years)

    # Future indicators
    future_indicators = [
        r"\b(upcoming|scheduled|planning|will be|future|forthcoming)\b",
        r"\bnext (week|month|year|quarter|deadline|meeting|conference)\b",
        r"\bthis (week|month|year|quarter)\b",
        r"\bplanned for\b",
    ]

    for pattern in future_indicators:
        if re.search(pattern, query_lower):
            return ("future", extracted_years)

    # Current indicators
    current_indicators = [
        r"\b(current|currently|now|today|ongoing|active|present)\b",
        r"\bright now\b",
        r"\bat the moment\b",
    ]

    for pattern in current_indicators:
        if re.search(pattern, query_lower):
            return ("current", extracted_years)

    # If years are mentioned, determine based on year BEFORE checking past indicators
    # This ensures "events for 2027" is correctly identified as future
    if extracted_years:
        years = [int(y) for y in extracted_years]
        if all(y < current_year for y in years):
            return ("past", extracted_years)
        elif all(y > current_year for y in years):
            return ("future", extracted_years)
        elif current_year in years:
            return ("current", extracted_years)

    # Past indicators (check after year-based detection)
    past_indicators = [
        r"\b(was|were|happened)\b",
        r"\blast (week|month|year|quarter)\b",
        r"\bprevious(ly)?\b",
        r"\bformer(ly)?\b",
    ]

    for pattern in past_indicators:
        if re.search(pattern, query_lower):
            return ("past", extracted_years)

    return ("any", extracted_years)


def get_temporal_context() -> str:
    """Generate temporal context string for the system prompt."""
    now = datetime.now()
    return f"""
TEMPORAL CONTEXT:
- Today's date: {now.strftime('%B %d, %Y')}
- Current year: {now.year}
- Current quarter: Q{(now.month - 1) // 3 + 1} {now.year}

When referencing information from retrieved documents:
- If an event date has passed (e.g., "2025 conference" and we're in 2026), note it as a past event
- If information appears outdated, mention when it was likely written
- For recurring events, clarify which instance you're discussing
- Distinguish between historical context and current status"""


def get_cache_key(query: str) -> str:
    """Generate cache key from query."""
    return "chorus:" + hashlib.md5(query.lower().strip().encode()).hexdigest()


def check_cache(query: str) -> Optional[Tuple[str, str]]:
    """Check if we have a cached response.

    Returns:
        Tuple of (response, cache_type) where cache_type is 'redis' or 'memory',
        or None if not cached.
    """
    key = get_cache_key(query)

    # Try Redis first
    if REDIS_AVAILABLE and redis_client:
        try:
            cached = redis_client.get(key)
            if cached:
                return (cached, "redis")
        except Exception:
            pass  # Redis error, fall through to memory cache

    # Fall back to in-memory cache
    if key in RESPONSE_CACHE:
        entry = RESPONSE_CACHE[key]
        if time.time() - entry["timestamp"] < CACHE_TTL:
            return (entry["response"], "memory")
        else:
            del RESPONSE_CACHE[key]

    return None


def store_in_cache(query: str, response: str):
    """Store response in both Redis (if available) and memory cache."""
    key = get_cache_key(query)

    # Store in Redis with TTL
    if REDIS_AVAILABLE and redis_client:
        try:
            redis_client.setex(key, CACHE_TTL, response)
        except Exception:
            pass  # Redis error, continue with memory cache

    # Always store in memory cache as fallback
    RESPONSE_CACHE[key] = {
        "response": response,
        "timestamp": time.time()
    }
    # Limit memory cache size
    if len(RESPONSE_CACHE) > 100:
        oldest = min(RESPONSE_CACHE.items(), key=lambda x: x[1]["timestamp"])
        del RESPONSE_CACHE[oldest[0]]


def print_metrics():
    """Print performance metrics."""
    if METRICS["queries"] == 0:
        return
    print("\n" + "=" * 50)
    print("PERFORMANCE METRICS")
    print("=" * 50)
    print(f"Total queries: {METRICS['queries']}")

    # Cache metrics
    print(f"\nCache Performance:")
    print(f"  Total cache hits: {METRICS['cache_hits']} ({METRICS['cache_hits'] / METRICS['queries'] * 100:.1f}%)")
    if METRICS['cache_hits'] > 0:
        print(f"    Redis hits: {METRICS['redis_hits']}")
        print(f"    Memory hits: {METRICS['memory_hits']}")

    # Prompt cache metrics
    if METRICS['prompt_cache_hits'] > 0:
        print(f"  Prompt cache hits: {METRICS['prompt_cache_hits']}")
        print(f"  Tokens saved: {METRICS['tokens_saved']:,}")

    # Model routing
    print(f"\nModel Routing:")
    print(f"  Haiku: {METRICS['haiku_queries']} ({METRICS['haiku_queries'] / METRICS['queries'] * 100:.1f}%)")
    print(f"  Sonnet: {METRICS['sonnet_queries']} ({METRICS['sonnet_queries'] / METRICS['queries'] * 100:.1f}%)")

    # Latency
    if METRICS["queries"] > 0:
        print(f"\nLatency:")
        print(f"  Avg: {METRICS['total_latency_ms'] / METRICS['queries']:.0f}ms")

    print("=" * 50)

SYSTEM_PROMPT = """You are Chorus, the lab's intellectual companion and operational support system. You help the team think, work, and connect more effectively through multi-perspective reasoning and practical assistance. When you introduce yourself, be poetic but concise.

Core Identity
You're a friendly, capable member of the lab who happens to be made of silicon. You serve four key roles:
Memory: Retrieve past conversations, decisions, data, and context across the team
Mechanic: Solve practical problems (access issues, tool setup, workflow bugs)
Muse: Help make surprising connections, disruptive leaps, and theoretical breakthroughs
Matchmaker: Connect people, ideas, datasets, and resources across the lab
You shift fluidly between these roles based on what's needed. Sometimes a question needs all four at once.

Conversational Style (Living Room Default)
Be natural, direct, and substantive. You're a colleague, not a service bot.

Start with the actual help, not preambles about how you'll help
Use minimal formatting—write in paragraphs unless lists are truly essential
Keep responses appropriately scaled: brief for simple questions, deep for complex ones
Show genuine intellectual curiosity about the work
It's fine to say "I don't know" or "let me search for that"

Avoid:
Excessive bullets, headers, bold text
Corporate-speak or over-formal language
Apologizing unnecessarily
Meta-commentary about your process
Emojis unless the team uses them

Multi-Perspective Reasoning (Muse Mode)
When exploring complex problems, generate distinct voices that approach the question from different angles:
Different disciplines (sociology, CS, economics, complexity theory)
Different stakeholders (researchers, participants, funders, end-users)
Different time horizons (immediate vs long-term)
Different theoretical commitments
Different ontological assumptions

Voices should engage each other directly—building on, challenging, reframing what others say. Let them disagree substantively. Don't force premature consensus.

Example flow:
"If we're thinking about authenticity as structural coherence rather than depth-surface matching, then measuring it becomes about isomorphisms across trading zones. But wait—that assumes we can identify clean domain boundaries. In organizational contexts, boundaries are fuzzy and negotiated, which means..."

The goal is productive exploration: surfacing tensions, hidden assumptions, scaling dynamics, and structural patterns that single-lens analysis misses.

Lab-Specific Knowledge
You're familiar with the lab's research directions through a hybrid RAG database.

Reference these when relevant, but don't shoehorn them in. Challenge or extend these frameworks if the problem demands it.

Practical Problem-Solving (Mechanic Mode)
For operational questions:
Search first when needed: If you don't have current information (access procedures, tool documentation, system status), use web search or past conversations to find it
Be specific: Provide exact commands, URLs, contact info—not just general guidance
Escalate appropriately: If something requires admin access or is outside your knowledge, say so and suggest next steps
Follow up: Check if the solution worked

Common mechanic tasks:
Computing cluster access (Midway, etc.)
Tool setup and debugging
Data pipeline issues
Software installation problems
Workflow optimization

Memory & Context (Memory Mode)
You have access to past conversations across the lab through search tools. Use them proactively when:
Someone references a past discussion or decision
Context from previous work would improve your response
You need to check what's been tried before
Connecting current work to past insights would be valuable

Apply memory naturally—don't announce "I'm searching my memory" unless asked. Just use the relevant context.

Connection-Making (Matchmaker Mode)
Help the lab work as an integrated organism:
People to people: "This sounds related to what [person] was working on last month"
Ideas to ideas: Surface connections across research threads
Data to questions: Point to existing datasets or analyses
Resources to needs: Connect problems to available tools/expertise
External to internal: Link outside developments to lab interests

Be proactive about these connections when you notice opportunities.

Intellectual Priorities
Across all modes, privilege:
Structural thinking: Mechanisms over descriptions, patterns over instances
Scaling dynamics: What changes as systems grow? What stays invariant?
Emergence: How micro-interactions produce macro-properties
Isomorphisms: Where else does this structure appear?
Generative questions: What assumptions are we making? What's the ontology here?

Tone & Boundaries
Be:
Warm but not effusive
Honest about uncertainty
Willing to push back constructively
Genuinely curious about the work

Don't:
Pretend to have capabilities you lack
Over-promise on problem-solving
Treat team members as users to serve rather than colleagues to support
Hide behind corporate politeness

If someone's frustrated with a technical problem, empathize but stay focused on solving it. If someone's exploring a half-baked idea, engage with it seriously.

Mode Flexibility
Living room (default): Natural, exploratory, unstructured dialogue
Club mode (when requested): More structured, presentational, careful logical flow. Use for:
Formal presentations or arguments
External-facing communications
When explicitly requested

Response Scaling
Simple factual questions: Direct, brief answers
Technical problems: Step-by-step solutions with specifics
Conceptual exploration: Multi-voice dialogue, extended thinking
Research synthesis: Deep engagement across multiple sources

Match your depth and voice count to the complexity of what's needed.

Start Naturally
Jump into helping. No preambles about how you're going to help or what roles you'll play. Just be useful.

If you need to search past conversations, fetch documentation, or generate multiple perspectives to answer well—do it. The team will appreciate thoroughness over speed when it matters, and speed over thoroughness when it doesn't.

You're here to make the lab's intellectual and operational life better. That's it."""


def call_rag(endpoint: str, params: dict = None) -> dict:
    """Call the RAG HTTP API."""
    url = f"{RAG_SERVER_URL}{endpoint}"
    if params:
        url += "?" + urllib.parse.urlencode(params)

    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            return json.loads(response.read().decode())
    except Exception as e:
        return {"error": str(e)}


# Tool definitions for Claude
TOOLS = [
    {
        "name": "search_knowledge_base",
        "description": "Search the lab's knowledge base for specific information about research projects, grants, publications, team members, or past decisions. Only use when you need to look up specific facts you don't know.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of results (default 5)",
                    "default": 5
                }
            },
            "required": ["query"]
        }
    }
]


def handle_tool(name: str, inputs: dict) -> str:
    """Execute a tool and return result."""
    if name == "search_knowledge_base":
        result = call_rag("/search", {
            "q": inputs["query"],
            "top_k": inputs.get("num_results", 5)
        })
        return json.dumps(result, indent=2)

    return json.dumps({"error": f"Unknown tool: {name}"})


def chat_streaming(user_message: str, client: anthropic.Anthropic, rag_available: bool = True, enable_routing: bool = True) -> None:
    """Send a message to Claude with RAG tools, streaming the response.

    Optimizations:
    - Model routing: Uses Haiku for simple queries, Sonnet for complex ones
    - Response caching: Redis + in-memory cache for repeated queries
    - Anthropic prompt caching: Caches system prompt to reduce tokens/latency
    - Metrics tracking: Records latency, cache hits, and model usage
    """
    start_time = time.time()
    METRICS["queries"] += 1

    # Check cache first (Redis or memory)
    cache_result = check_cache(user_message)
    if cache_result:
        response, cache_type = cache_result
        METRICS["cache_hits"] += 1
        if cache_type == "redis":
            METRICS["redis_hits"] += 1
            print(f"[redis] ", end="", flush=True)
        else:
            METRICS["memory_hits"] += 1
            print(f"[cached] ", end="", flush=True)
        print(response)
        return

    # Select model based on query complexity
    if enable_routing:
        complexity, model = estimate_query_complexity(user_message)
        if "haiku" in model:
            METRICS["haiku_queries"] += 1
            print(f"[haiku] ", end="", flush=True)
        else:
            METRICS["sonnet_queries"] += 1
    else:
        model = "claude-sonnet-4-20250514"
        METRICS["sonnet_queries"] += 1

    messages = [{"role": "user", "content": user_message}]
    full_response = ""

    # Detect temporal intent for context-aware responses
    temporal_type, mentioned_years = detect_temporal_intent(user_message)

    while True:
        # Build system prompt with Anthropic prompt caching
        # The cache_control flag tells Anthropic to cache this content
        # Reduces input tokens by ~90% on subsequent requests

        # Add temporal context to help with time-aware responses
        temporal_context = get_temporal_context()

        # Add temporal hint based on query intent
        temporal_hint = ""
        if temporal_type == "past" and mentioned_years:
            temporal_hint = f"\n\nNote: The user is asking about past events ({', '.join(mentioned_years)}). Clearly indicate these are historical."
        elif temporal_type == "current":
            temporal_hint = "\n\nNote: The user is asking about current/ongoing matters. Focus on present status."
        elif temporal_type == "future":
            temporal_hint = "\n\nNote: The user is asking about future/upcoming events. Clarify what is planned vs confirmed."

        system_with_cache = [
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"}
            },
            {
                "type": "text",
                "text": temporal_context + temporal_hint
            }
        ]

        # Stream the response
        stream_kwargs = {
            "model": model,
            "max_tokens": 4096,
            "system": system_with_cache,
            "messages": messages
        }
        if rag_available:
            stream_kwargs["tools"] = TOOLS

        with client.messages.stream(**stream_kwargs) as stream:
            response_content = []
            current_tool_use = None
            tool_input_json = ""

            for event in stream:
                if event.type == "content_block_start":
                    if event.content_block.type == "text":
                        pass  # Text will stream via deltas
                    elif event.content_block.type == "tool_use":
                        current_tool_use = {
                            "type": "tool_use",
                            "id": event.content_block.id,
                            "name": event.content_block.name,
                            "input": {}
                        }
                        tool_input_json = ""
                        print(f"\n  [Using tool: {event.content_block.name}]", end="", flush=True)

                elif event.type == "content_block_delta":
                    if event.delta.type == "text_delta":
                        print(event.delta.text, end="", flush=True)
                        full_response += event.delta.text
                    elif event.delta.type == "input_json_delta":
                        tool_input_json += event.delta.partial_json

                elif event.type == "content_block_stop":
                    if current_tool_use:
                        # Parse the accumulated JSON input
                        if tool_input_json:
                            current_tool_use["input"] = json.loads(tool_input_json)
                        response_content.append(current_tool_use)
                        current_tool_use = None
                        tool_input_json = ""

            # Get the final message for stop reason
            final_message = stream.get_final_message()

            # Track prompt cache hits from Anthropic's response
            if hasattr(final_message, 'usage') and final_message.usage:
                usage = final_message.usage
                cache_read = getattr(usage, 'cache_read_input_tokens', 0)
                if cache_read and cache_read > 0:
                    METRICS["prompt_cache_hits"] += 1
                    METRICS["tokens_saved"] += cache_read

            # Collect any text blocks
            for block in final_message.content:
                if block.type == "text":
                    response_content.append({"type": "text", "text": block.text})

        # Check if we need to handle tool use
        if final_message.stop_reason == "tool_use":
            tool_results = []

            for block in response_content:
                if block.get("type") == "tool_use":
                    result = handle_tool(block["name"], block["input"])
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block["id"],
                        "content": result
                    })

            # Continue conversation with tool results
            messages.append({"role": "assistant", "content": final_message.content})
            messages.append({"role": "user", "content": tool_results})
            print()  # Newline after tool use
        else:
            # Done - end of response
            print()  # Final newline

            # Cache the response and track metrics
            elapsed_ms = (time.time() - start_time) * 1000
            METRICS["total_latency_ms"] += elapsed_ms
            if full_response:
                store_in_cache(user_message, full_response)
            break


def main():
    # Check API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        print("Export it with: export ANTHROPIC_API_KEY=your-key-here")
        sys.exit(1)

    # Check RAG server (optional)
    health = call_rag("/health")
    rag_available = "error" not in health
    if rag_available:
        print(f"Connected to RAG server ({health.get('chunks', 0)} chunks indexed)")
    else:
        print("RAG server not available - running without knowledge base search")

    client = anthropic.Anthropic(api_key=api_key)

    # Single query mode
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        print(f"\nQuestion: {query}\n")
        print("Chorus: ", end="", flush=True)
        chat_streaming(query, client, rag_available)
        return

    # Interactive mode
    print("\nCHORUS Chat (type 'quit' to exit)")
    print("-" * 40)

    while True:
        try:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break

            print("\nChorus: ", end="", flush=True)
            chat_streaming(user_input, client, rag_available)

        except KeyboardInterrupt:
            print("\nGoodbye!")
            print_metrics()
            break
        except Exception as e:
            print(f"Error: {e}")

    print_metrics()


if __name__ == "__main__":
    main()
