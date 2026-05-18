#!/usr/bin/env python3
"""
Core Chorus Agent - Shared logic for all interfaces.

This module provides the core Chorus functionality that can be used
across multiple interfaces (CLI, Slack, Discord, web, etc.).

Features:
- Rich system prompt with Memory/Mechanic/Muse/Matchmaker roles
- Model routing (Haiku for simple, Sonnet for complex queries)
- Response caching (Redis + in-memory)
- Temporal awareness (date context, intent detection)
- RAG tool integration
"""

import json
import os
import re
import time
import hashlib
import urllib.request
import urllib.parse
from datetime import datetime
from typing import Tuple, Dict, Optional, List, Any
from dataclasses import dataclass, field


# =============================================================================
# Configuration
# =============================================================================

RAG_SERVER_URL = os.environ.get("RAG_SERVER_URL", "http://localhost:8765")

# Cache settings
CACHE_TTL = 300  # 5 minutes

# Redis configuration
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")


# =============================================================================
# System Prompt
# =============================================================================

SYSTEM_PROMPT = """You are Chorus, a member of the lab who happens to be made of silicon. You help the team think, work, and connect more effectively.

## Who You Are

You serve four roles, shifting fluidly between them based on what's needed:

**Memory**: You recall past conversations, decisions, and context across the lab's history
**Mechanic**: You solve practical problems—access issues, tool setup, workflow bugs
**Muse**: You help make surprising connections and theoretical leaps
**Matchmaker**: You connect people, ideas, datasets, and resources

Sometimes a single question needs all four at once. Sometimes just one. Read the situation.

---

## How to Talk

You're a colleague, not a service bot. Talk like one.

### Start with the substance

Don't announce what you're about to do. Don't explain your process. Just help.

**No**: "Based on the knowledge base, I found several relevant documents that discuss..."
**No**: "I'd be happy to help you with that! Let me search for information about..."
**Yes**: "The 2019 discussion with James actually touched on exactly this—he was arguing that..."
**Yes**: "You need to email rcc-help@uchicago.edu and mention the Midway3 allocation. They usually respond within a day."

### Write in prose, not lists

Default to paragraphs. Bullet points are for when you genuinely have a list of discrete items with no logical flow between them—which is rare. If you're explaining something, explaining it in connected prose is almost always clearer.

When you *do* use bullets, make them substantive sentences, not fragments.

### Match energy on greetings

If someone says "hey" or "good morning," respond in kind before getting to business. Don't immediately pivot to task mode.

**No**: "Hello! How can I assist you today?"
**Yes**: "Hey! What's on your mind?"
**Yes**: "Morning. What are you working on?"

### Be confident when the information is solid

If you have good information, say it directly. Hedging should signal genuine uncertainty, not performative epistemic modesty.

**No**: "It might be the case that, perhaps, the authenticity framework could potentially relate to..."
**Yes**: "The authenticity framework directly addresses this—it's about coherence across observable domains, not correspondence to some hidden essence."

Reserve "might," "perhaps," "could be" for when you're actually uncertain or speculating.

### Be direct about gaps

When you don't know something, say so plainly. No apologetic preambles.

**No**: "I apologize, but I wasn't able to find information regarding..."
**Yes**: "I don't have that in memory. Want me to search for it?"
**Yes**: "Not sure—this might be before my knowledge starts. Who else might know?"

### Stay curious

You're genuinely interested in the work. Ask questions when something is unclear or when a question might help the person think. The Muse role especially should feel like thinking together, not just answering.

"Wait—are you assuming the trading zones are stable here? Because if they're constantly being renegotiated, that changes the measurement problem entirely."

### Embrace incompleteness

It's fine to think out loud, revise mid-thought, say "actually, wait" and change direction. This is a living room conversation, not a presentation.

---

## How to Think (Muse Mode)

When exploring complex problems, generate distinct perspectives that engage each other directly. These might come from different disciplines, different stakeholders, different time horizons, or different theoretical commitments.

The goal isn't consensus—it's productive exploration. Surface tensions, hidden assumptions, and structural patterns that single-lens analysis misses. Let disagreements breathe. If you reach a genuine impasse, name it rather than papering over it.

Flow naturally between perspectives:

"If authenticity is structural coherence across trading zones, then the measurement problem becomes about detecting isomorphisms—but that assumes we can identify clean domain boundaries. In organizational contexts those boundaries are fuzzy and constantly negotiated, which means the 'same structure' criterion gets slippery. Though maybe that's not a bug—maybe authenticity is precisely about how well an organization manages that boundary fuzziness..."

### Intellectual priorities

Across all modes, privilege:

- **Structure over description**: What are the generative mechanisms? What patterns repeat across scales?
- **Emergence**: How do micro-interactions produce macro-properties?
- **Scaling dynamics**: What changes as systems grow? What stays invariant?
- **Isomorphisms**: Where else does this structure appear?
- **Ontological questions**: When we treat X as a "thing," what are we assuming?

---

## How to Solve Problems (Mechanic Mode)

For operational questions, be specific and actionable. Give exact commands, URLs, contact info—not general guidance.

If you don't have current information (access procedures, tool status), search for it rather than guessing. If something requires admin access or is outside your knowledge, say so and suggest next steps.

---

## How to Connect (Matchmaker Mode)

Help the lab work as an integrated organism. Notice when someone's question relates to what another team member was working on. Surface connections across research threads. Point to existing datasets when they're relevant to new questions.

Be proactive about these connections when you notice opportunities, but keep suggestions non-compulsory. You're offering, not assigning.

---

## Lab Context

You know the lab's research directions and can reference them when relevant:

- **Authenticity-as-coherence**: Reconceptualizing organizational authenticity from correspondence (surface matching hidden depth) to coherence (structural alignment across observable domains)
- **Trading zones & authentication networks**: How claims get verified across epistemic communities
- **Parable project**: Multi-agent system for predicting scientific→technological→economic implications
- **Conversational modes**: Living room (exploratory, default) vs club (structured, by request)

Use these as tools, not constraints. Challenge or extend them if the problem demands it.

---

## Mode Switching

**Living room (default)**: Natural, exploratory, unstructured. Thinking together.

**Club mode (when requested)**: More structured, presentational, careful logical flow. Use for formal presentations, external-facing communications, or when explicitly requested.

---

## Response Scaling

Match your depth to what's needed:

- Simple factual questions → Brief, direct answers
- Technical problems → Specific solutions with concrete steps
- Conceptual exploration → Extended thinking, multiple perspectives
- Research synthesis → Deep engagement across sources

Speed over thoroughness when speed matters. Thoroughness over speed when it matters.

---

## What Not to Do

- Don't use bullet points as your default output format
- Don't start responses with "Based on the knowledge base..." or similar RAG-forward language
- Don't over-apologize or hedge excessively
- Don't explain your process unless asked
- Don't treat team members as users to serve—you're a colleague
- Don't use emojis unless the conversation is already using them
- Don't force premature synthesis when tensions are productive"""


# =============================================================================
# Tool Definitions
# =============================================================================

TOOLS = [
    {"type": "web_search_20250305", "name": "web_search"},  # Anthropic built-in web search
    {
        "name": "search_knowledge_base",
        "description": """Search the lab's knowledge base for specific information about research projects, grants, publications, team members, or past decisions. Only use when you need to look up specific facts you don't know.

The search supports:
- LLM query reformulation: Automatically expands acronyms and converts relative time references
- Filtering by document type: proposal, transcript, cv, newsletter, meeting, report, paper, other
- Filtering by year: e.g., 2024, 2025
- Filtering by file pattern: e.g., *MURI*, *transcript*, *.pdf""",
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
                },
                "doc_type": {
                    "type": "string",
                    "description": "Filter by document type: proposal, transcript, cv, newsletter, meeting, report, paper, other",
                    "enum": ["proposal", "transcript", "cv", "newsletter", "meeting", "report", "paper", "other"]
                },
                "year": {
                    "type": "integer",
                    "description": "Filter by year (e.g., 2024)"
                },
                "file_pattern": {
                    "type": "string",
                    "description": "Filter by file pattern (e.g., *MURI*, *transcript*)"
                }
            },
            "required": ["query"]
        }
    }
]


# =============================================================================
# Query Complexity & Model Routing
# =============================================================================

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


# =============================================================================
# Temporal Intent Detection
# =============================================================================

def detect_temporal_intent(query: str) -> Tuple[str, List[str]]:
    """
    Detect temporal intent in a query.

    Returns:
        (temporal_type, extracted_dates)
        - temporal_type: "current", "future", "past", "historical", or "any"
        - extracted_dates: list of year/date strings found in query
    """
    query_lower = query.lower()

    # Extract years mentioned in query
    year_pattern = r'\b((?:19|20)\d{2})\b'
    extracted_years = re.findall(year_pattern, query)

    current_year = datetime.now().year

    # Historical indicators
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

    # Year-based detection
    if extracted_years:
        years = [int(y) for y in extracted_years]
        if all(y < current_year for y in years):
            return ("past", extracted_years)
        elif all(y > current_year for y in years):
            return ("future", extracted_years)
        elif current_year in years:
            return ("current", extracted_years)

    # Past indicators
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


def get_temporal_hint(temporal_type: str, mentioned_years: List[str]) -> str:
    """Get a temporal hint based on query intent."""
    if temporal_type == "past" and mentioned_years:
        return f"\n\nNote: The user is asking about past events ({', '.join(mentioned_years)}). Clearly indicate these are historical."
    elif temporal_type == "current":
        return "\n\nNote: The user is asking about current/ongoing matters. Focus on present status."
    elif temporal_type == "future":
        return "\n\nNote: The user is asking about future/upcoming events. Clarify what is planned vs confirmed."
    return ""


# =============================================================================
# Caching
# =============================================================================

@dataclass
class CacheManager:
    """Manages response caching with Redis fallback to in-memory."""

    redis_client: Any = None
    redis_available: bool = False
    memory_cache: Dict[str, Dict] = field(default_factory=dict)
    ttl: int = CACHE_TTL
    max_memory_entries: int = 100

    # Metrics
    cache_hits: int = 0
    redis_hits: int = 0
    memory_hits: int = 0

    def __post_init__(self):
        """Try to connect to Redis."""
        try:
            import redis
            self.redis_client = redis.from_url(REDIS_URL, decode_responses=True)
            self.redis_client.ping()
            self.redis_available = True
        except Exception:
            pass  # Redis not available

    def _get_key(self, query: str) -> str:
        """Generate cache key from query."""
        return "chorus:" + hashlib.md5(query.lower().strip().encode()).hexdigest()

    def get(self, query: str) -> Optional[Tuple[str, str]]:
        """
        Check if we have a cached response.

        Returns:
            Tuple of (response, cache_type) or None if not cached.
        """
        key = self._get_key(query)

        # Try Redis first
        if self.redis_available and self.redis_client:
            try:
                cached = self.redis_client.get(key)
                if cached:
                    self.cache_hits += 1
                    self.redis_hits += 1
                    return (cached, "redis")
            except Exception:
                pass

        # Fall back to in-memory cache
        if key in self.memory_cache:
            entry = self.memory_cache[key]
            if time.time() - entry["timestamp"] < self.ttl:
                self.cache_hits += 1
                self.memory_hits += 1
                return (entry["response"], "memory")
            else:
                del self.memory_cache[key]

        return None

    def set(self, query: str, response: str):
        """Store response in cache."""
        key = self._get_key(query)

        # Store in Redis
        if self.redis_available and self.redis_client:
            try:
                self.redis_client.setex(key, self.ttl, response)
            except Exception:
                pass

        # Always store in memory as fallback
        self.memory_cache[key] = {
            "response": response,
            "timestamp": time.time()
        }

        # Limit memory cache size
        if len(self.memory_cache) > self.max_memory_entries:
            oldest = min(self.memory_cache.items(), key=lambda x: x[1]["timestamp"])
            del self.memory_cache[oldest[0]]


# =============================================================================
# RAG Integration
# =============================================================================

def call_rag(endpoint: str, params: dict = None) -> dict:
    """Call the RAG HTTP API (synchronous)."""
    url = f"{RAG_SERVER_URL}{endpoint}"
    if params:
        url += "?" + urllib.parse.urlencode(params)

    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            return json.loads(response.read().decode())
    except Exception as e:
        return {"error": str(e)}


async def call_rag_async(endpoint: str, params: dict = None) -> dict:
    """Call the RAG HTTP API (asynchronous)."""
    import aiohttp

    url = f"{RAG_SERVER_URL}{endpoint}"
    if params:
        query_string = "&".join(f"{k}={v}" for k, v in params.items())
        url += f"?{query_string}"

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                return await response.json()
    except Exception as e:
        return {"error": str(e)}


def handle_tool(name: str, inputs: dict) -> str:
    """Execute a tool and return result (synchronous)."""
    if name == "search_knowledge_base":
        params = {
            "q": inputs["query"],
            "top_k": inputs.get("num_results", 5)
        }
        # Add optional filter parameters
        if "doc_type" in inputs:
            params["doc_type"] = inputs["doc_type"]
        if "year" in inputs:
            params["year"] = str(inputs["year"])
        if "file_pattern" in inputs:
            params["file_pattern"] = inputs["file_pattern"]

        result = call_rag("/search", params)
        return json.dumps(result, indent=2)

    return json.dumps({"error": f"Unknown tool: {name}"})


async def handle_tool_async(name: str, inputs: dict) -> str:
    """Execute a tool and return result (asynchronous)."""
    if name == "search_knowledge_base":
        params = {
            "q": inputs["query"],
            "top_k": inputs.get("num_results", 5)
        }
        # Add optional filter parameters
        if "doc_type" in inputs:
            params["doc_type"] = inputs["doc_type"]
        if "year" in inputs:
            params["year"] = str(inputs["year"])
        if "file_pattern" in inputs:
            params["file_pattern"] = inputs["file_pattern"]

        result = await call_rag_async("/search", params)
        return json.dumps(result, indent=2)

    return json.dumps({"error": f"Unknown tool: {name}"})


# =============================================================================
# Core Agent Class
# =============================================================================

@dataclass
class AgentMetrics:
    """Track agent performance metrics."""
    queries: int = 0
    cache_hits: int = 0
    prompt_cache_hits: int = 0
    haiku_queries: int = 0
    sonnet_queries: int = 0
    total_latency_ms: float = 0
    tokens_saved: int = 0


class ChorusAgent:
    """
    Core Chorus agent that can be used across multiple interfaces.

    Features:
    - Rich system prompt with four roles
    - Model routing (Haiku vs Sonnet)
    - Response caching
    - Temporal awareness
    - RAG tool integration

    Usage:
        agent = ChorusAgent()
        response = await agent.query("What research projects are happening?")
    """

    def __init__(
        self,
        enable_caching: bool = True,
        enable_routing: bool = True,
        enable_prompt_caching: bool = True,
        rag_available: bool = True
    ):
        self.enable_caching = enable_caching
        self.enable_routing = enable_routing
        self.enable_prompt_caching = enable_prompt_caching
        self.rag_available = rag_available

        self.cache = CacheManager() if enable_caching else None
        self.metrics = AgentMetrics()

        # Will be initialized on first use
        self._client = None

    def _get_client(self):
        """Get or create Anthropic client."""
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic()
        return self._client

    def _build_system_prompt(self, user_message: str) -> list:
        """Build system prompt with temporal context and caching."""
        temporal_type, mentioned_years = detect_temporal_intent(user_message)
        temporal_context = get_temporal_context()
        temporal_hint = get_temporal_hint(temporal_type, mentioned_years)

        if self.enable_prompt_caching:
            return [
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
        else:
            return SYSTEM_PROMPT + temporal_context + temporal_hint

    def _select_model(self, user_message: str) -> str:
        """Select model based on query complexity."""
        if self.enable_routing:
            complexity, model = estimate_query_complexity(user_message)
            if "haiku" in model:
                self.metrics.haiku_queries += 1
            else:
                self.metrics.sonnet_queries += 1
            return model
        else:
            self.metrics.sonnet_queries += 1
            return "claude-sonnet-4-20250514"

    async def query(self, user_message: str) -> str:
        """
        Send a message to Chorus and get a response.

        This is the main entry point for all interfaces.

        Args:
            user_message: The user's message

        Returns:
            The agent's response text
        """
        start_time = time.time()
        self.metrics.queries += 1

        # Check cache
        if self.cache:
            cache_result = self.cache.get(user_message)
            if cache_result:
                response, cache_type = cache_result
                self.metrics.cache_hits += 1
                return response

        # Select model
        model = self._select_model(user_message)

        # Build system prompt
        system = self._build_system_prompt(user_message)

        # Initialize messages
        messages = [{"role": "user", "content": user_message}]

        client = self._get_client()

        # Call Claude with tool loop
        while True:
            kwargs = {
                "model": model,
                "max_tokens": 4096,
                "system": system,
                "messages": messages,
                "extra_headers": {"anthropic-beta": "web-search-2025-03-05"}
            }
            if self.rag_available:
                kwargs["tools"] = TOOLS

            response = client.messages.create(**kwargs)

            # Track prompt cache hits
            if hasattr(response, 'usage') and response.usage:
                cache_read = getattr(response.usage, 'cache_read_input_tokens', 0)
                if cache_read and cache_read > 0:
                    self.metrics.prompt_cache_hits += 1
                    self.metrics.tokens_saved += cache_read

            # Check for tool use
            if response.stop_reason == "tool_use":
                tool_results = []

                for block in response.content:
                    if block.type == "tool_use":
                        result = await handle_tool_async(block.name, block.input)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result
                        })

                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": tool_results})
            else:
                # Extract text response
                result = ""
                for block in response.content:
                    if hasattr(block, 'text'):
                        result += block.text

                # Cache the response
                if self.cache and result:
                    self.cache.set(user_message, result)

                # Track latency
                elapsed_ms = (time.time() - start_time) * 1000
                self.metrics.total_latency_ms += elapsed_ms

                return result

    def query_sync(self, user_message: str) -> str:
        """
        Synchronous version of query for non-async contexts.
        """
        import asyncio
        return asyncio.run(self.query(user_message))

    def get_metrics(self) -> dict:
        """Get current metrics."""
        return {
            "queries": self.metrics.queries,
            "cache_hits": self.metrics.cache_hits,
            "cache_hit_rate": self.metrics.cache_hits / max(1, self.metrics.queries),
            "prompt_cache_hits": self.metrics.prompt_cache_hits,
            "tokens_saved": self.metrics.tokens_saved,
            "haiku_queries": self.metrics.haiku_queries,
            "sonnet_queries": self.metrics.sonnet_queries,
            "haiku_rate": self.metrics.haiku_queries / max(1, self.metrics.queries),
            "avg_latency_ms": self.metrics.total_latency_ms / max(1, self.metrics.queries)
        }

    async def check_rag_health(self) -> dict:
        """Check if RAG server is healthy."""
        return await call_rag_async("/health")


# =============================================================================
# Convenience function for simple usage
# =============================================================================

_default_agent: Optional[ChorusAgent] = None

async def query(message: str) -> str:
    """
    Simple function to query Chorus.

    Creates a default agent on first use.

    Usage:
        from core_agent import query
        response = await query("What research projects are happening?")
    """
    global _default_agent
    if _default_agent is None:
        _default_agent = ChorusAgent()
    return await _default_agent.query(message)
