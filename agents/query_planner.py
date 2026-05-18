"""
QueryPlanner - LLM-driven query planning for sophisticated fallback strategies.

This module provides intelligent query planning using Claude Haiku to:
1. Analyze query intent and complexity
2. Plan multi-step execution strategies
3. Generate fallback strategies when initial approaches fail
4. Support query reformulation, decomposition, and expansion
"""

import json
import logging
import time
import hashlib
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum

import anthropic


# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_PLANNING_MODEL = "claude-3-5-haiku-20241022"
MAX_PLANNING_TOKENS = 1024
DEFAULT_MAX_RETRIES = 3
SIMPLE_QUERY_THRESHOLD = 0.85  # Confidence threshold for fast path


# =============================================================================
# Data Structures
# =============================================================================

class QueryAction(str, Enum):
    """Types of actions in a query plan."""
    SEARCH_REGISTRY = "search_registry"
    SEARCH_DOCS = "search_docs"
    WEB_SEARCH = "web_search"
    COMBINE = "combine"
    REFORMULATE = "reformulate"
    DECOMPOSE = "decompose"
    DIRECT_ANSWER = "direct_answer"


class FallbackStrategy(str, Enum):
    """Types of fallback strategies."""
    REFORMULATION = "reformulation"      # Rephrase query for better matching
    DECOMPOSITION = "decomposition"      # Break compound query into parts
    EXPANSION = "expansion"              # Add synonyms or related terms
    NARROWING = "narrowing"              # Focus on most specific aspect
    WEB_FALLBACK = "web_fallback"        # Search web when local sources fail


@dataclass
class QueryStep:
    """A single step in a query execution plan."""
    step_id: int
    action: str  # QueryAction value as string
    query: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    depends_on: Optional[int] = None  # Previous step index for dependencies
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "step_id": self.step_id,
            "action": self.action,
            "query": self.query,
            "parameters": self.parameters,
            "depends_on": self.depends_on,
            "description": self.description,
        }


@dataclass
class QueryPlan:
    """Complete plan for executing a query."""
    original_query: str
    intent: str
    complexity: str  # "simple", "moderate", "complex"
    steps: List[QueryStep] = field(default_factory=list)
    fallback_strategies: List[str] = field(default_factory=list)
    max_retries: int = DEFAULT_MAX_RETRIES
    reasoning: str = ""
    confidence: float = 0.0
    requires_planning: bool = True  # False for simple queries (fast path)

    def to_dict(self) -> dict:
        return {
            "original_query": self.original_query,
            "intent": self.intent,
            "complexity": self.complexity,
            "steps": [s.to_dict() for s in self.steps],
            "fallback_strategies": self.fallback_strategies,
            "max_retries": self.max_retries,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "requires_planning": self.requires_planning,
        }


@dataclass
class StepResult:
    """Result of executing a single step."""
    step_id: int
    success: bool
    result: Any
    quality_score: float  # 0.0 to 1.0
    latency_ms: float
    error: Optional[str] = None


@dataclass
class PlannerMetrics:
    """Track query planner performance."""
    plans_created: int = 0
    simple_queries: int = 0  # Fast path (no planning needed)
    complex_queries: int = 0  # Required multi-step planning
    fallbacks_triggered: int = 0
    cache_hits: int = 0
    avg_planning_time_ms: float = 0.0
    total_planning_time_ms: float = 0.0


# =============================================================================
# Planning Prompt
# =============================================================================

PLANNING_PROMPT = """You are a query planning assistant for a research lab knowledge system. Your job is to analyze user queries and create execution plans.

## Available Actions

1. **search_registry**: Search the lab's structured registry (people, projects, grants, compute resources)
   - Best for: Who works on X? What projects exist? Lab member information

2. **search_docs**: Search unstructured documents (papers, notes, documentation)
   - Best for: Technical documentation, research content, historical information

3. **web_search**: Search the web for external information
   - Best for: Current events, recent papers, general knowledge, software documentation

4. **combine**: Synthesize results from previous steps
   - Best for: Multi-part questions requiring information from multiple sources

5. **direct_answer**: Answer directly without searching (for simple/meta questions)
   - Best for: "What can you do?", simple factual questions you can answer directly

## Query Complexity

- **simple**: Single intent, direct lookup, high confidence in approach
  - Example: "Who is James Evans?" -> search_registry
  - These should use fast path (no multi-step planning)

- **moderate**: May need 2-3 steps or one fallback strategy
  - Example: "What research is similar to James Evans' work?"

- **complex**: Multiple intents, comparisons, requires synthesis
  - Example: "Compare James Evans and Hyejin Youn's research and find their collaborations"

## Fallback Strategies

When initial searches fail, these strategies can help:

1. **reformulation**: Rephrase query with different terms
   - Example: "NLP research" -> "natural language processing work"

2. **decomposition**: Break compound queries into parts
   - Example: "James and Hyejin's collaborations" -> search each person, then find overlap

3. **expansion**: Add related terms or synonyms
   - Example: "ML projects" -> "machine learning, deep learning, AI projects"

4. **narrowing**: Focus on the most specific aspect
   - Example: "All research on networks" -> "network science publications 2024"

5. **web_fallback**: Try web search when local sources fail

## Output Format

Return a JSON object with this structure:
```json
{
  "intent": "<primary intent: lookup|technical|explore|connect>",
  "complexity": "<simple|moderate|complex>",
  "reasoning": "<brief explanation of your analysis>",
  "confidence": <0.0-1.0>,
  "steps": [
    {
      "step_id": 1,
      "action": "<action>",
      "query": "<search query or instruction>",
      "parameters": {},
      "depends_on": null,
      "description": "<what this step does>"
    }
  ],
  "fallback_strategies": ["<strategy1>", "<strategy2>"]
}
```

## Examples

Query: "Who is James Evans?"
```json
{
  "intent": "lookup",
  "complexity": "simple",
  "reasoning": "Direct person lookup, single registry search sufficient",
  "confidence": 0.95,
  "steps": [
    {"step_id": 1, "action": "search_registry", "query": "James Evans", "parameters": {"type": "person"}, "depends_on": null, "description": "Look up James Evans in lab registry"}
  ],
  "fallback_strategies": ["expansion", "web_fallback"]
}
```

Query: "Compare James Evans and Hyejin Youn's research and find their collaborations"
```json
{
  "intent": "explore",
  "complexity": "complex",
  "reasoning": "Requires looking up two people, comparing their work, and finding collaborations",
  "confidence": 0.85,
  "steps": [
    {"step_id": 1, "action": "search_registry", "query": "James Evans", "parameters": {"type": "person"}, "depends_on": null, "description": "Get James Evans profile"},
    {"step_id": 2, "action": "search_registry", "query": "Hyejin Youn", "parameters": {"type": "person"}, "depends_on": null, "description": "Get Hyejin Youn profile"},
    {"step_id": 3, "action": "search_docs", "query": "James Evans Hyejin Youn collaboration", "parameters": {}, "depends_on": null, "description": "Search for joint work"},
    {"step_id": 4, "action": "combine", "query": "Compare research areas and synthesize collaboration findings", "parameters": {"sources": [1, 2, 3]}, "depends_on": 3, "description": "Synthesize comparison"}
  ],
  "fallback_strategies": ["decomposition", "web_fallback"]
}
```

IMPORTANT:
- For simple queries (single intent, direct lookup), set complexity to "simple" and provide minimal steps
- Only output the JSON object, no other text
- Use the most efficient approach - don't over-plan simple queries
"""


# =============================================================================
# Query Planner
# =============================================================================

class QueryPlanner:
    """
    LLM-driven query planning for intelligent search strategies.

    Uses Claude Haiku to analyze queries and create execution plans with
    fallback strategies for handling failures.

    Features:
    - Fast path for simple queries (skip planning)
    - Multi-step planning for complex queries
    - Fallback strategy generation
    - Result quality evaluation
    - Caching of similar plans

    Usage:
        planner = QueryPlanner()
        plan = await planner.create_plan("Who is James Evans?")

        if plan.requires_planning:
            # Execute multi-step plan
            ...
        else:
            # Fast path - direct execution
            ...
    """

    def __init__(
        self,
        client: Optional[anthropic.Anthropic] = None,
        model: str = DEFAULT_PLANNING_MODEL,
        enable_caching: bool = True,
        cache_ttl: int = 3600,
    ):
        """
        Initialize the query planner.

        Args:
            client: Anthropic client (optional, will create if not provided)
            model: Model to use for planning
            enable_caching: Whether to cache plans
            cache_ttl: Cache time-to-live in seconds
        """
        self._client = client
        self.model = model
        self.enable_caching = enable_caching
        self.cache_ttl = cache_ttl

        # Plan cache
        self._plan_cache: Dict[str, tuple] = {}  # key -> (plan, timestamp)

        # Metrics
        self.metrics = PlannerMetrics()

    @property
    def client(self) -> anthropic.Anthropic:
        """Lazy initialization of Anthropic client."""
        if self._client is None:
            import os
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if api_key:
                self._client = anthropic.Anthropic(api_key=api_key)
            else:
                self._client = anthropic.Anthropic()
        return self._client

    def _get_cache_key(self, query: str) -> str:
        """Generate cache key from normalized query."""
        normalized = query.lower().strip()
        return hashlib.sha256(normalized.encode()).hexdigest()[:32]

    def _get_from_cache(self, query: str) -> Optional[QueryPlan]:
        """Get cached plan if available and not expired."""
        if not self.enable_caching:
            return None

        key = self._get_cache_key(query)
        if key in self._plan_cache:
            plan, timestamp = self._plan_cache[key]
            if time.time() - timestamp < self.cache_ttl:
                self.metrics.cache_hits += 1
                return plan
            else:
                del self._plan_cache[key]
        return None

    def _add_to_cache(self, query: str, plan: QueryPlan):
        """Add plan to cache."""
        if not self.enable_caching:
            return
        key = self._get_cache_key(query)
        self._plan_cache[key] = (plan, time.time())

        # Evict old entries if cache is too large
        if len(self._plan_cache) > 1000:
            oldest_keys = sorted(
                self._plan_cache.keys(),
                key=lambda k: self._plan_cache[k][1]
            )[:100]
            for k in oldest_keys:
                del self._plan_cache[k]

    def _is_simple_query(self, query: str) -> bool:
        """
        Fast heuristic check if query is simple enough for fast path.

        Simple queries:
        - Direct person lookups ("Who is X?")
        - Simple factual questions
        - Single-intent queries

        Returns True if we can skip LLM planning.
        """
        query_lower = query.lower().strip()

        # Very short queries are usually simple
        word_count = len(query.split())
        if word_count <= 5:
            # Direct person lookups
            if query_lower.startswith(("who is ", "who's ")):
                return True
            # Simple what is
            if query_lower.startswith("what is "):
                return True

        # Check for complexity indicators
        complexity_indicators = [
            " and ", " or ", " compare ", " versus ", " vs ",
            " relationship ", " between ", " both ", " multiple ",
            " all ", " similar to ", " like ",
            " as well as ", " along with ", " together ",
        ]

        has_complexity = any(ind in query_lower for ind in complexity_indicators)

        # Questions about comparisons or multiple entities are not simple
        if has_complexity:
            return False

        return word_count <= 10  # Short queries without complexity indicators

    def _create_simple_plan(self, query: str) -> QueryPlan:
        """
        Create a simple plan for fast-path queries.

        Doesn't require LLM call - uses heuristics to determine action.
        """
        query_lower = query.lower().strip()

        # Determine action based on query pattern
        if query_lower.startswith(("who is ", "who's ", "what is ")):
            action = QueryAction.SEARCH_REGISTRY.value
            intent = "lookup"
        elif any(kw in query_lower for kw in ["how do i", "how to", "setup", "install", "error"]):
            action = QueryAction.SEARCH_DOCS.value
            intent = "technical"
        elif any(kw in query_lower for kw in ["what can you", "help me", "your capabilities"]):
            action = QueryAction.DIRECT_ANSWER.value
            intent = "explore"
        else:
            action = QueryAction.SEARCH_REGISTRY.value
            intent = "lookup"

        step = QueryStep(
            step_id=1,
            action=action,
            query=query,
            parameters={},
            description="Direct search"
        )

        return QueryPlan(
            original_query=query,
            intent=intent,
            complexity="simple",
            steps=[step],
            fallback_strategies=[
                FallbackStrategy.EXPANSION.value,
                FallbackStrategy.WEB_FALLBACK.value
            ],
            max_retries=2,
            reasoning="Simple query - using fast path",
            confidence=0.9,
            requires_planning=False,  # Mark as fast path
        )

    def _parse_plan_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse LLM response JSON, handling markdown code blocks.
        """
        text = response_text.strip()

        # Handle markdown code blocks
        if text.startswith("```"):
            lines = text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        # Remove "json" language identifier if present
        if text.startswith("json"):
            text = text[4:].strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse plan response: {e}")
            raise ValueError(f"Invalid JSON in planning response: {e}")

    async def create_plan(self, query: str, context: Optional[Dict[str, Any]] = None) -> QueryPlan:
        """
        Create an execution plan for a query.

        Uses fast path for simple queries, LLM planning for complex ones.

        Args:
            query: The user's query
            context: Optional additional context

        Returns:
            QueryPlan with steps and fallback strategies
        """
        # Check cache first
        cached = self._get_from_cache(query)
        if cached:
            return cached

        # Check if simple enough for fast path
        if self._is_simple_query(query):
            self.metrics.simple_queries += 1
            plan = self._create_simple_plan(query)
            self._add_to_cache(query, plan)
            return plan

        # Complex query - use LLM planning
        return await self._create_plan_with_llm(query, context)

    async def _create_plan_with_llm(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> QueryPlan:
        """
        Create a plan using LLM for complex queries.
        """
        start_time = time.time()
        self.metrics.plans_created += 1
        self.metrics.complex_queries += 1

        try:
            # Build prompt with context
            user_prompt = f"Create an execution plan for this query:\n\n{query}"
            if context:
                if context.get("thread_context"):
                    user_prompt += f"\n\nConversation context:\n{context['thread_context']}"

            response = self.client.messages.create(
                model=self.model,
                max_tokens=MAX_PLANNING_TOKENS,
                system=PLANNING_PROMPT,
                messages=[{"role": "user", "content": user_prompt}]
            )

            # Parse response
            response_text = response.content[0].text
            data = self._parse_plan_response(response_text)

            # Build steps
            steps = []
            for step_data in data.get("steps", []):
                step = QueryStep(
                    step_id=step_data.get("step_id", len(steps) + 1),
                    action=step_data.get("action", "search_registry"),
                    query=step_data.get("query", query),
                    parameters=step_data.get("parameters", {}),
                    depends_on=step_data.get("depends_on"),
                    description=step_data.get("description", ""),
                )
                steps.append(step)

            # If no steps, create a default one
            if not steps:
                steps = [QueryStep(
                    step_id=1,
                    action="search_registry",
                    query=query,
                    parameters={},
                    description="Default search"
                )]

            # Build plan
            plan = QueryPlan(
                original_query=query,
                intent=data.get("intent", "lookup"),
                complexity=data.get("complexity", "moderate"),
                steps=steps,
                fallback_strategies=data.get("fallback_strategies", []),
                max_retries=DEFAULT_MAX_RETRIES,
                reasoning=data.get("reasoning", ""),
                confidence=float(data.get("confidence", 0.7)),
                requires_planning=True,
            )

            # Track timing
            latency_ms = (time.time() - start_time) * 1000
            self.metrics.total_planning_time_ms += latency_ms
            self.metrics.avg_planning_time_ms = (
                self.metrics.total_planning_time_ms /
                max(1, self.metrics.plans_created)
            )

            logger.debug(
                f"Created plan for '{query[:50]}...': {len(steps)} steps, "
                f"complexity={plan.complexity}, latency={latency_ms:.0f}ms"
            )

            # Cache the plan
            self._add_to_cache(query, plan)

            return plan

        except Exception as e:
            logger.error(f"Plan creation failed: {e}")
            # Return a simple fallback plan
            return self._create_simple_plan(query)

    async def generate_reformulation(
        self,
        original_query: str,
        strategy: str,
        previous_results: Optional[List[str]] = None
    ) -> str:
        """
        Generate a reformulated query based on the specified strategy.

        Args:
            original_query: The original user query
            strategy: The fallback strategy to use
            previous_results: Optional context from previous attempts

        Returns:
            Reformulated query string
        """
        prompt = f"""Reformulate this query using the '{strategy}' strategy.

Original query: {original_query}

Strategy definitions:
- reformulation: Rephrase using different words/phrasing for better matching
- decomposition: Return the most important single sub-question
- expansion: Add synonyms and related terms
- narrowing: Focus on the most specific, searchable aspect

{"Previous attempts returned no useful results." if previous_results else ""}

Return ONLY the reformulated query, nothing else."""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=150,
                messages=[{"role": "user", "content": prompt}]
            )

            reformulated = response.content[0].text.strip()

            # Sanity check
            if reformulated and len(reformulated) < len(original_query) * 3:
                return reformulated

        except Exception as e:
            logger.warning(f"Reformulation failed: {e}")

        return original_query  # Return original on failure

    def evaluate_result_quality(
        self,
        query: str,
        result: Any,
        step: QueryStep
    ) -> float:
        """
        Evaluate the quality of a step result (heuristic-based).

        Returns a score from 0.0 to 1.0.
        """
        if result is None:
            return 0.0

        if isinstance(result, dict):
            # Check for error
            if result.get("error"):
                return 0.0

            # Check for structured results
            structured = result.get("structured_results", [])
            semantic = result.get("semantic_results", result.get("results", []))

            if structured:
                # Structured results are high quality
                return min(1.0, 0.7 + 0.1 * len(structured))
            elif semantic:
                # Semantic results are moderate quality
                return min(0.8, 0.4 + 0.1 * len(semantic))
            else:
                return 0.2

        elif isinstance(result, str):
            # String result - check length and relevance
            if len(result) < 50:
                return 0.2
            elif "error" in result.lower() or "not found" in result.lower():
                return 0.1
            else:
                return 0.6

        return 0.5  # Default moderate score

    def get_metrics(self) -> dict:
        """Return planner metrics."""
        return {
            "plans_created": self.metrics.plans_created,
            "simple_queries": self.metrics.simple_queries,
            "complex_queries": self.metrics.complex_queries,
            "fallbacks_triggered": self.metrics.fallbacks_triggered,
            "cache_hits": self.metrics.cache_hits,
            "avg_planning_time_ms": self.metrics.avg_planning_time_ms,
            "cache_size": len(self._plan_cache),
        }

    def clear_cache(self):
        """Clear the plan cache."""
        self._plan_cache.clear()
        logger.info("Plan cache cleared")
