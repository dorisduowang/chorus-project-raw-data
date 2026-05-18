"""
ChorusOrchestrator - Routes queries to specialist agents.

This is the main entry point for the multi-agent Chorus system.

Features:
- Intent classification (regex + LLM fallback)
- Multi-agent routing with fallback chains
- Smart query planning with LLM-driven fallback strategies
- Response caching and parallel execution
"""

import asyncio
import logging
import os
import time
import hashlib
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple, Set
from dataclasses import dataclass, field

import anthropic

from .base import BaseAgent, AgentResponse, AgentMetrics, get_cache_key
from .classifier import QueryClassifier, QueryClassification
from .llm_classifier import LLMQueryClassifier, get_classifier
from .specialists import MemoryAgent, MechanicAgent, MuseAgent, MatchmakerAgent
from .query_planner import QueryPlanner, QueryPlan, QueryAction
from .plan_executor import PlanExecutor, ExecutionResult


# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# Agent Configuration
# =============================================================================

# Default enabled agents - can be overridden via constructor or environment
# Environment variable: CHORUS_ENABLED_AGENTS="memory,mechanic,muse,matchmaker"
DEFAULT_ENABLED_AGENTS = {"memory", "mechanic"}  # Muse and Matchmaker offline for now

# All available agents
ALL_AGENTS = {"memory", "mechanic", "muse", "matchmaker"}

# Intent to agent mapping
INTENT_AGENT_MAP = {
    "lookup": "memory",
    "technical": "mechanic",
    "explore": "muse",
    "connect": "matchmaker",
}

# Fallback chain when preferred agent is disabled
# Key = disabled agent, Value = ordered list of fallbacks to try
AGENT_FALLBACKS = {
    "muse": ["memory", "mechanic"],       # Explore queries → Memory (can use RAG)
    "matchmaker": ["memory", "mechanic"], # Connect queries → Memory (can look up people)
    "mechanic": ["memory"],               # Technical → Memory
    "memory": ["mechanic"],               # Lookup → Mechanic (unlikely to disable)
}


# =============================================================================
# Cache Manager (shared across agents)
# =============================================================================

@dataclass
class CacheEntry:
    response: str
    timestamp: float
    agent_type: str


@dataclass
class OrchestratorCache:
    """Simple in-memory cache for orchestrator responses."""

    entries: Dict[str, CacheEntry] = field(default_factory=dict)
    ttl: int = 300  # 5 minutes
    max_entries: int = 200

    def get(self, query: str, agent_type: str = "") -> Optional[str]:
        """Get cached response if available and not expired."""
        key = get_cache_key(query, agent_type)

        if key in self.entries:
            entry = self.entries[key]
            if time.time() - entry.timestamp < self.ttl:
                return entry.response
            else:
                del self.entries[key]

        return None

    def set(self, query: str, response: str, agent_type: str = ""):
        """Store response in cache."""
        key = get_cache_key(query, agent_type)

        self.entries[key] = CacheEntry(
            response=response,
            timestamp=time.time(),
            agent_type=agent_type
        )

        # Limit cache size
        if len(self.entries) > self.max_entries:
            # Remove oldest entries
            sorted_entries = sorted(
                self.entries.items(),
                key=lambda x: x[1].timestamp
            )
            for k, _ in sorted_entries[:50]:
                del self.entries[k]


# =============================================================================
# Temporal Context (from core_agent.py)
# =============================================================================

def get_temporal_context() -> str:
    """Generate temporal context string."""
    now = datetime.now()
    return f"Today is {now.strftime('%B %d, %Y')} (Q{(now.month - 1) // 3 + 1} {now.year})"


# =============================================================================
# Orchestrator
# =============================================================================

class ChorusOrchestrator:
    """
    Routes queries to specialist agents based on intent classification.

    Flow:
    1. Check response cache
    2. Classify query intent (fast regex, fallback to Haiku)
    3. Route to appropriate specialist(s)
    4. For multi-intent queries, run specialists in parallel
    5. Cache and return response

    Usage:
        orchestrator = ChorusOrchestrator()
        response = await orchestrator.query("What research projects are happening?")
    """

    def __init__(
        self,
        enable_caching: bool = True,
        enable_parallel: bool = True,
        rag_available: bool = True,
        enabled_agents: Optional[Set[str]] = None,
        enable_smart_fallback: Optional[bool] = None,
    ):
        """
        Initialize the orchestrator.

        Args:
            enable_caching: Whether to cache responses
            enable_parallel: Whether to run multi-intent queries in parallel
            rag_available: Whether RAG server is available
            enabled_agents: Set of agent names to enable (e.g., {"memory", "mechanic"})
                           If None, uses DEFAULT_ENABLED_AGENTS or CHORUS_ENABLED_AGENTS env var
            enable_smart_fallback: Whether to use LLM-driven query planning with smart fallbacks
                                  If None, reads from CHORUS_SMART_FALLBACK env var (default: True)
        """
        self.enable_caching = enable_caching
        self.enable_parallel = enable_parallel
        self.rag_available = rag_available

        # Configure enabled agents
        self.enabled_agents = self._resolve_enabled_agents(enabled_agents)

        # Configure smart fallback
        if enable_smart_fallback is None:
            env_val = os.environ.get("CHORUS_SMART_FALLBACK", "true").lower()
            self.enable_smart_fallback = env_val in ("true", "1", "yes")
        else:
            self.enable_smart_fallback = enable_smart_fallback

        # Initialize cache
        self.cache = OrchestratorCache() if enable_caching else None

        # Shared Anthropic client
        self._client: Optional[anthropic.Anthropic] = None

        # Initialize agents (lazy - created on first use)
        self._classifier: Optional[QueryClassifier] = None
        self._memory: Optional[MemoryAgent] = None
        self._mechanic: Optional[MechanicAgent] = None
        self._muse: Optional[MuseAgent] = None
        self._matchmaker: Optional[MatchmakerAgent] = None

        # Query planner and executor for smart fallback (lazy init)
        self._query_planner: Optional[QueryPlanner] = None
        self._plan_executor: Optional[PlanExecutor] = None

        # Fallback to original ChorusAgent for ambiguous queries
        self._fallback_agent = None

        # Aggregate metrics
        self.metrics = AgentMetrics()

        # Track smart fallback usage
        self._smart_fallback_attempts = 0
        self._smart_fallback_successes = 0

    def _resolve_enabled_agents(self, enabled_agents: Optional[Set[str]]) -> Set[str]:
        """Resolve which agents are enabled from args, env, or defaults."""
        if enabled_agents is not None:
            # Explicit configuration takes priority
            return enabled_agents & ALL_AGENTS

        # Check environment variable
        env_agents = os.environ.get("CHORUS_ENABLED_AGENTS")
        if env_agents:
            parsed = {a.strip().lower() for a in env_agents.split(",")}
            return parsed & ALL_AGENTS

        # Use defaults
        return DEFAULT_ENABLED_AGENTS.copy()

    def set_enabled_agents(self, agents: Set[str]):
        """
        Update which agents are enabled at runtime.

        Args:
            agents: Set of agent names to enable (e.g., {"memory", "mechanic", "muse"})
        """
        self.enabled_agents = agents & ALL_AGENTS

    def enable_agent(self, agent: str):
        """Enable a single agent."""
        if agent in ALL_AGENTS:
            self.enabled_agents.add(agent)

    def disable_agent(self, agent: str):
        """Disable a single agent (must keep at least one enabled)."""
        if agent in self.enabled_agents and len(self.enabled_agents) > 1:
            self.enabled_agents.discard(agent)

    @property
    def client(self) -> anthropic.Anthropic:
        """Lazy initialization of shared Anthropic client."""
        if self._client is None:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if api_key:
                self._client = anthropic.Anthropic(api_key=api_key)
            else:
                self._client = anthropic.Anthropic()
        return self._client

    @property
    def classifier(self) -> BaseAgent:
        """Get query classifier (LLM or regex based on config)."""
        if self._classifier is None:
            # Use factory function - respects CHORUS_USE_LLM_CLASSIFIER env var
            self._classifier = get_classifier(client=self.client)
        return self._classifier

    @property
    def memory(self) -> MemoryAgent:
        if self._memory is None:
            # Use fast_mode=True for pre-fetched RAG (single API call, ~50% faster)
            self._memory = MemoryAgent(client=self.client, fast_mode=True)
        return self._memory

    @property
    def mechanic(self) -> MechanicAgent:
        if self._mechanic is None:
            self._mechanic = MechanicAgent(client=self.client)
        return self._mechanic

    @property
    def muse(self) -> MuseAgent:
        if self._muse is None:
            self._muse = MuseAgent(client=self.client)
        return self._muse

    @property
    def matchmaker(self) -> MatchmakerAgent:
        if self._matchmaker is None:
            self._matchmaker = MatchmakerAgent(client=self.client)
        return self._matchmaker

    @property
    def query_planner(self) -> QueryPlanner:
        """Lazy initialization of query planner."""
        if self._query_planner is None:
            self._query_planner = QueryPlanner(client=self.client)
        return self._query_planner

    @property
    def plan_executor(self) -> PlanExecutor:
        """Lazy initialization of plan executor with search functions."""
        if self._plan_executor is None:
            self._plan_executor = PlanExecutor(
                search_registry=self._execute_registry_search,
                search_docs=self._execute_doc_search,
                web_search=self._execute_web_search,
                client=self.client,
            )
            self._plan_executor.set_planner(self.query_planner)
        return self._plan_executor

    async def _execute_registry_search(
        self,
        query: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a registry search for the plan executor."""
        from .specialists.memory import call_rag_async
        result = await call_rag_async("/hybrid", {"q": query, "top_k": 5})
        return result

    async def _execute_doc_search(
        self,
        query: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a document search for the plan executor."""
        from .specialists.memory import call_rag_async
        # Use semantic search for documents
        result = await call_rag_async("/search", {"q": query, "top_k": 5})
        return result

    async def _execute_web_search(
        self,
        query: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a web search for the plan executor."""
        from .specialists.memory import search_web_async
        result = await search_web_async(query, num_results=5)
        return result

    def _get_agent_instance(self, agent_name: str) -> BaseAgent:
        """Get agent instance by name."""
        agents = {
            "memory": self.memory,
            "mechanic": self.mechanic,
            "muse": self.muse,
            "matchmaker": self.matchmaker,
        }
        return agents.get(agent_name, self.memory)

    def _get_agent_for_intent(self, intent: str) -> BaseAgent:
        """
        Get the appropriate specialist agent for an intent.

        If the preferred agent is disabled, uses fallback chain.
        """
        # Get the preferred agent name for this intent
        preferred = INTENT_AGENT_MAP.get(intent, "memory")

        # Check if preferred agent is enabled
        if preferred in self.enabled_agents:
            return self._get_agent_instance(preferred)

        # Try fallback chain
        for fallback in AGENT_FALLBACKS.get(preferred, []):
            if fallback in self.enabled_agents:
                return self._get_agent_instance(fallback)

        # Last resort: return first enabled agent
        first_enabled = next(iter(self.enabled_agents))
        return self._get_agent_instance(first_enabled)

    async def _run_specialist(
        self,
        agent: BaseAgent,
        message: str,
        context: Dict[str, Any]
    ) -> AgentResponse:
        """Run a specialist agent with error handling."""
        try:
            return await agent.query(message, context)
        except Exception as e:
            return AgentResponse(
                content=f"Error from {agent.AGENT_TYPE}: {str(e)}",
                agent_type=agent.AGENT_TYPE,
                model_used=agent.DEFAULT_MODEL,
                error=str(e)
            )

    async def _run_parallel_specialists(
        self,
        intents: List[str],
        message: str,
        context: Dict[str, Any]
    ) -> List[AgentResponse]:
        """Run multiple specialists in parallel."""
        agents = [self._get_agent_for_intent(intent) for intent in intents]

        # Deduplicate agents (in case multiple intents map to same agent)
        unique_agents = list({id(a): a for a in agents}.values())

        tasks = [
            self._run_specialist(agent, message, context)
            for agent in unique_agents
        ]

        return await asyncio.gather(*tasks, return_exceptions=False)

    def _merge_responses(self, responses: List[AgentResponse]) -> str:
        """Merge responses from multiple specialists."""
        if len(responses) == 1:
            return responses[0].content

        # Multiple responses - format with attribution
        parts = []
        for response in responses:
            if response.error:
                continue
            # Don't add redundant headers for single substantial response
            if len(responses) == 2 and len(response.content) > 500:
                parts.append(response.content)
            else:
                parts.append(response.content)

        return "\n\n".join(parts)

    async def query(self, message: str, thread_context: Optional[str] = None) -> str:
        """
        Process a query through the orchestrator.

        This is the main entry point - routes to appropriate specialist(s)
        based on query classification.

        Flow:
        1. Check cache
        2. If smart fallback enabled: Use query planner for complex queries
        3. Otherwise: Classify and route to specialists
        4. Evaluate results and trigger fallbacks if needed
        5. Cache and return

        Args:
            message: The user's message
            thread_context: Optional conversation history for context-aware responses

        Returns:
            Response string
        """
        start_time = time.time()
        self.metrics.queries += 1

        # Check cache first (skip cache if we have thread context - responses are contextual)
        if self.cache and not thread_context:
            cached = self.cache.get(message)
            if cached:
                self.metrics.cache_hits += 1
                return cached

        # Build context
        context = {
            "temporal_context": get_temporal_context(),
            "rag_available": self.rag_available,
            "thread_context": thread_context or "",
        }

        # Use smart fallback path if enabled
        if self.enable_smart_fallback:
            final_response = await self._query_with_smart_fallback(message, context)
        else:
            final_response = await self._query_standard(message, context)

        # Cache the result
        if self.cache and final_response:
            self.cache.set(message, final_response)

        # Track latency
        latency_ms = (time.time() - start_time) * 1000
        self.metrics.total_latency_ms += latency_ms

        return final_response

    async def _query_with_smart_fallback(
        self,
        message: str,
        context: Dict[str, Any]
    ) -> str:
        """
        Process query using LLM-driven planning and smart fallback.

        This path:
        1. Creates a query plan (fast path for simple queries)
        2. Executes the plan with step tracking
        3. Evaluates result quality
        4. Triggers fallback strategies if needed
        5. Combines results into final response
        """
        self._smart_fallback_attempts += 1

        try:
            # Create query plan
            plan = await self.query_planner.create_plan(message, context)

            logger.debug(
                f"Query plan: complexity={plan.complexity}, "
                f"steps={len(plan.steps)}, requires_planning={plan.requires_planning}"
            )

            # For simple queries on fast path, route directly to specialist
            if not plan.requires_planning and plan.complexity == "simple":
                # Use standard routing for simple queries (faster)
                return await self._query_standard(message, context)

            # Execute the plan with fallback support
            result = await self.plan_executor.execute(plan)

            # If execution succeeded with good quality, use plan result
            if result.success and result.final_quality >= 0.5:
                self._smart_fallback_successes += 1

                # The plan executor returns formatted content
                # But we may want to run through a specialist for final formatting
                if result.content and len(result.content) > 50:
                    return result.content

            # If plan execution had poor results, fall back to standard path
            # with the classification-based routing
            logger.info(
                f"Plan execution quality {result.final_quality:.2f} below threshold, "
                f"falling back to standard routing"
            )
            return await self._query_standard(message, context)

        except Exception as e:
            logger.warning(f"Smart fallback failed: {e}, using standard routing")
            return await self._query_standard(message, context)

    async def _query_standard(
        self,
        message: str,
        context: Dict[str, Any]
    ) -> str:
        """
        Standard query processing with classification-based routing.

        This is the original query path, used when:
        - Smart fallback is disabled
        - Query is simple (fast path)
        - Smart fallback fails
        """
        # Classify the query
        classification = await self.classifier.classify(message)

        # Track classification
        self.metrics.classifier_calls += 1

        # Determine which agents to run
        intents_to_run = [classification.intent]

        # Add secondary intents if confident
        if classification.secondary_intents and classification.confidence > 0.7:
            intents_to_run.extend(classification.secondary_intents)

        # Handle ambiguous queries
        if classification.intent == "ambiguous" or classification.confidence < 0.5:
            # Fall back to explore intent (routes to best available agent via fallback chain)
            intents_to_run = ["explore"]
            self.metrics.fallback_calls += 1

        # Run specialist(s)
        if len(intents_to_run) > 1 and self.enable_parallel:
            # Parallel execution for multi-intent
            responses = await self._run_parallel_specialists(
                intents_to_run, message, context
            )
        else:
            # Single specialist
            agent = self._get_agent_for_intent(intents_to_run[0])
            response = await self._run_specialist(agent, message, context)
            responses = [response]

        # Update agent-specific metrics
        for response in responses:
            if response.agent_type == "memory":
                self.metrics.memory_calls += 1
            elif response.agent_type == "mechanic":
                self.metrics.mechanic_calls += 1
            elif response.agent_type == "muse":
                self.metrics.muse_calls += 1
            elif response.agent_type == "matchmaker":
                self.metrics.matchmaker_calls += 1

            # Track model usage
            if "haiku" in response.model_used:
                self.metrics.haiku_queries += 1
            else:
                self.metrics.sonnet_queries += 1

        # Merge responses
        return self._merge_responses(responses)

    def get_agent_status(self) -> dict:
        """Get current agent configuration status."""
        return {
            "enabled": sorted(self.enabled_agents),
            "disabled": sorted(ALL_AGENTS - self.enabled_agents),
            "all_available": sorted(ALL_AGENTS),
            "intent_routing": {
                intent: (
                    INTENT_AGENT_MAP[intent]
                    if INTENT_AGENT_MAP[intent] in self.enabled_agents
                    else AGENT_FALLBACKS.get(INTENT_AGENT_MAP[intent], ["memory"])[0]
                )
                for intent in INTENT_AGENT_MAP
            }
        }

    def get_metrics(self) -> dict:
        """Get orchestrator and agent metrics."""
        metrics = {
            "orchestrator": {
                "queries": self.metrics.queries,
                "cache_hits": self.metrics.cache_hits,
                "cache_hit_rate": self.metrics.cache_hits / max(1, self.metrics.queries),
                "avg_latency_ms": self.metrics.total_latency_ms / max(1, self.metrics.queries),
                "haiku_queries": self.metrics.haiku_queries,
                "sonnet_queries": self.metrics.sonnet_queries,
                "haiku_rate": self.metrics.haiku_queries / max(1, self.metrics.haiku_queries + self.metrics.sonnet_queries),
            },
            "routing": {
                "classifier_calls": self.metrics.classifier_calls,
                "memory_calls": self.metrics.memory_calls,
                "mechanic_calls": self.metrics.mechanic_calls,
                "muse_calls": self.metrics.muse_calls,
                "matchmaker_calls": self.metrics.matchmaker_calls,
                "fallback_calls": self.metrics.fallback_calls,
            },
            "smart_fallback": {
                "enabled": self.enable_smart_fallback,
                "attempts": self._smart_fallback_attempts,
                "successes": self._smart_fallback_successes,
                "success_rate": self._smart_fallback_successes / max(1, self._smart_fallback_attempts),
            },
            "agent_config": self.get_agent_status(),
        }

        # Add per-agent metrics if agents are initialized
        if self._classifier:
            metrics["classifier"] = self._classifier.get_metrics()
        if self._memory:
            metrics["memory"] = self._memory.get_metrics()
        if self._mechanic:
            metrics["mechanic"] = self._mechanic.get_metrics()
        if self._muse:
            metrics["muse"] = self._muse.get_metrics()
        if self._matchmaker:
            metrics["matchmaker"] = self._matchmaker.get_metrics()

        # Add planner and executor metrics if initialized
        if self._query_planner:
            metrics["query_planner"] = self._query_planner.get_metrics()
        if self._plan_executor:
            metrics["plan_executor"] = self._plan_executor.get_metrics()

        return metrics

    async def check_rag_health(self) -> dict:
        """Check if RAG server is healthy."""
        from .specialists.memory import call_rag_async
        return await call_rag_async("/health")
