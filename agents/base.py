"""
Base classes and data structures for the Chorus agent system.
"""

import os
import time
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, TYPE_CHECKING

import anthropic

# Type checking imports (avoid circular imports)
if TYPE_CHECKING:
    from .confidence import ConfidenceScore


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class QueryClassification:
    """Result of query classification."""
    intent: str  # "lookup", "technical", "explore", "connect", "ambiguous"
    confidence: float  # 0.0 to 1.0
    secondary_intents: List[str] = field(default_factory=list)
    requires_rag: bool = True
    temporal_context: Optional[str] = None  # "past", "current", "future", "any"
    reasoning: str = ""


@dataclass
class AgentResponse:
    """Unified response format from any agent."""
    content: str
    agent_type: str
    model_used: str
    tool_calls: List[dict] = field(default_factory=list)
    latency_ms: float = 0.0
    cache_hit: bool = False
    confidence: float = 1.0  # Legacy field (0.0-1.0 score)
    confidence_details: Optional["ConfidenceScore"] = None  # Full confidence with sources
    error: Optional[str] = None

    def format_with_confidence(self, format_type: str = "cli") -> str:
        """
        Format response content with confidence indicator and sources.

        Args:
            format_type: "cli" for terminal, "slack" for Slack formatting

        Returns:
            Formatted response string with confidence and sources appended
        """
        parts = [self.content]

        if self.confidence_details:
            parts.append("")
            parts.append(self.confidence_details.format_for_display(format_type=format_type))
        elif self.confidence < 1.0:
            # Legacy confidence display
            parts.append("")
            level = "HIGH" if self.confidence >= 0.8 else "MEDIUM" if self.confidence >= 0.6 else "LOW"
            parts.append(f"Confidence: {level}")

        return "\n".join(parts)


@dataclass
class AgentMetrics:
    """Track agent performance metrics."""
    queries: int = 0
    cache_hits: int = 0
    prompt_cache_hits: int = 0
    tokens_saved: int = 0
    total_latency_ms: float = 0
    errors: int = 0

    # Per-model tracking
    haiku_queries: int = 0
    sonnet_queries: int = 0

    # Per-agent tracking
    classifier_calls: int = 0
    memory_calls: int = 0
    mechanic_calls: int = 0
    muse_calls: int = 0
    matchmaker_calls: int = 0
    fallback_calls: int = 0


# =============================================================================
# Base Agent Class
# =============================================================================

class BaseAgent(ABC):
    """
    Abstract base class for all Chorus agents.

    Provides common functionality:
    - Anthropic client management
    - Prompt caching support
    - Metrics tracking
    - Response formatting
    """

    # Subclasses should override these
    AGENT_TYPE: str = "base"
    DEFAULT_MODEL: str = "claude-3-5-haiku-20241022"
    MAX_TOKENS: int = 1024

    def __init__(
        self,
        client: Optional[anthropic.Anthropic] = None,
        enable_prompt_caching: bool = True,
    ):
        self._client = client
        self.enable_prompt_caching = enable_prompt_caching
        self.metrics = AgentMetrics()

    @property
    def client(self) -> anthropic.Anthropic:
        """Lazy initialization of Anthropic client."""
        if self._client is None:
            # Try to get API key from environment
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if api_key:
                self._client = anthropic.Anthropic(api_key=api_key)
            else:
                self._client = anthropic.Anthropic()
        return self._client

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Return the system prompt for this agent."""
        pass

    def get_tools(self) -> List[dict]:
        """Return tool definitions. Override in subclasses that use tools."""
        return []

    def _build_system_with_caching(self, additional_context: str = "") -> list:
        """Build system prompt with Anthropic prompt caching."""
        base_prompt = self.get_system_prompt()

        if self.enable_prompt_caching:
            return [
                {
                    "type": "text",
                    "text": base_prompt,
                    "cache_control": {"type": "ephemeral"}
                },
                {
                    "type": "text",
                    "text": additional_context
                }
            ] if additional_context else [
                {
                    "type": "text",
                    "text": base_prompt,
                    "cache_control": {"type": "ephemeral"}
                }
            ]
        else:
            return base_prompt + additional_context

    @abstractmethod
    async def query(self, message: str, context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """
        Process a query and return a response.

        Args:
            message: The user's message
            context: Optional additional context (e.g., temporal info, conversation history)

        Returns:
            AgentResponse with the result
        """
        pass

    def _track_usage(self, response, model: str, latency_ms: float):
        """Track API usage from response."""
        self.metrics.queries += 1
        self.metrics.total_latency_ms += latency_ms

        if "haiku" in model:
            self.metrics.haiku_queries += 1
        else:
            self.metrics.sonnet_queries += 1

        # Track prompt cache hits
        if hasattr(response, 'usage') and response.usage:
            cache_read = getattr(response.usage, 'cache_read_input_tokens', 0)
            if cache_read and cache_read > 0:
                self.metrics.prompt_cache_hits += 1
                self.metrics.tokens_saved += cache_read

    def get_metrics(self) -> dict:
        """Return current metrics as a dictionary."""
        return {
            "agent_type": self.AGENT_TYPE,
            "queries": self.metrics.queries,
            "cache_hits": self.metrics.cache_hits,
            "prompt_cache_hits": self.metrics.prompt_cache_hits,
            "tokens_saved": self.metrics.tokens_saved,
            "avg_latency_ms": self.metrics.total_latency_ms / max(1, self.metrics.queries),
            "haiku_queries": self.metrics.haiku_queries,
            "sonnet_queries": self.metrics.sonnet_queries,
            "errors": self.metrics.errors,
        }


def get_cache_key(query: str, agent_type: str = "") -> str:
    """Generate cache key from query and optional agent type."""
    key_input = f"{agent_type}:{query.lower().strip()}" if agent_type else query.lower().strip()
    return "chorus:" + hashlib.md5(key_input.encode()).hexdigest()
