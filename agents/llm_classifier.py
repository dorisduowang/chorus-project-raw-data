"""
LLMQueryClassifier - LLM-based intent classification using Claude Haiku.

Replaces the brittle regex-based system with semantic understanding while
maintaining the same interface for drop-in replacement.
"""

import json
import logging
import os
import time
import hashlib
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from functools import lru_cache

import anthropic

from .base import BaseAgent, QueryClassification, AgentResponse, AgentMetrics
from .classifier import QueryClassifier


# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

# Environment variable to enable/disable LLM classification
# Set CHORUS_USE_LLM_CLASSIFIER=true to enable
LLM_CLASSIFIER_ENABLED_ENV = "CHORUS_USE_LLM_CLASSIFIER"

# Default model for classification
DEFAULT_CLASSIFICATION_MODEL = "claude-3-5-haiku-20241022"

# Maximum tokens for classification response
MAX_CLASSIFICATION_TOKENS = 256

# Cache settings
DEFAULT_CACHE_TTL = 3600  # 1 hour
DEFAULT_CACHE_MAX_SIZE = 2000


# =============================================================================
# Classification Prompt with Few-Shot Examples
# =============================================================================

LLM_CLASSIFICATION_PROMPT = """You are an intent classifier for a research lab assistant system called CHORUS. Your task is to classify user queries into exactly one primary intent category.

## Intent Categories

### LOOKUP
Factual retrieval questions seeking specific information that can be found in a knowledge base.
- Direct questions: who, what, when, where, how many
- Requests to list, find, show, or get specific information
- Questions about grants, publications, projects, people, dates, deadlines

Examples:
- "Who is working on the cancer genomics project?" -> LOOKUP
- "What grants does Professor Smith have?" -> LOOKUP
- "List all publications from 2023" -> LOOKUP
- "When is the next lab meeting?" -> LOOKUP
- "Show me the budget for Project X" -> LOOKUP

### TECHNICAL
Questions about setup, configuration, debugging, access issues, or computing resources.
- Problems with tools, software, or systems (errors, bugs, not working)
- How-to questions for technical tasks (install, configure, setup)
- Access and permission issues (SSH, VPN, credentials)
- Cluster/compute questions (Midway, Slurm, conda, Docker)

Examples:
- "How do I access Midway?" -> TECHNICAL
- "My Jupyter notebook won't connect to the kernel" -> TECHNICAL
- "I'm getting a permission denied error on the cluster" -> TECHNICAL
- "Help me set up conda environment for PyTorch" -> TECHNICAL
- "Git push is failing with authentication error" -> TECHNICAL

### EXPLORE
Deep analysis, synthesis, creative thinking, or multi-perspective reasoning.
- Requests to analyze, compare, evaluate, or synthesize
- Brainstorming and ideation
- "What if" and hypothetical questions
- Questions about approaches, methods, strategies
- Requests for opinions, thoughts, or recommendations
- Questions about CHORUS itself (capabilities, what it can do)

Examples:
- "Compare the approaches used in Project A vs Project B" -> EXPLORE
- "What are some ways we could improve our data pipeline?" -> EXPLORE
- "Analyze the trends in our publication output" -> EXPLORE
- "What do you think about using transformers for this task?" -> EXPLORE
- "What can you help me with?" -> EXPLORE
- "How might we approach the new NIH grant?" -> EXPLORE

### CONNECT
Finding relationships, collaborations, or connections between people, ideas, or resources.
- Questions about who works on what, who knows what
- Requests for introductions or to find collaborators
- Questions about relationships between projects, ideas, or people
- Finding similar work or related people

Examples:
- "Who else is working on single-cell analysis?" -> CONNECT
- "Connect me with someone who knows about CRISPR" -> CONNECT
- "How does Dr. Jones' work relate to the genomics project?" -> CONNECT
- "Who should I talk to about machine learning for biology?" -> CONNECT
- "Are there any collaborations between our lab and MIT?" -> CONNECT

## Classification Rules

1. Choose the SINGLE best-fitting intent based on the primary purpose of the query
2. Confidence should reflect how clearly the query fits one category:
   - 0.9-1.0: Very clear, unambiguous match
   - 0.7-0.89: Good match with some minor ambiguity
   - 0.5-0.69: Could fit multiple categories
   - Below 0.5: Very ambiguous, default to AMBIGUOUS

3. Use AMBIGUOUS only when:
   - The query is too vague to classify
   - Multiple intents are equally valid
   - The query doesn't fit any category well

4. requires_rag indicates if the query needs to search a knowledge base:
   - LOOKUP: almost always true
   - TECHNICAL: often true (for docs/guides)
   - CONNECT: usually true (to find people/projects)
   - EXPLORE: sometimes true (depends on if analysis needs data)

## Output Format

Respond with ONLY a JSON object (no markdown, no explanation):
{"intent": "<INTENT>", "confidence": <0.0-1.0>, "requires_rag": <true/false>, "secondary_intent": "<INTENT or null>", "reasoning": "<brief explanation>"}
"""


# =============================================================================
# Cache Entry for LLM Classifications
# =============================================================================

@dataclass
class ClassificationCacheEntry:
    """Cached classification result with timestamp."""
    classification: QueryClassification
    timestamp: float
    query_hash: str


# =============================================================================
# LLM Query Classifier
# =============================================================================

class LLMQueryClassifier(BaseAgent):
    """
    LLM-based query classification using Claude Haiku.

    Provides semantic understanding of queries instead of brittle regex patterns.
    Falls back to regex-based classifier if LLM call fails.

    Features:
    - Uses Claude Haiku for fast classification (~200-300ms)
    - Caches results to avoid redundant API calls
    - Includes fallback to regex for reliability
    - Same interface as QueryClassifier for drop-in replacement

    Usage:
        classifier = LLMQueryClassifier()
        classification = await classifier.classify("What grants does Prof Smith have?")
        print(classification.intent)  # "lookup"
        print(classification.confidence)  # 0.95

    Configuration:
        Set CHORUS_USE_LLM_CLASSIFIER=true to use LLM classification
        Set CHORUS_USE_LLM_CLASSIFIER=false to fall back to regex only
    """

    AGENT_TYPE = "llm_classifier"
    DEFAULT_MODEL = DEFAULT_CLASSIFICATION_MODEL
    MAX_TOKENS = MAX_CLASSIFICATION_TOKENS

    def __init__(
        self,
        client: Optional[anthropic.Anthropic] = None,
        enable_llm: Optional[bool] = None,
        enable_fallback: bool = True,
        cache_ttl: int = DEFAULT_CACHE_TTL,
        cache_max_size: int = DEFAULT_CACHE_MAX_SIZE,
        **kwargs
    ):
        """
        Initialize the LLM query classifier.

        Args:
            client: Anthropic client (optional, will be created if not provided)
            enable_llm: Whether to use LLM classification. If None, reads from
                       CHORUS_USE_LLM_CLASSIFIER environment variable (default: True)
            enable_fallback: Whether to fall back to regex if LLM fails (default: True)
            cache_ttl: Cache time-to-live in seconds (default: 3600)
            cache_max_size: Maximum cache entries (default: 2000)
        """
        super().__init__(client=client, **kwargs)

        # Resolve LLM enable flag
        if enable_llm is None:
            env_value = os.environ.get(LLM_CLASSIFIER_ENABLED_ENV, "true").lower()
            self.enable_llm = env_value in ("true", "1", "yes")
        else:
            self.enable_llm = enable_llm

        self.enable_fallback = enable_fallback
        self.cache_ttl = cache_ttl
        self.cache_max_size = cache_max_size

        # Initialize caches
        self._classification_cache: Dict[str, ClassificationCacheEntry] = {}

        # Fallback regex classifier (lazy initialization)
        self._regex_classifier: Optional[QueryClassifier] = None

        # Metrics
        self.llm_calls = 0
        self.llm_successes = 0
        self.llm_failures = 0
        self.regex_fallbacks = 0
        self.cache_hits = 0

    def get_system_prompt(self) -> str:
        """Return the classification system prompt."""
        return LLM_CLASSIFICATION_PROMPT

    @property
    def regex_classifier(self) -> QueryClassifier:
        """Lazy initialization of fallback regex classifier."""
        if self._regex_classifier is None:
            self._regex_classifier = QueryClassifier(client=self.client)
        return self._regex_classifier

    def _get_cache_key(self, query: str) -> str:
        """Generate cache key from normalized query."""
        normalized = query.lower().strip()
        return hashlib.sha256(normalized.encode()).hexdigest()[:32]

    def _get_from_cache(self, query: str) -> Optional[QueryClassification]:
        """
        Get classification from cache if available and not expired.

        Args:
            query: The query to look up

        Returns:
            Cached QueryClassification or None if not found/expired
        """
        cache_key = self._get_cache_key(query)

        if cache_key in self._classification_cache:
            entry = self._classification_cache[cache_key]

            # Check if expired
            if time.time() - entry.timestamp < self.cache_ttl:
                self.cache_hits += 1
                self.metrics.cache_hits += 1
                return entry.classification
            else:
                # Remove expired entry
                del self._classification_cache[cache_key]

        return None

    def _add_to_cache(self, query: str, classification: QueryClassification):
        """
        Add classification to cache.

        Args:
            query: The original query
            classification: The classification result
        """
        cache_key = self._get_cache_key(query)

        self._classification_cache[cache_key] = ClassificationCacheEntry(
            classification=classification,
            timestamp=time.time(),
            query_hash=cache_key
        )

        # Evict old entries if cache is too large
        if len(self._classification_cache) > self.cache_max_size:
            self._evict_oldest_entries()

    def _evict_oldest_entries(self, count: int = 100):
        """Remove oldest cache entries."""
        sorted_entries = sorted(
            self._classification_cache.items(),
            key=lambda x: x[1].timestamp
        )
        for key, _ in sorted_entries[:count]:
            del self._classification_cache[key]

    def _parse_llm_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse LLM response JSON, handling markdown code blocks.

        Args:
            response_text: Raw response from LLM

        Returns:
            Parsed JSON dictionary

        Raises:
            ValueError: If parsing fails
        """
        text = response_text.strip()

        # Handle markdown code blocks
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first and last lines (code block markers)
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        # Remove any leading "json" language identifier
        if text.startswith("json"):
            text = text[4:].strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse LLM response as JSON: {e}")

    def _validate_intent(self, intent: str) -> str:
        """
        Validate and normalize intent string.

        Args:
            intent: Raw intent from LLM

        Returns:
            Normalized intent string
        """
        valid_intents = {"lookup", "technical", "explore", "connect", "ambiguous"}
        normalized = intent.lower().strip()

        if normalized in valid_intents:
            return normalized

        # Try common variations
        intent_map = {
            "look_up": "lookup",
            "look-up": "lookup",
            "find": "lookup",
            "search": "lookup",
            "tech": "technical",
            "debug": "technical",
            "setup": "technical",
            "exploration": "explore",
            "analysis": "explore",
            "analyze": "explore",
            "connection": "connect",
            "collaborate": "connect",
            "collaboration": "connect",
            "unknown": "ambiguous",
            "unclear": "ambiguous",
        }

        return intent_map.get(normalized, "ambiguous")

    async def _classify_with_llm(self, query: str) -> QueryClassification:
        """
        Classify query using Claude Haiku.

        Args:
            query: User query to classify

        Returns:
            QueryClassification result

        Raises:
            Exception: If LLM call fails
        """
        start_time = time.time()
        self.llm_calls += 1

        try:
            response = self.client.messages.create(
                model=self.DEFAULT_MODEL,
                max_tokens=self.MAX_TOKENS,
                system=self.get_system_prompt(),
                messages=[
                    {
                        "role": "user",
                        "content": f"Classify this query:\n\n{query}"
                    }
                ]
            )

            latency_ms = (time.time() - start_time) * 1000
            self._track_usage(response, self.DEFAULT_MODEL, latency_ms)

            # Parse response
            response_text = response.content[0].text
            data = self._parse_llm_response(response_text)

            # Validate and extract fields
            intent = self._validate_intent(data.get("intent", "ambiguous"))
            confidence = float(data.get("confidence", 0.7))
            confidence = max(0.0, min(1.0, confidence))  # Clamp to [0, 1]

            requires_rag = data.get("requires_rag", True)
            if not isinstance(requires_rag, bool):
                requires_rag = str(requires_rag).lower() in ("true", "1", "yes")

            secondary_intent = data.get("secondary_intent")
            secondary_intents = []
            if secondary_intent and secondary_intent.lower() not in ("null", "none", ""):
                validated_secondary = self._validate_intent(secondary_intent)
                if validated_secondary != intent:
                    secondary_intents = [validated_secondary]

            reasoning = data.get("reasoning", "LLM classification")

            self.llm_successes += 1

            logger.debug(
                f"LLM classification: query='{query[:50]}...' intent={intent} "
                f"confidence={confidence:.2f} latency={latency_ms:.0f}ms"
            )

            return QueryClassification(
                intent=intent,
                confidence=confidence,
                secondary_intents=secondary_intents,
                requires_rag=requires_rag,
                reasoning=f"LLM: {reasoning}"
            )

        except Exception as e:
            self.llm_failures += 1
            self.metrics.errors += 1
            logger.warning(f"LLM classification failed: {e}")
            raise

    async def _classify_with_regex(self, query: str) -> QueryClassification:
        """
        Classify query using fallback regex classifier.

        Args:
            query: User query to classify

        Returns:
            QueryClassification result
        """
        self.regex_fallbacks += 1
        classification = await self.regex_classifier.classify(query)

        # Update reasoning to indicate fallback
        classification.reasoning = f"Regex fallback: {classification.reasoning}"

        return classification

    async def classify(self, query: str) -> QueryClassification:
        """
        Classify a query's intent.

        Uses LLM classification with caching and regex fallback.

        Args:
            query: The user's query

        Returns:
            QueryClassification with intent and metadata
        """
        # Check cache first
        cached = self._get_from_cache(query)
        if cached is not None:
            return cached

        self.metrics.classifier_calls += 1

        # Try LLM classification if enabled
        if self.enable_llm:
            try:
                classification = await self._classify_with_llm(query)
                self._add_to_cache(query, classification)
                return classification
            except Exception as e:
                logger.warning(f"LLM classification failed, trying fallback: {e}")

                if self.enable_fallback:
                    classification = await self._classify_with_regex(query)
                    self._add_to_cache(query, classification)
                    return classification
                else:
                    # Return ambiguous if no fallback
                    return QueryClassification(
                        intent="ambiguous",
                        confidence=0.5,
                        requires_rag=True,
                        reasoning=f"LLM failed, no fallback: {str(e)}"
                    )
        else:
            # LLM disabled, use regex directly
            classification = await self._classify_with_regex(query)
            self._add_to_cache(query, classification)
            return classification

    async def query(self, message: str, context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """
        Implement BaseAgent interface - wraps classify() in AgentResponse.
        """
        start_time = time.time()

        classification = await self.classify(message)

        latency_ms = (time.time() - start_time) * 1000

        return AgentResponse(
            content=json.dumps({
                "intent": classification.intent,
                "confidence": classification.confidence,
                "secondary_intents": classification.secondary_intents,
                "requires_rag": classification.requires_rag,
                "reasoning": classification.reasoning,
            }),
            agent_type=self.AGENT_TYPE,
            model_used=self.DEFAULT_MODEL,
            latency_ms=latency_ms,
            confidence=classification.confidence,
        )

    def get_metrics(self) -> dict:
        """Return classifier metrics."""
        base_metrics = super().get_metrics()
        base_metrics.update({
            "llm_calls": self.llm_calls,
            "llm_successes": self.llm_successes,
            "llm_failures": self.llm_failures,
            "llm_success_rate": self.llm_successes / max(1, self.llm_calls),
            "regex_fallbacks": self.regex_fallbacks,
            "cache_hits": self.cache_hits,
            "cache_size": len(self._classification_cache),
            "enable_llm": self.enable_llm,
            "enable_fallback": self.enable_fallback,
        })
        return base_metrics

    def clear_cache(self):
        """Clear the classification cache."""
        self._classification_cache.clear()
        logger.info("Classification cache cleared")


# =============================================================================
# Factory Function
# =============================================================================

def get_classifier(
    use_llm: Optional[bool] = None,
    client: Optional[anthropic.Anthropic] = None,
    **kwargs
) -> BaseAgent:
    """
    Factory function to get the appropriate classifier.

    Args:
        use_llm: Whether to use LLM classifier. If None, reads from environment.
        client: Anthropic client to use
        **kwargs: Additional arguments passed to classifier

    Returns:
        Either LLMQueryClassifier or QueryClassifier based on configuration
    """
    if use_llm is None:
        env_value = os.environ.get(LLM_CLASSIFIER_ENABLED_ENV, "true").lower()
        use_llm = env_value in ("true", "1", "yes")

    if use_llm:
        return LLMQueryClassifier(client=client, enable_llm=True, **kwargs)
    else:
        return QueryClassifier(client=client, **kwargs)


# =============================================================================
# Unit Tests
# =============================================================================

if __name__ == "__main__":
    import asyncio
    import unittest
    from unittest.mock import Mock, patch, MagicMock

    class TestLLMQueryClassifier(unittest.TestCase):
        """Unit tests for LLMQueryClassifier."""

        def setUp(self):
            """Set up test fixtures."""
            self.mock_client = Mock(spec=anthropic.Anthropic)

        def _create_mock_response(self, intent: str, confidence: float = 0.9):
            """Create a mock LLM response."""
            mock_response = Mock()
            mock_response.content = [Mock()]
            mock_response.content[0].text = json.dumps({
                "intent": intent,
                "confidence": confidence,
                "requires_rag": True,
                "secondary_intent": None,
                "reasoning": "Test classification"
            })
            mock_response.usage = Mock()
            mock_response.usage.cache_read_input_tokens = 0
            return mock_response

        def test_init_defaults(self):
            """Test default initialization."""
            with patch.dict(os.environ, {}, clear=True):
                classifier = LLMQueryClassifier(
                    client=self.mock_client,
                    enable_llm=True
                )
                self.assertTrue(classifier.enable_llm)
                self.assertTrue(classifier.enable_fallback)
                self.assertEqual(classifier.cache_ttl, DEFAULT_CACHE_TTL)

        def test_init_from_env(self):
            """Test initialization from environment variable."""
            with patch.dict(os.environ, {LLM_CLASSIFIER_ENABLED_ENV: "false"}):
                classifier = LLMQueryClassifier(client=self.mock_client)
                self.assertFalse(classifier.enable_llm)

        def test_cache_key_generation(self):
            """Test cache key generation is consistent."""
            classifier = LLMQueryClassifier(
                client=self.mock_client,
                enable_llm=True
            )
            key1 = classifier._get_cache_key("What is this?")
            key2 = classifier._get_cache_key("what is this?")
            key3 = classifier._get_cache_key("  What Is This?  ")

            # Should normalize to same key
            self.assertEqual(key1, key2)
            self.assertEqual(key2, key3)

        def test_parse_llm_response_json(self):
            """Test parsing plain JSON response."""
            classifier = LLMQueryClassifier(
                client=self.mock_client,
                enable_llm=True
            )

            response = '{"intent": "lookup", "confidence": 0.9}'
            result = classifier._parse_llm_response(response)

            self.assertEqual(result["intent"], "lookup")
            self.assertEqual(result["confidence"], 0.9)

        def test_parse_llm_response_markdown(self):
            """Test parsing JSON in markdown code block."""
            classifier = LLMQueryClassifier(
                client=self.mock_client,
                enable_llm=True
            )

            response = '```json\n{"intent": "technical", "confidence": 0.85}\n```'
            result = classifier._parse_llm_response(response)

            self.assertEqual(result["intent"], "technical")

        def test_validate_intent_valid(self):
            """Test intent validation with valid intents."""
            classifier = LLMQueryClassifier(
                client=self.mock_client,
                enable_llm=True
            )

            self.assertEqual(classifier._validate_intent("lookup"), "lookup")
            self.assertEqual(classifier._validate_intent("TECHNICAL"), "technical")
            self.assertEqual(classifier._validate_intent("Explore"), "explore")
            self.assertEqual(classifier._validate_intent("CONNECT"), "connect")

        def test_validate_intent_variations(self):
            """Test intent validation with common variations."""
            classifier = LLMQueryClassifier(
                client=self.mock_client,
                enable_llm=True
            )

            self.assertEqual(classifier._validate_intent("look-up"), "lookup")
            self.assertEqual(classifier._validate_intent("debug"), "technical")
            self.assertEqual(classifier._validate_intent("analysis"), "explore")
            self.assertEqual(classifier._validate_intent("collaborate"), "connect")

        def test_validate_intent_unknown(self):
            """Test intent validation with unknown intent."""
            classifier = LLMQueryClassifier(
                client=self.mock_client,
                enable_llm=True
            )

            self.assertEqual(classifier._validate_intent("unknown_intent"), "ambiguous")

        def test_caching(self):
            """Test that classification results are cached."""
            classifier = LLMQueryClassifier(
                client=self.mock_client,
                enable_llm=True
            )

            # Add to cache
            classification = QueryClassification(
                intent="lookup",
                confidence=0.9,
                requires_rag=True,
                reasoning="test"
            )
            classifier._add_to_cache("test query", classification)

            # Retrieve from cache
            cached = classifier._get_from_cache("test query")
            self.assertIsNotNone(cached)
            self.assertEqual(cached.intent, "lookup")

        def test_cache_expiration(self):
            """Test that expired cache entries are not returned."""
            classifier = LLMQueryClassifier(
                client=self.mock_client,
                enable_llm=True,
                cache_ttl=0  # Immediate expiration
            )

            classification = QueryClassification(
                intent="lookup",
                confidence=0.9,
                requires_rag=True,
                reasoning="test"
            )
            classifier._add_to_cache("test query", classification)

            # Should be expired immediately
            time.sleep(0.01)
            cached = classifier._get_from_cache("test query")
            self.assertIsNone(cached)

        def test_cache_eviction(self):
            """Test that old entries are evicted when cache is full."""
            classifier = LLMQueryClassifier(
                client=self.mock_client,
                enable_llm=True,
                cache_max_size=5
            )

            # Fill cache beyond max size
            for i in range(10):
                classification = QueryClassification(
                    intent="lookup",
                    confidence=0.9,
                    requires_rag=True,
                    reasoning=f"test {i}"
                )
                classifier._add_to_cache(f"query {i}", classification)

            # Cache should have been evicted
            self.assertLessEqual(len(classifier._classification_cache), 5)

    class TestLLMClassification(unittest.TestCase):
        """Test LLM classification with mocked API calls."""

        def setUp(self):
            self.mock_client = Mock(spec=anthropic.Anthropic)

        def _create_mock_response(self, intent: str, confidence: float = 0.9):
            mock_response = Mock()
            mock_response.content = [Mock()]
            mock_response.content[0].text = json.dumps({
                "intent": intent,
                "confidence": confidence,
                "requires_rag": True,
                "secondary_intent": None,
                "reasoning": "Test classification"
            })
            mock_response.usage = Mock()
            mock_response.usage.cache_read_input_tokens = 0
            return mock_response

        def test_lookup_queries(self):
            """Test classification of lookup queries."""
            queries = [
                "Who is working on the genomics project?",
                "What grants does Professor Smith have?",
                "List all publications from 2023",
                "When is the next lab meeting?",
                "Show me the budget for Project X",
            ]

            async def run_test():
                for query in queries:
                    self.mock_client.messages.create.return_value = \
                        self._create_mock_response("lookup", 0.95)

                    classifier = LLMQueryClassifier(
                        client=self.mock_client,
                        enable_llm=True
                    )
                    result = await classifier.classify(query)
                    self.assertEqual(result.intent, "lookup", f"Failed for query: {query}")

            asyncio.run(run_test())

        def test_technical_queries(self):
            """Test classification of technical queries."""
            queries = [
                "How do I access Midway?",
                "My Jupyter notebook won't connect",
                "I'm getting a permission denied error",
                "Help me set up conda environment",
                "Git push is failing",
            ]

            async def run_test():
                for query in queries:
                    self.mock_client.messages.create.return_value = \
                        self._create_mock_response("technical", 0.9)

                    classifier = LLMQueryClassifier(
                        client=self.mock_client,
                        enable_llm=True
                    )
                    classifier.clear_cache()  # Clear cache between queries
                    result = await classifier.classify(query)
                    self.assertEqual(result.intent, "technical", f"Failed for query: {query}")

            asyncio.run(run_test())

        def test_explore_queries(self):
            """Test classification of explore queries."""
            queries = [
                "Compare the approaches used in Project A vs Project B",
                "What are some ways we could improve our pipeline?",
                "Analyze the trends in our publication output",
                "What do you think about using transformers?",
                "What can you help me with?",
            ]

            async def run_test():
                for query in queries:
                    self.mock_client.messages.create.return_value = \
                        self._create_mock_response("explore", 0.85)

                    classifier = LLMQueryClassifier(
                        client=self.mock_client,
                        enable_llm=True
                    )
                    classifier.clear_cache()
                    result = await classifier.classify(query)
                    self.assertEqual(result.intent, "explore", f"Failed for query: {query}")

            asyncio.run(run_test())

        def test_connect_queries(self):
            """Test classification of connect queries."""
            queries = [
                "Who else is working on single-cell analysis?",
                "Connect me with someone who knows about CRISPR",
                "How does Dr. Jones' work relate to genomics?",
                "Who should I talk to about machine learning?",
                "Are there collaborations between our lab and MIT?",
            ]

            async def run_test():
                for query in queries:
                    self.mock_client.messages.create.return_value = \
                        self._create_mock_response("connect", 0.88)

                    classifier = LLMQueryClassifier(
                        client=self.mock_client,
                        enable_llm=True
                    )
                    classifier.clear_cache()
                    result = await classifier.classify(query)
                    self.assertEqual(result.intent, "connect", f"Failed for query: {query}")

            asyncio.run(run_test())

        def test_ambiguous_queries(self):
            """Test classification of ambiguous queries."""
            queries = [
                "hmm",
                "...",
                "thing",
            ]

            async def run_test():
                for query in queries:
                    self.mock_client.messages.create.return_value = \
                        self._create_mock_response("ambiguous", 0.4)

                    classifier = LLMQueryClassifier(
                        client=self.mock_client,
                        enable_llm=True
                    )
                    classifier.clear_cache()
                    result = await classifier.classify(query)
                    self.assertEqual(result.intent, "ambiguous", f"Failed for query: {query}")

            asyncio.run(run_test())

    class TestFallbackBehavior(unittest.TestCase):
        """Test fallback to regex classifier."""

        def test_fallback_on_llm_error(self):
            """Test fallback when LLM call fails."""
            mock_client = Mock(spec=anthropic.Anthropic)
            mock_client.messages.create.side_effect = Exception("API Error")

            async def run_test():
                classifier = LLMQueryClassifier(
                    client=mock_client,
                    enable_llm=True,
                    enable_fallback=True
                )

                result = await classifier.classify("What grants does Professor Smith have?")

                # Should have fallen back to regex
                self.assertEqual(classifier.regex_fallbacks, 1)
                self.assertIn("fallback", result.reasoning.lower())

            asyncio.run(run_test())

        def test_no_fallback_when_disabled(self):
            """Test that no fallback occurs when disabled."""
            mock_client = Mock(spec=anthropic.Anthropic)
            mock_client.messages.create.side_effect = Exception("API Error")

            async def run_test():
                classifier = LLMQueryClassifier(
                    client=mock_client,
                    enable_llm=True,
                    enable_fallback=False
                )

                result = await classifier.classify("What is this?")

                # Should return ambiguous without fallback
                self.assertEqual(result.intent, "ambiguous")
                self.assertEqual(classifier.regex_fallbacks, 0)

            asyncio.run(run_test())

        def test_regex_only_mode(self):
            """Test that regex is used when LLM is disabled."""
            mock_client = Mock(spec=anthropic.Anthropic)

            async def run_test():
                classifier = LLMQueryClassifier(
                    client=mock_client,
                    enable_llm=False
                )

                result = await classifier.classify("What grants does Professor Smith have?")

                # Should use regex directly
                self.assertEqual(classifier.llm_calls, 0)
                self.assertIn("regex", result.reasoning.lower())

            asyncio.run(run_test())

    class TestMetrics(unittest.TestCase):
        """Test metrics tracking."""

        def test_metrics_tracking(self):
            """Test that metrics are properly tracked."""
            mock_client = Mock(spec=anthropic.Anthropic)
            mock_response = Mock()
            mock_response.content = [Mock()]
            mock_response.content[0].text = json.dumps({
                "intent": "lookup",
                "confidence": 0.9,
                "requires_rag": True,
                "secondary_intent": None,
                "reasoning": "test"
            })
            mock_response.usage = Mock()
            mock_response.usage.cache_read_input_tokens = 0
            mock_client.messages.create.return_value = mock_response

            async def run_test():
                classifier = LLMQueryClassifier(
                    client=mock_client,
                    enable_llm=True
                )

                await classifier.classify("test query 1")
                await classifier.classify("test query 2")
                await classifier.classify("test query 1")  # Cache hit

                metrics = classifier.get_metrics()

                self.assertEqual(metrics["llm_calls"], 2)
                self.assertEqual(metrics["llm_successes"], 2)
                self.assertEqual(metrics["cache_hits"], 1)

            asyncio.run(run_test())

    class TestFactoryFunction(unittest.TestCase):
        """Test the factory function."""

        def test_get_classifier_llm(self):
            """Test factory returns LLM classifier when enabled."""
            mock_client = Mock(spec=anthropic.Anthropic)

            classifier = get_classifier(use_llm=True, client=mock_client)
            self.assertIsInstance(classifier, LLMQueryClassifier)

        def test_get_classifier_regex(self):
            """Test factory returns regex classifier when disabled."""
            mock_client = Mock(spec=anthropic.Anthropic)

            classifier = get_classifier(use_llm=False, client=mock_client)
            self.assertIsInstance(classifier, QueryClassifier)

        def test_get_classifier_from_env(self):
            """Test factory reads from environment."""
            mock_client = Mock(spec=anthropic.Anthropic)

            with patch.dict(os.environ, {LLM_CLASSIFIER_ENABLED_ENV: "true"}):
                classifier = get_classifier(client=mock_client)
                self.assertIsInstance(classifier, LLMQueryClassifier)

            with patch.dict(os.environ, {LLM_CLASSIFIER_ENABLED_ENV: "false"}):
                classifier = get_classifier(client=mock_client)
                self.assertIsInstance(classifier, QueryClassifier)

    # Run tests
    print("Running LLMQueryClassifier unit tests...")
    unittest.main(verbosity=2, exit=False)
