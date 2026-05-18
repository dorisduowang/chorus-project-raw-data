"""
Unit tests for LLMQueryClassifier.

This module tests the LLM-based query classification system, including:
- Basic classification functionality
- Caching behavior
- Fallback to regex classifier
- Metrics tracking
- Factory function
"""

import asyncio
import json
import os
import time
import unittest
from unittest.mock import Mock, patch

import anthropic

from agents import LLMQueryClassifier, QueryClassifier, QueryClassification, get_classifier
from agents.llm_classifier import (
    LLM_CLASSIFIER_ENABLED_ENV,
    DEFAULT_CACHE_TTL,
)


class TestLLMQueryClassifierInit(unittest.TestCase):
    """Test LLMQueryClassifier initialization."""

    def setUp(self):
        self.mock_client = Mock(spec=anthropic.Anthropic)

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
            self.assertEqual(classifier.AGENT_TYPE, "llm_classifier")

    def test_init_from_env_enabled(self):
        """Test initialization with LLM enabled via environment variable."""
        with patch.dict(os.environ, {LLM_CLASSIFIER_ENABLED_ENV: "true"}):
            classifier = LLMQueryClassifier(client=self.mock_client)
            self.assertTrue(classifier.enable_llm)

    def test_init_from_env_disabled(self):
        """Test initialization with LLM disabled via environment variable."""
        with patch.dict(os.environ, {LLM_CLASSIFIER_ENABLED_ENV: "false"}):
            classifier = LLMQueryClassifier(client=self.mock_client)
            self.assertFalse(classifier.enable_llm)

    def test_init_explicit_override(self):
        """Test that explicit enable_llm overrides environment."""
        with patch.dict(os.environ, {LLM_CLASSIFIER_ENABLED_ENV: "false"}):
            classifier = LLMQueryClassifier(
                client=self.mock_client,
                enable_llm=True  # Explicit override
            )
            self.assertTrue(classifier.enable_llm)


class TestCacheKeyGeneration(unittest.TestCase):
    """Test cache key generation."""

    def setUp(self):
        self.mock_client = Mock(spec=anthropic.Anthropic)
        self.classifier = LLMQueryClassifier(
            client=self.mock_client,
            enable_llm=True
        )

    def test_cache_key_consistent(self):
        """Test cache key is consistent for same query."""
        key1 = self.classifier._get_cache_key("What is this?")
        key2 = self.classifier._get_cache_key("What is this?")
        self.assertEqual(key1, key2)

    def test_cache_key_case_insensitive(self):
        """Test cache key is case insensitive."""
        key1 = self.classifier._get_cache_key("What is this?")
        key2 = self.classifier._get_cache_key("what is this?")
        key3 = self.classifier._get_cache_key("WHAT IS THIS?")
        self.assertEqual(key1, key2)
        self.assertEqual(key2, key3)

    def test_cache_key_strips_whitespace(self):
        """Test cache key ignores leading/trailing whitespace."""
        key1 = self.classifier._get_cache_key("What is this?")
        key2 = self.classifier._get_cache_key("  What is this?  ")
        key3 = self.classifier._get_cache_key("\n\tWhat is this?\n\t")
        self.assertEqual(key1, key2)
        self.assertEqual(key2, key3)

    def test_cache_key_different_queries(self):
        """Test different queries get different cache keys."""
        key1 = self.classifier._get_cache_key("What is this?")
        key2 = self.classifier._get_cache_key("Who is that?")
        self.assertNotEqual(key1, key2)


class TestResponseParsing(unittest.TestCase):
    """Test LLM response parsing."""

    def setUp(self):
        self.mock_client = Mock(spec=anthropic.Anthropic)
        self.classifier = LLMQueryClassifier(
            client=self.mock_client,
            enable_llm=True
        )

    def test_parse_plain_json(self):
        """Test parsing plain JSON response."""
        response = '{"intent": "lookup", "confidence": 0.9}'
        result = self.classifier._parse_llm_response(response)
        self.assertEqual(result["intent"], "lookup")
        self.assertEqual(result["confidence"], 0.9)

    def test_parse_json_with_whitespace(self):
        """Test parsing JSON with leading/trailing whitespace."""
        response = '  \n{"intent": "lookup", "confidence": 0.9}\n  '
        result = self.classifier._parse_llm_response(response)
        self.assertEqual(result["intent"], "lookup")

    def test_parse_markdown_code_block(self):
        """Test parsing JSON in markdown code block."""
        response = '```json\n{"intent": "technical", "confidence": 0.85}\n```'
        result = self.classifier._parse_llm_response(response)
        self.assertEqual(result["intent"], "technical")

    def test_parse_markdown_without_language(self):
        """Test parsing JSON in markdown code block without language."""
        response = '```\n{"intent": "explore", "confidence": 0.75}\n```'
        result = self.classifier._parse_llm_response(response)
        self.assertEqual(result["intent"], "explore")

    def test_parse_invalid_json(self):
        """Test that invalid JSON raises ValueError."""
        response = "not valid json"
        with self.assertRaises(ValueError):
            self.classifier._parse_llm_response(response)


class TestIntentValidation(unittest.TestCase):
    """Test intent validation."""

    def setUp(self):
        self.mock_client = Mock(spec=anthropic.Anthropic)
        self.classifier = LLMQueryClassifier(
            client=self.mock_client,
            enable_llm=True
        )

    def test_valid_intents(self):
        """Test that valid intents are returned as-is."""
        valid_intents = ["lookup", "technical", "explore", "connect", "ambiguous"]
        for intent in valid_intents:
            self.assertEqual(self.classifier._validate_intent(intent), intent)

    def test_case_normalization(self):
        """Test that intent case is normalized."""
        self.assertEqual(self.classifier._validate_intent("LOOKUP"), "lookup")
        self.assertEqual(self.classifier._validate_intent("Technical"), "technical")
        self.assertEqual(self.classifier._validate_intent("EXPLORE"), "explore")

    def test_common_variations(self):
        """Test that common intent variations are mapped correctly."""
        variations = {
            "look-up": "lookup",
            "look_up": "lookup",
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
        for variation, expected in variations.items():
            self.assertEqual(
                self.classifier._validate_intent(variation),
                expected,
                f"Failed for variation: {variation}"
            )

    def test_unknown_intent(self):
        """Test that unknown intents default to ambiguous."""
        self.assertEqual(self.classifier._validate_intent("random_thing"), "ambiguous")
        self.assertEqual(self.classifier._validate_intent(""), "ambiguous")


class TestCaching(unittest.TestCase):
    """Test caching behavior."""

    def setUp(self):
        self.mock_client = Mock(spec=anthropic.Anthropic)

    def test_cache_add_and_retrieve(self):
        """Test adding and retrieving from cache."""
        classifier = LLMQueryClassifier(
            client=self.mock_client,
            enable_llm=True
        )

        classification = QueryClassification(
            intent="lookup",
            confidence=0.9,
            requires_rag=True,
            reasoning="test"
        )
        classifier._add_to_cache("test query", classification)

        cached = classifier._get_from_cache("test query")
        self.assertIsNotNone(cached)
        self.assertEqual(cached.intent, "lookup")
        self.assertEqual(cached.confidence, 0.9)

    def test_cache_miss(self):
        """Test cache miss returns None."""
        classifier = LLMQueryClassifier(
            client=self.mock_client,
            enable_llm=True
        )

        cached = classifier._get_from_cache("nonexistent query")
        self.assertIsNone(cached)

    def test_cache_expiration(self):
        """Test that expired entries are not returned."""
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

    def test_clear_cache(self):
        """Test clearing the cache."""
        classifier = LLMQueryClassifier(
            client=self.mock_client,
            enable_llm=True
        )

        for i in range(5):
            classification = QueryClassification(
                intent="lookup",
                confidence=0.9,
                requires_rag=True,
                reasoning=f"test {i}"
            )
            classifier._add_to_cache(f"query {i}", classification)

        self.assertEqual(len(classifier._classification_cache), 5)
        classifier.clear_cache()
        self.assertEqual(len(classifier._classification_cache), 0)


class TestLLMClassification(unittest.TestCase):
    """Test LLM classification with mocked API calls."""

    def setUp(self):
        self.mock_client = Mock(spec=anthropic.Anthropic)

    def _create_mock_response(
        self,
        intent: str,
        confidence: float = 0.9,
        requires_rag: bool = True,
        secondary_intent: str = None,
        reasoning: str = "Test classification"
    ):
        """Create a mock LLM response."""
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = json.dumps({
            "intent": intent,
            "confidence": confidence,
            "requires_rag": requires_rag,
            "secondary_intent": secondary_intent,
            "reasoning": reasoning
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
                classifier.clear_cache()
                result = await classifier.classify(query)
                self.assertEqual(result.intent, "lookup", f"Failed for: {query}")

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
                classifier.clear_cache()
                result = await classifier.classify(query)
                self.assertEqual(result.intent, "technical", f"Failed for: {query}")

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
                self.assertEqual(result.intent, "explore", f"Failed for: {query}")

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
                self.assertEqual(result.intent, "connect", f"Failed for: {query}")

        asyncio.run(run_test())

    def test_secondary_intent(self):
        """Test that secondary intents are captured."""
        async def run_test():
            self.mock_client.messages.create.return_value = self._create_mock_response(
                "lookup",
                confidence=0.8,
                secondary_intent="connect"
            )

            classifier = LLMQueryClassifier(
                client=self.mock_client,
                enable_llm=True
            )
            result = await classifier.classify("Who works on genomics and what are they doing?")

            self.assertEqual(result.intent, "lookup")
            self.assertIn("connect", result.secondary_intents)

        asyncio.run(run_test())

    def test_confidence_clamping(self):
        """Test that confidence is clamped to [0, 1]."""
        async def run_test():
            # Test confidence > 1
            self.mock_client.messages.create.return_value = self._create_mock_response(
                "lookup",
                confidence=1.5
            )

            classifier = LLMQueryClassifier(
                client=self.mock_client,
                enable_llm=True
            )
            result = await classifier.classify("test query")
            self.assertLessEqual(result.confidence, 1.0)

            # Test confidence < 0
            classifier.clear_cache()
            self.mock_client.messages.create.return_value = self._create_mock_response(
                "lookup",
                confidence=-0.5
            )
            result = await classifier.classify("test query 2")
            self.assertGreaterEqual(result.confidence, 0.0)

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
            self.assertEqual(metrics["llm_failures"], 0)
            self.assertGreater(metrics["llm_success_rate"], 0)

        asyncio.run(run_test())

    def test_failure_metrics(self):
        """Test that failure metrics are tracked."""
        mock_client = Mock(spec=anthropic.Anthropic)
        mock_client.messages.create.side_effect = Exception("API Error")

        async def run_test():
            classifier = LLMQueryClassifier(
                client=mock_client,
                enable_llm=True,
                enable_fallback=True
            )

            await classifier.classify("test query")

            metrics = classifier.get_metrics()
            self.assertEqual(metrics["llm_failures"], 1)
            self.assertEqual(metrics["regex_fallbacks"], 1)

        asyncio.run(run_test())


class TestFactoryFunction(unittest.TestCase):
    """Test the get_classifier factory function."""

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

    def test_get_classifier_from_env_true(self):
        """Test factory reads true from environment."""
        mock_client = Mock(spec=anthropic.Anthropic)

        with patch.dict(os.environ, {LLM_CLASSIFIER_ENABLED_ENV: "true"}):
            classifier = get_classifier(client=mock_client)
            self.assertIsInstance(classifier, LLMQueryClassifier)

    def test_get_classifier_from_env_false(self):
        """Test factory reads false from environment."""
        mock_client = Mock(spec=anthropic.Anthropic)

        with patch.dict(os.environ, {LLM_CLASSIFIER_ENABLED_ENV: "false"}):
            classifier = get_classifier(client=mock_client)
            self.assertIsInstance(classifier, QueryClassifier)


class TestQueryMethod(unittest.TestCase):
    """Test the query() method (BaseAgent interface)."""

    def test_query_returns_agent_response(self):
        """Test that query() returns proper AgentResponse."""
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

            response = await classifier.query("What is this?")

            self.assertEqual(response.agent_type, "llm_classifier")
            self.assertGreater(response.latency_ms, 0)

            # Parse content as JSON
            content = json.loads(response.content)
            self.assertEqual(content["intent"], "lookup")
            self.assertEqual(content["confidence"], 0.9)

        asyncio.run(run_test())


if __name__ == "__main__":
    unittest.main(verbosity=2)
