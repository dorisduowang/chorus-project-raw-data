"""
Unit tests for QueryPlanner and PlanExecutor.

Tests the LLM-driven query planning system including:
- Simple vs complex query detection
- Plan creation with steps and fallbacks
- Plan execution with retry logic
- Result quality evaluation
- Caching behavior
"""

import asyncio
import json
import os
import time
import unittest
from unittest.mock import Mock, patch, AsyncMock

import anthropic

from agents.query_planner import (
    QueryPlanner,
    QueryPlan,
    QueryStep,
    QueryAction,
    FallbackStrategy,
    StepResult,
)
from agents.plan_executor import (
    PlanExecutor,
    ExecutionResult,
    execute_with_fallback,
)


class TestQueryPlannerInit(unittest.TestCase):
    """Test QueryPlanner initialization."""

    def setUp(self):
        self.mock_client = Mock(spec=anthropic.Anthropic)

    def test_init_defaults(self):
        """Test default initialization."""
        planner = QueryPlanner(client=self.mock_client)
        self.assertTrue(planner.enable_caching)
        self.assertEqual(planner.cache_ttl, 3600)
        self.assertEqual(planner.metrics.plans_created, 0)

    def test_init_custom_settings(self):
        """Test initialization with custom settings."""
        planner = QueryPlanner(
            client=self.mock_client,
            enable_caching=False,
            cache_ttl=1800,
        )
        self.assertFalse(planner.enable_caching)
        self.assertEqual(planner.cache_ttl, 1800)


class TestSimpleQueryDetection(unittest.TestCase):
    """Test simple query detection for fast path."""

    def setUp(self):
        self.mock_client = Mock(spec=anthropic.Anthropic)
        self.planner = QueryPlanner(client=self.mock_client)

    def test_simple_who_is_query(self):
        """Test that 'Who is X?' is detected as simple."""
        self.assertTrue(self.planner._is_simple_query("Who is James Evans?"))
        self.assertTrue(self.planner._is_simple_query("Who's Hyejin Youn?"))

    def test_simple_what_is_query(self):
        """Test that 'What is X?' is detected as simple."""
        self.assertTrue(self.planner._is_simple_query("What is CHORUS?"))

    def test_short_queries_simple(self):
        """Test that short queries without complexity indicators are simple."""
        self.assertTrue(self.planner._is_simple_query("James Evans"))
        self.assertTrue(self.planner._is_simple_query("Find the budget"))

    def test_complex_comparison_queries(self):
        """Test that comparison queries are detected as complex."""
        self.assertFalse(self.planner._is_simple_query(
            "Compare James Evans and Hyejin Youn's research"
        ))
        self.assertFalse(self.planner._is_simple_query(
            "What is the relationship between Project A and Project B?"
        ))

    def test_complex_multiple_entities(self):
        """Test that queries with multiple entities are complex."""
        self.assertFalse(self.planner._is_simple_query(
            "Find all publications from James Evans and Hyejin Youn"
        ))
        self.assertFalse(self.planner._is_simple_query(
            "Who works on NLP and computer vision?"
        ))

    def test_long_queries_complex(self):
        """Test that long queries are detected as complex."""
        long_query = "I want to find researchers who work on machine learning " \
                    "and also have experience with bioinformatics and genomics"
        self.assertFalse(self.planner._is_simple_query(long_query))


class TestSimplePlanCreation(unittest.TestCase):
    """Test simple plan creation (fast path)."""

    def setUp(self):
        self.mock_client = Mock(spec=anthropic.Anthropic)
        self.planner = QueryPlanner(client=self.mock_client)

    def test_simple_plan_structure(self):
        """Test that simple plans have correct structure."""
        plan = self.planner._create_simple_plan("Who is James Evans?")

        self.assertEqual(plan.original_query, "Who is James Evans?")
        self.assertEqual(plan.complexity, "simple")
        self.assertFalse(plan.requires_planning)
        self.assertEqual(len(plan.steps), 1)
        self.assertGreater(len(plan.fallback_strategies), 0)

    def test_simple_plan_lookup_intent(self):
        """Test that person lookup gets correct intent."""
        plan = self.planner._create_simple_plan("Who is James Evans?")
        self.assertEqual(plan.intent, "lookup")
        self.assertEqual(plan.steps[0].action, QueryAction.SEARCH_REGISTRY.value)

    def test_simple_plan_technical_intent(self):
        """Test that technical queries get correct intent."""
        plan = self.planner._create_simple_plan("How do I setup conda?")
        self.assertEqual(plan.intent, "technical")
        self.assertEqual(plan.steps[0].action, QueryAction.SEARCH_DOCS.value)

    def test_simple_plan_explore_intent(self):
        """Test that meta queries get explore intent."""
        plan = self.planner._create_simple_plan("What can you do?")
        self.assertEqual(plan.intent, "explore")
        self.assertEqual(plan.steps[0].action, QueryAction.DIRECT_ANSWER.value)

    def test_simple_plan_has_fallbacks(self):
        """Test that simple plans include fallback strategies."""
        plan = self.planner._create_simple_plan("Who is James Evans?")
        self.assertIn(FallbackStrategy.EXPANSION.value, plan.fallback_strategies)
        self.assertIn(FallbackStrategy.WEB_FALLBACK.value, plan.fallback_strategies)


class TestPlanCaching(unittest.TestCase):
    """Test plan caching behavior."""

    def setUp(self):
        self.mock_client = Mock(spec=anthropic.Anthropic)

    def test_cache_add_and_retrieve(self):
        """Test adding and retrieving from cache."""
        planner = QueryPlanner(client=self.mock_client, enable_caching=True)

        plan = planner._create_simple_plan("Who is James Evans?")
        planner._add_to_cache("Who is James Evans?", plan)

        cached = planner._get_from_cache("Who is James Evans?")
        self.assertIsNotNone(cached)
        self.assertEqual(cached.original_query, plan.original_query)

    def test_cache_case_insensitive(self):
        """Test that cache lookup is case insensitive."""
        planner = QueryPlanner(client=self.mock_client, enable_caching=True)

        plan = planner._create_simple_plan("Who is James Evans?")
        planner._add_to_cache("Who is James Evans?", plan)

        cached = planner._get_from_cache("who is james evans?")
        self.assertIsNotNone(cached)

    def test_cache_miss(self):
        """Test cache miss returns None."""
        planner = QueryPlanner(client=self.mock_client, enable_caching=True)

        cached = planner._get_from_cache("nonexistent query")
        self.assertIsNone(cached)

    def test_cache_expiration(self):
        """Test that expired entries are not returned."""
        planner = QueryPlanner(
            client=self.mock_client,
            enable_caching=True,
            cache_ttl=0  # Immediate expiration
        )

        plan = planner._create_simple_plan("Who is James Evans?")
        planner._add_to_cache("Who is James Evans?", plan)

        time.sleep(0.01)
        cached = planner._get_from_cache("Who is James Evans?")
        self.assertIsNone(cached)

    def test_cache_disabled(self):
        """Test that caching can be disabled."""
        planner = QueryPlanner(
            client=self.mock_client,
            enable_caching=False
        )

        plan = planner._create_simple_plan("Who is James Evans?")
        planner._add_to_cache("Who is James Evans?", plan)

        cached = planner._get_from_cache("Who is James Evans?")
        self.assertIsNone(cached)


class TestLLMPlanCreation(unittest.TestCase):
    """Test LLM-based plan creation for complex queries."""

    def setUp(self):
        self.mock_client = Mock(spec=anthropic.Anthropic)

    def _create_mock_response(self, plan_data: dict):
        """Create a mock LLM response."""
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = json.dumps(plan_data)
        return mock_response

    def test_llm_plan_creation(self):
        """Test that complex queries use LLM planning."""
        plan_data = {
            "intent": "explore",
            "complexity": "complex",
            "reasoning": "Comparison query requiring multiple lookups",
            "confidence": 0.85,
            "steps": [
                {
                    "step_id": 1,
                    "action": "search_registry",
                    "query": "James Evans",
                    "parameters": {"type": "person"},
                    "depends_on": None,
                    "description": "Get James Evans profile"
                },
                {
                    "step_id": 2,
                    "action": "search_registry",
                    "query": "Hyejin Youn",
                    "parameters": {"type": "person"},
                    "depends_on": None,
                    "description": "Get Hyejin Youn profile"
                }
            ],
            "fallback_strategies": ["decomposition", "web_fallback"]
        }

        self.mock_client.messages.create.return_value = self._create_mock_response(plan_data)

        async def run_test():
            planner = QueryPlanner(client=self.mock_client)
            plan = await planner._create_plan_with_llm(
                "Compare James Evans and Hyejin Youn's research"
            )

            self.assertEqual(plan.intent, "explore")
            self.assertEqual(plan.complexity, "complex")
            self.assertEqual(len(plan.steps), 2)
            self.assertTrue(plan.requires_planning)

        asyncio.run(run_test())

    def test_llm_plan_with_dependencies(self):
        """Test that step dependencies are preserved."""
        plan_data = {
            "intent": "explore",
            "complexity": "complex",
            "reasoning": "Multi-step with synthesis",
            "confidence": 0.8,
            "steps": [
                {
                    "step_id": 1,
                    "action": "search_registry",
                    "query": "James Evans",
                    "parameters": {},
                    "depends_on": None,
                    "description": "Step 1"
                },
                {
                    "step_id": 2,
                    "action": "combine",
                    "query": "Synthesize results",
                    "parameters": {"sources": [1]},
                    "depends_on": 1,
                    "description": "Step 2"
                }
            ],
            "fallback_strategies": []
        }

        self.mock_client.messages.create.return_value = self._create_mock_response(plan_data)

        async def run_test():
            planner = QueryPlanner(client=self.mock_client)
            plan = await planner._create_plan_with_llm("Complex query")

            self.assertEqual(plan.steps[1].depends_on, 1)

        asyncio.run(run_test())

    def test_llm_fallback_on_error(self):
        """Test that LLM errors fall back to simple plan."""
        self.mock_client.messages.create.side_effect = Exception("API Error")

        async def run_test():
            planner = QueryPlanner(client=self.mock_client)
            plan = await planner._create_plan_with_llm("Complex query")

            # Should fall back to simple plan
            self.assertEqual(plan.complexity, "simple")

        asyncio.run(run_test())


class TestPlanExecutorInit(unittest.TestCase):
    """Test PlanExecutor initialization."""

    def test_init_defaults(self):
        """Test default initialization."""
        executor = PlanExecutor()
        self.assertEqual(executor.metrics.executions, 0)
        self.assertIsNone(executor._search_registry)

    def test_init_with_search_functions(self):
        """Test initialization with search functions."""
        async def mock_search(q, p):
            return {"results": []}

        executor = PlanExecutor(
            search_registry=mock_search,
            search_docs=mock_search,
        )
        self.assertIsNotNone(executor._search_registry)
        self.assertIsNotNone(executor._search_docs)


class TestQualityEvaluation(unittest.TestCase):
    """Test result quality evaluation."""

    def setUp(self):
        self.executor = PlanExecutor()
        self.test_step = QueryStep(
            step_id=1,
            action="search_registry",
            query="test"
        )

    def test_quality_none_result(self):
        """Test that None results get 0 quality."""
        quality = self.executor._evaluate_quality("test", None, self.test_step)
        self.assertEqual(quality, 0.0)

    def test_quality_error_result(self):
        """Test that error results get 0 quality."""
        quality = self.executor._evaluate_quality(
            "test",
            {"error": "Something failed"},
            self.test_step
        )
        self.assertEqual(quality, 0.0)

    def test_quality_structured_results(self):
        """Test that structured results get high quality."""
        result = {
            "structured_results": [
                {"type": "person", "data": {"name": "James Evans"}}
            ]
        }
        quality = self.executor._evaluate_quality("test", result, self.test_step)
        self.assertGreater(quality, 0.7)

    def test_quality_semantic_results(self):
        """Test that semantic results get moderate quality."""
        result = {
            "semantic_results": [
                {"text": "Some relevant content here"}
            ]
        }
        quality = self.executor._evaluate_quality("test", result, self.test_step)
        self.assertGreater(quality, 0.3)
        self.assertLess(quality, 0.9)

    def test_quality_empty_results(self):
        """Test that empty results get low quality."""
        quality = self.executor._evaluate_quality(
            "test",
            {"structured_results": [], "semantic_results": []},
            self.test_step
        )
        self.assertLess(quality, 0.3)

    def test_quality_direct_answer(self):
        """Test that direct answer gets high quality."""
        quality = self.executor._evaluate_quality(
            "test",
            {"direct_answer": True, "query": "What can you do?"},
            self.test_step
        )
        self.assertGreater(quality, 0.8)


class TestStepExecution(unittest.TestCase):
    """Test individual step execution."""

    def test_execute_registry_search(self):
        """Test executing a registry search step."""
        async def mock_search(q, p):
            return {"structured_results": [{"type": "person", "data": {"name": "Test"}}]}

        executor = PlanExecutor(search_registry=mock_search)

        step = QueryStep(
            step_id=1,
            action=QueryAction.SEARCH_REGISTRY.value,
            query="test query",
            parameters={}
        )

        plan = QueryPlan(
            original_query="test query",
            intent="lookup",
            complexity="simple",
            steps=[step]
        )

        async def run_test():
            result = await executor._execute_step(step, plan)
            self.assertTrue(result.success)
            self.assertGreater(result.quality_score, 0.5)

        asyncio.run(run_test())

    def test_execute_step_without_function(self):
        """Test that missing search function returns error."""
        executor = PlanExecutor()  # No search functions

        step = QueryStep(
            step_id=1,
            action=QueryAction.SEARCH_REGISTRY.value,
            query="test query"
        )

        plan = QueryPlan(
            original_query="test query",
            intent="lookup",
            complexity="simple",
            steps=[step]
        )

        async def run_test():
            result = await executor._execute_step(step, plan)
            self.assertFalse(result.success)
            self.assertEqual(result.quality_score, 0.0)

        asyncio.run(run_test())


class TestDependencyOrdering(unittest.TestCase):
    """Test step dependency ordering."""

    def setUp(self):
        self.executor = PlanExecutor()

    def test_sort_no_dependencies(self):
        """Test sorting steps with no dependencies."""
        steps = [
            QueryStep(step_id=1, action="search_registry", query="a"),
            QueryStep(step_id=2, action="search_registry", query="b"),
            QueryStep(step_id=3, action="search_registry", query="c"),
        ]

        sorted_steps = self.executor._sort_steps_by_dependency(steps)
        self.assertEqual(len(sorted_steps), 3)

    def test_sort_with_dependencies(self):
        """Test that dependencies come before dependents."""
        steps = [
            QueryStep(step_id=3, action="combine", query="c", depends_on=2),
            QueryStep(step_id=1, action="search_registry", query="a"),
            QueryStep(step_id=2, action="search_registry", query="b", depends_on=1),
        ]

        sorted_steps = self.executor._sort_steps_by_dependency(steps)

        # Find positions
        step_positions = {s.step_id: i for i, s in enumerate(sorted_steps)}

        # Step 1 should come before step 2
        self.assertLess(step_positions[1], step_positions[2])
        # Step 2 should come before step 3
        self.assertLess(step_positions[2], step_positions[3])


class TestPlanExecution(unittest.TestCase):
    """Test full plan execution."""

    def test_execute_simple_plan(self):
        """Test executing a simple plan."""
        async def mock_search(q, p):
            return {"structured_results": [{"type": "person", "data": {"name": "Test"}}]}

        executor = PlanExecutor(search_registry=mock_search)

        plan = QueryPlan(
            original_query="Who is Test?",
            intent="lookup",
            complexity="simple",
            steps=[QueryStep(
                step_id=1,
                action=QueryAction.SEARCH_REGISTRY.value,
                query="Test"
            )],
            fallback_strategies=[],
        )

        async def run_test():
            result = await executor.execute(plan)
            self.assertTrue(result.success)
            self.assertEqual(result.steps_executed, 1)
            self.assertEqual(result.fallbacks_used, 0)

        asyncio.run(run_test())

    def test_execute_with_fallback(self):
        """Test that fallbacks are triggered on low quality."""
        call_count = [0]

        async def mock_search(q, p):
            call_count[0] += 1
            if call_count[0] == 1:
                return {"error": "Not found"}
            return {"structured_results": [{"type": "person", "data": {"name": "Test"}}]}

        async def mock_web_search(q, p):
            return {"results": [{"title": "Web Result"}]}

        executor = PlanExecutor(
            search_registry=mock_search,
            web_search=mock_web_search
        )

        plan = QueryPlan(
            original_query="Test query",
            intent="lookup",
            complexity="simple",
            steps=[QueryStep(
                step_id=1,
                action=QueryAction.SEARCH_REGISTRY.value,
                query="Test"
            )],
            fallback_strategies=[FallbackStrategy.EXPANSION.value],
            max_retries=2,
        )

        # Set up mock planner for reformulation
        mock_planner = Mock()
        mock_planner.generate_reformulation = AsyncMock(return_value="reformulated query")
        executor.set_planner(mock_planner)

        async def run_test():
            result = await executor.execute(plan)
            # Fallback should have been triggered
            self.assertGreater(result.fallbacks_used, 0)

        asyncio.run(run_test())


class TestMetrics(unittest.TestCase):
    """Test metrics tracking."""

    def test_planner_metrics(self):
        """Test planner metrics are tracked."""
        mock_client = Mock(spec=anthropic.Anthropic)
        planner = QueryPlanner(client=mock_client)

        # Create a simple plan (fast path)
        async def run_test():
            await planner.create_plan("Who is James Evans?")

            metrics = planner.get_metrics()
            self.assertEqual(metrics["simple_queries"], 1)
            self.assertEqual(metrics["complex_queries"], 0)

        asyncio.run(run_test())

    def test_executor_metrics(self):
        """Test executor metrics are tracked."""
        async def mock_search(q, p):
            return {"structured_results": [{"type": "person", "data": {"name": "Test"}}]}

        executor = PlanExecutor(search_registry=mock_search)

        plan = QueryPlan(
            original_query="test",
            intent="lookup",
            complexity="simple",
            steps=[QueryStep(step_id=1, action="search_registry", query="test")]
        )

        async def run_test():
            await executor.execute(plan)

            metrics = executor.get_metrics()
            self.assertEqual(metrics["executions"], 1)
            self.assertEqual(metrics["successful_executions"], 1)

        asyncio.run(run_test())


class TestIntegration(unittest.TestCase):
    """Test integration between planner and executor."""

    def test_execute_with_fallback_function(self):
        """Test the convenience function."""
        mock_client = Mock(spec=anthropic.Anthropic)

        async def mock_search(q, p):
            return {"structured_results": [{"type": "person", "data": {"name": "Test"}}]}

        planner = QueryPlanner(client=mock_client)
        executor = PlanExecutor(search_registry=mock_search)

        async def run_test():
            result = await execute_with_fallback(
                "Who is James Evans?",
                planner,
                executor
            )

            self.assertTrue(result.success)

        asyncio.run(run_test())


class TestDataclassSerialization(unittest.TestCase):
    """Test dataclass serialization."""

    def test_query_step_to_dict(self):
        """Test QueryStep serialization."""
        step = QueryStep(
            step_id=1,
            action="search_registry",
            query="test",
            parameters={"type": "person"},
            depends_on=None,
            description="Test step"
        )

        data = step.to_dict()
        self.assertEqual(data["step_id"], 1)
        self.assertEqual(data["action"], "search_registry")
        self.assertEqual(data["parameters"]["type"], "person")

    def test_query_plan_to_dict(self):
        """Test QueryPlan serialization."""
        plan = QueryPlan(
            original_query="test",
            intent="lookup",
            complexity="simple",
            steps=[QueryStep(step_id=1, action="search_registry", query="test")],
            fallback_strategies=["expansion"],
            confidence=0.9,
        )

        data = plan.to_dict()
        self.assertEqual(data["original_query"], "test")
        self.assertEqual(data["intent"], "lookup")
        self.assertEqual(len(data["steps"]), 1)

    def test_execution_result_to_dict(self):
        """Test ExecutionResult serialization."""
        result = ExecutionResult(
            success=True,
            content="Test content",
            steps_executed=1,
            fallbacks_used=0,
            total_latency_ms=100.0,
            final_quality=0.8
        )

        data = result.to_dict()
        self.assertTrue(data["success"])
        self.assertEqual(data["content"], "Test content")
        self.assertEqual(data["final_quality"], 0.8)


if __name__ == "__main__":
    unittest.main(verbosity=2)
