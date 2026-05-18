"""
PlanExecutor - Execute query plans with retry and fallback logic.

This module executes QueryPlans created by the QueryPlanner, handling:
1. Step-by-step execution with dependency tracking
2. Result quality evaluation
3. Automatic fallback and retry strategies
4. Result combination and synthesis
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Callable

import anthropic

from .query_planner import (
    QueryPlan, QueryStep, StepResult, QueryAction, FallbackStrategy
)


# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_EXECUTION_MODEL = "claude-3-5-haiku-20241022"
MIN_QUALITY_THRESHOLD = 0.3  # Minimum quality to consider result acceptable
GOOD_QUALITY_THRESHOLD = 0.6  # Quality threshold to skip fallbacks


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class ExecutionResult:
    """Result of executing a complete query plan."""
    success: bool
    content: str
    steps_executed: int
    fallbacks_used: int
    total_latency_ms: float
    step_results: List[StepResult] = field(default_factory=list)
    final_quality: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "content": self.content,
            "steps_executed": self.steps_executed,
            "fallbacks_used": self.fallbacks_used,
            "total_latency_ms": self.total_latency_ms,
            "final_quality": self.final_quality,
            "error": self.error,
        }


@dataclass
class ExecutorMetrics:
    """Track executor performance."""
    executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    fallbacks_triggered: int = 0
    retries_performed: int = 0
    avg_steps_per_plan: float = 0.0
    avg_execution_time_ms: float = 0.0
    total_execution_time_ms: float = 0.0


# =============================================================================
# Synthesis Prompt
# =============================================================================

SYNTHESIS_PROMPT = """You are a research assistant synthesizing search results to answer a user's question.

Given the search results below, provide a clear, concise answer. Follow these guidelines:
- Be direct and factual
- Cite sources when available
- If results are incomplete, acknowledge what's missing
- Don't apologize or hedge unnecessarily
- Keep the response focused and relevant

USER QUESTION: {query}

SEARCH RESULTS:
{results}

Provide a synthesized answer:"""


# =============================================================================
# Plan Executor
# =============================================================================

class PlanExecutor:
    """
    Execute query plans with retry and fallback logic.

    Handles:
    - Step execution in dependency order
    - Result quality evaluation
    - Automatic fallback strategies
    - Result synthesis from multiple steps

    Usage:
        executor = PlanExecutor(search_registry_fn, search_docs_fn, web_search_fn)
        result = await executor.execute(plan)
    """

    def __init__(
        self,
        search_registry: Optional[Callable] = None,
        search_docs: Optional[Callable] = None,
        web_search: Optional[Callable] = None,
        client: Optional[anthropic.Anthropic] = None,
        quality_threshold: float = MIN_QUALITY_THRESHOLD,
    ):
        """
        Initialize the plan executor.

        Args:
            search_registry: Async function to search the registry
            search_docs: Async function to search documents
            web_search: Async function to search the web
            client: Anthropic client for synthesis
            quality_threshold: Minimum quality score to accept results
        """
        self._search_registry = search_registry
        self._search_docs = search_docs
        self._web_search = web_search
        self._client = client
        self.quality_threshold = quality_threshold

        # Track step results for dependencies
        self._step_results: Dict[int, StepResult] = {}

        # Metrics
        self.metrics = ExecutorMetrics()

        # Query planner reference (for reformulation)
        self._planner = None

    def set_planner(self, planner):
        """Set reference to query planner for reformulation."""
        self._planner = planner

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

    def set_search_functions(
        self,
        search_registry: Optional[Callable] = None,
        search_docs: Optional[Callable] = None,
        web_search: Optional[Callable] = None,
    ):
        """Set or update search functions."""
        if search_registry:
            self._search_registry = search_registry
        if search_docs:
            self._search_docs = search_docs
        if web_search:
            self._web_search = web_search

    async def _execute_step(self, step: QueryStep, plan: QueryPlan) -> StepResult:
        """
        Execute a single step in the plan.

        Args:
            step: The step to execute
            plan: The full plan (for context)

        Returns:
            StepResult with outcome
        """
        start_time = time.time()

        try:
            action = step.action
            query = step.query

            # Handle different actions
            if action == QueryAction.SEARCH_REGISTRY.value:
                if self._search_registry:
                    result = await self._search_registry(query, step.parameters)
                else:
                    result = {"error": "Registry search not available"}

            elif action == QueryAction.SEARCH_DOCS.value:
                if self._search_docs:
                    result = await self._search_docs(query, step.parameters)
                else:
                    result = {"error": "Document search not available"}

            elif action == QueryAction.WEB_SEARCH.value:
                if self._web_search:
                    result = await self._web_search(query, step.parameters)
                else:
                    result = {"error": "Web search not available"}

            elif action == QueryAction.COMBINE.value:
                result = await self._combine_results(step, plan)

            elif action == QueryAction.DIRECT_ANSWER.value:
                result = {"direct_answer": True, "query": query}

            elif action == QueryAction.REFORMULATE.value:
                # Reformulation is handled at plan level, not step level
                result = {"reformulated": query}

            else:
                result = {"error": f"Unknown action: {action}"}

            # Evaluate quality
            quality = self._evaluate_quality(query, result, step)

            latency_ms = (time.time() - start_time) * 1000

            step_result = StepResult(
                step_id=step.step_id,
                success=quality >= self.quality_threshold,
                result=result,
                quality_score=quality,
                latency_ms=latency_ms,
            )

            # Store for dependencies
            self._step_results[step.step_id] = step_result

            logger.debug(
                f"Step {step.step_id} ({action}): quality={quality:.2f}, "
                f"success={step_result.success}, latency={latency_ms:.0f}ms"
            )

            return step_result

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            logger.error(f"Step {step.step_id} failed: {e}")

            step_result = StepResult(
                step_id=step.step_id,
                success=False,
                result=None,
                quality_score=0.0,
                latency_ms=latency_ms,
                error=str(e),
            )
            self._step_results[step.step_id] = step_result
            return step_result

    def _evaluate_quality(
        self,
        query: str,
        result: Any,
        step: QueryStep
    ) -> float:
        """
        Evaluate the quality of a step result.

        Returns a score from 0.0 to 1.0.
        """
        if result is None:
            return 0.0

        if isinstance(result, dict):
            # Check for error
            if result.get("error"):
                return 0.0

            # Direct answer is always good quality
            if result.get("direct_answer"):
                return 0.9

            # Check for structured results
            structured = result.get("structured_results", [])
            semantic = result.get("semantic_results", result.get("results", []))

            if structured:
                # Structured results from registry are high quality
                return min(1.0, 0.7 + 0.05 * len(structured))
            elif semantic:
                # Semantic results are moderate quality
                return min(0.8, 0.4 + 0.08 * len(semantic))
            else:
                return 0.2

        elif isinstance(result, str):
            # String result - check length and content
            if len(result) < 50:
                return 0.2
            elif "error" in result.lower() or "not found" in result.lower():
                return 0.1
            elif "no results" in result.lower():
                return 0.15
            else:
                # Decent content
                return min(0.8, 0.5 + min(0.3, len(result) / 1000))

        elif isinstance(result, list):
            if not result:
                return 0.2
            return min(0.9, 0.5 + 0.1 * len(result))

        return 0.5  # Default moderate score

    async def _combine_results(
        self,
        step: QueryStep,
        plan: QueryPlan
    ) -> Dict[str, Any]:
        """
        Combine results from previous steps.

        Uses LLM synthesis if multiple sources need to be combined.
        """
        source_ids = step.parameters.get("sources", [])

        # Gather results from source steps
        source_results = []
        for source_id in source_ids:
            if source_id in self._step_results:
                source_result = self._step_results[source_id]
                if source_result.success and source_result.result:
                    source_results.append(source_result.result)

        if not source_results:
            # No successful source results
            return {"error": "No source results to combine"}

        if len(source_results) == 1:
            # Just one source - return it directly
            return source_results[0]

        # Multiple sources - synthesize using LLM
        try:
            # Format results for synthesis
            formatted_results = []
            for i, result in enumerate(source_results, 1):
                if isinstance(result, dict):
                    formatted_results.append(f"Source {i}:\n{json.dumps(result, indent=2)}")
                else:
                    formatted_results.append(f"Source {i}:\n{result}")

            results_text = "\n\n".join(formatted_results)

            # Use synthesis prompt
            prompt = SYNTHESIS_PROMPT.format(
                query=plan.original_query,
                results=results_text
            )

            response = self.client.messages.create(
                model=DEFAULT_EXECUTION_MODEL,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}]
            )

            synthesized = response.content[0].text

            return {
                "synthesized": True,
                "content": synthesized,
                "source_count": len(source_results),
            }

        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            # Return concatenated results as fallback
            return {
                "combined": True,
                "results": source_results,
                "error": f"Synthesis failed: {str(e)}"
            }

    async def _execute_fallback(
        self,
        strategy: str,
        plan: QueryPlan,
        previous_results: List[StepResult]
    ) -> Optional[StepResult]:
        """
        Execute a fallback strategy.

        Args:
            strategy: The fallback strategy to use
            plan: The original plan
            previous_results: Results from previous attempts

        Returns:
            StepResult if fallback was attempted, None otherwise
        """
        self.metrics.fallbacks_triggered += 1

        logger.info(f"Attempting fallback strategy: {strategy}")

        try:
            if strategy == FallbackStrategy.REFORMULATION.value:
                # Reformulate the query
                if self._planner:
                    new_query = await self._planner.generate_reformulation(
                        plan.original_query,
                        strategy="reformulation",
                        previous_results=[str(r.result) for r in previous_results if r.result]
                    )
                else:
                    new_query = plan.original_query

                # Execute reformulated query
                step = QueryStep(
                    step_id=100,  # Fallback step ID
                    action=QueryAction.SEARCH_REGISTRY.value,
                    query=new_query,
                    description=f"Reformulated: {new_query}"
                )
                return await self._execute_step(step, plan)

            elif strategy == FallbackStrategy.DECOMPOSITION.value:
                # Get decomposed query (most important sub-question)
                if self._planner:
                    decomposed = await self._planner.generate_reformulation(
                        plan.original_query,
                        strategy="decomposition"
                    )
                else:
                    decomposed = plan.original_query

                step = QueryStep(
                    step_id=101,
                    action=QueryAction.SEARCH_REGISTRY.value,
                    query=decomposed,
                    description=f"Decomposed: {decomposed}"
                )
                return await self._execute_step(step, plan)

            elif strategy == FallbackStrategy.EXPANSION.value:
                # Expand with synonyms
                if self._planner:
                    expanded = await self._planner.generate_reformulation(
                        plan.original_query,
                        strategy="expansion"
                    )
                else:
                    expanded = plan.original_query

                step = QueryStep(
                    step_id=102,
                    action=QueryAction.SEARCH_REGISTRY.value,
                    query=expanded,
                    description=f"Expanded: {expanded}"
                )
                return await self._execute_step(step, plan)

            elif strategy == FallbackStrategy.NARROWING.value:
                # Narrow to specific aspect
                if self._planner:
                    narrowed = await self._planner.generate_reformulation(
                        plan.original_query,
                        strategy="narrowing"
                    )
                else:
                    narrowed = plan.original_query

                step = QueryStep(
                    step_id=103,
                    action=QueryAction.SEARCH_REGISTRY.value,
                    query=narrowed,
                    description=f"Narrowed: {narrowed}"
                )
                return await self._execute_step(step, plan)

            elif strategy == FallbackStrategy.WEB_FALLBACK.value:
                # Try web search
                if self._web_search:
                    step = QueryStep(
                        step_id=104,
                        action=QueryAction.WEB_SEARCH.value,
                        query=plan.original_query,
                        description="Web fallback"
                    )
                    return await self._execute_step(step, plan)
                else:
                    logger.warning("Web search not available for fallback")
                    return None

            else:
                logger.warning(f"Unknown fallback strategy: {strategy}")
                return None

        except Exception as e:
            logger.error(f"Fallback strategy '{strategy}' failed: {e}")
            return None

    def _format_results(
        self,
        step_results: List[StepResult],
        plan: QueryPlan
    ) -> str:
        """
        Format step results into final response content.
        """
        # Collect successful results
        successful = [r for r in step_results if r.success and r.result]

        if not successful:
            return "I couldn't find relevant information for your query."

        # Check for combined/synthesized result
        for result in reversed(successful):
            if isinstance(result.result, dict):
                if result.result.get("synthesized"):
                    return result.result.get("content", "")
                if result.result.get("direct_answer"):
                    return ""  # Will be handled by the specialist agent

        # Format individual results
        parts = []
        for result in successful:
            if isinstance(result.result, dict):
                if result.result.get("structured_results"):
                    parts.append(self._format_structured_results(result.result))
                elif result.result.get("content"):
                    parts.append(result.result["content"])
            elif isinstance(result.result, str):
                parts.append(result.result)

        return "\n\n".join(parts) if parts else "No relevant information found."

    def _format_structured_results(self, result: Dict) -> str:
        """Format structured results from registry."""
        structured = result.get("structured_results", [])
        if not structured:
            return ""

        parts = []
        for item in structured[:5]:  # Limit to top 5
            item_type = item.get("type", "")
            data = item.get("data", {})

            if item_type == "person":
                name = data.get("name", "Unknown")
                role = data.get("role", "")
                parts.append(f"- {name}" + (f" ({role})" if role else ""))
            elif item_type == "project":
                name = data.get("name", "Unknown")
                status = data.get("status", "")
                parts.append(f"- {name}" + (f" [{status}]" if status else ""))
            else:
                parts.append(f"- {data.get('name', str(data)[:100])}")

        return "\n".join(parts)

    async def execute(self, plan: QueryPlan) -> ExecutionResult:
        """
        Execute a complete query plan.

        Handles step execution, quality evaluation, and fallbacks.

        Args:
            plan: The plan to execute

        Returns:
            ExecutionResult with combined outcomes
        """
        start_time = time.time()
        self.metrics.executions += 1

        # Reset step results
        self._step_results.clear()

        step_results: List[StepResult] = []
        fallbacks_used = 0

        try:
            # Sort steps by dependencies (steps with no deps first)
            sorted_steps = self._sort_steps_by_dependency(plan.steps)

            # Execute steps
            for step in sorted_steps:
                # Check if dependencies are satisfied
                if step.depends_on is not None:
                    dep_result = self._step_results.get(step.depends_on)
                    if not dep_result or not dep_result.success:
                        logger.debug(f"Skipping step {step.step_id} - dependency not satisfied")
                        continue

                result = await self._execute_step(step, plan)
                step_results.append(result)

            # Check overall quality
            successful_results = [r for r in step_results if r.success]
            avg_quality = (
                sum(r.quality_score for r in successful_results) / len(successful_results)
                if successful_results else 0.0
            )

            # If quality is low, try fallbacks
            if avg_quality < GOOD_QUALITY_THRESHOLD and plan.fallback_strategies:
                logger.info(f"Quality {avg_quality:.2f} below threshold, trying fallbacks")

                for strategy in plan.fallback_strategies:
                    if fallbacks_used >= plan.max_retries:
                        break

                    fallback_result = await self._execute_fallback(
                        strategy, plan, step_results
                    )

                    if fallback_result and fallback_result.success:
                        step_results.append(fallback_result)
                        fallbacks_used += 1

                        # Check if we now have good quality
                        if fallback_result.quality_score >= GOOD_QUALITY_THRESHOLD:
                            break

            # Calculate final quality
            final_results = [r for r in step_results if r.success]
            final_quality = (
                max(r.quality_score for r in final_results)
                if final_results else 0.0
            )

            # Format final content
            content = self._format_results(step_results, plan)

            total_latency = (time.time() - start_time) * 1000

            # Update metrics
            if final_quality >= self.quality_threshold:
                self.metrics.successful_executions += 1
            else:
                self.metrics.failed_executions += 1

            self.metrics.total_execution_time_ms += total_latency
            self.metrics.avg_execution_time_ms = (
                self.metrics.total_execution_time_ms / self.metrics.executions
            )

            return ExecutionResult(
                success=final_quality >= self.quality_threshold,
                content=content,
                steps_executed=len(step_results),
                fallbacks_used=fallbacks_used,
                total_latency_ms=total_latency,
                step_results=step_results,
                final_quality=final_quality,
            )

        except Exception as e:
            logger.error(f"Plan execution failed: {e}")
            self.metrics.failed_executions += 1

            return ExecutionResult(
                success=False,
                content=f"An error occurred: {str(e)}",
                steps_executed=len(step_results),
                fallbacks_used=fallbacks_used,
                total_latency_ms=(time.time() - start_time) * 1000,
                step_results=step_results,
                final_quality=0.0,
                error=str(e),
            )

    def _sort_steps_by_dependency(self, steps: List[QueryStep]) -> List[QueryStep]:
        """
        Sort steps so dependencies come before dependents.

        Simple topological sort - assumes no cycles.
        """
        # Group steps by dependency status
        no_deps = [s for s in steps if s.depends_on is None]
        has_deps = [s for s in steps if s.depends_on is not None]

        # Build result list
        result = list(no_deps)
        added_ids = {s.step_id for s in no_deps}

        # Add steps with dependencies once their deps are satisfied
        remaining = list(has_deps)
        iterations = 0
        max_iterations = len(steps) * 2  # Prevent infinite loops

        while remaining and iterations < max_iterations:
            iterations += 1
            for step in remaining[:]:
                if step.depends_on in added_ids:
                    result.append(step)
                    added_ids.add(step.step_id)
                    remaining.remove(step)

        # Add any remaining steps (deps not found)
        result.extend(remaining)

        return result

    def get_metrics(self) -> dict:
        """Return executor metrics."""
        return {
            "executions": self.metrics.executions,
            "successful_executions": self.metrics.successful_executions,
            "failed_executions": self.metrics.failed_executions,
            "fallbacks_triggered": self.metrics.fallbacks_triggered,
            "retries_performed": self.metrics.retries_performed,
            "avg_execution_time_ms": self.metrics.avg_execution_time_ms,
            "success_rate": (
                self.metrics.successful_executions / max(1, self.metrics.executions)
            ),
        }


# =============================================================================
# Convenience Function
# =============================================================================

async def execute_with_fallback(
    query: str,
    planner,
    executor: PlanExecutor,
    context: Optional[Dict[str, Any]] = None
) -> ExecutionResult:
    """
    High-level function to plan and execute a query with fallbacks.

    Args:
        query: User query
        planner: QueryPlanner instance
        executor: PlanExecutor instance
        context: Optional context

    Returns:
        ExecutionResult
    """
    # Create plan
    plan = await planner.create_plan(query, context)

    # Set planner reference for reformulation
    executor.set_planner(planner)

    # Execute plan
    return await executor.execute(plan)
