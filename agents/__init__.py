"""
Chorus Agent System - Multi-agent orchestration for Knowledge Lab.

This package provides:
- QueryClassifier: Fast intent classification using regex patterns
- LLMQueryClassifier: LLM-based intent classification using Claude Haiku
- QueryPlanner: LLM-driven query planning with multi-step strategies
- PlanExecutor: Executes query plans with retry and fallback logic
- Specialist agents: Memory, Mechanic, Muse, Matchmaker
- ChorusOrchestrator: Routes queries to appropriate specialists

Usage:
    from agents import ChorusOrchestrator

    orchestrator = ChorusOrchestrator()
    response = await orchestrator.query("What research projects are happening?")

    # Use LLM-based classifier (recommended):
    from agents import LLMQueryClassifier, get_classifier

    classifier = get_classifier(use_llm=True)
    result = await classifier.classify("What grants does Prof Smith have?")

    # Use smart fallback with query planning:
    orchestrator = ChorusOrchestrator(enable_smart_fallback=True)
    response = await orchestrator.query("Compare James Evans and Hyejin Youn's research")
"""

from .base import BaseAgent, QueryClassification, AgentResponse
from .classifier import QueryClassifier
from .llm_classifier import LLMQueryClassifier, get_classifier
from .query_planner import QueryPlanner, QueryPlan, QueryStep, QueryAction, FallbackStrategy
from .plan_executor import PlanExecutor, ExecutionResult, execute_with_fallback
from .orchestrator import ChorusOrchestrator
from .confidence import (
    ConfidenceScore,
    ConfidenceCalculator,
    Source,
    SourceType,
    extract_sources_from_rag_result,
    extract_sources_from_web_results,
    create_inference_source,
)
from .legacy import ChorusAgent, call_rag_async

__all__ = [
    'BaseAgent',
    'QueryClassification',
    'AgentResponse',
    'QueryClassifier',
    'LLMQueryClassifier',
    'get_classifier',
    'QueryPlanner',
    'QueryPlan',
    'QueryStep',
    'QueryAction',
    'FallbackStrategy',
    'PlanExecutor',
    'ExecutionResult',
    'execute_with_fallback',
    'ChorusOrchestrator',
    # Confidence scoring
    'ConfidenceScore',
    'ConfidenceCalculator',
    'Source',
    'SourceType',
    'extract_sources_from_rag_result',
    'extract_sources_from_web_results',
    'create_inference_source',
    # Legacy (from core_agent.py)
    'ChorusAgent',
    'call_rag_async',
]
