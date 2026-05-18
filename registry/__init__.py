"""
Registry Module

Provides hybrid retrieval combining:
1. Structured queries against lab_registry.json (fast, deterministic)
2. Semantic search against embedded registry summaries + other docs
3. Typo correction and query logging for continuous improvement
4. Unified index for semantic search across all data types

Usage:
    from registry import RegistryRAG

    rag = RegistryRAG()
    results = rag.query("Who works on APTO?")

Or import specific components:
    from registry import RegistryLookup, QueryClassifier, RegistryTextGenerator
    from registry.models import RegistryMatch, QueryClassification
    from registry.typo_correction import TypoCorrector
    from registry.unified_index import UnifiedIndex
"""

from .models import RegistryMatch, QueryClassification
from .lookup import RegistryLookup
from .classifier import QueryClassifier
from .text_generator import RegistryTextGenerator
from .rag import RegistryRAG
from .topic_aliases import TOPIC_ALIASES, expand_topic_aliases
from .typo_correction import TypoCorrector, correct_query
from .query_logger import QueryLogger, log_query
from .entity_extractor import EntityExtractor, ExtractedEntities
from .unified_index import UnifiedIndex, build_unified_index
from .semantic_index import (
    SemanticRegistryIndex,
    build_semantic_registry_index,
    merge_with_structured,
)
from .query_router import (
    QueryRouter,
    RoutingDecision,
    RoutingResult,
    route_query,
    get_router,
)

__all__ = [
    # Main class
    "RegistryRAG",
    # Component classes
    "RegistryLookup",
    "QueryClassifier",
    "RegistryTextGenerator",
    # Data models
    "RegistryMatch",
    "QueryClassification",
    # Topic utilities
    "TOPIC_ALIASES",
    "expand_topic_aliases",
    # New Phase 1/2 components
    "TypoCorrector",
    "correct_query",
    "QueryLogger",
    "log_query",
    "EntityExtractor",
    "ExtractedEntities",
    "UnifiedIndex",
    "build_unified_index",
    # Semantic registry index
    "SemanticRegistryIndex",
    "build_semantic_registry_index",
    "merge_with_structured",
    # Query routing
    "QueryRouter",
    "RoutingDecision",
    "RoutingResult",
    "route_query",
    "get_router",
]
