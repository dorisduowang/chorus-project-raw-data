"""
Query Complexity Router Module

Routes queries based on complexity analysis to optimize performance:
- Simple queries (registry_only): Direct registry lookup, <10ms
- Complex queries (rag_only): Full RAG pipeline, 500-1000ms
- Hybrid queries: Registry + semantic search

Query Types:
- Simple: "Who is X?", "List all PhD students", "X's email"
- Complex: "papers about network analysis", conceptual questions
- Hybrid: Queries that benefit from both structured and semantic search
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple, TYPE_CHECKING

# Handle both package and direct imports
try:
    from .classifier import QueryClassifier
    from .models import QueryClassification
except ImportError:
    # Direct import when running standalone or in tests
    from classifier import QueryClassifier
    from models import QueryClassification


class RoutingDecision(Enum):
    """Routing decision for query processing."""
    REGISTRY_ONLY = "registry_only"
    RAG_ONLY = "rag_only"
    HYBRID = "hybrid"


@dataclass
class RoutingResult:
    """Result of query routing analysis."""
    decision: RoutingDecision
    confidence: float  # 0.0 to 1.0
    reasoning: str
    classification: Optional[QueryClassification] = None

    # Performance hints
    expected_latency_ms: Optional[int] = None
    skip_reranking: bool = False
    skip_mmr: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "decision": self.decision.value,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "expected_latency_ms": self.expected_latency_ms,
            "skip_reranking": self.skip_reranking,
            "skip_mmr": self.skip_mmr,
        }


class QueryRouter:
    """
    Routes queries based on complexity analysis.

    Simple queries go directly to the registry (fast path).
    Complex queries use the full RAG pipeline.
    Hybrid queries use both for comprehensive results.
    """

    # Patterns that indicate simple, registry-only queries
    # These are high-confidence patterns for direct lookup
    SIMPLE_PATTERNS = [
        # Direct person lookups
        (r"^who\s+is\s+[\w\s\-\.]+\??$", "person_lookup"),
        (r"^tell\s+me\s+about\s+[\w\s\-\.]+\??$", "person_lookup"),
        (r"^(?:info|information)\s+(?:on|about)\s+[\w\s\-\.]+\??$", "person_lookup"),

        # Contact queries
        (r"^[\w\s\-\.]+(?:'s)?\s+(?:email|contact)(?:\s+info(?:rmation)?)?\??$", "contact_lookup"),
        (r"^(?:what(?:'s| is)|get)\s+[\w\s\-\.]+(?:'s)?\s+email\??$", "contact_lookup"),
        (r"^how\s+(?:do\s+i|can\s+i|to)\s+contact\s+[\w\s\-\.]+\??$", "contact_lookup"),

        # Role listings
        (r"^list\s+(?:all\s+)?(?:the\s+)?(?:phd\s+students?|postdocs?|faculty|staff|members?)\??$", "role_list"),
        (r"^(?:show|get)\s+(?:all\s+)?(?:the\s+)?(?:phd\s+students?|postdocs?|faculty|staff|members?)\??$", "role_list"),
        (r"^who\s+are\s+(?:the\s+)?(?:phd\s+students?|postdocs?|faculty|staff|members?)\??$", "role_list"),
        (r"^how\s+many\s+(?:phd\s+students?|postdocs?|faculty|staff|members?)\??$", "count_query"),

        # Simple dataset/project/resource queries
        (r"^what\s+(?:datasets?|data)\s+do\s+we\s+have\??$", "list_datasets"),
        (r"^list\s+(?:all\s+)?(?:our\s+)?(?:datasets?|data)\??$", "list_datasets"),
        (r"^what\s+projects?\s+do\s+we\s+have\??$", "list_projects"),
        (r"^list\s+(?:all\s+)?(?:our\s+)?projects?\??$", "list_projects"),
        (r"^what\s+(?:funding|grants?)\s+do\s+we\s+have\??$", "list_funding"),
        (r"^what\s+(?:compute|resources?)\s+(?:do\s+we\s+have|are\s+available)\??$", "list_compute"),
        (r"^what\s+events?\s+(?:do\s+we\s+have|are\s+there)\??$", "list_events"),

        # Stats queries
        (r"^how\s+many\s+(?:people|members?|researchers?)\s+(?:do\s+we\s+have|are\s+there)\??$", "count_people"),

        # Institution lookup
        (r"^who\s+is\s+at\s+[\w\s]+\??$", "institution_lookup"),
        (r"^(?:people|members?)\s+(?:at|from)\s+[\w\s]+\??$", "institution_lookup"),
    ]

    # Patterns that indicate complex, RAG-required queries
    # These require semantic understanding
    COMPLEX_PATTERNS = [
        # Semantic/conceptual queries
        (r"papers?\s+(?:about|on|regarding)\s+.+", "semantic_search"),
        (r"publications?\s+(?:about|on|regarding)\s+.+", "semantic_search"),
        (r"research\s+(?:about|on|regarding)\s+.+", "semantic_search"),

        # Open-ended questions
        (r"how\s+does\s+(?:the\s+lab|our\s+lab|chorus)\s+.+", "conceptual"),
        (r"what\s+(?:approach|methodology|methods?)\s+.+", "conceptual"),
        (r"explain\s+.+", "conceptual"),
        (r"describe\s+(?:the\s+)?(?:research\s+)?(?:approach|methodology|process)\s+.+", "conceptual"),

        # Comparison requiring semantic understanding
        (r"(?:compare|contrast|difference)\s+.+\s+(?:and|with|vs)\s+.+", "comparison"),

        # Finding by topic (general)
        (r"who\s+(?:works?\s+on|studies|researches?)\s+.{15,}", "topic_search"),
        (r"researchers?\s+(?:working\s+on|studying|in)\s+.{15,}", "topic_search"),

        # Multi-hop reasoning
        (r"what\s+do\s+.+\s+and\s+.+\s+have\s+in\s+common", "multi_hop"),
        (r"how\s+(?:are|do)\s+.+\s+and\s+.+\s+(?:related|connected|collaborate)", "multi_hop"),
    ]

    # Keywords that suggest complexity
    COMPLEXITY_KEYWORDS = {
        "high": [
            "methodology", "approach", "explain", "describe", "compare",
            "contrast", "difference", "relationship", "impact", "influence",
            "conceptual", "theoretical", "framework", "paradigm",
        ],
        "medium": [
            "topic", "area", "field", "research", "work", "papers",
            "publications", "collaborat", "network", "connection",
        ],
        "low": [
            "who", "list", "show", "email", "contact", "name",
            "role", "position", "institution", "university",
        ],
    }

    # Confidence thresholds for routing decisions
    HIGH_CONFIDENCE_THRESHOLD = 0.9
    MEDIUM_CONFIDENCE_THRESHOLD = 0.7

    def __init__(
        self,
        classifier: Optional[QueryClassifier] = None,
        registry_path: str = "Data/lab_registry.json",
        midway_path: str = "Data/midway_registry.json",
    ):
        """
        Initialize the query router.

        Args:
            classifier: Optional QueryClassifier instance. Creates new one if not provided.
            registry_path: Path to lab registry JSON
            midway_path: Path to midway registry JSON
        """
        self.classifier = classifier or QueryClassifier(registry_path, midway_path)

    def route(self, query: str) -> RoutingResult:
        """
        Analyze query and determine optimal routing.

        Args:
            query: The user's query string

        Returns:
            RoutingResult with decision, confidence, and reasoning
        """
        query_normalized = query.lower().strip()

        # Step 1: Check for simple patterns (fast path)
        simple_match = self._match_simple_patterns(query_normalized)
        if simple_match:
            pattern_type, confidence = simple_match
            return RoutingResult(
                decision=RoutingDecision.REGISTRY_ONLY,
                confidence=confidence,
                reasoning=f"Simple pattern match: {pattern_type}",
                expected_latency_ms=10,
                skip_reranking=True,
                skip_mmr=True,
            )

        # Step 2: Check for complex patterns (semantic required)
        complex_match = self._match_complex_patterns(query_normalized)
        if complex_match:
            pattern_type, confidence = complex_match
            return RoutingResult(
                decision=RoutingDecision.RAG_ONLY,
                confidence=confidence,
                reasoning=f"Complex pattern match: {pattern_type}",
                expected_latency_ms=800,
                skip_reranking=False,
                skip_mmr=False,
            )

        # Step 3: Use classifier for more nuanced analysis
        classification = self.classifier.classify(query)

        # Step 4: Analyze classification results
        return self._route_from_classification(query, classification)

    def _match_simple_patterns(self, query: str) -> Optional[Tuple[str, float]]:
        """
        Check if query matches simple patterns.

        Returns (pattern_type, confidence) if matched, None otherwise.
        """
        for pattern, pattern_type in self.SIMPLE_PATTERNS:
            if re.match(pattern, query, re.IGNORECASE):
                # Base confidence is high for pattern matches
                confidence = 0.95

                # Adjust based on query length (shorter = more confident)
                query_words = len(query.split())
                if query_words <= 5:
                    confidence = min(confidence + 0.03, 1.0)
                elif query_words > 10:
                    confidence = max(confidence - 0.1, 0.7)

                return (pattern_type, confidence)

        return None

    def _match_complex_patterns(self, query: str) -> Optional[Tuple[str, float]]:
        """
        Check if query matches complex patterns.

        Returns (pattern_type, confidence) if matched, None otherwise.
        """
        for pattern, pattern_type in self.COMPLEX_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                # Base confidence for complex patterns
                confidence = 0.85

                # Boost confidence for longer, more specific queries
                query_words = len(query.split())
                if query_words > 10:
                    confidence = min(confidence + 0.08, 0.95)

                # Check for complexity keywords
                keyword_score = self._score_complexity_keywords(query)
                if keyword_score > 0.5:
                    confidence = min(confidence + 0.05, 0.95)

                return (pattern_type, confidence)

        return None

    def _score_complexity_keywords(self, query: str) -> float:
        """
        Score query based on complexity keywords.

        Returns a score from 0.0 (simple) to 1.0 (complex).
        """
        query_lower = query.lower()

        high_count = sum(1 for kw in self.COMPLEXITY_KEYWORDS["high"] if kw in query_lower)
        medium_count = sum(1 for kw in self.COMPLEXITY_KEYWORDS["medium"] if kw in query_lower)
        low_count = sum(1 for kw in self.COMPLEXITY_KEYWORDS["low"] if kw in query_lower)

        # Weighted scoring
        score = (high_count * 1.0 + medium_count * 0.5 + low_count * 0.0) / max(
            high_count + medium_count + low_count, 1
        )

        return min(score, 1.0)

    def _route_from_classification(
        self, query: str, classification: QueryClassification
    ) -> RoutingResult:
        """
        Determine routing based on classifier output.

        Args:
            query: Original query string
            classification: QueryClassification result from classifier

        Returns:
            RoutingResult based on classification analysis
        """
        # High confidence structured queries -> registry only
        if (
            classification.query_type == "structured"
            and classification.confidence >= self.HIGH_CONFIDENCE_THRESHOLD
        ):
            return RoutingResult(
                decision=RoutingDecision.REGISTRY_ONLY,
                confidence=classification.confidence,
                reasoning=f"High-confidence structured query (confidence={classification.confidence:.2f})",
                classification=classification,
                expected_latency_ms=10,
                skip_reranking=True,
                skip_mmr=True,
            )

        # Pure semantic queries -> RAG only
        if classification.query_type == "semantic":
            return RoutingResult(
                decision=RoutingDecision.RAG_ONLY,
                confidence=classification.confidence,
                reasoning=f"Semantic query type (confidence={classification.confidence:.2f})",
                classification=classification,
                expected_latency_ms=800,
                skip_reranking=False,
                skip_mmr=False,
            )

        # Low confidence structured -> use hybrid
        if (
            classification.query_type == "structured"
            and classification.confidence < self.MEDIUM_CONFIDENCE_THRESHOLD
        ):
            return RoutingResult(
                decision=RoutingDecision.HYBRID,
                confidence=classification.confidence,
                reasoning=f"Low-confidence structured query (confidence={classification.confidence:.2f})",
                classification=classification,
                expected_latency_ms=500,
                skip_reranking=False,
                skip_mmr=True,
            )

        # Entity type analysis for edge cases
        entity_routing = self._route_by_entity_types(classification.entity_types)
        if entity_routing:
            return RoutingResult(
                decision=entity_routing,
                confidence=classification.confidence,
                reasoning=f"Entity type routing: {classification.entity_types}",
                classification=classification,
                expected_latency_ms=500 if entity_routing == RoutingDecision.HYBRID else 10,
                skip_reranking=entity_routing == RoutingDecision.REGISTRY_ONLY,
                skip_mmr=entity_routing == RoutingDecision.REGISTRY_ONLY,
            )

        # Default to hybrid for medium confidence
        return RoutingResult(
            decision=RoutingDecision.HYBRID,
            confidence=classification.confidence,
            reasoning=f"Default hybrid routing (type={classification.query_type}, confidence={classification.confidence:.2f})",
            classification=classification,
            expected_latency_ms=500,
            skip_reranking=False,
            skip_mmr=False,
        )

    def _route_by_entity_types(self, entity_types: List[str]) -> Optional[RoutingDecision]:
        """
        Determine routing based on detected entity types.

        Returns routing decision or None if entity types don't dictate routing.
        """
        # Types that are well-served by registry alone
        registry_types = {"people", "projects", "funding", "datasets", "events", "compute", "contact"}

        # Types that benefit from semantic search
        semantic_types = {"topics", "publications", "comparison"}

        if not entity_types:
            return RoutingDecision.RAG_ONLY

        has_registry_types = any(t in registry_types for t in entity_types)
        has_semantic_types = any(t in semantic_types for t in entity_types)

        if has_registry_types and not has_semantic_types:
            return RoutingDecision.REGISTRY_ONLY
        elif has_semantic_types and not has_registry_types:
            return RoutingDecision.RAG_ONLY
        elif has_registry_types and has_semantic_types:
            return RoutingDecision.HYBRID

        return None

    def analyze_query(self, query: str) -> Dict[str, Any]:
        """
        Perform detailed analysis of query for debugging/inspection.

        Returns comprehensive analysis including all routing factors.
        """
        query_normalized = query.lower().strip()

        # Get basic routing result
        routing = self.route(query)

        # Get classifier results
        classification = self.classifier.classify(query)

        # Pattern analysis
        simple_match = self._match_simple_patterns(query_normalized)
        complex_match = self._match_complex_patterns(query_normalized)

        # Keyword analysis
        keyword_score = self._score_complexity_keywords(query)

        # Query structure analysis
        query_length = len(query)
        word_count = len(query.split())
        has_question_mark = query.strip().endswith("?")

        return {
            "query": query,
            "routing": routing.to_dict(),
            "classification": {
                "query_type": classification.query_type,
                "entity_types": classification.entity_types,
                "search_terms": classification.search_terms,
                "confidence": classification.confidence,
                "is_compound": classification.is_compound,
            },
            "pattern_analysis": {
                "simple_match": simple_match,
                "complex_match": complex_match,
            },
            "complexity_analysis": {
                "keyword_score": keyword_score,
                "query_length": query_length,
                "word_count": word_count,
                "has_question_mark": has_question_mark,
            },
        }


# Convenience function for quick routing
def route_query(query: str, classifier: Optional[QueryClassifier] = None) -> RoutingResult:
    """
    Quick routing function.

    Args:
        query: Query string to route
        classifier: Optional classifier instance

    Returns:
        RoutingResult with routing decision
    """
    router = QueryRouter(classifier=classifier)
    return router.route(query)


# Singleton instance for reuse
_router_instance: Optional[QueryRouter] = None


def get_router(
    registry_path: str = "Data/lab_registry.json",
    midway_path: str = "Data/midway_registry.json",
) -> QueryRouter:
    """
    Get or create the singleton QueryRouter instance.

    Args:
        registry_path: Path to lab registry
        midway_path: Path to midway registry

    Returns:
        QueryRouter instance
    """
    global _router_instance
    if _router_instance is None:
        _router_instance = QueryRouter(registry_path=registry_path, midway_path=midway_path)
    return _router_instance
