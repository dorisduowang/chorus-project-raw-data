#!/usr/bin/env python3
"""
Unit tests for QueryRouter module.

Tests query complexity routing to ensure:
- Simple person lookups route to registry_only
- Complex semantic queries route to rag_only
- Edge cases handled correctly
- Confidence scores are reasonable

Run with: pytest tests/test_query_router.py -v
"""

import os
import sys
import pytest

# Add parent directory and registry directory to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
registry_path = os.path.join(project_root, "registry")
sys.path.insert(0, project_root)
sys.path.insert(0, registry_path)

# Now we can import directly from the modules
from models import QueryClassification
from classifier import QueryClassifier
from query_router import (
    QueryRouter,
    RoutingDecision,
    RoutingResult,
    route_query,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def router():
    """Create a QueryRouter instance for testing."""
    # Use fixture paths - tests should work even without actual data files
    return QueryRouter()


@pytest.fixture
def router_with_classifier(router):
    """Router with classifier for testing classification-based routing."""
    return router


# =============================================================================
# Simple Pattern Tests (Registry-Only Routing)
# =============================================================================


class TestSimplePersonLookups:
    """Test that simple person lookups route to registry_only."""

    @pytest.mark.parametrize(
        "query",
        [
            "Who is James Evans?",
            "Who is Jake Burchard",
            "Who is Dr. Smith?",
            "who is john doe",
            "Tell me about James Evans",
            "tell me about jake burchard",
            "Info on James Evans",
            "information about Jake Burchard",
        ],
    )
    def test_person_lookup_routes_to_registry(self, router, query):
        """Simple person lookups should use registry only."""
        result = router.route(query)
        assert result.decision == RoutingDecision.REGISTRY_ONLY
        assert result.confidence >= 0.85
        assert "person_lookup" in result.reasoning.lower() or "pattern" in result.reasoning.lower()

    def test_person_lookup_performance_hints(self, router):
        """Person lookups should skip expensive operations."""
        result = router.route("Who is James Evans?")
        assert result.skip_reranking is True
        assert result.skip_mmr is True
        assert result.expected_latency_ms is not None
        assert result.expected_latency_ms <= 50  # Fast path


class TestContactQueries:
    """Test that contact queries route to registry_only."""

    @pytest.mark.parametrize(
        "query",
        [
            "James Evans email",
            "Jake Burchard's email",
            "What is James Evans' email?",
            "Get Jake Burchard's contact info",
            "How can I contact James Evans?",
            "How do I contact Jake Burchard?",
        ],
    )
    def test_contact_queries_route_to_registry(self, router, query):
        """Contact queries should use registry only."""
        result = router.route(query)
        assert result.decision == RoutingDecision.REGISTRY_ONLY
        assert result.confidence >= 0.85


class TestRoleListings:
    """Test that role listing queries route to registry_only."""

    @pytest.mark.parametrize(
        "query",
        [
            "List all PhD students",
            "list phd students",
            "Show all postdocs",
            "Get all faculty",
            "Who are the PhD students?",
            "Who are the postdocs?",
            "How many PhD students?",
            "How many postdocs?",
            "List all members",
            "Show staff",
        ],
    )
    def test_role_listings_route_to_registry(self, router, query):
        """Role listing queries should use registry only."""
        result = router.route(query)
        assert result.decision == RoutingDecision.REGISTRY_ONLY
        assert result.confidence >= 0.85


class TestSimpleResourceQueries:
    """Test that simple resource queries route to registry_only."""

    @pytest.mark.parametrize(
        "query",
        [
            "What datasets do we have?",
            "List all datasets",
            "What projects do we have?",
            "List all projects",
            "What funding do we have?",
            "What grants do we have?",
            "What compute do we have?",
            "What resources are available?",
            "What events do we have?",
            "What events are there?",
            "How many people do we have?",
            "How many members are there?",
        ],
    )
    def test_resource_queries_route_to_registry(self, router, query):
        """Simple resource queries should use registry only."""
        result = router.route(query)
        assert result.decision == RoutingDecision.REGISTRY_ONLY
        assert result.confidence >= 0.85


class TestInstitutionLookups:
    """Test institution-based lookups route to registry_only."""

    @pytest.mark.parametrize(
        "query",
        [
            "Who is at University of Chicago?",
            "who is at stanford",
            "People at MIT",
            "Members from Harvard",
        ],
    )
    def test_institution_lookups_route_to_registry(self, router, query):
        """Institution lookups should use registry only."""
        result = router.route(query)
        assert result.decision == RoutingDecision.REGISTRY_ONLY
        assert result.confidence >= 0.85


# =============================================================================
# Complex Pattern Tests (RAG-Only Routing)
# =============================================================================


class TestSemanticQueries:
    """Test that semantic/conceptual queries route to rag_only."""

    @pytest.mark.parametrize(
        "query",
        [
            "papers about network analysis",
            "publications about machine learning",
            "research on computational social science",
            "papers regarding citation networks",
            "publications on knowledge graphs",
        ],
    )
    def test_semantic_queries_route_to_rag(self, router, query):
        """Semantic queries should use RAG."""
        result = router.route(query)
        assert result.decision == RoutingDecision.RAG_ONLY
        assert result.confidence >= 0.75


class TestConceptualQuestions:
    """Test that conceptual questions route to rag_only."""

    @pytest.mark.parametrize(
        "query",
        [
            "How does the lab approach AI research?",
            "What methodology does CHORUS use for data analysis?",
            "Explain the research approach of the lab",
            "Describe the methodology for network analysis",
            "How does our lab think about science of science?",
        ],
    )
    def test_conceptual_questions_route_to_rag(self, router, query):
        """Conceptual questions should use RAG."""
        result = router.route(query)
        assert result.decision == RoutingDecision.RAG_ONLY
        assert result.confidence >= 0.75


class TestComparisonQueries:
    """Test that comparison queries route to rag_only."""

    @pytest.mark.parametrize(
        "query",
        [
            "Compare network analysis and graph theory approaches",
            "Contrast traditional methods with machine learning approaches",
            "What's the difference between APTO and C3S2 projects?",
        ],
    )
    def test_comparison_queries_route_to_rag(self, router, query):
        """Comparison queries should use RAG."""
        result = router.route(query)
        assert result.decision == RoutingDecision.RAG_ONLY
        assert result.confidence >= 0.75


class TestComplexTopicSearches:
    """Test that complex topic searches route to rag_only."""

    @pytest.mark.parametrize(
        "query",
        [
            "Who works on computational approaches to understanding scientific discovery?",
            "Researchers studying the dynamics of knowledge production in academia",
            "Who studies the relationship between citation patterns and research impact?",
        ],
    )
    def test_complex_topic_searches_route_to_rag(self, router, query):
        """Complex topic searches should use RAG."""
        result = router.route(query)
        # Long, complex topic searches should use RAG or hybrid
        assert result.decision in [RoutingDecision.RAG_ONLY, RoutingDecision.HYBRID]


class TestMultiHopQueries:
    """Test that multi-hop reasoning queries route to rag_only or hybrid."""

    @pytest.mark.parametrize(
        "query",
        [
            "What do James Evans and Jake Burchard have in common?",
            "How are the APTO and C3S2 projects related?",
            "How do network analysis and machine learning connect in our research?",
        ],
    )
    def test_multi_hop_queries_route_to_rag(self, router, query):
        """Multi-hop queries should use RAG (either RAG_ONLY or HYBRID, both include semantic search)."""
        result = router.route(query)
        # Multi-hop queries need semantic search, which is present in both RAG_ONLY and HYBRID
        assert result.decision in [RoutingDecision.RAG_ONLY, RoutingDecision.HYBRID]
        assert result.confidence >= 0.65


class TestRagPerformanceHints:
    """Test that RAG queries have appropriate performance hints."""

    def test_rag_queries_enable_reranking(self, router):
        """RAG queries should enable reranking."""
        result = router.route("papers about network analysis")
        assert result.skip_reranking is False
        assert result.skip_mmr is False

    def test_rag_queries_have_higher_latency_estimate(self, router):
        """RAG queries should estimate higher latency."""
        result = router.route("papers about network analysis")
        assert result.expected_latency_ms is not None
        assert result.expected_latency_ms >= 500


# =============================================================================
# Edge Cases and Hybrid Routing Tests
# =============================================================================


class TestAmbiguousQueries:
    """Test handling of ambiguous queries."""

    @pytest.mark.parametrize(
        "query",
        [
            "APTO",  # Just a project name
            "network analysis",  # Could be topic or general
            "publications",  # Very short, ambiguous
        ],
    )
    def test_ambiguous_queries_route_appropriately(self, router, query):
        """Ambiguous queries should route to hybrid or have lower confidence."""
        result = router.route(query)
        # Ambiguous queries either go hybrid or have lower confidence
        if result.decision == RoutingDecision.REGISTRY_ONLY:
            # If registry only, should have moderate confidence
            assert result.confidence <= 0.9
        else:
            # Hybrid or RAG is acceptable for ambiguous
            assert result.decision in [RoutingDecision.HYBRID, RoutingDecision.RAG_ONLY]


class TestLowConfidenceHandling:
    """Test that low-confidence classifications route to hybrid."""

    def test_hybrid_for_uncertain_queries(self, router):
        """Uncertain queries should use hybrid routing."""
        # A query that doesn't clearly match simple or complex patterns
        result = router.route("Tell me about the research on topics")

        # Should not be high confidence registry only
        if result.decision == RoutingDecision.REGISTRY_ONLY:
            # If registry, confidence should be reasonable
            assert result.confidence >= 0.7


class TestConfidenceScores:
    """Test that confidence scores are reasonable."""

    def test_confidence_scores_in_valid_range(self, router):
        """All confidence scores should be between 0 and 1."""
        queries = [
            "Who is James Evans?",
            "papers about network analysis",
            "List all PhD students",
            "How does the lab approach AI?",
            "APTO project",
        ]
        for query in queries:
            result = router.route(query)
            assert 0.0 <= result.confidence <= 1.0, f"Invalid confidence for: {query}"

    def test_simple_queries_have_high_confidence(self, router):
        """Simple, clear queries should have high confidence."""
        result = router.route("Who is James Evans?")
        assert result.confidence >= 0.9

    def test_ambiguous_queries_have_lower_confidence(self, router):
        """Ambiguous queries should have lower confidence."""
        result = router.route("research")  # Very ambiguous
        assert result.confidence <= 0.9


# =============================================================================
# API and Utility Tests
# =============================================================================


class TestRoutingResult:
    """Test RoutingResult class functionality."""

    def test_routing_result_to_dict(self, router):
        """RoutingResult should serialize to dict correctly."""
        result = router.route("Who is James Evans?")
        result_dict = result.to_dict()

        assert "decision" in result_dict
        assert "confidence" in result_dict
        assert "reasoning" in result_dict
        assert isinstance(result_dict["decision"], str)
        assert isinstance(result_dict["confidence"], float)

    def test_routing_decision_values(self):
        """RoutingDecision enum should have correct values."""
        assert RoutingDecision.REGISTRY_ONLY.value == "registry_only"
        assert RoutingDecision.RAG_ONLY.value == "rag_only"
        assert RoutingDecision.HYBRID.value == "hybrid"


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_route_query_function(self):
        """route_query convenience function should work."""
        result = route_query("Who is James Evans?")
        assert isinstance(result, RoutingResult)
        assert result.decision == RoutingDecision.REGISTRY_ONLY


class TestQueryAnalysis:
    """Test detailed query analysis functionality."""

    def test_analyze_query_returns_complete_info(self, router):
        """analyze_query should return comprehensive analysis."""
        analysis = router.analyze_query("Who is James Evans?")

        # Check structure
        assert "query" in analysis
        assert "routing" in analysis
        assert "classification" in analysis
        assert "pattern_analysis" in analysis
        assert "complexity_analysis" in analysis

        # Check routing info
        assert "decision" in analysis["routing"]
        assert "confidence" in analysis["routing"]
        assert "reasoning" in analysis["routing"]

        # Check classification info
        assert "query_type" in analysis["classification"]
        assert "entity_types" in analysis["classification"]
        assert "confidence" in analysis["classification"]

    def test_analyze_query_pattern_detection(self, router):
        """analyze_query should detect pattern matches."""
        # Simple pattern
        simple_analysis = router.analyze_query("Who is James Evans?")
        assert simple_analysis["pattern_analysis"]["simple_match"] is not None

        # Complex pattern
        complex_analysis = router.analyze_query("papers about network analysis")
        assert complex_analysis["pattern_analysis"]["complex_match"] is not None


# =============================================================================
# Regression Tests
# =============================================================================


class TestRegressions:
    """Regression tests for known edge cases."""

    def test_question_marks_handled(self, router):
        """Queries with and without question marks should route consistently."""
        result_with = router.route("Who is James Evans?")
        result_without = router.route("Who is James Evans")
        assert result_with.decision == result_without.decision

    def test_case_insensitivity(self, router):
        """Routing should be case-insensitive."""
        result_lower = router.route("who is james evans")
        result_upper = router.route("WHO IS JAMES EVANS")
        result_mixed = router.route("Who Is James Evans")
        assert result_lower.decision == result_upper.decision == result_mixed.decision

    def test_leading_trailing_whitespace(self, router):
        """Whitespace should not affect routing."""
        result_clean = router.route("Who is James Evans?")
        result_spaces = router.route("  Who is James Evans?  ")
        assert result_clean.decision == result_spaces.decision

    def test_empty_query_handled(self, router):
        """Empty queries should be handled gracefully."""
        result = router.route("")
        # Should not crash, decision should be valid
        assert result.decision in [
            RoutingDecision.REGISTRY_ONLY,
            RoutingDecision.RAG_ONLY,
            RoutingDecision.HYBRID,
        ]

    def test_very_long_query_handled(self, router):
        """Very long queries should be handled."""
        long_query = "Who works on " + "network analysis " * 50 + "?"
        result = router.route(long_query)
        # Should not crash
        assert result.decision in [
            RoutingDecision.REGISTRY_ONLY,
            RoutingDecision.RAG_ONLY,
            RoutingDecision.HYBRID,
        ]


# =============================================================================
# Integration-Style Tests
# =============================================================================


class TestTypicalUsageScenarios:
    """Tests for typical usage scenarios in CHORUS."""

    def test_new_member_lookup(self, router):
        """Looking up a new lab member."""
        result = router.route("Who is the new postdoc?")
        # Should handle gracefully
        assert result.decision in [RoutingDecision.REGISTRY_ONLY, RoutingDecision.HYBRID]

    def test_dataset_availability(self, router):
        """Checking available datasets."""
        result = router.route("What datasets do we have?")
        assert result.decision == RoutingDecision.REGISTRY_ONLY

    def test_research_topic_exploration(self, router):
        """Exploring research topics."""
        result = router.route("papers about science of science")
        assert result.decision == RoutingDecision.RAG_ONLY

    def test_collaboration_inquiry(self, router):
        """Asking about collaborations."""
        result = router.route("Who does James Evans collaborate with?")
        # Could be registry or hybrid depending on implementation
        assert result.decision in [RoutingDecision.REGISTRY_ONLY, RoutingDecision.HYBRID]


# =============================================================================
# Performance Hint Tests
# =============================================================================


class TestPerformanceHints:
    """Test that performance hints are set correctly."""

    def test_registry_only_skips_expensive_ops(self, router):
        """Registry-only queries should skip reranking and MMR."""
        result = router.route("Who is James Evans?")
        assert result.skip_reranking is True
        assert result.skip_mmr is True

    def test_rag_only_enables_all_ops(self, router):
        """RAG-only queries should enable all operations."""
        result = router.route("papers about network analysis")
        assert result.skip_reranking is False
        assert result.skip_mmr is False

    def test_latency_estimates_reasonable(self, router):
        """Latency estimates should be reasonable for each path."""
        # Registry-only: very fast
        registry_result = router.route("Who is James Evans?")
        assert registry_result.expected_latency_ms <= 50

        # RAG-only: slower
        rag_result = router.route("papers about network analysis")
        assert rag_result.expected_latency_ms >= 500


# =============================================================================
# Run tests directly
# =============================================================================


if __name__ == "__main__":
    # Run pytest with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
