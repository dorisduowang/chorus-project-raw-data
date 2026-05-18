"""
Unit Tests for Confidence Scoring Module

Tests for:
- Source creation and serialization
- ConfidenceScore calculation
- ConfidenceCalculator factor weighting
- Source extraction from RAG results
- Display formatting (CLI and Slack)
- Edge cases (no sources, single source, etc.)
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

# Add parent directory to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.confidence import (
    Source,
    SourceType,
    ConfidenceScore,
    ConfidenceCalculator,
    SOURCE_WEIGHTS,
    extract_sources_from_rag_result,
    extract_sources_from_web_results,
    create_inference_source,
)


# =============================================================================
# Test: Source Dataclass
# =============================================================================

class TestSource:
    """Tests for Source dataclass."""

    def test_source_creation_minimal(self):
        """Test creating a source with minimal fields."""
        source = Source(
            type=SourceType.REGISTRY,
            name="Jake Burchard",
            citation="Jake Burchard profile (lab_registry.json)",
        )

        assert source.type == SourceType.REGISTRY
        assert source.name == "Jake Burchard"
        assert source.citation == "Jake Burchard profile (lab_registry.json)"
        assert source.relevance == 1.0  # Default
        assert source.url is None  # Default

    def test_source_creation_full(self):
        """Test creating a source with all fields."""
        source = Source(
            type=SourceType.WEB,
            name="OpenAlex Author",
            citation="OpenAlex Author A5074990827",
            relevance=0.85,
            url="https://openalex.org/A5074990827",
        )

        assert source.type == SourceType.WEB
        assert source.relevance == 0.85
        assert source.url == "https://openalex.org/A5074990827"

    def test_source_to_dict(self):
        """Test serialization to dictionary."""
        source = Source(
            type=SourceType.DOCUMENT,
            name="APTO Proposal",
            citation="APTO Proposal 2024, p.12",
            relevance=0.92,
            url=None,
        )

        data = source.to_dict()

        assert data["type"] == "document"
        assert data["name"] == "APTO Proposal"
        assert data["citation"] == "APTO Proposal 2024, p.12"
        assert data["relevance"] == 0.92
        assert data["url"] is None

    def test_source_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "type": "registry",
            "name": "James Evans profile",
            "citation": "James Evans profile (lab_registry.json)",
            "relevance": 1.0,
            "url": "https://openalex.org/A123456",
        }

        source = Source.from_dict(data)

        assert source.type == SourceType.REGISTRY
        assert source.name == "James Evans profile"
        assert source.url == "https://openalex.org/A123456"

    def test_source_roundtrip(self):
        """Test that to_dict/from_dict roundtrip preserves data."""
        original = Source(
            type=SourceType.INFERENCE,
            name="LLM Inference",
            citation="Based on context analysis",
            relevance=0.65,
            url=None,
        )

        reconstructed = Source.from_dict(original.to_dict())

        assert reconstructed.type == original.type
        assert reconstructed.name == original.name
        assert reconstructed.citation == original.citation
        assert reconstructed.relevance == original.relevance
        assert reconstructed.url == original.url


# =============================================================================
# Test: ConfidenceScore Dataclass
# =============================================================================

class TestConfidenceScore:
    """Tests for ConfidenceScore dataclass."""

    def test_confidence_score_creation(self):
        """Test creating a confidence score."""
        sources = [
            Source(SourceType.REGISTRY, "Jake", "Jake profile"),
            Source(SourceType.DOCUMENT, "Paper", "Paper citation"),
        ]

        score = ConfidenceScore(
            level="high",
            score=0.85,
            reasoning="Found in registry and documents",
            sources=sources,
            factors={"source_count": 0.7, "source_quality": 0.9},
        )

        assert score.level == "high"
        assert score.score == 0.85
        assert len(score.sources) == 2
        assert "source_count" in score.factors

    def test_confidence_score_to_dict(self):
        """Test serialization to dictionary."""
        score = ConfidenceScore(
            level="medium",
            score=0.65,
            reasoning="Based on documents only",
            sources=[Source(SourceType.DOCUMENT, "Doc", "Doc citation")],
            factors={"source_count": 0.5},
        )

        data = score.to_dict()

        assert data["level"] == "medium"
        assert data["score"] == 0.65
        assert len(data["sources"]) == 1
        assert data["sources"][0]["type"] == "document"

    def test_confidence_score_format_cli(self):
        """Test CLI display formatting."""
        score = ConfidenceScore(
            level="high",
            score=0.85,
            reasoning="Strong evidence",
            sources=[
                Source(SourceType.REGISTRY, "Jake Burchard", "Jake Burchard profile"),
                Source(SourceType.WEB, "OpenAlex", "OpenAlex data", url="https://openalex.org/123"),
            ],
        )

        output = score.format_for_display(format_type="cli")

        assert "Confidence: HIGH" in output
        assert "Sources:" in output
        assert "[Registry]" in output
        assert "[Web]" in output

    def test_confidence_score_format_slack(self):
        """Test Slack display formatting."""
        score = ConfidenceScore(
            level="low",
            score=0.35,
            reasoning="Weak evidence",
            sources=[
                Source(SourceType.INFERENCE, "LLM", "Based on reasoning", relevance=0.4),
            ],
        )

        output = score.format_for_display(format_type="slack")

        assert ":large_orange_circle:" in output
        assert "*Confidence: LOW*" in output
        assert "*Sources:*" in output
        assert "[Inference]" in output
        assert "(relevance: 0.40)" in output

    def test_confidence_score_format_no_sources(self):
        """Test formatting with no sources."""
        score = ConfidenceScore(
            level="uncertain",
            score=0.2,
            reasoning="No sources found",
            sources=[],
        )

        output = score.format_for_display()

        assert "Confidence: UNCERTAIN" in output
        assert "Sources:" not in output  # Should not show empty sources section


# =============================================================================
# Test: ConfidenceCalculator
# =============================================================================

class TestConfidenceCalculator:
    """Tests for ConfidenceCalculator."""

    def test_calculate_high_confidence(self):
        """Test calculation resulting in high confidence."""
        calculator = ConfidenceCalculator(enable_llm_validation=False)

        sources = [
            Source(SourceType.REGISTRY, "Person1", "citation1", relevance=0.95),
            Source(SourceType.REGISTRY, "Person2", "citation2", relevance=0.90),
            Source(SourceType.DOCUMENT, "Doc1", "citation3", relevance=0.85),
        ]

        result = calculator.calculate(
            sources=sources,
            query="Who is Jake Burchard?",
            response_text="Jake Burchard is a PhD student...",
            classification_confidence=0.95,
            has_structured_data=True,
            similarity_scores=[0.92, 0.88, 0.85],
        )

        assert result.level == "high"
        assert result.score >= 0.8
        assert len(result.sources) == 3
        assert "structured registry data" in result.reasoning.lower()

    def test_calculate_low_confidence(self):
        """Test calculation resulting in low confidence."""
        calculator = ConfidenceCalculator(enable_llm_validation=False)

        sources = [
            Source(SourceType.INFERENCE, "LLM", "inference", relevance=0.4),
        ]

        result = calculator.calculate(
            sources=sources,
            query="What is the meaning of life?",
            response_text="The meaning of life is...",
            classification_confidence=0.5,
            has_structured_data=False,
            similarity_scores=[0.3, 0.25, 0.2],
        )

        assert result.level in ["low", "uncertain"]
        assert result.score < 0.6

    def test_calculate_no_sources(self):
        """Test calculation with no sources."""
        calculator = ConfidenceCalculator(enable_llm_validation=False)

        result = calculator.calculate(
            sources=[],
            query="Unknown query",
            response_text="I don't have information...",
            classification_confidence=0.6,
            has_structured_data=False,
        )

        assert result.level in ["low", "uncertain"]
        assert result.score < 0.5
        assert "no sources found" in result.reasoning.lower()

    def test_calculate_medium_confidence(self):
        """Test calculation resulting in medium confidence."""
        calculator = ConfidenceCalculator(enable_llm_validation=False)

        sources = [
            Source(SourceType.DOCUMENT, "Doc1", "citation1", relevance=0.7),
            Source(SourceType.DOCUMENT, "Doc2", "citation2", relevance=0.65),
        ]

        result = calculator.calculate(
            sources=sources,
            query="What research projects exist?",
            response_text="There are several projects...",
            classification_confidence=0.8,
            has_structured_data=False,
            similarity_scores=[0.75, 0.70],
        )

        assert result.level in ["medium", "high"]
        assert 0.5 <= result.score <= 0.85

    def test_source_type_weighting(self):
        """Test that registry sources score higher than inference sources."""
        calculator = ConfidenceCalculator(enable_llm_validation=False)

        # Registry-based response
        registry_result = calculator.calculate(
            sources=[Source(SourceType.REGISTRY, "Person", "citation", relevance=0.9)],
            query="Who is X?",
            response_text="X is...",
            classification_confidence=0.9,
            has_structured_data=True,
        )

        # Inference-based response
        inference_result = calculator.calculate(
            sources=[Source(SourceType.INFERENCE, "LLM", "citation", relevance=0.9)],
            query="Who is X?",
            response_text="X is...",
            classification_confidence=0.9,
            has_structured_data=False,
        )

        assert registry_result.score > inference_result.score
        assert registry_result.factors["source_quality"] > inference_result.factors["source_quality"]

    def test_factors_are_recorded(self):
        """Test that all factors are recorded in the result."""
        calculator = ConfidenceCalculator(enable_llm_validation=False)

        result = calculator.calculate(
            sources=[Source(SourceType.DOCUMENT, "Doc", "citation")],
            query="Test query",
            response_text="Test response",
            classification_confidence=0.7,
            has_structured_data=False,
            similarity_scores=[0.8],
        )

        expected_factors = [
            "source_count",
            "source_quality",
            "structured_data",
            "classification",
            "similarity",
            "source_relevance",
        ]

        for factor in expected_factors:
            assert factor in result.factors


# =============================================================================
# Test: Source Extraction Utilities
# =============================================================================

class TestSourceExtraction:
    """Tests for source extraction utilities."""

    def test_extract_from_rag_person(self):
        """Test extracting sources from RAG result with person data."""
        rag_result = {
            "structured_results": [
                {
                    "type": "person",
                    "data": {
                        "name": "Jake Burchard",
                        "role": "PhD Student",
                        "openalex": {"id": "https://openalex.org/A123"},
                    },
                },
            ],
            "semantic_results": [],
        }

        sources = extract_sources_from_rag_result(rag_result)

        assert len(sources) == 1
        assert sources[0].type == SourceType.REGISTRY
        assert "Jake Burchard" in sources[0].name
        assert sources[0].url == "https://openalex.org/A123"

    def test_extract_from_rag_mixed(self):
        """Test extracting from RAG result with mixed structured and semantic."""
        rag_result = {
            "structured_results": [
                {
                    "type": "project",
                    "data": {"name": "APTO Project"},
                },
            ],
            "semantic_results": [
                {
                    "citation": "research_notes.md",
                    "text": "Some research notes...",
                    "score": 0.85,
                },
            ],
        }

        sources = extract_sources_from_rag_result(rag_result)

        assert len(sources) == 2
        assert sources[0].type == SourceType.REGISTRY
        assert sources[1].type == SourceType.DOCUMENT
        assert sources[1].relevance == 0.85

    def test_extract_from_web_results(self):
        """Test extracting sources from web search results."""
        web_results = [
            {"title": "OpenAlex Author", "url": "https://openalex.org/123"},
            {"title": "Google Scholar", "url": "https://scholar.google.com/456"},
            {"title": "Wikipedia", "url": "https://en.wikipedia.org/789"},
        ]

        sources = extract_sources_from_web_results(web_results)

        assert len(sources) == 3
        assert all(s.type == SourceType.WEB for s in sources)
        # Relevance should decay for lower-ranked results
        assert sources[0].relevance > sources[2].relevance

    def test_create_inference_source(self):
        """Test creating an inference source."""
        source = create_inference_source("Based on context analysis", relevance=0.6)

        assert source.type == SourceType.INFERENCE
        assert source.name == "LLM Inference"
        assert source.relevance == 0.6


# =============================================================================
# Test: LLM Validation (Mocked)
# =============================================================================

class TestLLMValidation:
    """Tests for LLM-based confidence validation."""

    @pytest.mark.asyncio
    async def test_llm_validation_disabled(self):
        """Test that LLM validation returns original score when disabled."""
        calculator = ConfidenceCalculator(enable_llm_validation=False)

        original = ConfidenceScore(
            level="high",
            score=0.85,
            reasoning="Test",
            sources=[],
        )

        result = await calculator.validate_with_llm(
            query="Test",
            response_text="Response",
            sources=[],
            current_confidence=original,
        )

        assert result.score == original.score
        assert result.level == original.level

    @pytest.mark.asyncio
    async def test_llm_validation_with_mock(self):
        """Test LLM validation with mocked API response."""
        calculator = ConfidenceCalculator(enable_llm_validation=True)

        # Mock the Anthropic client
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"answers_question": true, "supported_by_sources": true, "should_say_unsure": false, "confidence_adjustment": 0, "issue": null}')]

        mock_client = MagicMock()
        mock_client.messages.create = MagicMock(return_value=mock_response)
        calculator._client = mock_client

        original = ConfidenceScore(
            level="high",
            score=0.85,
            reasoning="Test",
            sources=[Source(SourceType.REGISTRY, "Test", "Test citation")],
        )

        result = await calculator.validate_with_llm(
            query="Who is Jake?",
            response_text="Jake is a researcher...",
            sources=original.sources,
            current_confidence=original,
        )

        # Score should remain the same (no adjustment)
        assert result.score == original.score

    @pytest.mark.asyncio
    async def test_llm_validation_adjusts_down(self):
        """Test LLM validation adjusts confidence down when issues found."""
        calculator = ConfidenceCalculator(enable_llm_validation=True)

        # Mock response indicating issues
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"answers_question": false, "supported_by_sources": false, "should_say_unsure": true, "confidence_adjustment": -0.3, "issue": "Response does not address the question"}')]

        mock_client = MagicMock()
        mock_client.messages.create = MagicMock(return_value=mock_response)
        calculator._client = mock_client

        original = ConfidenceScore(
            level="high",
            score=0.85,
            reasoning="Test",
            sources=[],
        )

        result = await calculator.validate_with_llm(
            query="What is the weather?",
            response_text="Jake is a researcher...",
            sources=[],
            current_confidence=original,
        )

        # Score should be adjusted down
        assert result.score < original.score
        assert result.level in ["low", "uncertain"]
        assert "LLM validation" in result.reasoning


# =============================================================================
# Test: Source Type Weights
# =============================================================================

class TestSourceWeights:
    """Tests for source type weight constants."""

    def test_weights_order(self):
        """Test that weights are ordered correctly (registry > doc > web > inference)."""
        assert SOURCE_WEIGHTS[SourceType.REGISTRY] > SOURCE_WEIGHTS[SourceType.DOCUMENT]
        assert SOURCE_WEIGHTS[SourceType.DOCUMENT] > SOURCE_WEIGHTS[SourceType.WEB]
        assert SOURCE_WEIGHTS[SourceType.WEB] > SOURCE_WEIGHTS[SourceType.INFERENCE]

    def test_weights_range(self):
        """Test that all weights are between 0 and 1."""
        for source_type, weight in SOURCE_WEIGHTS.items():
            assert 0 <= weight <= 1, f"Weight for {source_type} out of range: {weight}"


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
