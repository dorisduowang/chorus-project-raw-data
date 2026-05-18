"""
Confidence Scoring Module for CHORUS

Provides confidence scoring and source citation tracking for agent responses.
Helps users understand answer reliability and information sources.

Features:
- Multi-factor confidence scoring (sources, type, similarity, consistency)
- Source citation formatting for CLI and Slack
- LLM-enhanced confidence validation using Claude Haiku
"""

import os
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum

import anthropic


# =============================================================================
# Data Structures
# =============================================================================

class SourceType(Enum):
    """Source types ordered by reliability (higher = more reliable)."""
    REGISTRY = "registry"      # Structured lab data (most reliable)
    DOCUMENT = "document"      # Indexed documents
    WEB = "web"                # External web sources
    INFERENCE = "inference"    # LLM reasoning/inference (least reliable)


@dataclass
class Source:
    """Represents a single information source."""
    type: SourceType
    name: str  # Source name/title
    citation: str  # Formatted citation string
    relevance: float = 1.0  # 0.0 to 1.0, how relevant to query
    url: Optional[str] = None  # Link if available

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "type": self.type.value,
            "name": self.name,
            "citation": self.citation,
            "relevance": self.relevance,
            "url": self.url,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Source":
        """Create from dictionary."""
        return cls(
            type=SourceType(data["type"]),
            name=data["name"],
            citation=data["citation"],
            relevance=data.get("relevance", 1.0),
            url=data.get("url"),
        )


@dataclass
class ConfidenceScore:
    """
    Confidence assessment for a response.

    Combines multiple factors into an overall confidence level.
    """
    level: str  # "high", "medium", "low", "uncertain"
    score: float  # 0.0 to 1.0
    reasoning: str  # Explanation for this confidence level
    sources: List[Source] = field(default_factory=list)
    factors: Dict[str, float] = field(default_factory=dict)  # Individual factor scores

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "level": self.level,
            "score": self.score,
            "reasoning": self.reasoning,
            "sources": [s.to_dict() for s in self.sources],
            "factors": self.factors,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConfidenceScore":
        """Create from dictionary."""
        return cls(
            level=data["level"],
            score=data["score"],
            reasoning=data["reasoning"],
            sources=[Source.from_dict(s) for s in data.get("sources", [])],
            factors=data.get("factors", {}),
        )

    def format_for_display(self, include_sources: bool = True, format_type: str = "cli") -> str:
        """
        Format confidence for display.

        Args:
            include_sources: Whether to include source list
            format_type: "cli" for terminal, "slack" for Slack formatting

        Returns:
            Formatted string
        """
        lines = []

        # Confidence indicator
        level_display = self.level.upper()
        if format_type == "slack":
            # Use Slack emoji indicators
            emoji_map = {
                "high": ":large_green_circle:",
                "medium": ":large_yellow_circle:",
                "low": ":large_orange_circle:",
                "uncertain": ":red_circle:",
            }
            emoji = emoji_map.get(self.level, ":white_circle:")
            lines.append(f"{emoji} *Confidence: {level_display}*")
        else:
            lines.append(f"Confidence: {level_display}")

        # Sources section
        if include_sources and self.sources:
            lines.append("")
            if format_type == "slack":
                lines.append("*Sources:*")
            else:
                lines.append("Sources:")

            for source in self.sources:
                source_line = self._format_source(source, format_type)
                lines.append(source_line)

        return "\n".join(lines)

    def _format_source(self, source: Source, format_type: str) -> str:
        """Format a single source for display."""
        type_label = f"[{source.type.value.title()}]"

        if source.url and format_type == "slack":
            # Slack link format
            citation = f"<{source.url}|{source.name}>"
        elif source.url:
            citation = f"{source.name} ({source.url})"
        else:
            citation = source.citation

        # Include relevance if not 1.0
        if source.relevance < 1.0:
            relevance_str = f" (relevance: {source.relevance:.2f})"
        else:
            relevance_str = ""

        return f"- {type_label} {citation}{relevance_str}"


# =============================================================================
# Confidence Calculator
# =============================================================================

# Source type reliability weights (higher = more reliable)
SOURCE_WEIGHTS = {
    SourceType.REGISTRY: 1.0,
    SourceType.DOCUMENT: 0.8,
    SourceType.WEB: 0.6,
    SourceType.INFERENCE: 0.4,
}


class ConfidenceCalculator:
    """
    Calculate confidence scores based on multiple factors.

    Factors considered:
    - Source count and quality
    - Source type weights (registry > docs > web > inference)
    - Semantic similarity scores
    - Presence of structured data
    - Consistency across sources
    """

    def __init__(self, enable_llm_validation: bool = True):
        """
        Initialize calculator.

        Args:
            enable_llm_validation: Whether to use LLM for confidence validation
        """
        self.enable_llm_validation = enable_llm_validation
        self._client = None

    @property
    def client(self) -> anthropic.Anthropic:
        """Lazy initialization of Anthropic client."""
        if self._client is None:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if api_key:
                self._client = anthropic.Anthropic(api_key=api_key)
            else:
                self._client = anthropic.Anthropic()
        return self._client

    def calculate(
        self,
        sources: List[Source],
        query: str,
        response_text: str,
        classification_confidence: float = 1.0,
        has_structured_data: bool = False,
        similarity_scores: Optional[List[float]] = None,
    ) -> ConfidenceScore:
        """
        Calculate confidence score for a response.

        Args:
            sources: List of sources used in the response
            query: Original user query
            response_text: Generated response text
            classification_confidence: Confidence from query classification
            has_structured_data: Whether structured registry data was found
            similarity_scores: Semantic similarity scores from search results

        Returns:
            ConfidenceScore with level, score, reasoning, and sources
        """
        factors = {}

        # Factor 1: Source count (more sources = higher confidence, up to a point)
        source_count = len(sources)
        if source_count == 0:
            factors["source_count"] = 0.1
        elif source_count == 1:
            factors["source_count"] = 0.5
        elif source_count <= 3:
            factors["source_count"] = 0.7
        else:
            factors["source_count"] = min(0.9, 0.7 + source_count * 0.05)

        # Factor 2: Source type quality (weighted average of source types)
        if sources:
            type_scores = [SOURCE_WEIGHTS.get(s.type, 0.5) for s in sources]
            factors["source_quality"] = sum(type_scores) / len(type_scores)
        else:
            factors["source_quality"] = 0.1

        # Factor 3: Structured data presence (strong positive signal)
        factors["structured_data"] = 0.9 if has_structured_data else 0.4

        # Factor 4: Query classification confidence
        factors["classification"] = classification_confidence

        # Factor 5: Semantic similarity (average of top similarity scores)
        if similarity_scores:
            top_scores = similarity_scores[:3]  # Top 3 scores
            avg_similarity = sum(top_scores) / len(top_scores)
            factors["similarity"] = avg_similarity
        else:
            factors["similarity"] = 0.5  # Neutral if no similarity scores

        # Factor 6: Source relevance (average relevance of all sources)
        if sources:
            avg_relevance = sum(s.relevance for s in sources) / len(sources)
            factors["source_relevance"] = avg_relevance
        else:
            factors["source_relevance"] = 0.1

        # Calculate weighted overall score
        weights = {
            "source_count": 0.15,
            "source_quality": 0.25,
            "structured_data": 0.20,
            "classification": 0.10,
            "similarity": 0.15,
            "source_relevance": 0.15,
        }

        overall_score = sum(factors.get(k, 0) * w for k, w in weights.items())

        # Determine level from score
        level = self._score_to_level(overall_score)

        # Generate reasoning
        reasoning = self._generate_reasoning(factors, sources, has_structured_data)

        return ConfidenceScore(
            level=level,
            score=overall_score,
            reasoning=reasoning,
            sources=sources,
            factors=factors,
        )

    def _score_to_level(self, score: float) -> str:
        """Convert numeric score to confidence level."""
        if score >= 0.8:
            return "high"
        elif score >= 0.6:
            return "medium"
        elif score >= 0.4:
            return "low"
        else:
            return "uncertain"

    def _generate_reasoning(
        self,
        factors: Dict[str, float],
        sources: List[Source],
        has_structured_data: bool,
    ) -> str:
        """Generate human-readable reasoning for confidence level."""
        reasons = []

        if has_structured_data:
            reasons.append("Found structured registry data")

        source_count = len(sources)
        if source_count == 0:
            reasons.append("No sources found")
        elif source_count == 1:
            reasons.append(f"Based on single source ({sources[0].type.value})")
        else:
            type_counts = {}
            for s in sources:
                type_counts[s.type.value] = type_counts.get(s.type.value, 0) + 1
            types_str = ", ".join(f"{c} {t}" for t, c in type_counts.items())
            reasons.append(f"Based on {source_count} sources ({types_str})")

        if factors.get("similarity", 0) < 0.5:
            reasons.append("Low semantic match to query")
        elif factors.get("similarity", 0) > 0.8:
            reasons.append("High semantic relevance")

        if factors.get("classification", 0) < 0.7:
            reasons.append("Query intent unclear")

        return ". ".join(reasons) if reasons else "Standard confidence"

    async def validate_with_llm(
        self,
        query: str,
        response_text: str,
        sources: List[Source],
        current_confidence: ConfidenceScore,
    ) -> ConfidenceScore:
        """
        Use LLM to validate if the response actually answers the question.

        This catches cases where:
        - Sources don't really support the answer
        - Response doesn't address the actual question
        - Model should say "I don't know" instead

        Args:
            query: Original user query
            response_text: Generated response
            sources: Sources used
            current_confidence: Pre-LLM confidence score

        Returns:
            Updated ConfidenceScore (may be adjusted down if LLM finds issues)
        """
        if not self.enable_llm_validation:
            return current_confidence

        try:
            # Build source summary for LLM
            source_summary = "\n".join(
                f"- [{s.type.value}] {s.citation}"
                for s in sources[:5]  # Limit to top 5 sources
            )
            if not source_summary:
                source_summary = "(no sources provided)"

            prompt = f"""Evaluate if this response adequately answers the user's question based on the sources provided.

USER QUESTION: {query}

RESPONSE: {response_text}

SOURCES USED:
{source_summary}

Evaluate:
1. Does the response directly answer what was asked?
2. Is the answer supported by the sources?
3. Are there any unsupported claims?
4. Should the system say "I don't know" instead?

Respond with a JSON object:
{{
    "answers_question": true/false,
    "supported_by_sources": true/false,
    "should_say_unsure": true/false,
    "confidence_adjustment": 0 to -0.3 (negative if issues found),
    "issue": "brief description if any issue found, or null"
}}

Only output the JSON, nothing else."""

            response = self.client.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}]
            )

            # Parse response
            import json
            result_text = response.content[0].text.strip()
            # Handle potential markdown code blocks
            if result_text.startswith("```"):
                result_text = result_text.split("```")[1]
                if result_text.startswith("json"):
                    result_text = result_text[4:]
                result_text = result_text.strip()

            result = json.loads(result_text)

            # Apply adjustments
            adjustment = result.get("confidence_adjustment", 0)
            new_score = max(0.0, current_confidence.score + adjustment)
            new_level = self._score_to_level(new_score)

            # Update reasoning if issues found
            issue = result.get("issue")
            new_reasoning = current_confidence.reasoning
            if issue:
                new_reasoning += f". LLM validation: {issue}"

            # If LLM says we should be unsure, force low confidence
            if result.get("should_say_unsure"):
                new_score = min(new_score, 0.35)
                new_level = "uncertain"
                new_reasoning += ". LLM suggests expressing uncertainty"

            return ConfidenceScore(
                level=new_level,
                score=new_score,
                reasoning=new_reasoning,
                sources=current_confidence.sources,
                factors={**current_confidence.factors, "llm_validation": 1.0 + adjustment},
            )

        except Exception as e:
            # On any error, return original confidence
            return current_confidence


# =============================================================================
# Source Extraction Utilities
# =============================================================================

def extract_sources_from_rag_result(rag_result: Dict[str, Any]) -> List[Source]:
    """
    Extract Source objects from RAG search results.

    Args:
        rag_result: Result from hybrid search endpoint

    Returns:
        List of Source objects
    """
    sources = []

    # Extract from structured results (registry data)
    for result in rag_result.get("structured_results", []):
        result_type = result.get("type", "")
        data = result.get("data", {})

        if result_type == "person":
            name = data.get("name", "Unknown")
            sources.append(Source(
                type=SourceType.REGISTRY,
                name=f"{name} profile",
                citation=f"{name} profile (lab_registry.json)",
                relevance=result.get("score", 1.0) if "score" in result else 1.0,
                url=data.get("openalex", {}).get("id"),
            ))
        elif result_type == "project":
            name = data.get("name", "Unknown Project")
            sources.append(Source(
                type=SourceType.REGISTRY,
                name=name,
                citation=f"{name} (lab_registry.json)",
                relevance=result.get("score", 1.0) if "score" in result else 1.0,
            ))
        elif result_type == "publication":
            title = data.get("title", "Untitled")[:50]
            sources.append(Source(
                type=SourceType.REGISTRY,
                name=title,
                citation=f"{title} (registry publication)",
                relevance=result.get("score", 1.0) if "score" in result else 1.0,
                url=data.get("url"),
            ))
        else:
            # Generic registry entry
            name = data.get("name", result_type)
            sources.append(Source(
                type=SourceType.REGISTRY,
                name=name,
                citation=f"{name} (lab_registry.json)",
                relevance=result.get("score", 1.0) if "score" in result else 1.0,
            ))

    # Extract from semantic results (documents)
    for result in rag_result.get("semantic_results", rag_result.get("results", [])):
        citation = result.get("citation", result.get("source_file", "unknown"))
        text_preview = result.get("text", "")[:30]
        score = result.get("score", 0.5)

        sources.append(Source(
            type=SourceType.DOCUMENT,
            name=citation,
            citation=citation,
            relevance=score,
        ))

    return sources


def extract_sources_from_web_results(web_results: List[Dict[str, Any]]) -> List[Source]:
    """
    Extract Source objects from web search results.

    Args:
        web_results: List of web search result dicts

    Returns:
        List of Source objects
    """
    sources = []

    for i, result in enumerate(web_results):
        title = result.get("title", "Web Result")
        url = result.get("url", "")

        # Decay relevance for lower-ranked results
        relevance = max(0.5, 1.0 - (i * 0.1))

        sources.append(Source(
            type=SourceType.WEB,
            name=title,
            citation=title,
            relevance=relevance,
            url=url,
        ))

    return sources


def create_inference_source(description: str, relevance: float = 0.5) -> Source:
    """
    Create a source for LLM inference/reasoning.

    Used when the response is based on the LLM's reasoning rather than
    specific sources.

    Args:
        description: Description of the inference
        relevance: Confidence in the inference

    Returns:
        Source object
    """
    return Source(
        type=SourceType.INFERENCE,
        name="LLM Inference",
        citation=description,
        relevance=relevance,
    )
