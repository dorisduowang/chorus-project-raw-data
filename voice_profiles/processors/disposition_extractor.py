"""
Disposition Extractor

Uses LLM analysis to extract disposition vectors and critique behaviors
from text samples (Slack messages, meeting transcripts, papers, etc.).
"""

import json
import os
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

try:
    import anthropic
except ImportError:
    anthropic = None

from ..models import (
    DispositionVector,
    CritiqueBehavior,
    IndividualProfile,
    QuestionTypes,
    CuriosityStyle,
)


EXTRACTION_PROMPT = """You are an expert in analyzing intellectual dispositions based on how people communicate, question, and engage with ideas.

Given text samples from a researcher, extract their intellectual disposition profile using this taxonomy:

## DIMENSION 1: EPISTEMIC APPETITE
- **curiosity_style**: "busybody" (broad collector), "hunter" (focused pursuer), or "dancer" (unexpected leaper)
- **scope**: 0=fox (many frameworks), 1=hedgehog (one big idea)
- **surprise_seeking**: 0=confirmation-seeking, 1=novelty-seeking

## DIMENSION 2: COGNITIVE STYLE
- **thinking_mode**: 0=convergent (selects best), 1=divergent (generates possibilities)
- **abstraction**: 0=frog (concrete details), 1=bird (abstract patterns)
- **classification**: 0=lumper (synthesizes), 1=splitter (differentiates)

## DIMENSION 3: METHODOLOGICAL
- **making_style**: 0=engineer (first principles), 1=bricoleur (recombination)
- **formalism**: 0=narrative/qualitative, 1=mathematical/quantitative
- **tacit_explicit**: 0=intuitive ("feeling for"), 1=explicit (codified rules)

## DIMENSION 4: SOCIAL
- **disclosure**: 0=private (shares finished), 1=open (thinks out loud)
- **inquiry**: 0=heads-down, 1=monitors others constantly
- **collaboration**: 0=solitary, 0.5=dyadic, 1=collective
- **competition**: 0=communal (gift), 1=agonistic (priority)

## DIMENSION 5: TEMPORAL
- **paradigm_orientation**: 0=normal science, 1=revolutionary
- **patience**: 0=quick iterations, 1=long-term investments
- **archival**: 0=present-focused, 1=historically-oriented

## DIMENSION 6: AESTHETIC
- **elegance_completeness**: 0=completeness/thoroughness, 1=elegance/parsimony
- **mechanism_phenomenon**: 0=phenomenological ("what is it like?"), 1=mechanist ("how does it work?")
- **risk**: 0=cautious (reliable progress), 1=bold (high-variance bets)

Also extract:
1. **Question types** they tend to ask (clarifying, challenging, building, reframing, connecting, scaling, mechanistic, historical)
2. **Critique patterns**: Characteristic phrases or question forms they use
3. **Signature moves**: Distinctive intellectual moves they make

Respond with a JSON object containing:
- disposition: object with the six dimension categories
- question_types: object with distribution across 8 types (should sum to ~1.0)
- critique_patterns: array of 3-5 characteristic phrases
- signature_moves: array of 2-4 distinctive moves
- confidence: 0-1 indicating how confident you are based on data quality/quantity
- reasoning: brief explanation of key signals in the text

TEXT SAMPLES:
{text_samples}

Respond ONLY with valid JSON."""


QUESTION_EXTRACTION_PROMPT = """Analyze these text samples and extract the types of questions this person asks.

Categorize each question into one of these types:
- clarifying: Seeks definition, scope, precision
- challenging: Questions assumptions, validity, evidence
- building: Extends, adds to, develops the idea further
- reframing: Offers alternative lens or perspective
- connecting: Links to other domains, finds analogies
- scaling: Asks about scale effects, what happens at larger N
- mechanistic: Asks how it works, causal pathways
- historical: References past work, precedent, tradition

TEXT SAMPLES:
{text_samples}

Return a JSON object with:
- questions_found: array of objects with {text, type, context}
- type_distribution: object mapping type -> proportion (sum to 1.0)
- example_patterns: array of characteristic question patterns

Respond ONLY with valid JSON."""


@dataclass
class TextSample:
    """A text sample with metadata."""
    text: str
    source_type: str  # "slack", "transcript", "paper", "review", etc.
    source_id: str
    timestamp: Optional[str] = None
    context: Optional[str] = None


class DispositionExtractor:
    """
    Extracts disposition profiles from text samples using LLM analysis.
    """

    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        """
        Initialize the extractor.

        Args:
            model: Anthropic model to use for extraction
        """
        if anthropic is None:
            raise ImportError("anthropic package required. Install with: pip install anthropic")

        self.model = model
        self.client = anthropic.Anthropic()

    def extract_disposition(
        self,
        samples: List[TextSample],
        person_name: str,
        max_samples: int = 50,
        max_chars_per_sample: int = 1000,
    ) -> Tuple[DispositionVector, CritiqueBehavior, float, str]:
        """
        Extract disposition and critique behavior from text samples.

        Args:
            samples: List of text samples from this person
            person_name: Name of the person
            max_samples: Maximum samples to include in prompt
            max_chars_per_sample: Truncate samples to this length

        Returns:
            Tuple of (DispositionVector, CritiqueBehavior, confidence, reasoning)
        """
        # Prepare samples for prompt
        selected = samples[:max_samples]
        formatted_samples = []

        for i, sample in enumerate(selected, 1):
            text = sample.text[:max_chars_per_sample]
            if len(sample.text) > max_chars_per_sample:
                text += "..."

            formatted_samples.append(
                f"[Sample {i} - {sample.source_type}]\n{text}\n"
            )

        samples_text = "\n---\n".join(formatted_samples)

        # Call LLM
        prompt = EXTRACTION_PROMPT.format(text_samples=samples_text)

        response = self.client.messages.create(
            model=self.model,
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )

        # Parse response
        response_text = response.content[0].text.strip()

        # Handle potential markdown code blocks
        if response_text.startswith("```"):
            lines = response_text.split("\n")
            response_text = "\n".join(lines[1:-1])

        try:
            result = json.loads(response_text)
        except json.JSONDecodeError as e:
            # Return defaults on parse failure
            return (
                DispositionVector(),
                CritiqueBehavior(),
                0.0,
                f"Failed to parse LLM response: {e}"
            )

        # Build DispositionVector
        disp_data = result.get("disposition", {})
        disposition = DispositionVector.from_dict(disp_data)

        # Build CritiqueBehavior
        qt_data = result.get("question_types", {})
        critique_behavior = CritiqueBehavior(
            question_types=QuestionTypes.from_dict(qt_data),
            critique_patterns=result.get("critique_patterns", []),
            signature_moves=result.get("signature_moves", []),
            productive_tensions=[],  # Determined by compound type matching
        )

        confidence = result.get("confidence", 0.5)
        reasoning = result.get("reasoning", "")

        return disposition, critique_behavior, confidence, reasoning

    def extract_questions(
        self,
        samples: List[TextSample],
        max_samples: int = 30,
    ) -> Dict:
        """
        Extract and categorize questions from text samples.

        Args:
            samples: Text samples to analyze
            max_samples: Maximum samples to include

        Returns:
            Dict with questions_found, type_distribution, example_patterns
        """
        selected = samples[:max_samples]
        formatted_samples = []

        for i, sample in enumerate(selected, 1):
            formatted_samples.append(
                f"[Sample {i} - {sample.source_type}]\n{sample.text}\n"
            )

        samples_text = "\n---\n".join(formatted_samples)
        prompt = QUESTION_EXTRACTION_PROMPT.format(text_samples=samples_text)

        response = self.client.messages.create(
            model=self.model,
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )

        response_text = response.content[0].text.strip()

        if response_text.startswith("```"):
            lines = response_text.split("\n")
            response_text = "\n".join(lines[1:-1])

        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            return {
                "questions_found": [],
                "type_distribution": {},
                "example_patterns": [],
            }

    def match_compound_type(
        self,
        disposition: DispositionVector,
        compound_types: Dict[str, "CompoundType"],
    ) -> Tuple[Optional[str], float]:
        """
        Find the best-matching compound type for a disposition.

        Args:
            disposition: The disposition vector to match
            compound_types: Dict of compound types to compare against

        Returns:
            Tuple of (compound_type_name, similarity_score)
        """
        best_match = None
        best_score = 0.0

        for name, compound in compound_types.items():
            score = self._disposition_similarity(disposition, compound.disposition)
            if score > best_score:
                best_score = score
                best_match = name

        return best_match, best_score

    def _disposition_similarity(
        self,
        d1: DispositionVector,
        d2: DispositionVector,
    ) -> float:
        """
        Calculate similarity between two disposition vectors.

        Uses simple Euclidean distance on the numeric dimensions.

        Returns:
            Similarity score 0-1 (1 = identical)
        """
        # Extract all numeric values
        v1 = [
            d1.epistemic_appetite.scope,
            d1.epistemic_appetite.surprise_seeking,
            d1.cognitive_style.thinking_mode,
            d1.cognitive_style.abstraction,
            d1.cognitive_style.classification,
            d1.methodological.making_style,
            d1.methodological.formalism,
            d1.methodological.tacit_explicit,
            d1.social.disclosure,
            d1.social.inquiry,
            d1.social.collaboration,
            d1.social.competition,
            d1.temporal.paradigm_orientation,
            d1.temporal.patience,
            d1.temporal.archival,
            d1.aesthetic.elegance_completeness,
            d1.aesthetic.mechanism_phenomenon,
            d1.aesthetic.risk,
        ]

        v2 = [
            d2.epistemic_appetite.scope,
            d2.epistemic_appetite.surprise_seeking,
            d2.cognitive_style.thinking_mode,
            d2.cognitive_style.abstraction,
            d2.cognitive_style.classification,
            d2.methodological.making_style,
            d2.methodological.formalism,
            d2.methodological.tacit_explicit,
            d2.social.disclosure,
            d2.social.inquiry,
            d2.social.collaboration,
            d2.social.competition,
            d2.temporal.paradigm_orientation,
            d2.temporal.patience,
            d2.temporal.archival,
            d2.aesthetic.elegance_completeness,
            d2.aesthetic.mechanism_phenomenon,
            d2.aesthetic.risk,
        ]

        # Euclidean distance
        distance = sum((a - b) ** 2 for a, b in zip(v1, v2)) ** 0.5

        # Normalize to similarity (max distance is sqrt(18) ≈ 4.24)
        max_distance = len(v1) ** 0.5
        similarity = 1 - (distance / max_distance)

        # Account for curiosity style (categorical)
        if d1.epistemic_appetite.curiosity_style == d2.epistemic_appetite.curiosity_style:
            similarity = (similarity + 0.1) / 1.1  # Small boost for matching style

        return max(0.0, min(1.0, similarity))

    def build_profile(
        self,
        person_id: str,
        person_name: str,
        samples: List[TextSample],
        affiliation: Optional[str] = None,
        compound_types: Optional[Dict] = None,
    ) -> IndividualProfile:
        """
        Build a complete individual profile from text samples.

        Args:
            person_id: Unique identifier
            person_name: Display name
            samples: Text samples from this person
            affiliation: Optional institution/affiliation
            compound_types: Optional dict of compound types for matching

        Returns:
            IndividualProfile with extracted disposition and behavior
        """
        disposition, critique_behavior, confidence, reasoning = self.extract_disposition(
            samples, person_name
        )

        # Match to compound type if provided
        matched_type = None
        if compound_types:
            matched_type, match_score = self.match_compound_type(disposition, compound_types)
            if match_score < 0.6:
                matched_type = None  # Not a strong enough match

            # Copy productive tensions from matched type
            if matched_type and matched_type in compound_types:
                critique_behavior.productive_tensions = (
                    compound_types[matched_type].critique_behavior.productive_tensions
                )

        # Build data sources summary
        source_counts = {}
        for sample in samples:
            source_counts[sample.source_type] = source_counts.get(sample.source_type, 0) + 1

        data_sources = {
            "internal": {
                "sample_counts": source_counts,
                "total_samples": len(samples),
            },
            "external": {},
        }

        return IndividualProfile(
            id=person_id,
            name=person_name,
            affiliation=affiliation,
            compound_type=matched_type,
            disposition=disposition,
            critique_behavior=critique_behavior,
            data_sources=data_sources,
            confidence=confidence,
        )
