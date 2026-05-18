"""
Adaptive Search Weighting

Dynamically adjusts semantic vs BM25 weighting based on query characteristics.

Query Types:
- Name lookups: 50/50 (exact match matters)
- Conceptual queries: 85/15 (semantic understanding matters)
- Mixed: 70/30 (default balance)
"""

import re
from dataclasses import dataclass
from typing import Tuple

# Patterns for query classification
NAME_PATTERNS = re.compile(
    r'^(who is|what is|tell me about|find)\s+[A-Z][a-z]+(\s+[A-Z][a-z]+)*$|'
    r'^[A-Z][a-z]+(\s+[A-Z][a-z]+){1,3}$|'
    r'\b(h-index|publication|citations?|papers?)\s+(of|for|by)\b',
    re.IGNORECASE
)

CONCEPTUAL_PATTERNS = re.compile(
    r'\b(how|why|what is the|explain|describe|compare|difference|relationship|'
    r'concept|theory|method|approach|impact|effect|trend|overview|'
    r'research on|studies about|work on|field of)\b',
    re.IGNORECASE
)

EXACT_MATCH_INDICATORS = re.compile(
    r'^"[^"]+"|'
    r'\b(exactly|specifically|called|named|titled)\b',
    re.IGNORECASE
)


@dataclass
class SearchWeights:
    """Weights for hybrid search components."""
    semantic: float
    bm25: float
    query_type: str

    def __post_init__(self):
        total = self.semantic + self.bm25
        if abs(total - 1.0) > 0.001:
            self.semantic = self.semantic / total
            self.bm25 = self.bm25 / total


class AdaptiveWeighting:
    """Calculates adaptive weights for semantic vs BM25 search."""

    # Weight presets for different query types
    PRESETS = {
        "name_lookup": SearchWeights(semantic=0.50, bm25=0.50, query_type="name_lookup"),
        "conceptual": SearchWeights(semantic=0.85, bm25=0.15, query_type="conceptual"),
        "exact_match": SearchWeights(semantic=0.30, bm25=0.70, query_type="exact_match"),
        "mixed": SearchWeights(semantic=0.70, bm25=0.30, query_type="mixed"),
    }

    def classify_query(self, query: str) -> str:
        """Classify query type based on patterns."""
        # Check for exact match requests first
        if EXACT_MATCH_INDICATORS.search(query):
            return "exact_match"

        # Check for name/person lookups
        if NAME_PATTERNS.search(query):
            return "name_lookup"

        # Check for conceptual questions
        if CONCEPTUAL_PATTERNS.search(query):
            return "conceptual"

        return "mixed"

    def calculate_weights(self, query: str) -> SearchWeights:
        """
        Calculate adaptive weights based on query characteristics.

        Args:
            query: The search query

        Returns:
            SearchWeights with semantic and bm25 weights
        """
        query_type = self.classify_query(query)
        return self.PRESETS[query_type]

    def get_weight_tuple(self, query: str) -> Tuple[float, float]:
        """
        Get weights as a simple tuple for backward compatibility.

        Returns:
            (semantic_weight, bm25_weight)
        """
        weights = self.calculate_weights(query)
        return (weights.semantic, weights.bm25)


# Singleton instance
_adaptive_weighting = None


def get_adaptive_weighting() -> AdaptiveWeighting:
    """Get the singleton AdaptiveWeighting instance."""
    global _adaptive_weighting
    if _adaptive_weighting is None:
        _adaptive_weighting = AdaptiveWeighting()
    return _adaptive_weighting
