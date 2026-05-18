"""
Data models for the registry module.

Contains dataclasses used across the registry components.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any


@dataclass
class RegistryMatch:
    """A match from structured registry lookup."""
    entity_type: str  # people, projects, funding, etc.
    entity_id: str
    entity_data: Dict[str, Any]
    match_reason: str  # Why this matched
    score: float = 1.0


@dataclass
class QueryClassification:
    """Classification of a query for routing."""
    query_type: str  # structured, semantic, hybrid
    entity_types: List[str]  # Which registry sections to search
    search_terms: List[str]  # Extracted search terms
    confidence: float
    is_compound: bool = False  # Whether this is a compound/intersection query
    compound_filters: Dict[str, Any] = field(default_factory=dict)  # Extracted filters for compound queries
