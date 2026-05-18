"""
Hypergraph Builders

Modules for constructing hypergraph from structured data sources:
- openalex.py: Publications from OpenAlex data in lab_registry
- registry.py: Projects, funding, datasets from lab_registry
- relationships.py: Additional relationships (co-authorship, topics, usage)
"""

from .openalex import build_from_openalex
from .registry import build_from_registry
from .relationships import build_relationships

__all__ = [
    "build_from_openalex",
    "build_from_registry",
    "build_relationships",
]
