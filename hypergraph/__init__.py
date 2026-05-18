"""
Hypergraph Module for CHORUS

A directed hypergraph representation of Knowledge Lab as a "digital twin".
Supports structural queries, temporal analysis, and retrieval augmentation.

Usage:
    from hypergraph import HyperGraph, HyperNode, HyperEdge

    # Load or create
    graph = HyperGraph.load("Data/hypergraph.json")

    # Query
    collaborators = graph.collaborators_of("person-james-evans")
    outputs = graph.what_produced("person-sarah-chen")

    # Analytics
    dead_ends = graph.terminal_nodes()
"""

from .models import HyperNode, HyperEdge, HyperGraph, NodeType, EdgeType
from .storage import save_hypergraph, load_hypergraph

__all__ = [
    "HyperNode",
    "HyperEdge",
    "HyperGraph",
    "NodeType",
    "EdgeType",
    "save_hypergraph",
    "load_hypergraph",
]
