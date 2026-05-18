"""
Build the Hypergraph

Script to construct the hypergraph from all structured sources
and save it to Data/hypergraph.json.

Usage:
    python -m hypergraph.build
    # or
    python hypergraph/build.py
"""

import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from hypergraph.models import HyperGraph, NodeType, EdgeType
from hypergraph.storage import save_hypergraph
from hypergraph.builders.openalex import build_from_openalex
from hypergraph.builders.registry import build_from_registry
from hypergraph.builders.relationships import build_relationships


def build_hypergraph(
    lab_registry_path: str = "Data/lab_registry.json",
    output_path: str = "Data/hypergraph.json",
) -> HyperGraph:
    """
    Build the complete hypergraph from all sources.

    Args:
        lab_registry_path: Path to lab_registry.json
        output_path: Where to save the hypergraph

    Returns:
        The constructed HyperGraph
    """
    print("=" * 60)
    print("Building Knowledge Lab Hypergraph")
    print("=" * 60)

    # Start with empty graph
    graph = HyperGraph()

    # Phase 1: OpenAlex publications (includes Person, Document, Concept, Venue nodes)
    print("\n[1/3] Building from OpenAlex publication data...")
    graph = build_from_openalex(lab_registry_path, graph)

    # Phase 2: Registry entities (Projects, Funding, Datasets, Code)
    print("\n[2/3] Building from registry entities...")
    graph = build_from_registry(lab_registry_path, graph)

    # Phase 3: Additional relationships (co-authorship, topics, usage)
    print("\n[3/3] Building relationship edges...")
    graph = build_relationships(lab_registry_path, "Data/midway_registry.json", graph)

    # Save
    print(f"\nSaving to {output_path}...")
    save_hypergraph(graph, output_path)

    # Summary
    print("\n" + "=" * 60)
    print("Hypergraph Summary")
    print("=" * 60)
    stats = graph.stats()

    print(f"\nTotal nodes: {stats['total_nodes']}")
    print("By type:")
    for node_type, count in stats['node_types'].items():
        if count > 0:
            print(f"  - {node_type}: {count}")

    print(f"\nTotal edges: {stats['total_edges']}")
    print("By type:")
    for edge_type, count in stats['edge_types'].items():
        if count > 0:
            print(f"  - {edge_type}: {count}")

    if stats['time_range']:
        print(f"\nTime range: {stats['time_range']['earliest'][:10]} to {stats['time_range']['latest'][:10]}")

    return graph


def run_analytics(graph: HyperGraph) -> None:
    """Run basic analytics on the hypergraph."""
    print("\n" + "=" * 60)
    print("Analytics: Terminal Nodes")
    print("=" * 60)

    terminal = graph.terminal_nodes()
    print(f"\nFound {len(terminal)} terminal nodes (outputs never reused as inputs)")

    # Group by type
    by_type = {}
    for node in terminal:
        t = node.type.value if hasattr(node.type, 'value') else str(node.type)
        if t not in by_type:
            by_type[t] = []
        by_type[t].append(node)

    for node_type, nodes in sorted(by_type.items()):
        print(f"\n{node_type} ({len(nodes)}):")
        for node in nodes[:5]:
            reuse = graph.node_reuse_count(node.id)
            print(f"  - {node.name[:60]}...")
            print(f"    (output {reuse['as_output']}x, input {reuse['as_input']}x)")
        if len(nodes) > 5:
            print(f"  ... and {len(nodes) - 5} more")


def demo_queries(graph: HyperGraph) -> None:
    """Demonstrate some queries."""
    print("\n" + "=" * 60)
    print("Demo: Sample Queries")
    print("=" * 60)

    # Find a person with publications
    people = graph.get_nodes_by_type(NodeType.PERSON)
    if people:
        person = people[0]
        print(f"\n1. What has {person.name} produced?")
        outputs = graph.what_produced(person.id)
        for out in outputs[:5]:
            print(f"   - [{out.type.value if hasattr(out.type, 'value') else out.type}] {out.name[:50]}")
        if len(outputs) > 5:
            print(f"   ... and {len(outputs) - 5} more")

        print(f"\n2. Who has collaborated with {person.name}?")
        collabs = graph.collaborators_of(person.id)
        for c in collabs[:5]:
            print(f"   - {c.name}")
        if len(collabs) > 5:
            print(f"   ... and {len(collabs) - 5} more")

    # Find a concept
    concepts = graph.get_nodes_by_type(NodeType.CONCEPT)
    if concepts:
        concept = concepts[0]
        print(f"\n3. Who has worked with '{concept.name}'?")
        workers = graph.who_worked_with(concept.id)
        for w in workers[:5]:
            print(f"   - {w.name}")
        if len(workers) > 5:
            print(f"   ... and {len(workers) - 5} more")


if __name__ == "__main__":
    # Build the hypergraph
    graph = build_hypergraph()

    # Run analytics
    run_analytics(graph)

    # Demo queries
    demo_queries(graph)

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
