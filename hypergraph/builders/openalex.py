"""
OpenAlex Publication Builder

Builds hyperedges from OpenAlex publication data in lab_registry.json.

Each publication becomes a hyperedge:
- agents: authors who are in the lab registry
- inputs: concepts/topics from OpenAlex
- outputs: the Document node for the paper itself
- context: venue
- timestamp: publication year
"""

import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Set, Optional

from ..models import HyperGraph, HyperNode, HyperEdge, NodeType, EdgeType


def _make_id(prefix: str, *parts: str) -> str:
    """Create a stable ID from parts."""
    combined = "-".join(str(p).lower().replace(" ", "-") for p in parts if p)
    # Use hash for very long IDs
    if len(combined) > 60:
        hash_suffix = hashlib.md5(combined.encode()).hexdigest()[:8]
        combined = combined[:50] + "-" + hash_suffix
    return f"{prefix}-{combined}"


def _normalize_name(name: str) -> str:
    """Normalize a name for matching."""
    return name.lower().strip()


def build_from_openalex(
    registry_path: str = "Data/lab_registry.json",
    graph: Optional[HyperGraph] = None,
) -> HyperGraph:
    """
    Build hypergraph nodes and edges from OpenAlex publication data.

    Extracts from lab_registry.json:
    - Person nodes (from people.members)
    - Document nodes (from person.openalex.recent_publications)
    - Concept nodes (from person.openalex.topics)
    - Venue nodes (from publication venues)
    - Publication hyperedges connecting them

    Args:
        registry_path: Path to lab_registry.json
        graph: Existing graph to add to, or None to create new

    Returns:
        HyperGraph with nodes and edges added
    """
    if graph is None:
        graph = HyperGraph()

    # Load registry
    registry_path = Path(registry_path)
    if not registry_path.exists():
        print(f"Warning: Registry not found at {registry_path}")
        return graph

    with open(registry_path) as f:
        registry = json.load(f)

    # Track what we've seen
    seen_documents: Set[str] = set()  # DOIs
    seen_concepts: Set[str] = set()
    seen_venues: Set[str] = set()

    # Build name -> person_id mapping for author matching
    name_to_person: Dict[str, str] = {}
    members = registry.get("people", {}).get("members", [])

    for person in members:
        person_id = person.get("id", "")
        name = person.get("name", "")

        if not person_id or not name:
            continue

        # Create Person node
        person_node = HyperNode(
            id=person_id,
            type=NodeType.PERSON,
            name=name,
            attributes={
                "role": person.get("role"),
                "institution": person.get("institution"),
                "email": person.get("email"),
                "openalex_id": person.get("openalex", {}).get("author_id"),
                "orcid": person.get("openalex", {}).get("orcid"),
            }
        )
        graph.add_node(person_node)

        # Build name index
        name_to_person[_normalize_name(name)] = person_id
        # Also index by last name
        parts = name.split()
        if len(parts) > 1:
            name_to_person[_normalize_name(parts[-1])] = person_id
        # Also index by first name for disambiguation
        if parts:
            name_to_person[_normalize_name(parts[0])] = person_id

    # Process each person's publications
    publication_count = 0
    for person in members:
        person_id = person.get("id", "")
        openalex = person.get("openalex", {})
        publications = openalex.get("recent_publications", [])

        # Get person's topics as concepts
        person_topics = openalex.get("topics", [])
        for topic in person_topics:
            if topic and topic not in seen_concepts:
                concept_id = _make_id("concept", topic)
                concept_node = HyperNode(
                    id=concept_id,
                    type=NodeType.CONCEPT,
                    name=topic,
                    attributes={"source": "openalex_topic"}
                )
                graph.add_node(concept_node)
                seen_concepts.add(topic)

        for pub in publications:
            doi = pub.get("doi", "")
            title = pub.get("title", "")

            if not title:
                continue

            # Skip duplicates (same DOI)
            if doi and doi in seen_documents:
                continue

            # Create Document node
            doc_id = _make_id("doc", doi or title)
            if doi:
                seen_documents.add(doi)

            doc_node = HyperNode(
                id=doc_id,
                type=NodeType.DOCUMENT,
                name=title,
                attributes={
                    "doi": doi,
                    "year": pub.get("year"),
                    "venue": pub.get("venue"),
                    "citation_count": pub.get("citation_count"),
                    "openalex_id": pub.get("openalex_id"),
                }
            )
            graph.add_node(doc_node)

            # Create Venue node if exists
            venue = pub.get("venue")
            venue_id = None
            if venue and venue not in seen_venues:
                venue_id = _make_id("venue", venue)
                venue_node = HyperNode(
                    id=venue_id,
                    type=NodeType.VENUE,
                    name=venue,
                    attributes={}
                )
                graph.add_node(venue_node)
                seen_venues.add(venue)
            elif venue:
                venue_id = _make_id("venue", venue)

            # Find lab authors from publication authors
            pub_authors = pub.get("authors", [])
            lab_author_ids = []

            for author in pub_authors:
                author_normalized = _normalize_name(author)
                # Try exact match
                if author_normalized in name_to_person:
                    lab_author_ids.append(name_to_person[author_normalized])
                else:
                    # Try partial match (last name)
                    author_parts = author.split()
                    for part in author_parts:
                        part_normalized = _normalize_name(part)
                        if part_normalized in name_to_person:
                            lab_author_ids.append(name_to_person[part_normalized])
                            break

            # Ensure the person whose profile this is from is included
            if person_id and person_id not in lab_author_ids:
                lab_author_ids.append(person_id)

            # Remove duplicates while preserving order
            lab_author_ids = list(dict.fromkeys(lab_author_ids))

            # Get publication topics as concepts (inputs)
            pub_topics = pub.get("topics", [])
            concept_ids = []
            for topic in pub_topics:
                if topic:
                    concept_id = _make_id("concept", topic)
                    if topic not in seen_concepts:
                        concept_node = HyperNode(
                            id=concept_id,
                            type=NodeType.CONCEPT,
                            name=topic,
                            attributes={"source": "publication_topic"}
                        )
                        graph.add_node(concept_node)
                        seen_concepts.add(topic)
                    concept_ids.append(concept_id)

            # Create publication hyperedge
            year = pub.get("year")
            timestamp = None
            if year:
                try:
                    timestamp = datetime(year=int(year), month=6, day=15)  # Mid-year
                except (ValueError, TypeError):
                    pass

            edge_id = _make_id("pub-edge", doc_id)
            edge = HyperEdge(
                id=edge_id,
                edge_type=EdgeType.PUBLICATION,
                agents=lab_author_ids,
                inputs=concept_ids,  # Topics as inputs
                outputs=[doc_id],    # The paper as output
                context=[venue_id] if venue_id else [],
                timestamp=timestamp,
                evidence=[],
                confidence=1.0,
                created_by="structured_import",
            )
            graph.add_edge(edge)
            publication_count += 1

    print(f"Built {publication_count} publication hyperedges from OpenAlex data")
    print(f"  - {len(graph.get_nodes_by_type(NodeType.PERSON))} Person nodes")
    print(f"  - {len(graph.get_nodes_by_type(NodeType.DOCUMENT))} Document nodes")
    print(f"  - {len(graph.get_nodes_by_type(NodeType.CONCEPT))} Concept nodes")
    print(f"  - {len(graph.get_nodes_by_type(NodeType.VENUE))} Venue nodes")

    return graph
