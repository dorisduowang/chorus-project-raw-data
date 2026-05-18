"""
Relationship Builder

Extracts additional relationships from existing data to enrich the hypergraph:
- Co-authorship strength (repeated collaborations)
- Topic-based connections
- Dataset usage from Midway directories
- Institution affiliations
"""

import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Set, Optional, Tuple

from ..models import HyperGraph, HyperNode, HyperEdge, NodeType, EdgeType


def build_relationships(
    lab_registry_path: str = "Data/lab_registry.json",
    midway_registry_path: str = "Data/midway_registry.json",
    graph: Optional[HyperGraph] = None,
) -> HyperGraph:
    """
    Build additional relationship edges from registry data.

    Adds:
    - Co-authorship strength edges (people who co-author frequently)
    - Topic collaboration edges (people sharing research topics)
    - Dataset usage edges (people linked to datasets they likely use)
    - Institution affiliation edges

    Args:
        lab_registry_path: Path to lab_registry.json
        midway_registry_path: Path to midway_registry.json
        graph: Existing graph to add to, or None to create new

    Returns:
        HyperGraph with additional edges
    """
    if graph is None:
        graph = HyperGraph()

    # Load registries
    lab_path = Path(lab_registry_path)
    midway_path = Path(midway_registry_path)

    if not lab_path.exists():
        print(f"Warning: Lab registry not found at {lab_path}")
        return graph

    with open(lab_path) as f:
        lab_registry = json.load(f)

    midway_registry = {}
    if midway_path.exists():
        with open(midway_path) as f:
            midway_registry = json.load(f)

    members = lab_registry.get("people", {}).get("members", [])
    person_ids = {p.get("id") for p in members if p.get("id")}

    # Track edge counts
    coauthor_count = 0
    topic_count = 0
    usage_count = 0
    affiliation_count = 0

    # === 1. Co-authorship Strength Edges ===
    # Count how many times each pair of people co-authored
    coauthor_pairs = defaultdict(int)

    for person in members:
        person_id = person.get("id", "")
        publications = person.get("openalex", {}).get("recent_publications", [])

        for pub in publications:
            authors = pub.get("authors", [])
            # Find lab members among authors
            lab_authors = []
            for author in authors:
                author_lower = author.lower()
                for member in members:
                    member_name = member.get("name", "").lower()
                    member_id = member.get("id", "")
                    if author_lower in member_name or member_name in author_lower:
                        lab_authors.append(member_id)
                        break

            # Add person_id if not already included
            if person_id and person_id not in lab_authors:
                lab_authors.append(person_id)

            # Count all pairs
            lab_authors = list(set(lab_authors))
            for i in range(len(lab_authors)):
                for j in range(i + 1, len(lab_authors)):
                    pair = tuple(sorted([lab_authors[i], lab_authors[j]]))
                    coauthor_pairs[pair] += 1

    # Create edges for frequent collaborators (3+ papers)
    for (p1, p2), count in coauthor_pairs.items():
        if count >= 3 and p1 in person_ids and p2 in person_ids:
            edge = HyperEdge(
                id=f"coauthor-strength-{p1}-{p2}",
                edge_type=EdgeType.COLLABORATION,
                agents=[p1, p2],
                inputs=[],
                outputs=[],
                context=[],
                timestamp=None,
                evidence=[],
                confidence=min(1.0, count / 10),  # Scale by collaboration count
                created_by="relationship_extraction",
            )
            # Store count in edge for reference
            edge.evidence = [f"co-authored {count} publications"]
            graph.add_edge(edge)
            coauthor_count += 1

    print(f"Added {coauthor_count} co-authorship strength edges")

    # === 2. Topic Collaboration Edges ===
    # Connect people who share research topics
    person_topics: Dict[str, Set[str]] = {}
    topic_to_people: Dict[str, Set[str]] = defaultdict(set)

    for person in members:
        person_id = person.get("id", "")
        topics = person.get("openalex", {}).get("topics", [])

        if person_id and topics:
            person_topics[person_id] = set(t.lower() for t in topics)
            for topic in topics:
                topic_to_people[topic.lower()].add(person_id)

    # Create edges for topics with 2+ people
    for topic, people in topic_to_people.items():
        if len(people) >= 2:
            people_list = list(people)

            # Create a topic node if it doesn't exist
            topic_id = f"topic-{topic.replace(' ', '-')[:50]}"
            if not graph.get_node(topic_id):
                topic_node = HyperNode(
                    id=topic_id,
                    type=NodeType.CONCEPT,
                    name=topic.title(),
                    attributes={"source": "shared_research_topic"}
                )
                graph.add_node(topic_node)

            # Create collaboration edge around this topic
            edge = HyperEdge(
                id=f"topic-collab-{topic_id}",
                edge_type=EdgeType.COLLABORATION,
                agents=people_list,
                inputs=[topic_id],
                outputs=[],
                context=[],
                timestamp=None,
                evidence=[f"{len(people)} researchers share this topic"],
                confidence=0.7,
                created_by="relationship_extraction",
            )
            graph.add_edge(edge)
            topic_count += 1

    print(f"Added {topic_count} topic collaboration edges")

    # === 3. Dataset Usage Edges from Midway ===
    # Link people to datasets based on their Midway directories

    def get_midway_items(category: str) -> List[Dict]:
        data = midway_registry.get(category, {})
        if isinstance(data, dict) and "items" in data:
            return data["items"]
        return data if isinstance(data, list) else []

    all_datasets = (
        get_midway_items("data_snapshots") +
        get_midway_items("embeddings") +
        get_midway_items("precomputed_resources")
    )

    researcher_dirs = get_midway_items("researcher_directories")

    # Map usernames to person IDs
    username_to_person = {}
    for person in members:
        midway_info = person.get("midway", {})
        username = midway_info.get("username", "").lower()
        if username:
            username_to_person[username] = person.get("id", "")

    # For each researcher directory, link to datasets they might use
    for researcher in researcher_dirs:
        username = researcher.get("username", "").lower()
        person_id = username_to_person.get(username)

        if not person_id:
            continue

        focus = researcher.get("primary_focus", "").lower()
        key_datasets = researcher.get("key_datasets", [])

        # Find matching datasets
        for dataset in all_datasets:
            dataset_id = dataset.get("id", "")
            dataset_name = dataset.get("name", "").lower()
            dataset_desc = dataset.get("description", "").lower()

            # Check if researcher's focus mentions this dataset
            match_score = 0
            if dataset_name in focus or dataset_name in str(key_datasets).lower():
                match_score = 0.9
            elif any(word in focus for word in dataset_name.split()):
                match_score = 0.5

            if match_score > 0:
                # Create or get dataset node
                if not graph.get_node(dataset_id):
                    dataset_node = HyperNode(
                        id=dataset_id,
                        type=NodeType.DATASET,
                        name=dataset.get("name", ""),
                        attributes={
                            "path": dataset.get("path"),
                            "source": "midway"
                        }
                    )
                    graph.add_node(dataset_node)

                edge = HyperEdge(
                    id=f"usage-{person_id}-{dataset_id}",
                    edge_type=EdgeType.USAGE,
                    agents=[person_id],
                    inputs=[dataset_id],
                    outputs=[],
                    context=[],
                    timestamp=None,
                    evidence=[f"Researcher focus: {focus[:50]}"],
                    confidence=match_score,
                    created_by="relationship_extraction",
                )
                graph.add_edge(edge)
                usage_count += 1

    print(f"Added {usage_count} dataset usage edges")

    # === 4. Institution Affiliation Edges ===
    # Connect people to their institutions
    institution_to_people: Dict[str, List[str]] = defaultdict(list)

    for person in members:
        person_id = person.get("id", "")
        institution = person.get("institution", "")

        if person_id and institution and isinstance(institution, str):
            inst_id = f"inst-{institution.lower().replace(' ', '-')}"
            institution_to_people[inst_id].append(person_id)

    for inst_id, people in institution_to_people.items():
        if len(people) >= 1:
            # Get institution name from ID
            inst_name = inst_id.replace("inst-", "").replace("-", " ").title()

            edge = HyperEdge(
                id=f"affiliation-{inst_id}",
                edge_type=EdgeType.COLLABORATION,  # Using collaboration for affiliation
                agents=people,
                inputs=[],
                outputs=[],
                context=[inst_id],
                timestamp=None,
                evidence=[f"{len(people)} people at {inst_name}"],
                confidence=1.0,
                created_by="relationship_extraction",
            )
            graph.add_edge(edge)
            affiliation_count += 1

    print(f"Added {affiliation_count} institution affiliation edges")

    total_new = coauthor_count + topic_count + usage_count + affiliation_count
    print(f"\nTotal new relationship edges: {total_new}")

    return graph
