"""
Registry Builder

Builds hyperedges from lab_registry.json entities:
- Projects (with members, leads, datasets)
- Funding (with PIs, projects)
- Datasets (with creators, projects)
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..models import HyperGraph, HyperNode, HyperEdge, NodeType, EdgeType


def build_from_registry(
    registry_path: str = "Data/lab_registry.json",
    graph: Optional[HyperGraph] = None,
) -> HyperGraph:
    """
    Build hypergraph nodes and edges from lab registry entities.

    Adds:
    - Project nodes and membership edges
    - Funding nodes and funding edges
    - Dataset nodes and usage edges
    - Institution nodes

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

    # Build person ID set and name mapping for validation
    members = registry.get("people", {}).get("members", [])
    person_ids = {p.get("id") for p in members if p.get("id")}

    # Build name -> person_id mapping (names in projects may be full names, not IDs)
    name_to_id = {}
    for person in members:
        person_id = person.get("id", "")
        name = person.get("name", "")
        if person_id and name:
            # Map full name
            name_to_id[name.lower().strip()] = person_id
            # Map by last name
            parts = name.split()
            if len(parts) > 1:
                name_to_id[parts[-1].lower().strip()] = person_id

    def resolve_person(ref: str) -> str:
        """Resolve a person reference (name or ID) to a person ID."""
        if ref in person_ids:
            return ref
        # Try name lookup
        return name_to_id.get(ref.lower().strip(), "")

    # Track institutions we've seen
    seen_institutions = set()

    # Add Institution nodes from people
    for person in members:
        institution = person.get("institution")
        # Skip if not a valid string (could be NaN/None)
        if institution and isinstance(institution, str) and institution not in seen_institutions:
            inst_id = f"inst-{institution.lower().replace(' ', '-')}"
            inst_node = HyperNode(
                id=inst_id,
                type=NodeType.INSTITUTION,
                name=institution,
                attributes={}
            )
            graph.add_node(inst_node)
            seen_institutions.add(institution)

    # === Process Projects ===
    projects = registry.get("projects", [])
    project_count = 0

    for project in projects:
        project_id = project.get("id", "")
        project_name = project.get("name", "")

        if not project_id:
            continue

        # Create Project node
        project_node = HyperNode(
            id=project_id,
            type=NodeType.PROJECT,
            name=project_name,
            attributes={
                "status": project.get("status"),
                "description": project.get("description"),
                "start_date": project.get("start_date"),
                "end_date": project.get("end_date"),
                "tags": project.get("tags", []),
            }
        )
        graph.add_node(project_node)

        # Collect members
        leads = project.get("leads", [])
        members_list = project.get("members", [])
        all_members = list(set(leads + members_list))

        # Resolve names/IDs to valid person IDs
        valid_members = [resolve_person(m) for m in all_members]
        valid_members = [m for m in valid_members if m]  # Filter empty

        if valid_members:
            # Parse start date for timestamp
            timestamp = None
            start_date = project.get("start_date")
            if start_date:
                try:
                    timestamp = datetime.fromisoformat(start_date)
                except (ValueError, TypeError):
                    try:
                        # Try year only
                        timestamp = datetime(year=int(start_date[:4]), month=1, day=1)
                    except:
                        pass

            # Create collaboration edge
            edge = HyperEdge(
                id=f"collab-{project_id}",
                edge_type=EdgeType.COLLABORATION,
                agents=valid_members,
                inputs=[],
                outputs=[],
                context=[project_id],
                timestamp=timestamp,
                evidence=[],
                confidence=1.0,
                created_by="structured_import",
            )
            graph.add_edge(edge)
            project_count += 1

        # Link datasets to project
        project_datasets = project.get("datasets", [])
        for dataset_id in project_datasets:
            # Create usage edge if dataset exists
            if graph.get_node(dataset_id):
                edge = HyperEdge(
                    id=f"usage-{project_id}-{dataset_id}",
                    edge_type=EdgeType.USAGE,
                    agents=valid_members[:3] if valid_members else [],  # Limit agents
                    inputs=[dataset_id],
                    outputs=[],
                    context=[project_id],
                    timestamp=timestamp,
                    evidence=[],
                    confidence=0.9,
                    created_by="structured_import",
                )
                graph.add_edge(edge)

    print(f"Built {project_count} project collaboration edges")

    # === Process Funding ===
    funding_list = registry.get("funding", [])
    funding_count = 0

    for funding in funding_list:
        funding_id = funding.get("id", "")
        funding_name = funding.get("name", "")

        if not funding_id:
            continue

        # Create a node representing the funding source/grant
        funding_node = HyperNode(
            id=funding_id,
            type=NodeType.PROJECT,  # Treat grants as a type of project
            name=funding_name,
            attributes={
                "entity_type": "funding",
                "source": funding.get("source"),
                "type": funding.get("type"),
                "amount": funding.get("amount"),
                "currency": funding.get("currency"),
                "status": funding.get("status"),
            }
        )
        graph.add_node(funding_node)

        # Collect PIs
        pi = funding.get("pi")
        co_pis = funding.get("co_pis", [])
        all_pis = ([pi] if pi else []) + co_pis
        valid_pis = [resolve_person(p) for p in all_pis]
        valid_pis = [p for p in valid_pis if p]  # Filter empty

        # Get funded projects
        funded_projects = funding.get("projects", [])

        if valid_pis:
            # Create funding edge
            edge = HyperEdge(
                id=f"funding-{funding_id}",
                edge_type=EdgeType.FUNDING,
                agents=valid_pis,
                inputs=[],
                outputs=funded_projects,  # Projects as outputs of funding
                context=[funding_id],
                timestamp=None,
                evidence=[],
                confidence=1.0,
                created_by="structured_import",
            )
            graph.add_edge(edge)
            funding_count += 1

    print(f"Built {funding_count} funding edges")

    # === Process Datasets ===
    datasets = registry.get("datasets", [])
    dataset_count = 0

    for dataset in datasets:
        dataset_id = dataset.get("id", "")
        dataset_name = dataset.get("name", "")

        if not dataset_id:
            continue

        # Create Dataset node
        dataset_node = HyperNode(
            id=dataset_id,
            type=NodeType.DATASET,
            name=dataset_name,
            attributes={
                "description": dataset.get("description"),
                "format": dataset.get("format"),
                "size_gb": dataset.get("size_gb"),
                "access_level": dataset.get("access_level"),
                "tags": dataset.get("tags", []),
            }
        )
        graph.add_node(dataset_node)
        dataset_count += 1

        # Create derivation edge if dataset has creator
        created_by = dataset.get("created_by")
        if created_by and created_by in person_ids:
            edge = HyperEdge(
                id=f"created-{dataset_id}",
                edge_type=EdgeType.DERIVATION,
                agents=[created_by],
                inputs=[],
                outputs=[dataset_id],
                context=[],
                timestamp=None,
                evidence=[],
                confidence=1.0,
                created_by="structured_import",
            )
            graph.add_edge(edge)

    print(f"Added {dataset_count} Dataset nodes")

    # === Process Code Repos ===
    repos = registry.get("code_repos", [])
    repo_count = 0

    for repo in repos:
        repo_id = repo.get("id", "")
        repo_name = repo.get("name", "")

        if not repo_id:
            continue

        # Create Code node
        code_node = HyperNode(
            id=repo_id,
            type=NodeType.CODE,
            name=repo_name,
            attributes={
                "url": repo.get("url"),
                "language": repo.get("language"),
                "status": repo.get("status"),
            }
        )
        graph.add_node(code_node)
        repo_count += 1

        # Create derivation edge if repo has maintainers
        maintainers = repo.get("maintainers", [])
        valid_maintainers = [m for m in maintainers if m in person_ids]

        if valid_maintainers:
            edge = HyperEdge(
                id=f"maintains-{repo_id}",
                edge_type=EdgeType.DERIVATION,
                agents=valid_maintainers,
                inputs=[],
                outputs=[repo_id],
                context=repo.get("projects", []),
                timestamp=None,
                evidence=[],
                confidence=1.0,
                created_by="structured_import",
            )
            graph.add_edge(edge)

    print(f"Added {repo_count} Code nodes")

    return graph
