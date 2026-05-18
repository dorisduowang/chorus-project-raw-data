"""
Hypergraph Data Models

Core dataclasses for the directed hypergraph representation.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Optional, Set, Tuple


class NodeType(str, Enum):
    """Types of nodes in the hypergraph."""
    PERSON = "Person"
    DATASET = "Dataset"
    DOCUMENT = "Document"
    MODEL = "Model"
    CODE = "Code"
    CONCEPT = "Concept"
    PROJECT = "Project"
    EVENT = "Event"
    VENUE = "Venue"
    INSTITUTION = "Institution"


class EdgeType(str, Enum):
    """Types of hyperedges."""
    PUBLICATION = "publication"           # Authors produce a paper
    COMPUTATION = "computation"           # Person runs analysis
    COLLABORATION = "collaboration"       # People work together on project
    FUNDING = "funding"                   # Funder supports project/people
    MEMBERSHIP = "membership"             # Person belongs to institution/project
    USAGE = "usage"                       # Person/system uses dataset
    DERIVATION = "derivation"             # Output derived from inputs
    DISCUSSION = "discussion"             # People discuss concepts


@dataclass
class HyperNode:
    """
    A node in the hypergraph representing an entity.

    Node lifetime is inferred from edge participation:
    - First appearance = min(edge.timestamp) where node participates
    - Last appearance = max(edge.timestamp) where node participates
    """
    id: str
    type: NodeType
    name: str
    attributes: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if isinstance(other, HyperNode):
            return self.id == other.id
        return False

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "type": self.type.value if isinstance(self.type, NodeType) else self.type,
            "name": self.name,
            "attributes": self.attributes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HyperNode":
        """Deserialize from dictionary."""
        return cls(
            id=data["id"],
            type=NodeType(data["type"]) if data["type"] in [t.value for t in NodeType] else data["type"],
            name=data["name"],
            attributes=data.get("attributes", {}),
        )


@dataclass
class HyperEdge:
    """
    A directed hyperedge connecting multiple nodes.

    Direction is encoded in the slot structure:
    - agents: Person nodes who acted (always sources)
    - inputs: Nodes consumed/used (sources)
    - outputs: Nodes produced/created (targets)
    - context: Project/Event nodes for scope (neither source nor target)
    """
    id: str
    edge_type: EdgeType
    agents: List[str] = field(default_factory=list)      # Person IDs
    inputs: List[str] = field(default_factory=list)      # Node IDs consumed
    outputs: List[str] = field(default_factory=list)     # Node IDs produced
    context: List[str] = field(default_factory=list)     # Project/Event IDs
    timestamp: Optional[datetime] = None
    evidence: List[str] = field(default_factory=list)    # Chunk/source IDs
    confidence: float = 1.0
    created_by: str = "structured_import"

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if isinstance(other, HyperEdge):
            return self.id == other.id
        return False

    @property
    def all_nodes(self) -> Set[str]:
        """All node IDs participating in this hyperedge."""
        return set(self.agents) | set(self.inputs) | set(self.outputs) | set(self.context)

    @property
    def source_nodes(self) -> Set[str]:
        """Nodes on the source side (agents + inputs)."""
        return set(self.agents) | set(self.inputs)

    @property
    def target_nodes(self) -> Set[str]:
        """Nodes on the target side (outputs)."""
        return set(self.outputs)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "edge_type": self.edge_type.value if isinstance(self.edge_type, EdgeType) else self.edge_type,
            "agents": self.agents,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "context": self.context,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "evidence": self.evidence,
            "confidence": self.confidence,
            "created_by": self.created_by,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HyperEdge":
        """Deserialize from dictionary."""
        timestamp = None
        if data.get("timestamp"):
            timestamp = datetime.fromisoformat(data["timestamp"])

        edge_type = data["edge_type"]
        if edge_type in [t.value for t in EdgeType]:
            edge_type = EdgeType(edge_type)

        return cls(
            id=data["id"],
            edge_type=edge_type,
            agents=data.get("agents", []),
            inputs=data.get("inputs", []),
            outputs=data.get("outputs", []),
            context=data.get("context", []),
            timestamp=timestamp,
            evidence=data.get("evidence", []),
            confidence=data.get("confidence", 1.0),
            created_by=data.get("created_by", "structured_import"),
        )


@dataclass
class HyperGraph:
    """
    A directed hypergraph representing Knowledge Lab.

    Provides:
    - Node and edge storage with indexing
    - Structural queries (who worked with X, what produced Y)
    - Temporal queries (network at time T)
    - Analytics (terminal nodes, underutilization)
    """
    nodes: Dict[str, HyperNode] = field(default_factory=dict)
    edges: Dict[str, HyperEdge] = field(default_factory=dict)

    # Indexes for fast lookup
    _node_to_edges: Dict[str, Set[str]] = field(default_factory=dict)
    _type_to_nodes: Dict[str, Set[str]] = field(default_factory=dict)
    _type_to_edges: Dict[str, Set[str]] = field(default_factory=dict)

    def add_node(self, node: HyperNode) -> None:
        """Add a node to the graph."""
        self.nodes[node.id] = node

        # Update type index
        type_key = node.type.value if isinstance(node.type, NodeType) else node.type
        if type_key not in self._type_to_nodes:
            self._type_to_nodes[type_key] = set()
        self._type_to_nodes[type_key].add(node.id)

    def add_edge(self, edge: HyperEdge) -> None:
        """Add a hyperedge to the graph."""
        self.edges[edge.id] = edge

        # Update node-to-edge index
        for node_id in edge.all_nodes:
            if node_id not in self._node_to_edges:
                self._node_to_edges[node_id] = set()
            self._node_to_edges[node_id].add(edge.id)

        # Update type index
        type_key = edge.edge_type.value if isinstance(edge.edge_type, EdgeType) else edge.edge_type
        if type_key not in self._type_to_edges:
            self._type_to_edges[type_key] = set()
        self._type_to_edges[type_key].add(edge.id)

    def get_node(self, node_id: str) -> Optional[HyperNode]:
        """Get a node by ID."""
        return self.nodes.get(node_id)

    def get_edge(self, edge_id: str) -> Optional[HyperEdge]:
        """Get an edge by ID."""
        return self.edges.get(edge_id)

    def get_nodes_by_type(self, node_type: NodeType) -> List[HyperNode]:
        """Get all nodes of a given type."""
        type_key = node_type.value if isinstance(node_type, NodeType) else node_type
        node_ids = self._type_to_nodes.get(type_key, set())
        return [self.nodes[nid] for nid in node_ids if nid in self.nodes]

    def get_edges_by_type(self, edge_type: EdgeType) -> List[HyperEdge]:
        """Get all edges of a given type."""
        type_key = edge_type.value if isinstance(edge_type, EdgeType) else edge_type
        edge_ids = self._type_to_edges.get(type_key, set())
        return [self.edges[eid] for eid in edge_ids if eid in self.edges]

    def get_edges_for_node(self, node_id: str) -> List[HyperEdge]:
        """Get all edges containing a node."""
        edge_ids = self._node_to_edges.get(node_id, set())
        return [self.edges[eid] for eid in edge_ids if eid in self.edges]

    def node_lifetime(self, node_id: str) -> Tuple[Optional[datetime], Optional[datetime]]:
        """
        Infer a node's lifetime from edge participation.

        Returns (first_appearance, last_appearance) based on edge timestamps.
        """
        edges = self.get_edges_for_node(node_id)
        timestamps = [e.timestamp for e in edges if e.timestamp]

        if not timestamps:
            return (None, None)

        return (min(timestamps), max(timestamps))

    # === Structural Queries ===

    def who_worked_with(self, node_id: str) -> List[HyperNode]:
        """Find all people who have worked with a given node (dataset, concept, etc.)."""
        person_ids = set()
        for edge in self.get_edges_for_node(node_id):
            person_ids.update(edge.agents)
        return [self.nodes[pid] for pid in person_ids if pid in self.nodes]

    def what_produced(self, person_id: str) -> List[HyperNode]:
        """Find all outputs produced by a person."""
        output_ids = set()
        for edge in self.get_edges_for_node(person_id):
            if person_id in edge.agents:
                output_ids.update(edge.outputs)
        return [self.nodes[oid] for oid in output_ids if oid in self.nodes]

    def collaborators_of(self, person_id: str) -> List[HyperNode]:
        """Find all people who have collaborated with a given person."""
        collaborator_ids = set()
        for edge in self.get_edges_for_node(person_id):
            if person_id in edge.agents:
                collaborator_ids.update(edge.agents)
        collaborator_ids.discard(person_id)  # Remove self
        return [self.nodes[cid] for cid in collaborator_ids if cid in self.nodes]

    def inputs_for(self, node_id: str) -> List[HyperNode]:
        """Find all inputs that went into producing a node."""
        input_ids = set()
        for edge in self.get_edges_for_node(node_id):
            if node_id in edge.outputs:
                input_ids.update(edge.inputs)
        return [self.nodes[iid] for iid in input_ids if iid in self.nodes]

    # === Temporal Queries ===

    def edges_in_range(self, start: datetime, end: datetime) -> List[HyperEdge]:
        """Get all edges within a time range."""
        return [
            e for e in self.edges.values()
            if e.timestamp and start <= e.timestamp <= end
        ]

    def subgraph_at_time(self, start: datetime, end: datetime) -> "HyperGraph":
        """Extract a subgraph for a specific time range."""
        subgraph = HyperGraph()

        # Get edges in range
        edges = self.edges_in_range(start, end)

        # Collect all nodes referenced
        node_ids = set()
        for edge in edges:
            node_ids.update(edge.all_nodes)
            subgraph.add_edge(edge)

        # Add nodes
        for node_id in node_ids:
            if node_id in self.nodes:
                subgraph.add_node(self.nodes[node_id])

        return subgraph

    # === Analytics ===

    def terminal_nodes(self) -> List[HyperNode]:
        """
        Find nodes that appear as outputs but never as inputs to later edges.

        These are "dead ends" - work products that weren't built upon.
        """
        # Find all nodes that appear as outputs
        output_appearances: Dict[str, datetime] = {}  # node_id -> latest output time
        for edge in self.edges.values():
            if edge.timestamp:
                for output_id in edge.outputs:
                    if output_id not in output_appearances:
                        output_appearances[output_id] = edge.timestamp
                    else:
                        output_appearances[output_id] = max(
                            output_appearances[output_id], edge.timestamp
                        )

        # Find nodes that appear as inputs
        input_appearances: Dict[str, datetime] = {}  # node_id -> earliest input time after output
        for edge in self.edges.values():
            if edge.timestamp:
                for input_id in edge.inputs:
                    if input_id in output_appearances:
                        # Only count if this input use is after some output
                        if edge.timestamp > output_appearances.get(input_id, datetime.min):
                            if input_id not in input_appearances:
                                input_appearances[input_id] = edge.timestamp
                            else:
                                input_appearances[input_id] = min(
                                    input_appearances[input_id], edge.timestamp
                                )

        # Terminal = appeared as output but never as input afterward
        terminal_ids = set(output_appearances.keys()) - set(input_appearances.keys())

        return [self.nodes[tid] for tid in terminal_ids if tid in self.nodes]

    def node_reuse_count(self, node_id: str) -> Dict[str, int]:
        """Count how many times a node appears as input vs output."""
        as_input = 0
        as_output = 0
        as_agent = 0

        for edge in self.get_edges_for_node(node_id):
            if node_id in edge.inputs:
                as_input += 1
            if node_id in edge.outputs:
                as_output += 1
            if node_id in edge.agents:
                as_agent += 1

        return {
            "as_input": as_input,
            "as_output": as_output,
            "as_agent": as_agent,
        }

    # === Statistics ===

    def stats(self) -> Dict[str, Any]:
        """Get summary statistics about the hypergraph."""
        node_type_counts = {
            t.value: len(self._type_to_nodes.get(t.value, set()))
            for t in NodeType
        }
        edge_type_counts = {
            t.value: len(self._type_to_edges.get(t.value, set()))
            for t in EdgeType
        }

        # Time range
        timestamps = [e.timestamp for e in self.edges.values() if e.timestamp]
        time_range = None
        if timestamps:
            time_range = {
                "earliest": min(timestamps).isoformat(),
                "latest": max(timestamps).isoformat(),
            }

        return {
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
            "node_types": node_type_counts,
            "edge_types": edge_type_counts,
            "time_range": time_range,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the entire hypergraph to a dictionary."""
        return {
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "edges": [e.to_dict() for e in self.edges.values()],
            "stats": self.stats(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HyperGraph":
        """Deserialize a hypergraph from a dictionary."""
        graph = cls()

        for node_data in data.get("nodes", []):
            graph.add_node(HyperNode.from_dict(node_data))

        for edge_data in data.get("edges", []):
            graph.add_edge(HyperEdge.from_dict(edge_data))

        return graph
