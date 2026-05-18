"""
Unified Index Module

Creates a single vector index containing both document chunks and registry entries.
This enables semantic search across all data types.
"""

import json
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field


@dataclass
class IndexEntry:
    """A single entry in the unified index."""
    id: str
    doc_type: str  # "document", "person", "dataset", "project", "publication", etc.
    text: str
    embedding: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)
    mentioned_entities: List[str] = field(default_factory=list)


class UnifiedIndex:
    """
    Unified vector index for documents and registry entries.

    Supports:
    - Semantic search across all entry types
    - Filtering by doc_type
    - Entity-based boosting
    """

    def __init__(self, embedding_model=None):
        """
        Initialize the unified index.

        Args:
            embedding_model: A SentenceTransformer model or similar.
                           If None, must call set_model() before adding entries.
        """
        self.model = embedding_model
        self.entries: List[IndexEntry] = []
        self.id_to_index: Dict[str, int] = {}
        self.embedding_dim: Optional[int] = None
        self._index = None  # FAISS index, built lazily

    def set_model(self, model):
        """Set the embedding model."""
        self.model = model

    def _encode(self, text: str) -> np.ndarray:
        """Encode text to embedding vector."""
        if self.model is None:
            raise ValueError("No embedding model set. Call set_model() first.")
        return self.model.encode(text, convert_to_numpy=True)

    def add_entry(
        self,
        id: str,
        doc_type: str,
        text: str,
        metadata: Dict[str, Any] = None,
        mentioned_entities: List[str] = None,
        embedding: np.ndarray = None,
    ):
        """
        Add an entry to the index.

        Args:
            id: Unique identifier for this entry
            doc_type: Type of entry (document, person, dataset, etc.)
            text: Text content to embed (or use pre-computed embedding)
            metadata: Additional metadata for the entry
            mentioned_entities: Entity IDs mentioned in this entry
            embedding: Pre-computed embedding (if None, will compute from text)
        """
        if id in self.id_to_index:
            return  # Already exists

        if embedding is None:
            embedding = self._encode(text)

        if self.embedding_dim is None:
            self.embedding_dim = len(embedding)

        entry = IndexEntry(
            id=id,
            doc_type=doc_type,
            text=text,
            embedding=embedding,
            metadata=metadata or {},
            mentioned_entities=mentioned_entities or [],
        )

        self.id_to_index[id] = len(self.entries)
        self.entries.append(entry)
        self._index = None  # Invalidate FAISS index

    def build_faiss_index(self):
        """Build FAISS index from entries."""
        if not self.entries:
            return

        try:
            import faiss
        except ImportError:
            print("Warning: FAISS not available, using numpy for search")
            return

        embeddings = np.array([e.embedding for e in self.entries]).astype('float32')
        self._index = faiss.IndexFlatIP(self.embedding_dim)
        faiss.normalize_L2(embeddings)
        self._index.add(embeddings)

    def search(
        self,
        query: str,
        top_k: int = 10,
        doc_types: List[str] = None,
        boost_entities: List[str] = None,
        boost_factor: float = 0.2,
    ) -> List[Dict[str, Any]]:
        """
        Search the index.

        Args:
            query: Search query text
            top_k: Number of results to return
            doc_types: Filter to these document types (None = all)
            boost_entities: Entity IDs to boost in results
            boost_factor: How much to boost entity matches (0-1)

        Returns:
            List of results with scores and metadata
        """
        if not self.entries:
            return []

        query_embedding = self._encode(query)

        # Use FAISS if available
        if self._index is not None:
            query_vec = query_embedding.astype('float32').reshape(1, -1)
            import faiss
            faiss.normalize_L2(query_vec)
            scores, indices = self._index.search(query_vec, min(top_k * 3, len(self.entries)))
            scores = scores[0]
            indices = indices[0]
        else:
            # Fallback to numpy
            embeddings = np.array([e.embedding for e in self.entries])
            query_norm = query_embedding / np.linalg.norm(query_embedding)
            emb_norms = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            scores = np.dot(emb_norms, query_norm)
            indices = np.argsort(scores)[::-1][:top_k * 3]
            scores = scores[indices]

        # Build results with filtering and boosting
        results = []
        for score, idx in zip(scores, indices):
            if idx < 0:
                continue

            entry = self.entries[idx]

            # Filter by doc_type
            if doc_types and entry.doc_type not in doc_types:
                continue

            # Apply entity boost
            boost = 0
            if boost_entities:
                for entity in entry.mentioned_entities:
                    if entity in boost_entities:
                        boost += boost_factor

            results.append({
                "id": entry.id,
                "doc_type": entry.doc_type,
                "text": entry.text,
                "score": float(score) + boost,
                "metadata": entry.metadata,
                "mentioned_entities": entry.mentioned_entities,
            })

        # Re-sort by boosted scores
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def get_entry(self, id: str) -> Optional[IndexEntry]:
        """Get an entry by ID."""
        if id in self.id_to_index:
            return self.entries[self.id_to_index[id]]
        return None

    def get_by_type(self, doc_type: str) -> List[IndexEntry]:
        """Get all entries of a given type."""
        return [e for e in self.entries if e.doc_type == doc_type]

    def save(self, path: str):
        """Save the index to disk."""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Save entries (without FAISS index)
        data = {
            "embedding_dim": self.embedding_dim,
            "entries": [
                {
                    "id": e.id,
                    "doc_type": e.doc_type,
                    "text": e.text,
                    "embedding": e.embedding.tolist(),
                    "metadata": e.metadata,
                    "mentioned_entities": e.mentioned_entities,
                }
                for e in self.entries
            ],
        }

        with open(save_path, "wb") as f:
            pickle.dump(data, f)

        print(f"Saved unified index: {len(self.entries)} entries to {path}")

    def load(self, path: str):
        """Load the index from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)

        self.embedding_dim = data["embedding_dim"]
        self.entries = []
        self.id_to_index = {}

        for entry_data in data["entries"]:
            entry = IndexEntry(
                id=entry_data["id"],
                doc_type=entry_data["doc_type"],
                text=entry_data["text"],
                embedding=np.array(entry_data["embedding"]),
                metadata=entry_data["metadata"],
                mentioned_entities=entry_data["mentioned_entities"],
            )
            self.id_to_index[entry.id] = len(self.entries)
            self.entries.append(entry)

        # Rebuild FAISS index
        self.build_faiss_index()
        print(f"Loaded unified index: {len(self.entries)} entries")

    def stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        type_counts = {}
        for entry in self.entries:
            type_counts[entry.doc_type] = type_counts.get(entry.doc_type, 0) + 1

        return {
            "total_entries": len(self.entries),
            "embedding_dim": self.embedding_dim,
            "by_type": type_counts,
            "has_faiss": self._index is not None,
        }


def create_registry_text(entry_type: str, data: Dict) -> str:
    """
    Create searchable text from a registry entry.

    Args:
        entry_type: Type of registry entry (person, dataset, project, etc.)
        data: The entry data

    Returns:
        Text suitable for embedding
    """
    if entry_type == "person":
        topics = data.get("openalex", {}).get("topics", [])
        topics_str = ", ".join(topics[:10]) if topics else ""

        pubs = data.get("openalex", {}).get("recent_publications", [])
        pub_titles = [p.get("title", "") for p in pubs[:3]]
        pubs_str = "; ".join(pub_titles) if pub_titles else ""

        return f"""
        {data.get("name", "")} is a {data.get("role", "")} at {data.get("institution", "")}.
        Research areas: {topics_str}.
        Recent publications: {pubs_str}.
        """.strip()

    elif entry_type == "dataset":
        tech = data.get("technical", {}) or {}
        access = data.get("access", {}) or {}
        prov = data.get("provenance", {}) or {}
        sensitivity = data.get("sensitivity", {}) or {}
        parts = [
            f"{data.get('name', '')} - {data.get('description', '')}",
            f"Format: {tech.get('format', '')}",
            f"Access: {access.get('level', '')}",
            f"Source: {prov.get('source', '')}",
        ]
        if sensitivity.get("contains_pii") is not None:
            parts.append(f"Contains PII: {sensitivity['contains_pii']}")
        if sensitivity.get("risk_level"):
            parts.append(f"Risk: {sensitivity['risk_level']}")
        tags = data.get("tags", [])
        if tags:
            parts.append(f"Topics: {', '.join(tags)}")
        return "\n        ".join(parts).strip()

    elif entry_type == "midway_snapshot":
        return f"""
        {data.get("name", "")} database snapshot on Midway.
        Path: {data.get("path", "")}
        Size: {data.get("size_gb", "")} GB
        Description: {data.get("description", "")}
        """.strip()

    elif entry_type == "midway_embedding":
        return f"""
        {data.get("name", "")} - embedding/vector data on Midway.
        Path: {data.get("path", "")}
        Model: {data.get("model", "")}
        Description: {data.get("description", "")}
        """.strip()

    elif entry_type == "midway_social":
        platform = data.get("platform", "")
        if isinstance(platform, list):
            platform = ", ".join(platform)
        return f"""
        {data.get("name", "")} - social media dataset.
        Platform: {platform}
        Path: {data.get("path", "")}
        Content: {data.get("content_description", "")}
        """.strip()

    elif entry_type == "project":
        members = data.get("members", [])
        members_str = ", ".join(members[:5]) if members else ""
        return f"""
        {data.get("name", "")} project.
        Description: {data.get("description", "")}
        Team members: {members_str}
        Status: {data.get("status", "")}
        """.strip()

    elif entry_type == "publication":
        authors = data.get("authors", [])
        authors_str = ", ".join(authors[:5]) if authors else ""
        return f"""
        {data.get("title", "")}
        Authors: {authors_str}
        Venue: {data.get("venue", "")} ({data.get("year", "")})
        Citations: {data.get("cited_by_count", 0)}
        """.strip()

    else:
        # Generic fallback
        return f"{data.get('name', '')} - {data.get('description', '')}"


def build_unified_index(
    registry_path: str = "Data/lab_registry.json",
    midway_path: str = "Data/midway_registry.json",
    output_path: str = "Data/indexes/unified/index.pkl",
    model_name: str = "all-MiniLM-L6-v2",
) -> UnifiedIndex:
    """
    Build a unified index from registry data.

    Args:
        registry_path: Path to lab_registry.json
        midway_path: Path to midway_registry.json
        output_path: Where to save the index
        model_name: SentenceTransformer model to use

    Returns:
        The built UnifiedIndex
    """
    from sentence_transformers import SentenceTransformer

    print("Loading embedding model...")
    model = SentenceTransformer(model_name)

    index = UnifiedIndex(embedding_model=model)

    # Load lab registry
    print("Processing lab registry...")
    with open(registry_path) as f:
        registry = json.load(f)

    # Add people
    for person in registry.get("people", {}).get("members", []):
        text = create_registry_text("person", person)
        index.add_entry(
            id=f"person-{person.get('id', person.get('name', '').lower().replace(' ', '-'))}",
            doc_type="person",
            text=text,
            metadata={
                "name": person.get("name"),
                "role": person.get("role"),
                "institution": person.get("institution"),
            },
        )

    # Add projects
    for project in registry.get("projects", []):
        text = create_registry_text("project", project)
        index.add_entry(
            id=f"project-{project.get('id', project.get('name', '').lower().replace(' ', '-'))}",
            doc_type="project",
            text=text,
            metadata=project,
        )

    # Add datasets
    for dataset in registry.get("datasets", []):
        text = create_registry_text("dataset", dataset)
        index.add_entry(
            id=f"dataset-{dataset.get('id', dataset.get('name', '').lower().replace(' ', '-'))}",
            doc_type="dataset",
            text=text,
            metadata=dataset,
        )

    # Add publications
    for person in registry.get("people", {}).get("members", []):
        for pub in person.get("openalex", {}).get("recent_publications", []):
            doi = pub.get("doi") or ""  # Handle None DOI
            pub_id = doi.replace("https://doi.org/", "").replace("/", "-")
            if not pub_id:
                title = pub.get("title") or ""  # Handle None title
                pub_id = title[:50].lower().replace(" ", "-")

            text = create_registry_text("publication", pub)
            index.add_entry(
                id=f"pub-{pub_id}",
                doc_type="publication",
                text=text,
                metadata=pub,
                mentioned_entities=[person.get("id", "")],
            )

    # Load midway registry
    print("Processing midway registry...")
    with open(midway_path) as f:
        midway = json.load(f)

    def get_items(category: str) -> List[Dict]:
        data = midway.get(category, {})
        if isinstance(data, dict) and "items" in data:
            return data["items"]
        return data if isinstance(data, list) else []

    # Add snapshots
    for item in get_items("data_snapshots"):
        text = create_registry_text("midway_snapshot", item)
        index.add_entry(
            id=f"snapshot-{item.get('id', '')}",
            doc_type="midway_snapshot",
            text=text,
            metadata=item,
        )

    # Add embeddings
    for item in get_items("embeddings"):
        text = create_registry_text("midway_embedding", item)
        index.add_entry(
            id=f"embedding-{item.get('id', '')}",
            doc_type="midway_embedding",
            text=text,
            metadata=item,
        )

    # Add social media datasets
    for item in get_items("social_media_datasets"):
        text = create_registry_text("midway_social", item)
        index.add_entry(
            id=f"social-{item.get('id', '')}",
            doc_type="midway_social",
            text=text,
            metadata=item,
        )

    # Add precomputed resources
    for item in get_items("precomputed_resources"):
        text = create_registry_text("midway_snapshot", item)  # Similar format
        index.add_entry(
            id=f"precomputed-{item.get('id', '')}",
            doc_type="midway_precomputed",
            text=text,
            metadata=item,
        )

    # Build FAISS index
    print("Building FAISS index...")
    index.build_faiss_index()

    # Save
    print(f"Saving to {output_path}...")
    index.save(output_path)

    # Print stats
    stats = index.stats()
    print(f"\nUnified Index Stats:")
    print(f"  Total entries: {stats['total_entries']}")
    print(f"  By type:")
    for doc_type, count in stats['by_type'].items():
        print(f"    - {doc_type}: {count}")

    return index


if __name__ == "__main__":
    build_unified_index()
