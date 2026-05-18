"""
Semantic Registry Index Module

Provides embedding-based semantic search for registry entities (people, datasets, projects).
Uses FAISS for fast similarity search and sentence-transformers for embeddings.

Usage:
    from registry.semantic_index import SemanticRegistryIndex

    index = SemanticRegistryIndex()
    index.build_index()

    # Search for researchers by topic
    results = index.search("researchers working on network science")

    # Find similar entities
    similar = index.find_similar("person-james-evans")
"""

import json
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class RegistryEntity:
    """A single entity in the semantic registry index."""
    id: str
    entity_type: str  # "person", "dataset", "project", "funding"
    name: str
    text: str  # Rich text representation for embedding
    embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class SemanticRegistryIndex:
    """
    Semantic search index for registry entities.

    Embeds all registry entities (people, datasets, projects) with their descriptions
    and uses FAISS for fast similarity search.

    Supports queries like:
    - "researchers working on network science"
    - "datasets about social media"
    - "projects related to AI prediction"
    """

    def __init__(
        self,
        registry_path: str = "Data/lab_registry.json",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        """
        Initialize the semantic registry index.

        Args:
            registry_path: Path to lab_registry.json
            model_name: SentenceTransformer model to use for embeddings
        """
        self.registry_path = Path(registry_path)
        self.model_name = model_name

        self.model = None
        self.faiss_index = None
        self.entities: List[RegistryEntity] = []
        self.id_to_index: Dict[str, int] = {}
        self.type_to_indices: Dict[str, List[int]] = {}
        self.embedding_dim: Optional[int] = None

        # Registry data
        self.registry: Dict[str, Any] = {}

    def _load_model(self):
        """Lazy-load the embedding model."""
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            print(f"Loading embedding model: {self.model_name}...")
            self.model = SentenceTransformer(self.model_name)

    def _encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts to embedding vectors."""
        self._load_model()
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=len(texts) > 10)
        return embeddings.astype('float32')

    def _create_person_text(self, person: Dict[str, Any]) -> str:
        """
        Create rich text representation for a person entity.

        Combines: name + role + institution + research topics + bio + recent publications
        """
        parts = []

        # Basic info
        name = person.get("name", "Unknown")
        role = person.get("role", "")
        institution = person.get("institution", "")

        parts.append(f"{name} is a {role}" + (f" at {institution}" if institution else "") + ".")

        # Bio if available
        bio = person.get("bio", "")
        if bio:
            # Truncate very long bios
            if len(bio) > 500:
                bio = bio[:500] + "..."
            parts.append(bio)

        # Research topics from OpenAlex
        openalex = person.get("openalex", {})
        topics = openalex.get("topics", [])
        if topics:
            # Topics can be strings or dicts
            topic_names = []
            for t in topics[:10]:
                if isinstance(t, str):
                    topic_names.append(t)
                elif isinstance(t, dict):
                    topic_names.append(t.get("display_name", ""))
            if topic_names:
                parts.append(f"Research areas: {', '.join(topic_names)}.")

        # Recent publication titles
        publications = openalex.get("recent_publications", [])
        if publications:
            pub_titles = [p.get("title", "") for p in publications[:5] if p.get("title")]
            if pub_titles:
                parts.append(f"Recent publications: {'; '.join(pub_titles)}.")

        # Bibliometric info
        if openalex.get("h_index"):
            parts.append(f"h-index: {openalex['h_index']}, cited by: {openalex.get('cited_by_count', 0)}.")

        return " ".join(parts)

    def _create_dataset_text(self, dataset: Dict[str, Any]) -> str:
        """
        Create rich text representation for a dataset entity.

        Combines: name + description + technical details + access + sensitivity +
        creators + preprocessing + models + tags
        """
        parts = []

        name = dataset.get("name", "Unknown Dataset")
        description = dataset.get("description", "")

        parts.append(f"{name}.")
        if description:
            parts.append(description)

        # Technical info (nested)
        tech = dataset.get("technical", {})
        if tech:
            fmt = tech.get("format", "")
            if fmt:
                parts.append(f"Format: {fmt}.")
            size = tech.get("size_gb")
            if size:
                parts.append(f"Size: {size} GB.")
            location = tech.get("location", "")
            if location:
                parts.append(f"Location: {location}.")

        # Access info (nested)
        access = dataset.get("access", {})
        if access:
            level = access.get("level", "")
            if level:
                parts.append(f"Access: {level}.")

        # Provenance
        prov = dataset.get("provenance", {})
        if prov:
            source = prov.get("source", "")
            if source:
                parts.append(f"Source: {source}.")
            license_str = prov.get("license", "")
            if license_str:
                parts.append(f"License: {license_str}.")

        # Sensitivity (Data Cards)
        sensitivity = dataset.get("sensitivity", {}) or {}
        risk = sensitivity.get("risk_level")
        if risk:
            parts.append(f"Risk level: {risk}.")
        if sensitivity.get("contains_pii"):
            parts.append("Contains PII.")

        # Creators
        creators = dataset.get("creators", [])
        if creators:
            affiliations = [c.get("affiliation", "") for c in creators if c.get("affiliation")]
            if affiliations:
                parts.append(f"Created by: {', '.join(affiliations)}.")

        # Preprocessing
        preprocessing = dataset.get("preprocessing", {}) or {}
        steps = preprocessing.get("steps", [])
        if steps:
            parts.append(f"Preprocessing: {'; '.join(steps[:3])}.")

        # Known models
        models_info = dataset.get("known_models_and_benchmarks", {}) or {}
        models = models_info.get("models_trained", [])
        if models:
            parts.append(f"Models: {', '.join(models[:5])}.")

        # Tags
        tags = dataset.get("tags", [])
        if tags:
            parts.append(f"Topics: {', '.join(tags)}.")

        return " ".join(parts)

    def _create_project_text(self, project: Dict[str, Any]) -> str:
        """
        Create rich text representation for a project entity.

        Combines: name + description + status + tags + leads
        """
        parts = []

        name = project.get("name", "Unknown Project")
        description = project.get("description", "")
        status = project.get("status", "")

        parts.append(f"{name} project" + (f" ({status})" if status else "") + ".")

        if description:
            parts.append(description)

        # Project leads
        leads = project.get("leads", [])
        if leads:
            parts.append(f"Led by: {', '.join(leads)}.")

        # Team members
        members = project.get("members", [])
        if members:
            parts.append(f"Team: {', '.join(members[:10])}.")

        # Tags/topics
        tags = project.get("tags", [])
        if tags:
            parts.append(f"Topics: {', '.join(tags)}.")

        # Start date
        start_date = project.get("start_date")
        if start_date:
            parts.append(f"Started: {start_date}.")

        return " ".join(parts)

    def _create_funding_text(self, funding: Dict[str, Any]) -> str:
        """
        Create rich text representation for a funding entity.

        Combines: name + source + type + PI + amount + status
        """
        parts = []

        name = funding.get("name", "Unknown Grant")
        source = funding.get("source", "")
        status = funding.get("status", "")

        parts.append(f"{name}" + (f" from {source}" if source else "") + (f" ({status})" if status else "") + ".")

        # Grant type
        grant_type = funding.get("type", "")
        if grant_type:
            parts.append(f"Type: {grant_type}.")

        # Amount
        amount = funding.get("amount")
        if amount:
            currency = funding.get("currency", "USD")
            parts.append(f"Amount: ${amount:,} {currency}.")

        # PI
        pi = funding.get("pi", "")
        if pi:
            parts.append(f"Principal Investigator: {pi}.")

        # Co-PIs
        co_pis = funding.get("co_pis", [])
        if co_pis:
            parts.append(f"Co-PIs: {', '.join(co_pis)}.")

        # Duration
        start = funding.get("start_date")
        end = funding.get("end_date")
        if start:
            duration = f"From {start}"
            if end:
                duration += f" to {end}"
            parts.append(duration + ".")

        return " ".join(parts)

    def build_index(self) -> None:
        """
        Load registry data and create embeddings for all entities.

        Builds the FAISS index for fast similarity search.
        """
        print(f"Loading registry from {self.registry_path}...")

        if not self.registry_path.exists():
            raise FileNotFoundError(f"Registry not found at {self.registry_path}")

        with open(self.registry_path) as f:
            self.registry = json.load(f)

        # Clear existing data
        self.entities = []
        self.id_to_index = {}
        self.type_to_indices = {
            "person": [],
            "dataset": [],
            "project": [],
            "funding": [],
        }

        texts_to_embed = []

        # Process people
        print("Processing people...")
        members = self.registry.get("people", {}).get("members", [])
        for person in members:
            person_id = person.get("id", person.get("name", "").lower().replace(" ", "-"))
            entity_id = f"person-{person_id}"

            text = self._create_person_text(person)

            entity = RegistryEntity(
                id=entity_id,
                entity_type="person",
                name=person.get("name", "Unknown"),
                text=text,
                metadata={
                    "role": person.get("role"),
                    "institution": person.get("institution"),
                    "email": person.get("email"),
                    "topics": person.get("openalex", {}).get("topics", []),
                    "h_index": person.get("openalex", {}).get("h_index"),
                    "cited_by_count": person.get("openalex", {}).get("cited_by_count"),
                }
            )

            self.id_to_index[entity_id] = len(self.entities)
            self.type_to_indices["person"].append(len(self.entities))
            self.entities.append(entity)
            texts_to_embed.append(text)

        # Process datasets
        print("Processing datasets...")
        for dataset in self.registry.get("datasets", []):
            dataset_id = dataset.get("id", dataset.get("name", "").lower().replace(" ", "-"))
            entity_id = f"dataset-{dataset_id}"

            text = self._create_dataset_text(dataset)

            tech = dataset.get("technical", {})
            entity = RegistryEntity(
                id=entity_id,
                entity_type="dataset",
                name=dataset.get("name", "Unknown"),
                text=text,
                metadata={
                    "description": dataset.get("description"),
                    "format": tech.get("format") if tech else None,
                    "size_gb": tech.get("size_gb") if tech else None,
                    "access_level": dataset.get("access", {}).get("level"),
                    "location": tech.get("location") if tech else None,
                    "tags": dataset.get("tags", []),
                }
            )

            self.id_to_index[entity_id] = len(self.entities)
            self.type_to_indices["dataset"].append(len(self.entities))
            self.entities.append(entity)
            texts_to_embed.append(text)

        # Process projects
        print("Processing projects...")
        for project in self.registry.get("projects", []):
            project_id = project.get("id", project.get("name", "").lower().replace(" ", "-"))
            entity_id = f"project-{project_id}"

            text = self._create_project_text(project)

            entity = RegistryEntity(
                id=entity_id,
                entity_type="project",
                name=project.get("name", "Unknown"),
                text=text,
                metadata={
                    "description": project.get("description"),
                    "status": project.get("status"),
                    "leads": project.get("leads", []),
                    "tags": project.get("tags", []),
                    "start_date": project.get("start_date"),
                }
            )

            self.id_to_index[entity_id] = len(self.entities)
            self.type_to_indices["project"].append(len(self.entities))
            self.entities.append(entity)
            texts_to_embed.append(text)

        # Process funding
        print("Processing funding...")
        for funding in self.registry.get("funding", []):
            funding_id = funding.get("id", funding.get("name", "").lower().replace(" ", "-"))
            entity_id = f"funding-{funding_id}"

            text = self._create_funding_text(funding)

            entity = RegistryEntity(
                id=entity_id,
                entity_type="funding",
                name=funding.get("name", "Unknown"),
                text=text,
                metadata={
                    "source": funding.get("source"),
                    "type": funding.get("type"),
                    "amount": funding.get("amount"),
                    "pi": funding.get("pi"),
                    "status": funding.get("status"),
                }
            )

            self.id_to_index[entity_id] = len(self.entities)
            self.type_to_indices["funding"].append(len(self.entities))
            self.entities.append(entity)
            texts_to_embed.append(text)

        # Create embeddings
        print(f"Creating embeddings for {len(texts_to_embed)} entities...")
        embeddings = self._encode(texts_to_embed)

        # Assign embeddings to entities
        for i, entity in enumerate(self.entities):
            entity.embedding = embeddings[i]

        self.embedding_dim = embeddings.shape[1]

        # Build FAISS index
        print("Building FAISS index...")
        self._build_faiss_index(embeddings)

        # Print summary
        print(f"\nSemantic Registry Index built:")
        print(f"  - People: {len(self.type_to_indices['person'])}")
        print(f"  - Datasets: {len(self.type_to_indices['dataset'])}")
        print(f"  - Projects: {len(self.type_to_indices['project'])}")
        print(f"  - Funding: {len(self.type_to_indices['funding'])}")
        print(f"  - Total: {len(self.entities)} entities")
        print(f"  - Embedding dim: {self.embedding_dim}")

    def _build_faiss_index(self, embeddings: np.ndarray) -> None:
        """Build the FAISS index from embeddings."""
        try:
            import faiss
        except ImportError:
            print("Warning: FAISS not available, falling back to numpy search")
            self.faiss_index = None
            return

        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)

        # Use inner product index (equivalent to cosine sim after normalization)
        self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
        self.faiss_index.add(embeddings)

    def search(
        self,
        query: str,
        top_k: int = 5,
        entity_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for entities matching a natural language query.

        Args:
            query: Natural language search query (e.g., "researchers working on network science")
            top_k: Number of results to return
            entity_type: Filter to specific entity type ("person", "dataset", "project", "funding")
                        or None for all types

        Returns:
            List of matching entities with scores and metadata
        """
        if not self.entities:
            return []

        # Encode query
        query_embedding = self._encode([query])[0]

        # Determine which indices to search
        if entity_type and entity_type in self.type_to_indices:
            search_indices = self.type_to_indices[entity_type]
            if not search_indices:
                return []
        else:
            search_indices = None  # Search all

        # Search using FAISS or numpy fallback
        if self.faiss_index is not None:
            results = self._faiss_search(query_embedding, top_k, search_indices)
        else:
            results = self._numpy_search(query_embedding, top_k, search_indices)

        return results

    def _faiss_search(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        filter_indices: Optional[List[int]] = None,
    ) -> List[Dict[str, Any]]:
        """Search using FAISS index."""
        import faiss

        query_vec = query_embedding.reshape(1, -1).astype('float32')
        faiss.normalize_L2(query_vec)

        if filter_indices is not None:
            # Need to search more candidates and filter
            search_k = min(len(self.entities), top_k * 10)
        else:
            search_k = min(len(self.entities), top_k)

        scores, indices = self.faiss_index.search(query_vec, search_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue

            # Apply type filter
            if filter_indices is not None and idx not in filter_indices:
                continue

            entity = self.entities[idx]
            results.append({
                "id": entity.id,
                "entity_type": entity.entity_type,
                "name": entity.name,
                "score": float(score),
                "text": entity.text,
                "metadata": entity.metadata,
            })

            if len(results) >= top_k:
                break

        return results

    def _numpy_search(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        filter_indices: Optional[List[int]] = None,
    ) -> List[Dict[str, Any]]:
        """Fallback search using numpy."""
        # Normalize query
        query_norm = query_embedding / np.linalg.norm(query_embedding)

        # Get all embeddings
        if filter_indices is not None:
            indices = filter_indices
            embeddings = np.array([self.entities[i].embedding for i in indices])
        else:
            indices = list(range(len(self.entities)))
            embeddings = np.array([e.embedding for e in self.entities])

        # Normalize embeddings
        emb_norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings_normalized = embeddings / emb_norms

        # Compute cosine similarity
        scores = np.dot(embeddings_normalized, query_norm)

        # Get top k
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for i in top_indices:
            actual_idx = indices[i]
            entity = self.entities[actual_idx]
            results.append({
                "id": entity.id,
                "entity_type": entity.entity_type,
                "name": entity.name,
                "score": float(scores[i]),
                "text": entity.text,
                "metadata": entity.metadata,
            })

        return results

    def find_similar(
        self,
        entity_id: str,
        top_k: int = 5,
        same_type_only: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Find entities similar to a given entity.

        Args:
            entity_id: ID of the entity to find similar items for (e.g., "person-james-evans")
            top_k: Number of similar entities to return
            same_type_only: If True, only return entities of the same type

        Returns:
            List of similar entities with scores
        """
        if entity_id not in self.id_to_index:
            return []

        idx = self.id_to_index[entity_id]
        entity = self.entities[idx]

        # Determine filter
        filter_indices = None
        if same_type_only:
            filter_indices = self.type_to_indices.get(entity.entity_type)

        # Search using the entity's embedding
        if self.faiss_index is not None:
            # Need to exclude the entity itself
            results = self._faiss_search(entity.embedding, top_k + 1, filter_indices)
            results = [r for r in results if r["id"] != entity_id][:top_k]
        else:
            results = self._numpy_search(entity.embedding, top_k + 1, filter_indices)
            results = [r for r in results if r["id"] != entity_id][:top_k]

        return results

    def get_entity(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """Get an entity by ID."""
        if entity_id not in self.id_to_index:
            return None

        entity = self.entities[self.id_to_index[entity_id]]
        return {
            "id": entity.id,
            "entity_type": entity.entity_type,
            "name": entity.name,
            "text": entity.text,
            "metadata": entity.metadata,
        }

    def save_index(self, path: str) -> None:
        """
        Save the index to disk for fast loading later.

        Args:
            path: Path to save the index (e.g., "Data/indexes/semantic_registry.pkl")
        """
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Prepare data for saving
        data = {
            "embedding_dim": self.embedding_dim,
            "model_name": self.model_name,
            "entities": [
                {
                    "id": e.id,
                    "entity_type": e.entity_type,
                    "name": e.name,
                    "text": e.text,
                    "embedding": e.embedding.tolist() if e.embedding is not None else None,
                    "metadata": e.metadata,
                }
                for e in self.entities
            ],
        }

        with open(save_path, "wb") as f:
            pickle.dump(data, f)

        print(f"Saved semantic registry index: {len(self.entities)} entities to {path}")

    def load_index(self, path: str) -> None:
        """
        Load a previously saved index.

        Args:
            path: Path to the saved index
        """
        with open(path, "rb") as f:
            data = pickle.load(f)

        self.embedding_dim = data["embedding_dim"]
        self.model_name = data.get("model_name", self.model_name)

        # Rebuild entities
        self.entities = []
        self.id_to_index = {}
        self.type_to_indices = {
            "person": [],
            "dataset": [],
            "project": [],
            "funding": [],
        }

        embeddings_list = []

        for entry_data in data["entities"]:
            embedding = np.array(entry_data["embedding"]).astype('float32') if entry_data["embedding"] else None

            entity = RegistryEntity(
                id=entry_data["id"],
                entity_type=entry_data["entity_type"],
                name=entry_data["name"],
                text=entry_data["text"],
                embedding=embedding,
                metadata=entry_data["metadata"],
            )

            self.id_to_index[entity.id] = len(self.entities)
            if entity.entity_type in self.type_to_indices:
                self.type_to_indices[entity.entity_type].append(len(self.entities))
            self.entities.append(entity)

            if embedding is not None:
                embeddings_list.append(embedding)

        # Rebuild FAISS index
        if embeddings_list:
            embeddings = np.array(embeddings_list)
            self._build_faiss_index(embeddings)

        print(f"Loaded semantic registry index: {len(self.entities)} entities")

    def stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        return {
            "total_entities": len(self.entities),
            "embedding_dim": self.embedding_dim,
            "model": self.model_name,
            "by_type": {
                entity_type: len(indices)
                for entity_type, indices in self.type_to_indices.items()
            },
            "has_faiss": self.faiss_index is not None,
        }


def merge_with_structured(
    semantic_results: List[Dict[str, Any]],
    structured_results: List[Dict[str, Any]],
    boost_overlap: float = 0.3,
) -> List[Dict[str, Any]]:
    """
    Merge semantic search results with structured lookup results.

    Boosts results that appear in both, creating a hybrid ranking.

    Args:
        semantic_results: Results from SemanticRegistryIndex.search()
        structured_results: Results from RegistryLookup
        boost_overlap: Score boost for results in both lists

    Returns:
        Merged and re-ranked results
    """
    # Index structured results by ID
    structured_ids = set()
    structured_by_id = {}

    for r in structured_results:
        # Handle different ID formats
        entity_id = r.get("data", {}).get("id", "")
        entity_type = r.get("type", "")

        if entity_id:
            full_id = f"{entity_type}-{entity_id}"
            structured_ids.add(full_id)
            structured_by_id[full_id] = r

    # Score and merge results
    merged = {}

    # Add semantic results
    for r in semantic_results:
        entity_id = r["id"]
        score = r["score"]

        # Boost if also in structured results
        if entity_id in structured_ids:
            score += boost_overlap
            r["match_type"] = "both"
        else:
            r["match_type"] = "semantic"

        r["merged_score"] = score
        merged[entity_id] = r

    # Add structured results not in semantic
    for full_id, r in structured_by_id.items():
        if full_id not in merged:
            merged[full_id] = {
                "id": full_id,
                "entity_type": r.get("type", ""),
                "name": r.get("data", {}).get("name", "Unknown"),
                "score": 0.5,  # Base score for structured-only
                "merged_score": 0.5,
                "text": "",
                "metadata": r.get("data", {}),
                "match_type": "structured",
                "match_reason": r.get("match_reason", ""),
            }

    # Sort by merged score
    sorted_results = sorted(merged.values(), key=lambda x: x["merged_score"], reverse=True)

    return sorted_results


# Convenience function for building and saving index
def build_semantic_registry_index(
    registry_path: str = "Data/lab_registry.json",
    output_path: str = "Data/indexes/semantic_registry.pkl",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> SemanticRegistryIndex:
    """
    Build and save a semantic registry index.

    Args:
        registry_path: Path to lab_registry.json
        output_path: Where to save the index
        model_name: SentenceTransformer model to use

    Returns:
        The built SemanticRegistryIndex
    """
    index = SemanticRegistryIndex(
        registry_path=registry_path,
        model_name=model_name,
    )
    index.build_index()
    index.save_index(output_path)
    return index


if __name__ == "__main__":
    # Build and test the index
    index = build_semantic_registry_index()

    print("\n" + "="*60)
    print("Testing semantic search...")
    print("="*60)

    # Test searches
    test_queries = [
        ("researchers working on network science", None),
        ("datasets about scholarly publications", "dataset"),
        ("AI prediction projects", "project"),
        ("NSF funding", "funding"),
    ]

    for query, entity_type in test_queries:
        print(f"\nQuery: '{query}'" + (f" (type={entity_type})" if entity_type else ""))
        print("-" * 40)
        results = index.search(query, top_k=3, entity_type=entity_type)
        for r in results:
            print(f"  [{r['score']:.3f}] {r['name']} ({r['entity_type']})")

    # Test find_similar
    print("\n" + "="*60)
    print("Testing find_similar...")
    print("="*60)

    if index.entities:
        first_person = next(
            (e for e in index.entities if e.entity_type == "person"),
            None
        )
        if first_person:
            print(f"\nSimilar to: {first_person.name}")
            print("-" * 40)
            similar = index.find_similar(first_person.id, top_k=3)
            for r in similar:
                print(f"  [{r['score']:.3f}] {r['name']} ({r['entity_type']})")
