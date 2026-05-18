"""
Entity Extraction Module

Extracts and links entities (people, datasets, projects, topics) from text.
Used during document ingestion to create entity links.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class ExtractedEntities:
    """Container for extracted entities from a text."""
    people: List[str] = field(default_factory=list)
    datasets: List[str] = field(default_factory=list)
    projects: List[str] = field(default_factory=list)
    topics: List[str] = field(default_factory=list)
    institutions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, List[str]]:
        return {
            "people": self.people,
            "datasets": self.datasets,
            "projects": self.projects,
            "topics": self.topics,
            "institutions": self.institutions,
        }

    def all_entities(self) -> List[str]:
        """Get all entity IDs as a flat list."""
        return self.people + self.datasets + self.projects + self.topics + self.institutions

    def is_empty(self) -> bool:
        return not (self.people or self.datasets or self.projects or self.topics or self.institutions)


class EntityExtractor:
    """
    Extracts entities from text using pattern matching against registry data.
    """

    def __init__(
        self,
        registry_path: str = "Data/lab_registry.json",
        midway_path: str = "Data/midway_registry.json",
    ):
        # Entity lookup dictionaries: name_variant -> entity_id
        self.person_names: Dict[str, str] = {}
        self.dataset_names: Dict[str, str] = {}
        self.project_names: Dict[str, str] = {}
        self.topic_keywords: Dict[str, str] = {}
        self.institution_names: Dict[str, str] = {}

        self._load_entities(registry_path, midway_path)

    def _load_entities(self, registry_path: str, midway_path: str):
        """Load entities from registry files."""
        # Load lab registry
        lab_path = Path(registry_path)
        if lab_path.exists():
            with open(lab_path) as f:
                registry = json.load(f)

            # Load people
            for person in registry.get("people", {}).get("members", []):
                person_id = person.get("id", person.get("name", "").lower().replace(" ", "-"))
                name = person.get("name", "")

                # Full name
                self.person_names[name.lower()] = person_id

                # First name + last name separately (for partial matching)
                parts = name.split()
                if len(parts) >= 2:
                    # Last name is usually most distinctive
                    self.person_names[parts[-1].lower()] = person_id
                    # First + last
                    self.person_names[f"{parts[0].lower()} {parts[-1].lower()}"] = person_id

            # Load projects
            for project in registry.get("projects", []):
                project_id = project.get("id", project.get("name", "").lower().replace(" ", "-"))
                name = project.get("name", "")
                self.project_names[name.lower()] = project_id

                # Also add acronyms
                if project.get("acronym"):
                    self.project_names[project["acronym"].lower()] = project_id

            # Load datasets
            for dataset in registry.get("datasets", []):
                dataset_id = dataset.get("id", dataset.get("name", "").lower().replace(" ", "-"))
                name = dataset.get("name", "")
                self.dataset_names[name.lower()] = dataset_id

            # Load topics from people
            for person in registry.get("people", {}).get("members", []):
                topics = person.get("openalex", {}).get("topics", [])
                for topic in topics:
                    if isinstance(topic, str) and len(topic) >= 5:
                        topic_id = topic.lower().replace(" ", "-")[:50]
                        self.topic_keywords[topic.lower()] = topic_id

            # Load institutions
            institutions_seen = set()
            for person in registry.get("people", {}).get("members", []):
                inst = person.get("institution", "")
                if isinstance(inst, str) and inst and inst.lower() not in institutions_seen:
                    inst_id = inst.lower().replace(" ", "-")
                    self.institution_names[inst.lower()] = inst_id
                    institutions_seen.add(inst.lower())

        # Load midway registry
        midway_path_obj = Path(midway_path)
        if midway_path_obj.exists():
            with open(midway_path_obj) as f:
                midway = json.load(f)

            # Load all midway items as datasets
            for category in ["data_snapshots", "embeddings", "precomputed_resources",
                           "social_media_datasets"]:
                items = midway.get(category, {})
                if isinstance(items, dict) and "items" in items:
                    items = items["items"]
                elif not isinstance(items, list):
                    items = []

                for item in items:
                    item_id = item.get("id", "")
                    name = item.get("name", "")
                    if name:
                        self.dataset_names[name.lower()] = f"midway-{item_id}"

        # Add common aliases for datasets and resources
        self._add_common_aliases()

        print(f"EntityExtractor loaded: {len(self.person_names)} people, "
              f"{len(self.dataset_names)} datasets, {len(self.project_names)} projects, "
              f"{len(self.topic_keywords)} topics")

    def _add_common_aliases(self):
        """Add common shorthand names that people use in conversation."""
        # Dataset aliases - map common terms to canonical dataset IDs
        dataset_aliases = {
            "reddit": "reddit-data",
            "reddit data": "reddit-data",
            "twitter": "twitter-data",
            "twitter data": "twitter-data",
            "openalex": "openalex-data",
            "open alex": "openalex-data",
            "semantic scholar": "semantic-scholar-data",
            "web of science": "wos-data",
            "wos": "wos-data",
            "pubmed": "pubmed-data",
            "mag": "mag-data",
            "microsoft academic": "mag-data",
            "dblp": "dblp-data",
            "proquest": "proquest-data",
            "patent": "patent-data",
            "patents": "patent-data",
        }
        for alias, dataset_id in dataset_aliases.items():
            if alias not in self.dataset_names:
                self.dataset_names[alias] = dataset_id

        # Computing resource aliases
        resource_aliases = {
            "midway": "midway-cluster",
            "midway cluster": "midway-cluster",
            "rcc": "rcc-resources",
            "research computing": "rcc-resources",
            "gpu": "gpu-resources",
            "gpus": "gpu-resources",
        }
        for alias, resource_id in resource_aliases.items():
            if alias not in self.dataset_names:
                self.dataset_names[alias] = resource_id

        # Project aliases
        project_aliases = {
            "apto": "apto-project",
            "knowledge lab": "knowledge-lab",
            "klab": "knowledge-lab",
        }
        for alias, project_id in project_aliases.items():
            if alias not in self.project_names:
                self.project_names[alias] = project_id

    def extract(self, text: str) -> ExtractedEntities:
        """
        Extract entities from text.

        Args:
            text: Text to extract entities from

        Returns:
            ExtractedEntities containing all found entities
        """
        text_lower = text.lower()
        result = ExtractedEntities()

        # Extract people (use word boundaries for names)
        found_people = set()
        for name, person_id in self.person_names.items():
            # Use word boundary matching for names
            if len(name) >= 4:  # Avoid short names causing false positives
                pattern = r'\b' + re.escape(name) + r'\b'
                if re.search(pattern, text_lower):
                    found_people.add(person_id)
        result.people = list(found_people)

        # Extract datasets
        found_datasets = set()
        for name, dataset_id in self.dataset_names.items():
            if len(name) >= 4 and name in text_lower:
                found_datasets.add(dataset_id)
        result.datasets = list(found_datasets)

        # Extract projects
        found_projects = set()
        for name, project_id in self.project_names.items():
            if name in text_lower:
                found_projects.add(project_id)
        result.projects = list(found_projects)

        # Extract topics (be more selective - require longer matches)
        found_topics = set()
        for topic, topic_id in self.topic_keywords.items():
            if len(topic) >= 10 and topic in text_lower:
                found_topics.add(topic_id)
        result.topics = list(found_topics)[:5]  # Limit topics per chunk

        # Extract institutions
        found_institutions = set()
        for inst, inst_id in self.institution_names.items():
            if len(inst) >= 5 and inst in text_lower:
                found_institutions.add(inst_id)
        result.institutions = list(found_institutions)

        return result

    def extract_from_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """
        Extract entities from a list of document chunks.

        Args:
            chunks: List of chunk dictionaries with 'text' field

        Returns:
            List of chunks with 'entities' field added
        """
        enriched_chunks = []
        for chunk in chunks:
            text = chunk.get("text", "")
            entities = self.extract(text)

            enriched_chunk = dict(chunk)
            enriched_chunk["entities"] = entities.to_dict()
            enriched_chunk["entity_ids"] = entities.all_entities()
            enriched_chunks.append(enriched_chunk)

        return enriched_chunks


def extract_entities_from_file(
    file_path: str,
    extractor: EntityExtractor = None,
) -> Dict[str, List[str]]:
    """
    Extract entities from a text file.

    Args:
        file_path: Path to text file
        extractor: EntityExtractor instance (will create if None)

    Returns:
        Dictionary of entity types to lists of entity IDs
    """
    if extractor is None:
        extractor = EntityExtractor()

    with open(file_path) as f:
        text = f.read()

    entities = extractor.extract(text)
    return entities.to_dict()


def create_entity_links_file(
    chunks_path: str = "Data/rag_indexes/hybrid/chunks.pkl",
    output_path: str = "Data/entity_links.json",
    registry_path: str = "Data/lab_registry.json",
    midway_path: str = "Data/midway_registry.json",
):
    """
    Create entity links file from existing chunks.

    Args:
        chunks_path: Path to pickled chunks from RAG indexing
        output_path: Where to save entity links
        registry_path: Path to lab registry
        midway_path: Path to midway registry
    """
    import pickle

    print("Loading chunks...")
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)

    print(f"Loaded {len(chunks)} chunks")

    print("Initializing entity extractor...")
    extractor = EntityExtractor(registry_path, midway_path)

    print("Extracting entities from chunks...")
    entity_links = {}

    for i, chunk in enumerate(chunks):
        if i % 1000 == 0:
            print(f"  Processing chunk {i}/{len(chunks)}...")

        text = chunk.get("text", "")
        source = chunk.get("source", f"chunk_{i}")

        entities = extractor.extract(text)

        if not entities.is_empty():
            chunk_id = chunk.get("id", f"{source}_chunk_{i}")
            entity_links[chunk_id] = {
                "source": source,
                "entities": entities.to_dict(),
            }

    print(f"\nExtracted entities from {len(entity_links)} chunks")

    # Count entity types
    people_count = sum(len(v["entities"].get("people", [])) for v in entity_links.values())
    dataset_count = sum(len(v["entities"].get("datasets", [])) for v in entity_links.values())
    project_count = sum(len(v["entities"].get("projects", [])) for v in entity_links.values())

    print(f"  People mentions: {people_count}")
    print(f"  Dataset mentions: {dataset_count}")
    print(f"  Project mentions: {project_count}")

    print(f"\nSaving to {output_path}...")
    with open(output_path, "w") as f:
        json.dump(entity_links, f, indent=2)

    print("Done!")


# Singleton instance
_extractor: Optional[EntityExtractor] = None


def get_extractor() -> EntityExtractor:
    """Get or create the singleton EntityExtractor instance."""
    global _extractor
    if _extractor is None:
        _extractor = EntityExtractor()
    return _extractor


if __name__ == "__main__":
    # Test the extractor
    extractor = EntityExtractor()

    test_texts = [
        "James Evans and his collaborators at the University of Chicago have published research on network science.",
        "The APTO project uses OpenAlex data to study science of science.",
        "We collected Reddit and Twitter data for social media analysis.",
        "The patent similarity matrices are stored on Midway.",
    ]

    for text in test_texts:
        print(f"\nText: {text[:80]}...")
        entities = extractor.extract(text)
        print(f"  People: {entities.people}")
        print(f"  Datasets: {entities.datasets}")
        print(f"  Projects: {entities.projects}")
