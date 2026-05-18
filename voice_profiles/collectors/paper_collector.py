"""
Paper Collector

Extracts text samples from publications (abstracts, full text) for disposition analysis.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional

from ..processors.disposition_extractor import TextSample


class PaperCollector:
    """
    Collects text samples from papers and publications.

    Sources:
    - Lab registry (OpenAlex data)
    - Parsed publication abstracts
    - Full paper text if available
    """

    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize collector.

        Args:
            data_path: Path to Data directory
        """
        if data_path:
            self.data_path = Path(data_path)
        else:
            self.data_path = Path(__file__).parent.parent.parent / "Data"

        self.registry_path = self.data_path / "lab_registry.json"
        self.parsed_path = self.data_path / "parsed"
        self._registry: Optional[Dict] = None

    def _load_registry(self) -> Dict:
        """Load lab registry if not already loaded."""
        if self._registry is None:
            if self.registry_path.exists():
                with open(self.registry_path, "r") as f:
                    self._registry = json.load(f)
            else:
                self._registry = {}
        return self._registry

    def get_person_publications(self, person_name: str) -> List[Dict]:
        """
        Get publications for a person from the registry.

        Args:
            person_name: Name to search for

        Returns:
            List of publication dicts with title, abstract, year, etc.
        """
        registry = self._load_registry()
        name_lower = person_name.lower()

        publications = []

        # Search through people
        for member in registry.get("people", {}).get("members", []):
            member_name = member.get("name", "").lower()

            if name_lower in member_name or member_name in name_lower:
                # Found the person - get their OpenAlex publications
                openalex = member.get("openalex", {})
                recent_pubs = openalex.get("recent_publications", [])

                for pub in recent_pubs:
                    publications.append({
                        "title": pub.get("title", ""),
                        "year": pub.get("year"),
                        "cited_by_count": pub.get("cited_by_count", 0),
                        "doi": pub.get("doi"),
                        "abstract": pub.get("abstract"),  # May not be present
                        "topics": pub.get("topics", []),
                    })

        return publications

    def get_person_topics(self, person_name: str) -> List[str]:
        """
        Get research topics for a person.

        Args:
            person_name: Name to search for

        Returns:
            List of topic strings
        """
        registry = self._load_registry()
        name_lower = person_name.lower()

        for member in registry.get("people", {}).get("members", []):
            member_name = member.get("name", "").lower()

            if name_lower in member_name or member_name in name_lower:
                openalex = member.get("openalex", {})
                return openalex.get("topics", [])

        return []

    def collect_samples_for_author(
        self,
        author_name: str,
        max_samples: int = 50,
        include_titles: bool = True,
        include_abstracts: bool = True,
    ) -> List[TextSample]:
        """
        Collect text samples from an author's publications.

        Args:
            author_name: Author name to search
            max_samples: Maximum samples to collect
            include_titles: Include paper titles as samples
            include_abstracts: Include abstracts as samples

        Returns:
            List of TextSample objects
        """
        samples = []
        publications = self.get_person_publications(author_name)

        for pub in publications:
            if len(samples) >= max_samples:
                break

            # Title as sample
            if include_titles and pub.get("title"):
                samples.append(TextSample(
                    text=pub["title"],
                    source_type="paper_title",
                    source_id=f"paper:{pub.get('doi', pub['title'][:30])}",
                    timestamp=str(pub.get("year")) if pub.get("year") else None,
                    context=f"Paper title ({pub.get('year', 'unknown year')})",
                ))

            # Abstract as sample
            if include_abstracts and pub.get("abstract"):
                samples.append(TextSample(
                    text=pub["abstract"],
                    source_type="paper_abstract",
                    source_id=f"abstract:{pub.get('doi', pub['title'][:30])}",
                    timestamp=str(pub.get("year")) if pub.get("year") else None,
                    context=f"Paper abstract: {pub.get('title', 'Unknown')[:50]}",
                ))

        # Also check parsed publication data
        pub_chunks = self._get_publication_chunks(author_name)
        for chunk in pub_chunks:
            if len(samples) >= max_samples:
                break

            samples.append(TextSample(
                text=chunk.get("text", ""),
                source_type="paper_chunk",
                source_id=f"chunk:{chunk.get('metadata', {}).get('source', 'unknown')}",
                timestamp=None,
                context="Parsed publication text",
            ))

        return samples

    def _get_publication_chunks(self, author_name: str) -> List[Dict]:
        """Get parsed chunks related to an author's publications."""
        chunks_path = self.parsed_path / "chunks.json"

        if not chunks_path.exists():
            return []

        with open(chunks_path, "r") as f:
            all_chunks = json.load(f)

        # Look for chunks from publication files that mention the author
        name_lower = author_name.lower()
        matching = []

        for chunk in all_chunks:
            source = chunk.get("metadata", {}).get("source", "").lower()
            text = chunk.get("text", "").lower()

            # Check if it's from publications and mentions the author
            if "pub" in source and name_lower in text:
                matching.append(chunk)

        return matching

    def collect_by_topic(
        self,
        topic: str,
        max_samples: int = 30,
    ) -> List[TextSample]:
        """
        Collect paper samples related to a topic.

        Args:
            topic: Topic to search for
            max_samples: Maximum samples

        Returns:
            List of TextSample objects
        """
        samples = []
        registry = self._load_registry()
        topic_lower = topic.lower()

        # Search all members' publications
        for member in registry.get("people", {}).get("members", []):
            openalex = member.get("openalex", {})

            for pub in openalex.get("recent_publications", []):
                if len(samples) >= max_samples:
                    return samples

                # Check if topic matches
                title = pub.get("title", "").lower()
                pub_topics = [t.lower() for t in pub.get("topics", [])]

                if topic_lower in title or any(topic_lower in t for t in pub_topics):
                    if pub.get("abstract"):
                        samples.append(TextSample(
                            text=pub["abstract"],
                            source_type="paper_abstract",
                            source_id=f"abstract:{pub.get('doi', 'unknown')}",
                            timestamp=str(pub.get("year")) if pub.get("year") else None,
                            context=f"Topic: {topic}, Author: {member.get('name', 'Unknown')}",
                        ))

        return samples

    def get_coauthor_network(self, person_name: str) -> List[str]:
        """
        Get coauthors for a person (useful for finding related voices).

        Args:
            person_name: Person to find coauthors for

        Returns:
            List of coauthor names from registry
        """
        registry = self._load_registry()
        name_lower = person_name.lower()

        # Find the person's topics
        person_topics = set()
        for member in registry.get("people", {}).get("members", []):
            if name_lower in member.get("name", "").lower():
                person_topics = set(member.get("openalex", {}).get("topics", []))
                break

        if not person_topics:
            return []

        # Find others with overlapping topics
        coauthors = []
        for member in registry.get("people", {}).get("members", []):
            member_name = member.get("name", "")
            if name_lower in member_name.lower():
                continue  # Skip self

            member_topics = set(member.get("openalex", {}).get("topics", []))
            overlap = person_topics & member_topics

            if len(overlap) >= 2:  # At least 2 topics in common
                coauthors.append(member_name)

        return coauthors

    def list_authors_by_topic(self, topic: str) -> List[Dict]:
        """
        List authors who work on a topic.

        Args:
            topic: Topic to search for

        Returns:
            List of author info dicts
        """
        registry = self._load_registry()
        topic_lower = topic.lower()

        authors = []
        for member in registry.get("people", {}).get("members", []):
            openalex = member.get("openalex", {})
            member_topics = [t.lower() for t in openalex.get("topics", [])]

            if any(topic_lower in t for t in member_topics):
                authors.append({
                    "name": member.get("name"),
                    "role": member.get("role"),
                    "institution": member.get("institution"),
                    "topics": openalex.get("topics", []),
                    "works_count": openalex.get("works_count", 0),
                })

        # Sort by works count
        authors.sort(key=lambda x: x.get("works_count", 0), reverse=True)

        return authors
