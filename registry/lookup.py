"""
Registry Lookup Module

Provides fast structured lookup against lab_registry.json.

Handles queries like:
- "Who works on APTO?"
- "List all PhD students"
- "What datasets do we have?"
- "Who is Jake Burchard?"
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from .topic_aliases import expand_topic_aliases


class RegistryLookup:
    """
    Fast structured lookup against lab_registry.json.

    Handles queries like:
    - "Who works on APTO?"
    - "List all PhD students"
    - "What datasets do we have?"
    - "Who is Jake Burchard?"
    """

    def __init__(self, registry_path: str = "Data/lab_registry.json"):
        self.registry_path = Path(registry_path)
        self.registry: Dict[str, Any] = {}

        # Hot-reload tracking
        self._last_mtime: Optional[float] = None
        self._reload_check_interval = 10  # seconds
        self._last_reload_check = 0.0

        # Build search indexes
        self._name_index: Dict[str, Tuple[str, str]] = {}  # name -> (entity_type, entity_id)

        # Initial load
        self._load_and_index()

    def _load_and_index(self):
        """Load registry and rebuild indexes."""
        if self.registry_path.exists():
            with open(self.registry_path) as f:
                self.registry = json.load(f)
            self._last_mtime = self.registry_path.stat().st_mtime
            self._build_indexes()
            members = self.registry.get('people', {}).get('members', [])
            pub_count = sum(
                len(m.get("openalex", {}).get("recent_publications", []))
                for m in members
            )
            print(f"Loaded registry: {self.registry_path} ({len(members)} people, {pub_count} publications)")
        else:
            print(f"Warning: Registry not found at {self.registry_path}")
            self.registry = {}

    def reload_if_changed(self) -> bool:
        """
        Check if registry file changed and reload if needed.

        Returns True if registry was reloaded.
        """
        now = time.time()

        # Only check every N seconds
        if now - self._last_reload_check < self._reload_check_interval:
            return False
        self._last_reload_check = now

        if not self.registry_path.exists():
            return False

        current_mtime = self.registry_path.stat().st_mtime
        if current_mtime != self._last_mtime:
            old_count = len(self.registry.get('people', {}).get('members', []))
            self._load_and_index()
            new_count = len(self.registry.get('people', {}).get('members', []))
            print(f"Registry reloaded: {old_count} -> {new_count} people")
            return True

        return False

    def force_reload(self) -> Dict[str, Any]:
        """Force reload the registry. Returns reload status."""
        old_count = len(self.registry.get('people', {}).get('members', []))
        self._load_and_index()
        new_count = len(self.registry.get('people', {}).get('members', []))
        return {
            "status": "reloaded",
            "previous_members": old_count,
            "current_members": new_count,
            "timestamp": datetime.now().isoformat()
        }

    def load(self):
        """Load or reload the registry. Deprecated: use _load_and_index()."""
        self._load_and_index()

    def _build_indexes(self):
        """Build search indexes for fast lookup."""
        self._name_index = {}

        # Index people by name
        if "people" in self.registry and "members" in self.registry["people"]:
            for person in self.registry["people"]["members"]:
                name = person.get("name", "").lower()
                person_id = person.get("id", "")
                if name:
                    self._name_index[name] = ("people", person_id)
                    # Also index by last name
                    parts = name.split()
                    if len(parts) > 1:
                        self._name_index[parts[-1]] = ("people", person_id)

        # Index projects by name
        for project in self.registry.get("projects", []):
            name = project.get("name", "").lower()
            proj_id = project.get("id", "")
            if name:
                self._name_index[name] = ("projects", proj_id)
            if proj_id:
                self._name_index[proj_id] = ("projects", proj_id)

        # Index funding by name
        for funding in self.registry.get("funding", []):
            name = funding.get("name", "").lower()
            fund_id = funding.get("id", "")
            if name:
                self._name_index[name] = ("funding", fund_id)
            if fund_id:
                self._name_index[fund_id] = ("funding", fund_id)

    def find_person(self, query: str) -> Optional[Dict[str, Any]]:
        """Find a person by name (fuzzy match)."""
        query_lower = query.lower().strip()

        if "people" not in self.registry or "members" not in self.registry["people"]:
            return None

        for person in self.registry["people"]["members"]:
            name = person.get("name", "").lower()
            # Exact match
            if query_lower == name:
                return person
            # Partial match (first or last name)
            if query_lower in name or any(query_lower == part for part in name.split()):
                return person

        return None

    def find_people_by_role(self, role: str) -> List[Dict[str, Any]]:
        """Find all people with a specific role."""
        if "people" not in self.registry or "members" not in self.registry["people"]:
            return []

        role_lower = role.lower()
        matches = []
        for person in self.registry["people"]["members"]:
            person_role = person.get("role", "").lower()
            if role_lower in person_role:
                matches.append(person)
        return matches

    def find_people_by_institution(self, institution: str) -> List[Dict[str, Any]]:
        """Find all people at a specific institution."""
        if "people" not in self.registry or "members" not in self.registry["people"]:
            return []

        inst_lower = institution.lower()
        matches = []
        for person in self.registry["people"]["members"]:
            person_inst = person.get("institution")
            # Handle None, NaN, or non-string values
            if person_inst is None or (isinstance(person_inst, float) and person_inst != person_inst):
                continue
            if not isinstance(person_inst, str):
                continue
            if inst_lower in person_inst.lower():
                matches.append(person)
        return matches

    def find_people_by_topic(self, topic: str) -> List[Dict[str, Any]]:
        """
        Find all people who work on a specific research topic.

        Searches through members' openalex.topics arrays with fuzzy/partial matching.
        Also expands the search using topic aliases (e.g., "NLP" -> "natural language processing").

        Args:
            topic: The topic to search for (e.g., "network analysis", "machine learning", "NLP")

        Returns:
            List of dicts with person data and matched topics:
            [{"person": {...}, "matched_topics": ["Topic Name 1", "Topic Name 2"]}]
        """
        if "people" not in self.registry or "members" not in self.registry["people"]:
            return []

        topic_lower = topic.lower().strip()

        # Expand topic to include aliases (e.g., "nlp" -> {"nlp", "natural language processing", ...})
        expanded_topics = expand_topic_aliases(topic_lower)

        # Build word sets for each expanded topic for partial matching
        topic_word_sets = {t: set(t.split()) for t in expanded_topics}

        matches = []
        for person in self.registry["people"]["members"]:
            openalex = person.get("openalex", {})
            person_topics = openalex.get("topics", [])

            if not person_topics:
                continue

            matched_topics = []
            for t in person_topics:
                t_lower = t.lower()

                # Check against all expanded topic variants
                for search_term in expanded_topics:
                    # Exact substring match
                    if search_term in t_lower:
                        if t not in matched_topics:
                            matched_topics.append(t)
                        break

                    # Word-level partial match (all query words must appear in topic)
                    search_words = topic_word_sets.get(search_term, set())
                    if search_words and all(word in t_lower for word in search_words):
                        if t not in matched_topics:
                            matched_topics.append(t)
                        break

                    # Fuzzy match: check if most query words appear
                    if len(search_words) >= 2:
                        matching_words = sum(1 for word in search_words if word in t_lower)
                        if matching_words >= len(search_words) * 0.6:  # At least 60% of words match
                            if t not in matched_topics:
                                matched_topics.append(t)
                            break

            if matched_topics:
                matches.append({
                    "person": person,
                    "matched_topics": matched_topics
                })

        # Sort by number of matched topics (most relevant first)
        matches.sort(key=lambda x: len(x["matched_topics"]), reverse=True)
        return matches

    def get_all_topics(self) -> Dict[str, int]:
        """
        Get all unique topics from all members with their counts.

        Returns:
            Dict mapping topic name to count of people with that topic
        """
        if "people" not in self.registry or "members" not in self.registry["people"]:
            return {}

        topic_counts: Dict[str, int] = {}
        for person in self.registry["people"]["members"]:
            openalex = person.get("openalex", {})
            person_topics = openalex.get("topics", [])

            for topic in person_topics:
                topic_counts[topic] = topic_counts.get(topic, 0) + 1

        return topic_counts

    def find_publications(self, query: str) -> List[Dict[str, Any]]:
        """
        Search publications by title, venue, author name, or topic.

        Searches through all members' recent_publications from OpenAlex data
        in the registry.

        Args:
            query: Search term (can match title, venue, author names, or topics)

        Returns:
            List of publication dicts with added 'author_name' field:
            [{"title": ..., "year": ..., "venue": ..., "authors": [...], "author_name": "Person from registry", ...}]
        """
        members = self.registry.get('people', {}).get('members', [])
        if not members:
            return []

        query_lower = query.lower().strip()
        query_words = set(query_lower.split())
        matches = []

        for person in members:
            openalex = person.get("openalex", {})
            publications = openalex.get("recent_publications", [])
            person_name = person.get("name", "Unknown")
            person_name_lower = person_name.lower()

            # Check if query matches person's name (for "papers by X" queries)
            name_match = (
                query_lower in person_name_lower or
                any(query_lower == part for part in person_name_lower.split()) or
                all(word in person_name_lower for word in query_words if len(word) > 2)
            )

            for pub in publications:
                matched = False
                match_reason = ""

                # If query matches person name, include all their publications
                if name_match:
                    matched = True
                    match_reason = f"Author: {person_name}"
                else:
                    # Search in publication title
                    title = pub.get("title", "").lower()
                    if query_lower in title:
                        matched = True
                        match_reason = "Title match"
                    elif all(word in title for word in query_words if len(word) > 2):
                        matched = True
                        match_reason = "Title (partial)"

                    # Search in venue
                    if not matched:
                        venue = pub.get("venue") or ""
                        venue_lower = venue.lower()
                        if query_lower in venue_lower:
                            matched = True
                            match_reason = f"Venue: {venue}"

                    # Search in publication topics
                    if not matched:
                        pub_topics = pub.get("topics", [])
                        for topic in pub_topics:
                            topic_lower = topic.lower()
                            if query_lower in topic_lower or all(word in topic_lower for word in query_words):
                                matched = True
                                match_reason = f"Topic: {topic}"
                                break

                    # Search in co-authors (with fuzzy name matching)
                    if not matched:
                        authors = pub.get("authors", [])
                        for author in authors:
                            author_lower = author.lower()
                            # Exact substring match
                            if query_lower in author_lower:
                                matched = True
                                match_reason = f"Co-author: {author}"
                                break
                            # Word-based match (handles "James Evans" matching "James A. Evans")
                            author_words = set(author_lower.replace(".", "").split())
                            if query_words and all(qw in author_words for qw in query_words if len(qw) > 1):
                                matched = True
                                match_reason = f"Co-author: {author}"
                                break

                if matched:
                    # Create a copy with additional metadata
                    pub_result = dict(pub)
                    pub_result["registry_author"] = person_name
                    pub_result["match_reason"] = match_reason
                    matches.append(pub_result)

        # Sort by year (newest first), then by citation count
        matches.sort(key=lambda x: (-x.get("year", 0), -x.get("citation_count", 0)))

        # Remove duplicates (same DOI)
        seen_dois = set()
        unique_matches = []
        for pub in matches:
            doi = pub.get("doi", "")
            if doi and doi in seen_dois:
                continue
            if doi:
                seen_dois.add(doi)
            unique_matches.append(pub)

        return unique_matches

    def get_all_publications(self) -> List[Dict[str, Any]]:
        """
        Get all publications from all members.

        Returns:
            List of all publications with author info, sorted by year desc.
        """
        members = self.registry.get('people', {}).get('members', [])
        if not members:
            return []

        all_pubs = []
        seen_dois = set()

        for person in members:
            openalex = person.get("openalex", {})
            publications = openalex.get("recent_publications", [])
            person_name = person.get("name", "Unknown")

            for pub in publications:
                doi = pub.get("doi", "")
                if doi and doi in seen_dois:
                    continue
                if doi:
                    seen_dois.add(doi)

                pub_result = dict(pub)
                pub_result["registry_author"] = person_name
                all_pubs.append(pub_result)

        # Sort by year (newest first), then by citation count
        all_pubs.sort(key=lambda x: (-x.get("year", 0), -x.get("citation_count", 0)))
        return all_pubs

    def find_project(self, query: str) -> Optional[Dict[str, Any]]:
        """Find a project by name or ID."""
        query_lower = query.lower().strip()

        for project in self.registry.get("projects", []):
            if query_lower == project.get("id", "").lower():
                return project
            if query_lower in project.get("name", "").lower():
                return project

        return None

    def find_funding(self, query: str) -> Optional[Dict[str, Any]]:
        """Find a funding source by name or ID."""
        query_lower = query.lower().strip()

        for funding in self.registry.get("funding", []):
            if query_lower == funding.get("id", "").lower():
                return funding
            if query_lower in funding.get("name", "").lower():
                return funding

        return None

    def find_dataset(self, query: str) -> Optional[Dict[str, Any]]:
        """Find a dataset by ID, name, or keyword."""
        query_lower = query.lower().strip()
        datasets = self.registry.get("datasets", [])

        # First try exact ID match
        for ds in datasets:
            if query_lower == ds.get("id", "").lower():
                return ds

        # Try name match
        for ds in datasets:
            name = ds.get("name", "").lower()
            if query_lower in name or name in query_lower:
                return ds

        # Try keyword/tag match
        for ds in datasets:
            tags = ds.get("tags", [])
            for tag in tags:
                if query_lower in tag.lower() or tag.lower() in query_lower:
                    return ds

        return None

    def find_datasets_by_tag(self, tag: str) -> List[Dict[str, Any]]:
        """Find all datasets matching a tag."""
        tag_lower = tag.lower().strip()
        matches = []
        datasets = self.registry.get("datasets", [])

        for ds in datasets:
            tags = ds.get("tags", [])
            if any(tag_lower in t.lower() for t in tags):
                matches.append(ds)

        return matches

    def find_datasets_for_use(self, use_case: str) -> List[Dict[str, Any]]:
        """Find datasets suitable for a given use case."""
        use_lower = use_case.lower().strip()
        matches = []
        datasets = self.registry.get("datasets", [])

        for ds in datasets:
            intended_use = ds.get("intended_use", {})
            use_cases = intended_use.get("use_cases", [])
            primary_use = intended_use.get("primary_use", "")

            # Check if use case matches
            if use_lower in primary_use.lower():
                matches.append(ds)
                continue
            for uc in use_cases:
                if use_lower in uc.lower():
                    matches.append(ds)
                    break

        return matches

    def list_datasets(self) -> List[Dict[str, Any]]:
        """List all datasets."""
        return self.registry.get("datasets", [])

    def find_guide(self, query: str) -> Optional[Dict[str, Any]]:
        """Find a guide by ID, title, or keyword."""
        query_lower = query.lower().strip()
        guides = self.registry.get("guides", [])

        # First try exact ID match
        for guide in guides:
            if query_lower == guide.get("id", "").lower():
                return guide

        # Try title match
        for guide in guides:
            title = guide.get("title", "").lower()
            if query_lower in title or title in query_lower:
                return guide

        # Try keyword match
        for guide in guides:
            keywords = guide.get("keywords", [])
            for kw in keywords:
                if kw.lower() in query_lower or query_lower in kw.lower():
                    return guide

        return None

    def find_guides_by_keyword(self, keyword: str) -> List[Dict[str, Any]]:
        """Find all guides matching a keyword."""
        keyword_lower = keyword.lower().strip()
        matches = []
        guides = self.registry.get("guides", [])

        for guide in guides:
            # Check title
            if keyword_lower in guide.get("title", "").lower():
                matches.append(guide)
                continue
            # Check keywords
            for kw in guide.get("keywords", []):
                if keyword_lower in kw.lower() or kw.lower() in keyword_lower:
                    matches.append(guide)
                    break
            # Check content
            if keyword_lower in guide.get("content", "").lower():
                if guide not in matches:
                    matches.append(guide)

        return matches

    def list_guides(self) -> List[Dict[str, Any]]:
        """List all available guides."""
        return self.registry.get("guides", [])

    def find_guides_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Find guides by category (compute, data, software, access, general)."""
        category_lower = category.lower().strip()
        return [
            g for g in self.registry.get("guides", [])
            if g.get("category", "").lower() == category_lower
        ]

    def list_by_type(self, entity_type: str) -> List[Dict[str, Any]]:
        """List all entities of a given type."""
        if entity_type == "people":
            return self.registry.get("people", {}).get("members", [])
        return self.registry.get(entity_type, [])

    def get_stats(self) -> Dict[str, int]:
        """Get counts of each entity type."""
        stats = {}
        for key in ["projects", "funding", "datasets", "code_repos", "compute", "events", "tools", "decisions", "guides"]:
            items = self.registry.get(key, [])
            stats[key] = len(items) if isinstance(items, list) else 0

        if "people" in self.registry and "members" in self.registry["people"]:
            stats["people"] = len(self.registry["people"]["members"])

        return stats


class MidwayLookup:
    """
    Fast structured lookup against midway_registry.json.

    Handles queries like:
    - "Where is the OpenAlex data?"
    - "What precomputed resources exist?"
    - "Who has Reddit data on Midway?"
    - "What embeddings are available?"
    """

    def __init__(self, registry_path: str = "Data/midway_registry.json"):
        self.registry_path = Path(registry_path)
        self.registry: Dict[str, Any] = {}

        # Hot-reload tracking
        self._last_mtime: Optional[float] = None
        self._reload_check_interval = 10  # seconds
        self._last_reload_check = 0.0

        # Initial load
        self._load_registry()

    def _load_registry(self):
        """Load the Midway registry."""
        if self.registry_path.exists():
            with open(self.registry_path) as f:
                self.registry = json.load(f)
            self._last_mtime = self.registry_path.stat().st_mtime
            snapshot_count = len(self.registry.get("data_snapshots", []))
            precomputed_count = len(self.registry.get("precomputed_resources", []))
            print(f"Loaded Midway registry: {self.registry_path} ({snapshot_count} snapshots, {precomputed_count} precomputed)")
        else:
            print(f"Warning: Midway registry not found at {self.registry_path}")
            self.registry = {}

    def reload_if_changed(self) -> bool:
        """Check if registry file changed and reload if needed."""
        now = time.time()
        if now - self._last_reload_check < self._reload_check_interval:
            return False
        self._last_reload_check = now

        if not self.registry_path.exists():
            return False

        current_mtime = self.registry_path.stat().st_mtime
        if current_mtime != self._last_mtime:
            self._load_registry()
            return True
        return False

    def _get_items(self, key: str) -> List[Dict[str, Any]]:
        """Helper to get items from nested or flat structure."""
        data = self.registry.get(key, {})
        if isinstance(data, dict):
            return data.get("items", [])
        return data if isinstance(data, list) else []

    def find_snapshot(self, query: str) -> Optional[Dict[str, Any]]:
        """Find a data snapshot by name, ID, or keyword."""
        query_lower = query.lower().strip()

        for snapshot in self._get_items("data_snapshots"):
            # Match by ID
            if query_lower == snapshot.get("id", "").lower():
                return snapshot
            # Match by name
            if query_lower in snapshot.get("name", "").lower():
                return snapshot
            # Match by description
            if query_lower in snapshot.get("description", "").lower():
                return snapshot

        return None

    def find_snapshots_by_keyword(self, keyword: str) -> List[Dict[str, Any]]:
        """Find all snapshots matching a keyword."""
        keyword_lower = keyword.lower().strip()
        matches = []

        for snapshot in self._get_items("data_snapshots"):
            if (keyword_lower in snapshot.get("name", "").lower() or
                keyword_lower in snapshot.get("id", "").lower() or
                keyword_lower in snapshot.get("description", "").lower() or
                keyword_lower in str(snapshot.get("entity_types", [])).lower()):
                matches.append(snapshot)

        return matches

    def list_snapshots(self) -> List[Dict[str, Any]]:
        """List all data snapshots."""
        return self._get_items("data_snapshots")

    def find_precomputed(self, query: str) -> Optional[Dict[str, Any]]:
        """Find a precomputed resource by name, ID, or keyword."""
        query_lower = query.lower().strip()

        for resource in self._get_items("precomputed_resources"):
            if query_lower == resource.get("id", "").lower():
                return resource
            if query_lower in resource.get("name", "").lower():
                return resource
            if query_lower in resource.get("description", "").lower():
                return resource

        return None

    def list_precomputed(self) -> List[Dict[str, Any]]:
        """List all precomputed resources."""
        return self._get_items("precomputed_resources")

    def find_embedding(self, query: str) -> Optional[Dict[str, Any]]:
        """Find an embedding resource by name, ID, or domain."""
        query_lower = query.lower().strip()

        for embedding in self._get_items("embeddings"):
            if query_lower == embedding.get("id", "").lower():
                return embedding
            if query_lower in embedding.get("name", "").lower():
                return embedding
            if query_lower in embedding.get("domain", "").lower():
                return embedding

        return None

    def list_embeddings(self) -> List[Dict[str, Any]]:
        """List all embedding resources."""
        return self._get_items("embeddings")

    def find_social_dataset(self, query: str) -> Optional[Dict[str, Any]]:
        """Find a social media dataset by name, platform, or keyword."""
        query_lower = query.lower().strip()

        for dataset in self._get_items("social_media_datasets"):
            if query_lower == dataset.get("id", "").lower():
                return dataset
            if query_lower in dataset.get("name", "").lower():
                return dataset
            # Handle platform as string or list
            plat = dataset.get("platform", "")
            if isinstance(plat, list):
                if any(query_lower in p.lower() for p in plat):
                    return dataset
            elif isinstance(plat, str) and query_lower in plat.lower():
                return dataset
            if query_lower in dataset.get("content_description", "").lower():
                return dataset

        return None

    def find_social_by_platform(self, platform: str) -> List[Dict[str, Any]]:
        """Find all social media datasets for a specific platform."""
        platform_lower = platform.lower().strip()
        matches = []

        for dataset in self._get_items("social_media_datasets"):
            plat = dataset.get("platform", "")
            # Handle both string and list platform values
            if isinstance(plat, list):
                if any(platform_lower in p.lower() for p in plat):
                    matches.append(dataset)
            elif isinstance(plat, str) and platform_lower in plat.lower():
                matches.append(dataset)

        return matches

    def list_social_datasets(self) -> List[Dict[str, Any]]:
        """List all social media datasets."""
        return self._get_items("social_media_datasets")

    def find_researcher_directory(self, query: str) -> Optional[Dict[str, Any]]:
        """Find a researcher's directory by username or name."""
        query_lower = query.lower().strip()

        for researcher in self._get_items("researcher_directories"):
            if query_lower == researcher.get("username", "").lower():
                return researcher
            if query_lower in researcher.get("username", "").lower():
                return researcher
            # Also check directory path
            if query_lower in researcher.get("directory_path", "").lower():
                return researcher

        return None

    def find_researchers_with_data(self, data_type: str) -> List[Dict[str, Any]]:
        """Find researchers who have specific types of data."""
        data_type_lower = data_type.lower().strip()
        matches = []

        for researcher in self._get_items("researcher_directories"):
            # Check in primary focus
            if data_type_lower in researcher.get("primary_focus", "").lower():
                matches.append(researcher)
                continue
            # Check in active projects
            projects = researcher.get("active_projects", [])
            if any(data_type_lower in proj.lower() for proj in projects):
                matches.append(researcher)
                continue
            # Check in assessment
            if data_type_lower in researcher.get("assessment", "").lower():
                matches.append(researcher)

        return matches

    def list_researcher_directories(self) -> List[Dict[str, Any]]:
        """List all researcher directories."""
        return self._get_items("researcher_directories")

    def find_infrastructure(self, query: str) -> Optional[Dict[str, Any]]:
        """Find shared infrastructure by name or type."""
        query_lower = query.lower().strip()

        for infra in self._get_items("shared_infrastructure"):
            if query_lower == infra.get("id", "").lower():
                return infra
            if query_lower in infra.get("name", "").lower():
                return infra
            if query_lower in infra.get("type", "").lower():
                return infra

        return None

    def list_infrastructure(self) -> List[Dict[str, Any]]:
        """List all shared infrastructure."""
        return self._get_items("shared_infrastructure")

    def get_stats(self) -> Dict[str, int]:
        """Get counts of each Midway entity type."""
        return {
            "data_snapshots": len(self._get_items("data_snapshots")),
            "precomputed_resources": len(self._get_items("precomputed_resources")),
            "embeddings": len(self._get_items("embeddings")),
            "social_media_datasets": len(self._get_items("social_media_datasets")),
            "researcher_directories": len(self._get_items("researcher_directories")),
            "shared_infrastructure": len(self._get_items("shared_infrastructure")),
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get summary information from the registry."""
        return self.registry.get("summary", {})
