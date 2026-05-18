"""
Cross-Registry Linking

Links lab_registry.json and midway_registry.json to enable queries like:
- "What data does Jake use?"
- "Who works with OpenAlex?"
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from difflib import SequenceMatcher


class CrossRegistryLinker:
    """
    Creates and queries links between lab registry people and Midway resources.
    """

    def __init__(
        self,
        lab_registry_path: str = "Data/lab_registry.json",
        midway_registry_path: str = "Data/midway_registry.json"
    ):
        self.lab_path = Path(lab_registry_path)
        self.midway_path = Path(midway_registry_path)

        self.lab_registry: Dict[str, Any] = {}
        self.midway_registry: Dict[str, Any] = {}

        # Computed links
        self.person_to_resources: Dict[str, List[Dict]] = {}
        self.resource_to_people: Dict[str, List[Dict]] = {}
        self.topic_to_resources: Dict[str, List[Dict]] = {}

        self._load_registries()
        self._build_links()

    def _load_registries(self):
        """Load both registry files."""
        if self.lab_path.exists():
            with open(self.lab_path) as f:
                self.lab_registry = json.load(f)

        if self.midway_path.exists():
            with open(self.midway_path) as f:
                self.midway_registry = json.load(f)

    def _get_midway_items(self, category: str) -> List[Dict]:
        """Get items from a midway registry category."""
        data = self.midway_registry.get(category, {})
        if isinstance(data, dict) and "items" in data:
            return data["items"]
        elif isinstance(data, list):
            return data
        return []

    def _all_midway_resources(self) -> List[Dict]:
        """Get all Midway resources with category labels."""
        resources = []
        categories = [
            ("data_snapshots", "snapshot"),
            ("precomputed_resources", "precomputed"),
            ("embeddings", "embedding"),
            ("social_media_datasets", "social"),
            ("researcher_directories", "researcher"),
            ("shared_infrastructure", "infrastructure"),
        ]
        for cat_key, cat_label in categories:
            for item in self._get_midway_items(cat_key):
                item_copy = dict(item)
                item_copy["_category"] = cat_label
                resources.append(item_copy)
        return resources

    def _normalize_name(self, name: str) -> str:
        """Normalize a name for matching."""
        return name.lower().strip().replace("-", " ").replace("_", " ")

    def _extract_username(self, path: str) -> Optional[str]:
        """Extract username from a Midway path like /project/jevans/username/."""
        if not path:
            return None
        match = re.search(r'/project/jevans/([^/]+)', path)
        if match:
            return match.group(1).lower()
        return None

    def _name_similarity(self, name1: str, name2: str) -> float:
        """Calculate similarity between two names."""
        n1 = self._normalize_name(name1)
        n2 = self._normalize_name(name2)
        return SequenceMatcher(None, n1, n2).ratio()

    def _build_links(self):
        """Build all cross-registry links."""
        people = self.lab_registry.get("people", {}).get("members", [])
        all_resources = self._all_midway_resources()

        # Build person ID to name mapping
        person_names = {}
        for person in people:
            pid = person.get("id", "")
            name = person.get("name", "")
            if pid:
                person_names[pid] = name
                # Also map by username if they have midway info
                midway_info = person.get("midway", {})
                if midway_info.get("username"):
                    person_names[midway_info["username"].lower()] = name

        # Link people to resources
        for person in people:
            pid = person.get("id", "")
            name = person.get("name", "")
            if not pid:
                continue

            linked_resources = []

            # 1. Link via Midway username
            midway_info = person.get("midway", {})
            username = midway_info.get("username", "").lower()

            for resource in all_resources:
                score = 0
                reasons = []

                # Check path ownership
                path = resource.get("path", "")
                path_username = self._extract_username(path)

                if username and path_username == username:
                    score += 1.0
                    reasons.append("owns_directory")

                # Check owner/created_by fields
                owner = resource.get("owner", "").lower()
                created_by = resource.get("created_by", "").lower()

                if username and (owner == username or created_by == username):
                    score += 0.9
                    reasons.append("listed_owner")

                # Check name similarity
                if owner and self._name_similarity(name, owner) > 0.8:
                    score += 0.7
                    reasons.append("name_match")

                # Check if person's research topics match resource
                person_topics = person.get("openalex", {}).get("topics", [])
                resource_desc = " ".join([
                    resource.get("name", ""),
                    resource.get("description", ""),
                    " ".join(resource.get("entity_types", []) if isinstance(resource.get("entity_types"), list) else [])
                ]).lower()

                for topic in person_topics:
                    if topic.lower() in resource_desc:
                        score += 0.3
                        reasons.append(f"topic_match:{topic}")
                        break

                if score > 0:
                    linked_resources.append({
                        "resource": resource,
                        "score": score,
                        "reasons": reasons
                    })

            # Sort by score
            linked_resources.sort(key=lambda x: x["score"], reverse=True)
            self.person_to_resources[pid] = linked_resources

        # Build reverse mapping: resource to people
        for pid, resources in self.person_to_resources.items():
            for link in resources:
                resource_id = link["resource"].get("id", link["resource"].get("username", ""))
                if resource_id:
                    if resource_id not in self.resource_to_people:
                        self.resource_to_people[resource_id] = []
                    self.resource_to_people[resource_id].append({
                        "person_id": pid,
                        "person_name": person_names.get(pid, pid),
                        "score": link["score"],
                        "reasons": link["reasons"]
                    })

        # Build topic to resources mapping
        self._build_topic_links(all_resources)

    def _build_topic_links(self, resources: List[Dict]):
        """Map research topics to relevant resources."""
        # Common research topics and their resource keywords
        topic_keywords = {
            "bibliometrics": ["openalex", "semantic scholar", "mag", "wos", "citation", "publication", "author"],
            "natural language processing": ["embedding", "word2vec", "specter", "nlp", "text", "language"],
            "machine learning": ["embedding", "model", "neural", "deep learning", "ai"],
            "network science": ["network", "graph", "citation", "collaboration"],
            "social media": ["reddit", "twitter", "4chan", "weibo", "telegram", "social"],
            "patents": ["patent", "patstat", "uspto", "invention", "innovation"],
            "biomedical": ["pubmed", "mesh", "medical", "health", "clinical"],
            "computational social science": ["social", "behavior", "survey", "experiment"],
        }

        for topic, keywords in topic_keywords.items():
            matched_resources = []
            for resource in resources:
                searchable = " ".join([
                    resource.get("name", ""),
                    resource.get("description", ""),
                    resource.get("path", ""),
                ]).lower()

                matches = [kw for kw in keywords if kw in searchable]
                if matches:
                    matched_resources.append({
                        "resource": resource,
                        "matched_keywords": matches,
                        "score": len(matches) / len(keywords)
                    })

            matched_resources.sort(key=lambda x: x["score"], reverse=True)
            self.topic_to_resources[topic] = matched_resources[:10]

    def get_person_resources(self, person_id: str, min_score: float = 0.0) -> List[Dict]:
        """Get Midway resources linked to a person."""
        links = self.person_to_resources.get(person_id, [])
        return [
            {
                "id": l["resource"].get("id", l["resource"].get("username")),
                "name": l["resource"].get("name", l["resource"].get("username")),
                "category": l["resource"].get("_category"),
                "path": l["resource"].get("path"),
                "score": l["score"],
                "reasons": l["reasons"]
            }
            for l in links if l["score"] >= min_score
        ]

    def get_resource_users(self, resource_id: str) -> List[Dict]:
        """Get people who use or own a resource."""
        return self.resource_to_people.get(resource_id, [])

    def get_topic_resources(self, topic: str) -> List[Dict]:
        """Get Midway resources relevant to a research topic."""
        topic_lower = topic.lower()

        # Try exact match first
        if topic_lower in self.topic_to_resources:
            return self.topic_to_resources[topic_lower]

        # Try partial match
        for t, resources in self.topic_to_resources.items():
            if topic_lower in t or t in topic_lower:
                return resources

        # Fallback: search all resources
        all_resources = self._all_midway_resources()
        matches = []
        for resource in all_resources:
            searchable = " ".join([
                resource.get("name", ""),
                resource.get("description", ""),
            ]).lower()
            if topic_lower in searchable:
                matches.append({
                    "resource": resource,
                    "score": 0.5,
                    "matched_keywords": [topic]
                })
        return matches[:10]

    def find_person_by_name(self, name: str) -> Optional[str]:
        """Find a person ID by name or username."""
        name_lower = name.lower().strip()
        people = self.lab_registry.get("people", {}).get("members", [])

        for person in people:
            pid = person.get("id", "")
            pname = person.get("name", "").lower()

            # Exact match on ID or name
            if name_lower == pid or name_lower == pname:
                return pid

            # Partial name match
            if name_lower in pname or pname in name_lower:
                return pid

            # Match on username
            username = person.get("midway", {}).get("username", "").lower()
            if username and name_lower == username:
                return pid

            # Match on first or last name
            name_parts = pname.split()
            if name_lower in name_parts:
                return pid

        return None

    def query_person_resources(self, query: str) -> Dict[str, Any]:
        """
        Handle queries like "What data does Jake use?"
        Returns person info and their linked resources.
        """
        # Extract person name from query
        patterns = [
            r"what\s+(?:data|resources?)\s+does\s+(\w+)\s+(?:have|use|work)",
            r"(\w+)(?:'s)?\s+(?:data|resources?|midway)",
            r"show\s+(\w+)(?:'s)?\s+resources?",
        ]

        person_name = None
        for pattern in patterns:
            match = re.search(pattern, query.lower())
            if match:
                person_name = match.group(1)
                break

        if not person_name:
            return {"error": "Could not identify person in query"}

        person_id = self.find_person_by_name(person_name)
        if not person_id:
            return {"error": f"Person '{person_name}' not found"}

        resources = self.get_person_resources(person_id, min_score=0.3)

        # Get person details
        people = self.lab_registry.get("people", {}).get("members", [])
        person_info = next((p for p in people if p.get("id") == person_id), {})

        return {
            "person": {
                "id": person_id,
                "name": person_info.get("name"),
                "role": person_info.get("role"),
            },
            "resources": resources,
            "count": len(resources)
        }


# Module-level singleton
_linker: Optional[CrossRegistryLinker] = None


def get_linker() -> CrossRegistryLinker:
    """Get or create the cross-registry linker."""
    global _linker
    if _linker is None:
        _linker = CrossRegistryLinker()
    return _linker
