"""
Query Suggestions

Provides helpful suggestions when queries return no or few results:
- Typo correction
- Alternative query suggestions
- Related terms
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from difflib import SequenceMatcher, get_close_matches


class QuerySuggester:
    """
    Generates query suggestions and typo corrections.
    """

    # Common English words that should NEVER be corrected
    # Imported from TypoCorrector to maintain consistency
    STOPWORDS = {
        # Common verbs
        "have", "has", "had", "are", "is", "was", "were", "be", "been", "being",
        "want", "wants", "wanted", "work", "works", "worked", "working",
        "like", "likes", "liked", "plan", "plans", "planned", "planning",
        "make", "makes", "made", "making", "take", "takes", "took", "taking",
        "give", "gives", "gave", "giving", "find", "finds", "found", "finding",
        "use", "uses", "used", "using", "tell", "tells", "told", "telling",
        "get", "gets", "got", "getting", "come", "comes", "came", "coming",
        "know", "knows", "knew", "knowing", "think", "thinks", "thought",
        "see", "sees", "saw", "seeing", "look", "looks", "looked", "looking",
        # Common prepositions/articles/conjunctions
        "the", "and", "for", "with", "from", "that", "this", "which", "when",
        "where", "what", "who", "how", "why", "all", "can", "will", "would",
        "could", "should", "may", "might", "must", "shall", "into", "about",
        # Common adjectives/adverbs
        "new", "old", "most", "more", "some", "any", "each", "many", "much",
        "other", "first", "last", "same", "different", "recent", "latest",
        # Question words
        "does", "did", "do", "there", "here", "then", "now", "just", "also",
        # Common nouns
        "name", "names", "time", "times", "year", "years", "day", "days",
        "data", "both", "nlp", "help", "need", "show", "list", "tell",
    }

    def __init__(
        self,
        lab_registry_path: str = "Data/lab_registry.json",
        midway_registry_path: str = "Data/midway_registry.json"
    ):
        self.lab_path = Path(lab_registry_path)
        self.midway_path = Path(midway_registry_path)

        # Vocabulary for suggestions
        self.person_names: List[str] = []
        self.person_ids: List[str] = []
        self.dataset_names: List[str] = []
        self.resource_names: List[str] = []
        self.topics: List[str] = []
        self.all_terms: List[str] = []

        # Category suggestions
        self.category_queries: Dict[str, List[str]] = {
            "people": [
                "Who works on [topic]?",
                "List all researchers",
                "Tell me about [name]",
            ],
            "datasets": [
                "What datasets are available?",
                "Where is the [name] data?",
                "List all datasets",
            ],
            "midway_snapshots": [
                "What data snapshots are on Midway?",
                "Where is the OpenAlex snapshot?",
                "What bibliometric databases exist?",
            ],
            "midway_embeddings": [
                "What embeddings are available?",
                "Where are the word2vec embeddings?",
                "List embedding resources",
            ],
            "midway_social": [
                "What social media datasets exist?",
                "Where is the Reddit data?",
                "Do we have Twitter data?",
            ],
            "midway_precomputed": [
                "What precomputed resources exist?",
                "Where are the similarity matrices?",
                "List precomputed data",
            ],
            "compute": [
                "What GPUs are available?",
                "How do I use Midway?",
                "What compute resources exist?",
            ],
        }

        self._build_vocabulary()

    def _build_vocabulary(self):
        """Build vocabulary from registries."""
        # Load lab registry
        if self.lab_path.exists():
            with open(self.lab_path) as f:
                lab = json.load(f)

            # Person names and IDs
            for person in lab.get("people", {}).get("members", []):
                name = person.get("name", "")
                pid = person.get("id", "")
                if name:
                    self.person_names.append(name.lower())
                    self.all_terms.append(name.lower())
                if pid:
                    self.person_ids.append(pid.lower())
                    self.all_terms.append(pid.lower())

                # Topics
                for topic in person.get("openalex", {}).get("topics", []):
                    topic_lower = topic.lower()
                    if topic_lower not in self.topics:
                        self.topics.append(topic_lower)
                        self.all_terms.append(topic_lower)

            # Datasets
            for dataset in lab.get("datasets", []):
                name = dataset.get("name", "")
                if name:
                    self.dataset_names.append(name.lower())
                    self.all_terms.append(name.lower())

        # Load midway registry
        if self.midway_path.exists():
            with open(self.midway_path) as f:
                midway = json.load(f)

            # Extract all resource names
            categories = [
                "data_snapshots", "precomputed_resources", "embeddings",
                "social_media_datasets", "researcher_directories", "shared_infrastructure"
            ]

            for cat in categories:
                data = midway.get(cat, {})
                items = data.get("items", []) if isinstance(data, dict) else data
                for item in items:
                    name = item.get("name", item.get("username", ""))
                    if name:
                        self.resource_names.append(name.lower())
                        self.all_terms.append(name.lower())

        # Add common data names
        common_names = [
            "openalex", "semantic scholar", "mag", "dblp", "pubmed", "patstat",
            "uspto", "wos", "web of science", "reddit", "twitter", "weibo",
            "4chan", "word2vec", "specter", "mat2vec", "hyperbolic",
            "embeddings", "snapshots", "precomputed", "social media"
        ]
        for name in common_names:
            if name not in self.all_terms:
                self.all_terms.append(name)

    def find_typos(self, query: str, threshold: float = 0.85) -> List[Tuple[str, str]]:
        """
        Find potential typos in the query.
        Returns list of (typo, correction) tuples.

        NOTE: Threshold raised from 0.7 to 0.85 to prevent over-correction
        of common English words to similar-looking registry terms.
        """
        corrections = []
        words = query.lower().split()

        for word in words:
            if len(word) < 3:
                continue

            # Skip common English words - use comprehensive stopword list
            if word in self.STOPWORDS:
                continue

            # Find close matches
            matches = get_close_matches(word, self.all_terms, n=1, cutoff=threshold)
            if matches and matches[0] != word:
                # Make sure it's actually a typo (not just similar)
                similarity = SequenceMatcher(None, word, matches[0]).ratio()
                if threshold <= similarity < 1.0:
                    corrections.append((word, matches[0]))

        return corrections

    def suggest_alternatives(self, query: str, classification: Dict) -> List[str]:
        """
        Suggest alternative queries based on the original query.
        """
        suggestions = []
        query_lower = query.lower()

        # Get entity types from classification
        entity_types = classification.get("entity_types", [])

        # Suggest broader category queries
        for entity_type in entity_types:
            if entity_type in self.category_queries:
                suggestions.extend(self.category_queries[entity_type][:2])

        # If query mentions specific terms, suggest related queries
        if "embedding" in query_lower:
            suggestions.append("What embeddings are on Midway?")
            suggestions.append("Where are the word2vec embeddings?")
        elif "data" in query_lower or "dataset" in query_lower:
            suggestions.append("What datasets are available?")
            suggestions.append("What data snapshots are on Midway?")
        elif "who" in query_lower or "person" in query_lower:
            suggestions.append("List all researchers")
            suggestions.append("Who works on [topic]?")

        # Suggest based on extracted terms
        words = set(query_lower.split())
        data_terms = {"openalex", "semantic", "scholar", "patent", "reddit", "twitter", "pubmed"}
        if words & data_terms:
            suggestions.append("What data snapshots are on Midway?")

        # Remove duplicates and limit
        seen = set()
        unique_suggestions = []
        for s in suggestions:
            if s not in seen:
                seen.add(s)
                unique_suggestions.append(s)

        return unique_suggestions[:5]

    def get_suggestions(
        self,
        query: str,
        classification: Dict,
        num_results: int = 0
    ) -> Dict[str, Any]:
        """
        Get all suggestions for a query.

        Args:
            query: The original query
            classification: Query classification result
            num_results: Number of results found (0 = no results)

        Returns:
            Dict with typo corrections and alternative suggestions
        """
        result = {
            "has_suggestions": False,
            "typo_corrections": [],
            "did_you_mean": None,
            "alternative_queries": [],
            "message": None
        }

        # Check for typos
        typos = self.find_typos(query)
        if typos:
            result["typo_corrections"] = typos
            result["has_suggestions"] = True

            # Build "did you mean" suggestion
            corrected_query = query
            for typo, correction in typos:
                corrected_query = corrected_query.replace(typo, correction)
            if corrected_query != query:
                result["did_you_mean"] = corrected_query

        # Get alternative queries if no results
        if num_results == 0:
            alternatives = self.suggest_alternatives(query, classification)
            if alternatives:
                result["alternative_queries"] = alternatives
                result["has_suggestions"] = True

        # Build user-friendly message
        if result["has_suggestions"]:
            messages = []
            if result["did_you_mean"]:
                messages.append(f"Did you mean: \"{result['did_you_mean']}\"?")
            if result["alternative_queries"]:
                messages.append("Try these queries:")
                for alt in result["alternative_queries"][:3]:
                    messages.append(f"  - {alt}")
            result["message"] = "\n".join(messages)

        return result

    def format_no_results_message(
        self,
        query: str,
        classification: Dict
    ) -> str:
        """
        Format a helpful message when no results are found.
        """
        suggestions = self.get_suggestions(query, classification, num_results=0)

        lines = [f"No results found for: {query}"]

        if suggestions["did_you_mean"]:
            lines.append(f"\nDid you mean: \"{suggestions['did_you_mean']}\"?")

        if suggestions["alternative_queries"]:
            lines.append("\nTry one of these queries:")
            for alt in suggestions["alternative_queries"][:3]:
                lines.append(f"  - {alt}")

        return "\n".join(lines)


# Module-level singleton
_suggester: Optional[QuerySuggester] = None


def get_suggester() -> QuerySuggester:
    """Get or create the query suggester."""
    global _suggester
    if _suggester is None:
        _suggester = QuerySuggester()
    return _suggester
