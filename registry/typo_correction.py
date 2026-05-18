"""
Typo Correction Module

Provides fuzzy matching and typo correction using edit distance
against a vocabulary built from registry data.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from functools import lru_cache


def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def similarity_ratio(s1: str, s2: str) -> float:
    """Calculate similarity ratio (0-1) between two strings."""
    if not s1 or not s2:
        return 0.0
    distance = levenshtein_distance(s1.lower(), s2.lower())
    max_len = max(len(s1), len(s2))
    return 1 - (distance / max_len)


class TypoCorrector:
    """
    Corrects typos in queries using a vocabulary built from registry data.
    """

    # Common English words that should NEVER be corrected
    # These were being incorrectly changed (e.g., "have" → "hate", "are" → "care")
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
        # Common nouns that shouldn't be corrected to names
        "name", "names", "time", "times", "year", "years", "day", "days",
    }

    def __init__(
        self,
        registry_path: str = "Data/lab_registry.json",
        midway_path: str = "Data/midway_registry.json",
        jevans_path: str = "Data/jevans_resources.json",
        min_word_length: int = 3,
        correction_threshold: float = 0.85,  # Raised from 0.70 to prevent over-correction
    ):
        self.min_word_length = min_word_length
        self.correction_threshold = correction_threshold
        self.vocabulary: Set[str] = set()
        self.word_to_canonical: Dict[str, str] = {}  # lowercase -> preferred form

        self._build_vocabulary(registry_path, midway_path, jevans_path)

    def _build_vocabulary(self, registry_path: str, midway_path: str, jevans_path: str = ""):
        """Build vocabulary from registry data."""
        # Common search terms
        common_terms = {
            # Roles
            "researcher", "researchers", "student", "students", "phd",
            "postdoc", "postdocs", "faculty", "professor", "staff",
            "member", "members", "scientist", "scientists",
            # Actions
            "working", "studying", "researching", "published", "collaborating",
            # Data terms
            "dataset", "datasets", "data", "database", "snapshot", "embedding",
            "embeddings", "precomputed", "matrix", "matrices",
            # Platforms
            "reddit", "twitter", "weibo", "telegram", "4chan",
            # Topics
            "machine", "learning", "network", "networks", "science",
            "social", "computational", "artificial", "intelligence",
            "natural", "language", "processing", "neural", "deep",
        }

        for term in common_terms:
            self._add_word(term)

        # Load lab registry
        lab_path = Path(registry_path)
        if lab_path.exists():
            with open(lab_path) as f:
                registry = json.load(f)

            # Add person names
            for person in registry.get("people", {}).get("members", []):
                name = person.get("name", "")
                for word in name.split():
                    self._add_word(word, canonical=word)

                # Add institution
                inst = person.get("institution", "")
                if isinstance(inst, str):
                    for word in inst.split():
                        if len(word) >= self.min_word_length:
                            self._add_word(word)

                # Add topics
                topics = person.get("openalex", {}).get("topics", [])
                for topic in topics:
                    if isinstance(topic, str):
                        for word in topic.lower().split():
                            if len(word) >= self.min_word_length:
                                self._add_word(word)

            # Add project names
            for project in registry.get("projects", []):
                name = project.get("name", "")
                for word in name.split():
                    self._add_word(word)

            # Add dataset names
            for dataset in registry.get("datasets", []):
                name = dataset.get("name", "")
                for word in name.split():
                    if len(word) >= self.min_word_length:
                        self._add_word(word)

        # Load midway registry
        midway_path_obj = Path(midway_path)
        if midway_path_obj.exists():
            with open(midway_path_obj) as f:
                midway = json.load(f)

            # Add from all categories
            for category in ["data_snapshots", "embeddings", "precomputed_resources",
                           "social_media_datasets", "researcher_directories"]:
                items = midway.get(category, {})
                if isinstance(items, dict) and "items" in items:
                    items = items["items"]
                elif not isinstance(items, list):
                    items = []

                for item in items:
                    name = item.get("name", "")
                    for word in name.split():
                        if len(word) >= self.min_word_length:
                            self._add_word(word)

        # Load jevans resources
        jevans_path_obj = Path(jevans_path) if jevans_path else None
        if jevans_path_obj and jevans_path_obj.exists():
            with open(jevans_path_obj) as f:
                jevans = json.load(f)

            for resource in jevans.get("resources", []):
                # Add resource names
                name = resource.get("name", "")
                if len(name) >= self.min_word_length:
                    self._add_word(name)

                # Add owner usernames (important for queries like "robbie's data")
                owner = resource.get("owner_username", "")
                if owner and len(owner) >= self.min_word_length:
                    self._add_word(owner, canonical=owner)

        print(f"Typo corrector built vocabulary: {len(self.vocabulary)} words")

    def _add_word(self, word: str, canonical: str = None):
        """Add a word to the vocabulary."""
        word_lower = word.lower().strip()
        if len(word_lower) >= self.min_word_length:
            self.vocabulary.add(word_lower)
            if canonical:
                self.word_to_canonical[word_lower] = canonical

    @lru_cache(maxsize=1000)
    def find_closest(self, word: str, max_distance: int = 2) -> Optional[Tuple[str, float]]:
        """
        Find the closest vocabulary word to the given word.

        Returns (closest_word, similarity) or None if no close match.
        """
        word_lower = word.lower()

        # Exact match
        if word_lower in self.vocabulary:
            return (word_lower, 1.0)

        # Find closest by edit distance
        best_match = None
        best_similarity = 0.0

        for vocab_word in self.vocabulary:
            # Quick length check to skip obviously different words
            if abs(len(vocab_word) - len(word_lower)) > max_distance:
                continue

            sim = similarity_ratio(word_lower, vocab_word)
            if sim > best_similarity and sim >= self.correction_threshold:
                best_similarity = sim
                best_match = vocab_word

        if best_match:
            return (best_match, best_similarity)
        return None

    def correct_query(self, query: str) -> Tuple[str, List[Dict]]:
        """
        Correct typos in a query.

        Returns:
            Tuple of (corrected_query, list of corrections made)
        """
        words = query.split()
        corrected_words = []
        corrections = []

        for word in words:
            # Skip short words and words with special characters
            if len(word) < self.min_word_length or not word.isalpha():
                corrected_words.append(word)
                continue

            word_lower = word.lower()

            # Skip common English stopwords - these should NEVER be corrected
            if word_lower in self.STOPWORDS:
                corrected_words.append(word)
                continue

            # Check if already in vocabulary
            if word_lower in self.vocabulary:
                corrected_words.append(word)
                continue

            # Try to find a correction
            result = self.find_closest(word_lower)
            if result:
                corrected, similarity = result
                # Get canonical form if available
                canonical = self.word_to_canonical.get(corrected, corrected)

                # Preserve original case pattern if possible
                if word[0].isupper() and canonical[0].islower():
                    canonical = canonical.capitalize()

                corrected_words.append(canonical)
                corrections.append({
                    "original": word,
                    "corrected": canonical,
                    "similarity": round(similarity, 3)
                })
            else:
                corrected_words.append(word)

        return " ".join(corrected_words), corrections

    def suggest_corrections(self, word: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Get top-k correction suggestions for a word.
        """
        word_lower = word.lower()

        suggestions = []
        for vocab_word in self.vocabulary:
            sim = similarity_ratio(word_lower, vocab_word)
            if sim >= 0.5:  # Lower threshold for suggestions
                suggestions.append((vocab_word, sim))

        suggestions.sort(key=lambda x: x[1], reverse=True)
        return suggestions[:top_k]


# Singleton instance
_corrector: Optional[TypoCorrector] = None


def get_corrector() -> TypoCorrector:
    """Get or create the singleton TypoCorrector instance."""
    global _corrector
    if _corrector is None:
        _corrector = TypoCorrector()
    return _corrector


def correct_query(query: str) -> Tuple[str, List[Dict]]:
    """Convenience function to correct a query."""
    return get_corrector().correct_query(query)
