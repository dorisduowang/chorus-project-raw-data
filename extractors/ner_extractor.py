#!/usr/bin/env python3
"""
Named Entity Recognition (NER) Extractor for RAG System

Extracts people, organizations, projects, and grant IDs from text content.
Uses spaCy for NER with custom patterns for research-specific entities.

Usage:
    from extractors import extract_entities, get_ner_extractor

    result = extract_entities("Dr. James Evans from NSF grant 2033345...")
    print(result.people)  # ["James Evans"]
    print(result.grant_ids)  # ["NSF-2033345"]
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Any

# Lazy load spaCy to avoid import overhead
_nlp = None


def _get_nlp():
    """Lazy load spaCy model."""
    global _nlp
    if _nlp is None:
        try:
            import spacy
            try:
                _nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
            except OSError:
                # Model not installed, try to download
                import subprocess
                subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
                _nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
        except ImportError:
            raise ImportError("spaCy is required. Install with: pip install spacy")
    return _nlp


@dataclass
class EntityExtractionResult:
    """Result of entity extraction from text."""
    people: List[str] = field(default_factory=list)
    organizations: List[str] = field(default_factory=list)
    projects: List[str] = field(default_factory=list)
    grant_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, List[str]]:
        """Convert to dictionary for JSON serialization."""
        return {
            "people": self.people,
            "organizations": self.organizations,
            "projects": self.projects,
            "grant_ids": self.grant_ids,
        }

    def has_entities(self) -> bool:
        """Check if any entities were found."""
        return bool(self.people or self.organizations or self.projects or self.grant_ids)

    def total_count(self) -> int:
        """Total number of entities found."""
        return len(self.people) + len(self.organizations) + len(self.projects) + len(self.grant_ids)


# Grant ID patterns for various funding agencies
GRANT_PATTERNS = [
    # NSF: NSF-2033345, NSF 2033345, NSF #2033345
    (r'\bNSF[-\s#]?(\d{7})\b', 'NSF'),
    # NIH: R01-GM123456, U01-CA123456, P01-HL123456, etc.
    (r'\b([RUFKPT]\d{2})[-\s]?([A-Z]{2})[-\s]?(\d{6})\b', 'NIH'),
    # DARPA: DARPA-HR001121S0001
    (r'\bDARPA[-\s]?([A-Z0-9]{10,})\b', 'DARPA'),
    # DOE: DE-SC0012345, DE-FG02-12345
    (r'\bDE[-\s]?([A-Z]{2}\d{2}[-]?\d{5,})\b', 'DOE'),
    # NASA: NNX12AB34C, 80NSSC21K1234
    (r'\b(NNX\d{2}[A-Z]{2}\d{2}[A-Z]|80NSSC\d{2}[A-Z]\d{4})\b', 'NASA'),
    # Air Force: FA9550-12-1-0123
    (r'\b(FA\d{4}[-\s]?\d{2}[-\s]?\d[-\s]?\d{4})\b', 'AFOSR'),
    # Army: W911NF-12-1-0123
    (r'\b(W911NF[-\s]?\d{2}[-\s]?\d[-\s]?\d{4})\b', 'ARO'),
    # ONR: N00014-12-1-0123
    (r'\b(N\d{5}[-\s]?\d{2}[-\s]?\d[-\s]?\d{4})\b', 'ONR'),
]

# Project/acronym patterns
PROJECT_PATTERNS = [
    r'\b(MURI)\b',
    r'\b(APTO)\b',
    r'\b(CAREER)\b',
    r'\b(EAGER)\b',
    r'\b(RAPID)\b',
    r'\b(STTR)\b',
    r'\b(SBIR)\b',
]

# Common funding organizations to look for
FUNDING_ORGS = {
    'NSF', 'NIH', 'DARPA', 'DOE', 'NASA', 'DOD', 'IARPA',
    'ONR', 'ARO', 'AFOSR', 'AFRL', 'NCI', 'NIMH', 'NIGMS',
    'Wellcome Trust', 'Howard Hughes', 'Sloan Foundation',
    'MacArthur Foundation', 'Moore Foundation', 'Simons Foundation',
}

# Title prefixes to remove from names
TITLE_PREFIXES = {
    'Dr.', 'Dr', 'Prof.', 'Prof', 'Professor', 'Mr.', 'Mr',
    'Mrs.', 'Mrs', 'Ms.', 'Ms', 'Sir', 'Dame',
}


class NERExtractor:
    """
    Named Entity Recognition extractor using spaCy with custom patterns.

    Extracts:
    - People names (PERSON entities from spaCy)
    - Organizations (ORG entities + funding agency keywords)
    - Projects (research project acronyms)
    - Grant IDs (various funding agency patterns)
    """

    def __init__(self):
        """Initialize the NER extractor."""
        # Compile grant patterns
        self.grant_patterns = [
            (re.compile(pattern, re.IGNORECASE), agency)
            for pattern, agency in GRANT_PATTERNS
        ]

        # Compile project patterns
        self.project_patterns = [
            re.compile(pattern) for pattern in PROJECT_PATTERNS
        ]

    def extract(
        self,
        text: str,
        known_entities: Optional[Dict[str, List[str]]] = None
    ) -> EntityExtractionResult:
        """
        Extract named entities from text.

        Args:
            text: The text to extract entities from
            known_entities: Optional dict of known entities to boost matching
                           {"people": [...], "projects": [...], "orgs": [...]}

        Returns:
            EntityExtractionResult with lists of extracted entities
        """
        if not text or not text.strip():
            return EntityExtractionResult()

        # Extract each entity type
        people = self._extract_people(text)
        organizations = self._extract_organizations(text)
        projects = self._extract_projects(text)
        grant_ids = self._extract_grant_ids(text)

        # Boost known entities if provided
        if known_entities:
            people = self._boost_known(people, known_entities.get('people', []), text)
            organizations = self._boost_known(organizations, known_entities.get('orgs', []), text)
            projects = self._boost_known(projects, known_entities.get('projects', []), text)

        # Deduplicate
        people = self._deduplicate_names(people)
        organizations = list(set(organizations))
        projects = list(set(projects))
        grant_ids = list(set(grant_ids))

        return EntityExtractionResult(
            people=people,
            organizations=organizations,
            projects=projects,
            grant_ids=grant_ids
        )

    def _extract_people(self, text: str) -> List[str]:
        """Extract person names using spaCy NER."""
        nlp = _get_nlp()
        doc = nlp(text)

        people = []
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                name = self._clean_person_name(ent.text)
                if name and len(name) > 2:
                    people.append(name)

        return people

    def _extract_organizations(self, text: str) -> List[str]:
        """Extract organization names using spaCy NER + keyword matching."""
        nlp = _get_nlp()
        doc = nlp(text)

        organizations = []

        # Get ORG entities from spaCy
        for ent in doc.ents:
            if ent.label_ == "ORG":
                org = ent.text.strip()
                if org and len(org) > 1:
                    organizations.append(org)

        # Also check for funding org keywords
        text_upper = text.upper()
        for org in FUNDING_ORGS:
            if org.upper() in text_upper:
                organizations.append(org)

        return organizations

    def _extract_projects(self, text: str) -> List[str]:
        """Extract project names/acronyms using patterns."""
        projects = []

        for pattern in self.project_patterns:
            matches = pattern.findall(text)
            projects.extend(matches)

        return projects

    def _extract_grant_ids(self, text: str) -> List[str]:
        """Extract grant IDs using agency-specific patterns."""
        grant_ids = []

        for pattern, agency in self.grant_patterns:
            matches = pattern.findall(text)
            for match in matches:
                if isinstance(match, tuple):
                    # NIH-style: combine parts
                    grant_id = f"{agency}-{''.join(match)}"
                else:
                    grant_id = f"{agency}-{match}"
                grant_ids.append(grant_id.upper())

        return grant_ids

    def _clean_person_name(self, name: str) -> str:
        """Clean a person name by removing titles and extra whitespace."""
        name = name.strip()

        # Remove title prefixes
        for title in TITLE_PREFIXES:
            if name.startswith(title + ' '):
                name = name[len(title):].strip()
            elif name.startswith(title):
                name = name[len(title):].strip()

        # Remove extra whitespace
        name = ' '.join(name.split())

        return name

    def _boost_known(
        self,
        found: List[str],
        known: List[str],
        text: str
    ) -> List[str]:
        """Add known entities that appear in text but weren't extracted."""
        text_lower = text.lower()

        for entity in known:
            if entity.lower() in text_lower:
                if entity not in found:
                    found.append(entity)

        return found

    def _deduplicate_names(self, names: List[str]) -> List[str]:
        """
        Deduplicate names, preferring longer versions.
        E.g., keep "James Evans" over "Evans" if both are found.
        """
        if not names:
            return []

        # Sort by length (longest first)
        sorted_names = sorted(names, key=len, reverse=True)

        deduped = []
        for name in sorted_names:
            # Check if this name is a substring of any already-kept name
            name_lower = name.lower()
            is_substring = any(
                name_lower in kept.lower() or kept.lower() in name_lower
                for kept in deduped
            )

            if not is_substring:
                deduped.append(name)
            elif not any(name_lower in kept.lower() for kept in deduped):
                # This name contains a kept name, so replace
                deduped = [
                    name if kept.lower() in name_lower else kept
                    for kept in deduped
                ]

        return deduped


# Singleton instance
_extractor: Optional[NERExtractor] = None


def get_ner_extractor() -> NERExtractor:
    """Get or create the singleton NERExtractor instance."""
    global _extractor
    if _extractor is None:
        _extractor = NERExtractor()
    return _extractor


def extract_entities(
    text: str,
    known_entities: Optional[Dict[str, List[str]]] = None
) -> EntityExtractionResult:
    """
    Convenience function to extract entities from text.

    Args:
        text: Text to extract entities from
        known_entities: Optional dict of known entities to boost matching

    Returns:
        EntityExtractionResult with all found entities

    Example:
        >>> result = extract_entities("Dr. James Evans from NSF grant 2033345")
        >>> print(result.people)
        ["James Evans"]
        >>> print(result.grant_ids)
        ["NSF-2033345"]
    """
    extractor = get_ner_extractor()
    return extractor.extract(text, known_entities)


def extract_entities_batch(
    texts: List[str],
    known_entities: Optional[Dict[str, List[str]]] = None
) -> List[EntityExtractionResult]:
    """
    Extract entities from multiple texts efficiently.

    Args:
        texts: List of texts to process
        known_entities: Optional dict of known entities

    Returns:
        List of EntityExtractionResult objects
    """
    extractor = get_ner_extractor()
    return [extractor.extract(text, known_entities) for text in texts]


def enrich_chunks_with_entities(
    chunks: List[Dict[str, Any]],
    known_entities: Optional[Dict[str, List[str]]] = None,
    text_field: str = "text"
) -> List[Dict[str, Any]]:
    """
    Enrich chunk dictionaries with extracted entities.

    Args:
        chunks: List of chunk dicts with text content
        known_entities: Optional known entities for boosting
        text_field: Field name containing the text

    Returns:
        Same chunks with added 'entities' field
    """
    extractor = get_ner_extractor()

    for chunk in chunks:
        text = chunk.get(text_field, "")
        result = extractor.extract(text, known_entities)
        chunk["entities"] = result.to_dict()

    return chunks


if __name__ == "__main__":
    # Test examples
    test_texts = [
        "Dr. James Evans from the University of Chicago received NSF grant 2033345.",
        "The MURI project is funded by DARPA-HR001121S0001 and involves Prof. Smith.",
        "NIH R01-GM123456 supports research by the Evans lab.",
        "Our collaboration with IARPA and DOE-SC0012345 continues.",
    ]

    print("Testing NER Extractor\n" + "=" * 60)

    for text in test_texts:
        print(f"\nText: {text}")
        result = extract_entities(text)
        print(f"  People: {result.people}")
        print(f"  Organizations: {result.organizations}")
        print(f"  Projects: {result.projects}")
        print(f"  Grant IDs: {result.grant_ids}")
