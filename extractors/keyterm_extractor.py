#!/usr/bin/env python3
"""
Key Terms Extraction Module for RAG System

Extracts important/salient terms from text using TF-IDF and RAKE-inspired
scoring. Optimized for academic and research text.

Usage:
    from extractors import extract_key_terms, get_keyterm_extractor

    result = extract_key_terms("Deep learning models use neural networks...")
    for term in result.terms:
        print(f"{term.term}: {term.score:.3f}")
"""

import re
import math
import json
import pickle
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Any, Tuple
from collections import Counter
from pathlib import Path


@dataclass
class KeyTerm:
    """A single extracted key term."""
    term: str
    score: float
    frequency: int
    is_phrase: bool  # True if multi-word

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "term": self.term,
            "score": round(self.score, 4),
            "frequency": self.frequency,
            "is_phrase": self.is_phrase,
        }


@dataclass
class KeyTermResult:
    """Result of key term extraction."""
    terms: List[KeyTerm] = field(default_factory=list)
    total_candidates: int = 0
    extraction_method: str = "tf-idf+rake"
    has_idf: bool = False

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "terms": [t.to_dict() for t in self.terms],
            "total_candidates": self.total_candidates,
            "extraction_method": self.extraction_method,
            "has_idf": self.has_idf,
        }

    def has_terms(self) -> bool:
        """Check if any terms were extracted."""
        return len(self.terms) > 0

    def term_list(self) -> List[str]:
        """Get list of term strings."""
        return [t.term for t in self.terms]


# Comprehensive stopwords for academic text
STOPWORDS: Set[str] = {
    # Common English
    'a', 'an', 'the', 'and', 'or', 'but', 'if', 'then', 'else', 'when',
    'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
    'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from',
    'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
    'further', 'once', 'here', 'there', 'where', 'why', 'how', 'all',
    'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
    'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'can',
    'will', 'just', 'should', 'now', 'also', 'well', 'may', 'much',

    # Pronouns
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
    'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his',
    'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself',
    'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
    'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having',
    'do', 'does', 'did', 'doing', 'would', 'could', 'ought',

    # Academic filler
    'however', 'therefore', 'thus', 'hence', 'moreover', 'furthermore',
    'although', 'though', 'whereas', 'while', 'since', 'because',
    'nevertheless', 'nonetheless', 'despite', 'regarding', 'according',

    # Common verbs
    'said', 'says', 'found', 'showed', 'shown', 'show', 'shows',
    'used', 'using', 'use', 'based', 'proposed', 'present', 'presented',
    'described', 'discussed', 'reported', 'observed', 'noted',

    # Numbers and units
    'one', 'two', 'three', 'first', 'second', 'third', 'new', 'old',
}


class KeyTermExtractor:
    """
    Key term extractor using TF-IDF with RAKE-inspired co-occurrence scoring.

    Handles academic text well, including acronyms, hyphenated terms, and
    technical phrases.
    """

    def __init__(self, idf_path: Optional[str] = None):
        """
        Initialize the key term extractor.

        Args:
            idf_path: Optional path to load pre-computed IDF scores
        """
        self.idf_scores: Dict[str, float] = {}
        self.doc_count: int = 0

        if idf_path:
            self.load_idf(idf_path)

    def fit(self, corpus: List[str], verbose: bool = False) -> None:
        """
        Compute IDF scores from a corpus of documents.

        Args:
            corpus: List of document texts
            verbose: Print progress information
        """
        if verbose:
            print(f"Fitting on {len(corpus)} documents...")

        self.doc_count = len(corpus)
        doc_freq: Counter = Counter()

        for i, doc in enumerate(corpus):
            # Get unique terms in document
            terms = set(self._tokenize(doc))
            doc_freq.update(terms)

            if verbose and (i + 1) % 1000 == 0:
                print(f"  Processed {i + 1}/{len(corpus)} documents")

        # Compute IDF: log(N / df)
        self.idf_scores = {}
        for term, df in doc_freq.items():
            self.idf_scores[term] = math.log(self.doc_count / df)

        if verbose:
            print(f"Computed IDF for {len(self.idf_scores)} unique terms")

    def extract(
        self,
        text: str,
        top_k: int = 10,
        min_score: float = 0.1
    ) -> KeyTermResult:
        """
        Extract key terms from text.

        Args:
            text: Input text
            top_k: Maximum number of terms to return
            min_score: Minimum score threshold

        Returns:
            KeyTermResult with extracted terms
        """
        if not text or not text.strip():
            return KeyTermResult()

        # Tokenize and get candidates
        words = self._tokenize(text)
        word_counts = Counter(words)
        total_words = len(words)

        if total_words == 0:
            return KeyTermResult()

        # Get unigram candidates
        candidates: Dict[str, Tuple[float, int, bool]] = {}

        for word, count in word_counts.items():
            if self._is_valid_term(word):
                score = self._score_term(word, count, total_words)
                if score > 0:
                    candidates[word] = (score, count, False)

        # Get n-gram candidates (bigrams, trigrams)
        ngram_candidates = self._extract_ngrams(text, total_words)
        candidates.update(ngram_candidates)

        # Filter and sort
        filtered = [
            KeyTerm(term=term, score=score, frequency=freq, is_phrase=is_phrase)
            for term, (score, freq, is_phrase) in candidates.items()
            if score >= min_score
        ]

        filtered.sort(key=lambda x: x.score, reverse=True)
        filtered = filtered[:top_k]

        return KeyTermResult(
            terms=filtered,
            total_candidates=len(candidates),
            extraction_method="tf-idf+rake",
            has_idf=bool(self.idf_scores)
        )

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        # Preserve hyphenated terms and acronyms
        text_lower = text.lower()
        # Match words, hyphenated terms, and acronyms
        tokens = re.findall(r'\b[a-z]+(?:-[a-z]+)*\b|\b[A-Z]{2,}\b', text)
        return [t.lower() for t in tokens]

    def _is_valid_term(self, term: str) -> bool:
        """Check if a term is valid (not stopword, long enough)."""
        return (
            len(term) >= 2 and
            term not in STOPWORDS and
            not term.isdigit()
        )

    def _score_term(self, term: str, count: int, total_words: int) -> float:
        """Score a single term using TF-IDF."""
        # Term frequency
        tf = count / total_words

        # IDF (use default if not fitted)
        idf = self.idf_scores.get(term, 2.0)

        # Base TF-IDF score
        score = tf * idf

        # Boost academic terms
        if self._is_academic_term(term):
            score *= 1.3

        return score

    def _is_academic_term(self, term: str) -> bool:
        """Check if term appears to be academic/technical."""
        # Acronyms (2+ uppercase)
        if term.isupper() and len(term) >= 2:
            return True

        # Hyphenated terms
        if '-' in term:
            return True

        # Technical suffixes
        suffixes = ('tion', 'ology', 'metric', 'graph', 'ism', 'ist', 'ity')
        if any(term.endswith(s) for s in suffixes):
            return True

        return False

    def _extract_ngrams(
        self,
        text: str,
        total_words: int
    ) -> Dict[str, Tuple[float, int, bool]]:
        """Extract and score bigrams and trigrams."""
        candidates = {}

        # Clean text for phrase extraction
        text_clean = re.sub(r'[^\w\s-]', ' ', text.lower())
        words = text_clean.split()

        # Bigrams
        for i in range(len(words) - 1):
            w1, w2 = words[i], words[i + 1]
            if self._is_valid_term(w1) and self._is_valid_term(w2):
                phrase = f"{w1} {w2}"
                count = text.lower().count(phrase)
                if count > 0:
                    score = self._score_phrase(phrase, count, total_words, 2)
                    if phrase not in candidates or score > candidates[phrase][0]:
                        candidates[phrase] = (score, count, True)

        # Trigrams
        for i in range(len(words) - 2):
            w1, w2, w3 = words[i], words[i + 1], words[i + 2]
            if (self._is_valid_term(w1) and self._is_valid_term(w3) and
                (self._is_valid_term(w2) or w2 in ('of', 'and', 'for', 'the'))):
                phrase = f"{w1} {w2} {w3}"
                count = text.lower().count(phrase)
                if count > 0:
                    score = self._score_phrase(phrase, count, total_words, 3)
                    if phrase not in candidates or score > candidates[phrase][0]:
                        candidates[phrase] = (score, count, True)

        return candidates

    def _score_phrase(
        self,
        phrase: str,
        count: int,
        total_words: int,
        n: int
    ) -> float:
        """Score a multi-word phrase."""
        tf = count / total_words

        # Average IDF of component words
        words = phrase.split()
        idfs = [self.idf_scores.get(w, 2.0) for w in words]
        avg_idf = sum(idfs) / len(idfs)

        # Base score
        score = tf * avg_idf

        # Length bonus (longer phrases are often more specific)
        score *= (1.0 + 0.1 * n)

        # Phrase boost
        score *= 1.5

        return score

    def save_idf(self, path: str) -> None:
        """Save IDF scores to disk."""
        path = Path(path)
        data = {
            "idf_scores": self.idf_scores,
            "doc_count": self.doc_count
        }

        if path.suffix == '.json':
            with open(path, 'w') as f:
                json.dump(data, f)
        else:
            with open(path, 'wb') as f:
                pickle.dump(data, f)

    def load_idf(self, path: str) -> None:
        """Load IDF scores from disk."""
        path = Path(path)

        if not path.exists():
            return

        if path.suffix == '.json':
            with open(path) as f:
                data = json.load(f)
        else:
            with open(path, 'rb') as f:
                data = pickle.load(f)

        self.idf_scores = data.get("idf_scores", {})
        self.doc_count = data.get("doc_count", 0)

    def extract_from_chunks(
        self,
        chunks: List[Dict[str, Any]],
        top_k: int = 10,
        min_score: float = 0.1,
        text_field: str = "text"
    ) -> List[Dict[str, Any]]:
        """
        Extract key terms from multiple chunks.

        Args:
            chunks: List of chunk dictionaries
            top_k: Maximum terms per chunk
            min_score: Minimum score threshold
            text_field: Field name containing text

        Returns:
            Chunks with added 'key_terms' and 'key_terms_list' fields
        """
        for chunk in chunks:
            text = chunk.get(text_field, "")
            result = self.extract(text, top_k, min_score)
            chunk["key_terms"] = [t.to_dict() for t in result.terms]
            chunk["key_terms_list"] = result.term_list()

        return chunks


# Singleton instance
_extractor: Optional[KeyTermExtractor] = None


def get_keyterm_extractor(idf_path: Optional[str] = None) -> KeyTermExtractor:
    """Get or create the singleton KeyTermExtractor instance."""
    global _extractor
    if _extractor is None:
        _extractor = KeyTermExtractor(idf_path)
    return _extractor


def extract_key_terms(
    text: str,
    top_k: int = 10,
    min_score: float = 0.1
) -> KeyTermResult:
    """
    Convenience function to extract key terms from text.

    Args:
        text: Text to extract terms from
        top_k: Maximum number of terms
        min_score: Minimum score threshold

    Returns:
        KeyTermResult with extracted terms
    """
    extractor = get_keyterm_extractor()
    return extractor.extract(text, top_k, min_score)


if __name__ == "__main__":
    test_texts = [
        """Deep learning models use neural networks to learn representations
        from data. Transformer architectures like BERT have revolutionized
        natural language processing through self-attention mechanisms.""",

        """Network analysis reveals community structure in citation networks.
        We apply graph-based methods to identify research clusters and
        measure collaboration patterns across institutions.""",

        """The grant proposal outlines a five-year research plan focusing
        on computational social science. Funding will support graduate
        students and computational infrastructure.""",
    ]

    print("Testing Key Term Extractor\n" + "=" * 60)

    for i, text in enumerate(test_texts, 1):
        print(f"\n--- Text {i} ---")
        print(text[:100] + "...")

        result = extract_key_terms(text, top_k=8)

        if result.has_terms():
            print("\nKey terms:")
            for term in result.terms:
                phrase_marker = " (phrase)" if term.is_phrase else ""
                print(f"  {term.term}: {term.score:.3f}{phrase_marker}")
        else:
            print("  No key terms found")
