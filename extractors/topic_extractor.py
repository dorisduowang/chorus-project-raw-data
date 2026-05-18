#!/usr/bin/env python3
"""
Topic/Theme Classification Module for RAG System

Classifies text into predefined research topics using keyword/phrase
matching with TF-IDF weighting. Optimized for speed over accuracy.

Usage:
    from extractors import classify_topics, get_topic_extractor

    result = classify_topics("We study network analysis and graph theory...")
    for topic in result.topics:
        print(f"{topic.topic}: {topic.confidence:.2f}")
"""

import re
import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple, Any
from collections import Counter


@dataclass
class TopicClassification:
    """A single topic classification with confidence."""
    topic: str
    confidence: float  # 0.0-1.0
    matched_terms: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "topic": self.topic,
            "confidence": self.confidence,
            "matched_terms": self.matched_terms[:10],  # Limit for storage
        }


@dataclass
class TopicClassificationResult:
    """Result of topic classification."""
    topics: List[TopicClassification] = field(default_factory=list)
    text_length: int = 0
    unique_terms_found: int = 0

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "topics": [t.to_dict() for t in self.topics],
            "text_length": self.text_length,
            "unique_terms_found": self.unique_terms_found,
        }

    def has_topics(self) -> bool:
        """Check if any topics were classified."""
        return len(self.topics) > 0

    def primary_topic(self) -> Optional[str]:
        """Get the highest-confidence topic."""
        return self.topics[0].topic if self.topics else None

    def topic_labels(self) -> List[str]:
        """Get list of topic names."""
        return [t.topic for t in self.topics]


# Topic taxonomy with weighted keywords and phrases
TOPIC_TAXONOMY: Dict[str, Dict[str, Dict[str, float]]] = {
    "Network Analysis": {
        "keywords": {
            "network": 1.5, "graph": 1.5, "node": 1.3, "edge": 1.3,
            "centrality": 2.0, "clustering": 1.5, "community": 1.3,
            "degree": 1.0, "connectivity": 1.5, "topology": 1.8,
            "adjacency": 1.8, "path": 0.8, "betweenness": 2.0,
        },
        "phrases": {
            "network analysis": 2.5, "graph theory": 2.5, "social network": 2.0,
            "citation network": 2.5, "collaboration network": 2.5,
            "community detection": 2.5, "network structure": 2.0,
            "network science": 2.5, "complex network": 2.0,
        },
    },
    "Natural Language Processing": {
        "keywords": {
            "nlp": 2.5, "text": 1.0, "language": 1.3, "linguistic": 1.8,
            "semantic": 1.8, "syntactic": 1.8, "parser": 1.5, "tokenize": 2.0,
            "embedding": 1.8, "transformer": 2.0, "bert": 2.5, "gpt": 2.0,
            "sentiment": 2.0, "corpus": 1.5, "vocabulary": 1.3,
        },
        "phrases": {
            "natural language": 2.5, "text analysis": 2.0, "text mining": 2.0,
            "language model": 2.5, "word embedding": 2.5, "named entity": 2.5,
            "sentiment analysis": 2.5, "topic model": 2.0, "document classification": 2.0,
        },
    },
    "Science of Science": {
        "keywords": {
            "citation": 2.0, "publication": 1.5, "journal": 1.3, "paper": 0.8,
            "bibliometric": 2.5, "scientometric": 2.5, "h-index": 2.5,
            "impact": 1.3, "scholarly": 1.8, "academic": 1.0, "researcher": 1.0,
        },
        "phrases": {
            "science of science": 3.0, "citation analysis": 2.5, "research impact": 2.0,
            "scientific discovery": 2.0, "knowledge production": 2.0,
            "research output": 2.0, "scientific collaboration": 2.5,
            "publication patterns": 2.5, "citation patterns": 2.5,
        },
    },
    "Machine Learning": {
        "keywords": {
            "learning": 1.0, "model": 0.8, "training": 1.3, "prediction": 1.5,
            "classifier": 2.0, "regression": 1.8, "neural": 2.0, "deep": 1.5,
            "supervised": 2.0, "unsupervised": 2.0, "reinforcement": 2.0,
            "overfitting": 2.0, "validation": 1.3, "accuracy": 1.0,
        },
        "phrases": {
            "machine learning": 2.5, "deep learning": 2.5, "neural network": 2.5,
            "random forest": 2.5, "gradient descent": 2.5, "feature extraction": 2.0,
            "cross validation": 2.0, "training data": 1.8, "test set": 1.5,
        },
    },
    "Social Media": {
        "keywords": {
            "twitter": 2.5, "facebook": 2.5, "social": 1.0, "viral": 1.8,
            "hashtag": 2.5, "tweet": 2.5, "post": 0.8, "follower": 2.0,
            "influencer": 2.0, "platform": 1.0, "online": 1.0, "digital": 1.0,
        },
        "phrases": {
            "social media": 2.5, "online community": 2.0, "viral spread": 2.0,
            "information diffusion": 2.5, "online behavior": 2.0,
            "social platform": 2.0, "user engagement": 2.0,
        },
    },
    "Knowledge Discovery": {
        "keywords": {
            "discovery": 1.5, "retrieval": 1.8, "search": 1.0, "query": 1.5,
            "index": 1.3, "ranking": 1.5, "relevance": 1.8, "database": 1.3,
            "knowledge": 1.0, "information": 0.8, "extraction": 1.5,
        },
        "phrases": {
            "knowledge discovery": 2.5, "information retrieval": 2.5,
            "knowledge graph": 2.5, "data mining": 2.0, "pattern discovery": 2.0,
            "knowledge extraction": 2.5, "search engine": 2.0,
        },
    },
    "Computational Methods": {
        "keywords": {
            "algorithm": 1.8, "computational": 2.0, "simulation": 1.8,
            "optimization": 1.8, "heuristic": 2.0, "complexity": 1.5,
            "scalable": 1.5, "parallel": 1.5, "efficient": 1.0, "implementation": 1.0,
        },
        "phrases": {
            "computational method": 2.5, "algorithm design": 2.0,
            "computational approach": 2.0, "monte carlo": 2.5,
            "numerical method": 2.0, "high performance": 1.8,
        },
    },
    "Policy and Ethics": {
        "keywords": {
            "policy": 1.8, "ethics": 2.0, "regulation": 1.8, "governance": 1.8,
            "privacy": 2.0, "fairness": 2.0, "bias": 1.8, "accountability": 2.0,
            "transparency": 1.8, "legal": 1.3, "compliance": 1.5,
        },
        "phrases": {
            "science policy": 2.5, "research ethics": 2.5, "data privacy": 2.5,
            "algorithmic bias": 2.5, "responsible ai": 2.5, "ethical considerations": 2.0,
            "policy implications": 2.0, "regulatory framework": 2.0,
        },
    },
    "Collaboration": {
        "keywords": {
            "collaboration": 2.0, "team": 1.3, "interdisciplinary": 2.0,
            "coauthor": 2.5, "partnership": 1.5, "cooperative": 1.5,
            "collective": 1.5, "joint": 1.0, "together": 0.8,
        },
        "phrases": {
            "team science": 2.5, "research collaboration": 2.5,
            "interdisciplinary research": 2.5, "collaborative work": 2.0,
            "research team": 2.0, "cross-disciplinary": 2.5,
        },
    },
    "Education": {
        "keywords": {
            "education": 1.8, "teaching": 1.8, "student": 1.5, "learning": 1.0,
            "curriculum": 2.0, "training": 1.0, "mentor": 2.0, "workshop": 1.5,
            "course": 1.3, "pedagogy": 2.5, "instructor": 1.8,
        },
        "phrases": {
            "research training": 2.5, "graduate student": 2.0, "education program": 2.0,
            "teaching method": 2.0, "curriculum development": 2.5,
            "student training": 2.0, "mentorship program": 2.5,
        },
    },
    "Administrative": {
        "keywords": {
            "budget": 1.8, "funding": 1.5, "grant": 1.5, "proposal": 1.5,
            "report": 1.0, "deadline": 1.5, "meeting": 1.0, "schedule": 1.3,
            "administrative": 2.0, "management": 1.3, "resource": 1.0,
        },
        "phrases": {
            "grant proposal": 2.5, "budget planning": 2.0, "project management": 2.0,
            "annual report": 2.0, "progress report": 2.0, "funding source": 2.0,
            "resource allocation": 2.0, "lab meeting": 2.0,
        },
    },
}


class TopicExtractor:
    """
    Topic classifier using keyword/phrase matching with TF-IDF weighting.

    Optimized for speed: uses pre-compiled regex patterns and single-pass
    text processing.
    """

    def __init__(self):
        """Initialize the topic extractor with compiled patterns."""
        self.taxonomy = TOPIC_TAXONOMY
        self._compile_patterns()

    def _compile_patterns(self):
        """Compile regex patterns for phrase matching."""
        self.phrase_patterns: Dict[str, Dict[str, Tuple[re.Pattern, float]]] = {}

        for topic, content in self.taxonomy.items():
            self.phrase_patterns[topic] = {}
            for phrase, weight in content.get("phrases", {}).items():
                # Word boundary pattern for phrases
                pattern = re.compile(rf'\b{re.escape(phrase)}\b', re.IGNORECASE)
                self.phrase_patterns[topic][phrase] = (pattern, weight)

    def classify(
        self,
        text: str,
        top_k: int = 3,
        min_confidence: float = 0.1
    ) -> TopicClassificationResult:
        """
        Classify text into research topics.

        Args:
            text: Text to classify
            top_k: Maximum number of topics to return
            min_confidence: Minimum confidence threshold

        Returns:
            TopicClassificationResult with ranked topics
        """
        if not text or not text.strip():
            return TopicClassificationResult()

        # Tokenize
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        word_counts = Counter(words)
        total_words = len(words)

        if total_words == 0:
            return TopicClassificationResult()

        # Score each topic
        topic_scores: Dict[str, Tuple[float, List[str]]] = {}

        for topic, content in self.taxonomy.items():
            score, matched = self._score_topic(
                text, text_lower, word_counts, total_words, content
            )
            if score > 0:
                topic_scores[topic] = (score, matched)

        if not topic_scores:
            return TopicClassificationResult(text_length=len(text))

        # Normalize scores to 0-1 range
        max_score = max(s[0] for s in topic_scores.values())

        # Build classifications
        classifications = []
        for topic, (score, matched) in topic_scores.items():
            # Normalize and apply sigmoid for better spread
            normalized = score / max_score if max_score > 0 else 0
            confidence = self._sigmoid(normalized * 2 - 1)  # Map to sigmoid

            if confidence >= min_confidence:
                classifications.append(TopicClassification(
                    topic=topic,
                    confidence=round(confidence, 3),
                    matched_terms=matched[:10]
                ))

        # Sort by confidence and take top_k
        classifications.sort(key=lambda x: x.confidence, reverse=True)
        classifications = classifications[:top_k]

        return TopicClassificationResult(
            topics=classifications,
            text_length=len(text),
            unique_terms_found=len(set(
                term for c in classifications for term in c.matched_terms
            ))
        )

    def _score_topic(
        self,
        text: str,
        text_lower: str,
        word_counts: Counter,
        total_words: int,
        topic_content: Dict
    ) -> Tuple[float, List[str]]:
        """Score a single topic against the text."""
        score = 0.0
        matched = []

        # Score phrases (higher weight)
        phrases = topic_content.get("phrases", {})
        for phrase, (pattern, weight) in self.phrase_patterns.get(
            next((t for t, c in self.taxonomy.items() if c is topic_content), ""), {}
        ).items():
            matches = pattern.findall(text)
            if matches:
                tf = len(matches) / total_words
                score += tf * weight * 2.0  # Phrases get 2x boost
                matched.append(phrase)

        # Score keywords
        keywords = topic_content.get("keywords", {})
        for keyword, weight in keywords.items():
            count = word_counts.get(keyword, 0)
            if count > 0:
                tf = count / total_words
                score += tf * weight
                if keyword not in matched:
                    matched.append(keyword)

        return score, matched

    def _sigmoid(self, x: float) -> float:
        """Apply sigmoid function for score normalization."""
        return 1.0 / (1.0 + math.exp(-x * 3))

    def get_topics_list(self) -> List[str]:
        """Get list of available topics."""
        return list(self.taxonomy.keys())


# Singleton instance
_extractor: Optional[TopicExtractor] = None


def get_topic_extractor() -> TopicExtractor:
    """Get or create the singleton TopicExtractor instance."""
    global _extractor
    if _extractor is None:
        _extractor = TopicExtractor()
    return _extractor


def classify_topics(
    text: str,
    top_k: int = 3,
    min_confidence: float = 0.1
) -> TopicClassificationResult:
    """
    Convenience function to classify text into topics.

    Args:
        text: Text to classify
        top_k: Maximum number of topics to return
        min_confidence: Minimum confidence threshold

    Returns:
        TopicClassificationResult with ranked topics
    """
    extractor = get_topic_extractor()
    return extractor.classify(text, top_k, min_confidence)


def classify_chunks(
    chunks: List[Dict[str, Any]],
    top_k: int = 3,
    min_confidence: float = 0.1,
    text_field: str = "text"
) -> List[Dict[str, Any]]:
    """
    Classify multiple chunks and add topic metadata.

    Args:
        chunks: List of chunk dictionaries
        top_k: Maximum topics per chunk
        min_confidence: Minimum confidence threshold
        text_field: Field name containing text

    Returns:
        Chunks with added 'topics' and 'topic_labels' fields
    """
    extractor = get_topic_extractor()

    for chunk in chunks:
        text = chunk.get(text_field, "")
        result = extractor.classify(text, top_k, min_confidence)
        chunk["topics"] = [t.to_dict() for t in result.topics]
        chunk["topic_labels"] = result.topic_labels()
        chunk["primary_topic"] = result.primary_topic()

    return chunks


if __name__ == "__main__":
    test_texts = [
        "We use graph theory and network analysis to study citation networks.",
        "This paper presents a deep learning model for text classification.",
        "Our grant proposal focuses on science policy and research ethics.",
        "The team collaboration led to interdisciplinary research outcomes.",
        "The budget planning meeting discussed FY24 funding allocations.",
    ]

    print("Testing Topic Extractor\n" + "=" * 60)

    for text in test_texts:
        print(f"\nText: {text}")
        result = classify_topics(text, top_k=3)

        if result.has_topics():
            for topic in result.topics:
                print(f"  - {topic.topic}: {topic.confidence:.2f}")
                print(f"    Matched: {', '.join(topic.matched_terms[:5])}")
        else:
            print("  No topics matched")
