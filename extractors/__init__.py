"""
Extractors Module

Feature extraction utilities for RAG document processing pipelines.

Provides:
- NER extraction (people, organizations, projects, grants)
- Date extraction from content
- Topic/theme classification
- Key term extraction

Usage:
    from extractors import (
        extract_entities,
        extract_dates,
        classify_topics,
        extract_key_terms,
    )

    # Extract entities
    entities = extract_entities("Dr. James Evans from NSF grant 2033345")
    print(entities.people)  # ["James Evans"]

    # Extract dates
    dates = extract_dates("The deadline is March 15, 2024")
    print(dates.primary_date.normalized)  # "2024-03-15"

    # Classify topics
    topics = classify_topics("Network analysis of citation patterns")
    print(topics.primary_topic())  # "Network Analysis"

    # Extract key terms
    terms = extract_key_terms("Deep learning and neural networks...")
    print(terms.term_list())  # ["deep learning", "neural networks", ...]
"""

# NER extraction
from .ner_extractor import (
    NERExtractor,
    EntityExtractionResult,
    extract_entities,
    extract_entities_batch,
    enrich_chunks_with_entities,
    get_ner_extractor,
)

# Date extraction
from .date_extractor import (
    DateExtractor,
    ExtractedDate,
    DateExtractionResult,
    extract_dates,
    get_date_extractor,
)

# Topic classification
from .topic_extractor import (
    TopicExtractor,
    TopicClassification,
    TopicClassificationResult,
    classify_topics,
    classify_chunks,
    get_topic_extractor,
    TOPIC_TAXONOMY,
)

# Key term extraction
from .keyterm_extractor import (
    KeyTermExtractor,
    KeyTerm,
    KeyTermResult,
    extract_key_terms,
    get_keyterm_extractor,
)

# Enrichment pipeline
from .enrichment import (
    enrich_chunk,
    enrich_chunks_batch,
    get_enrichment_stats,
)


__all__ = [
    # NER
    "NERExtractor",
    "EntityExtractionResult",
    "extract_entities",
    "extract_entities_batch",
    "enrich_chunks_with_entities",
    "get_ner_extractor",
    # Dates
    "DateExtractor",
    "ExtractedDate",
    "DateExtractionResult",
    "extract_dates",
    "get_date_extractor",
    # Topics
    "TopicExtractor",
    "TopicClassification",
    "TopicClassificationResult",
    "classify_topics",
    "classify_chunks",
    "get_topic_extractor",
    "TOPIC_TAXONOMY",
    # Key terms
    "KeyTermExtractor",
    "KeyTerm",
    "KeyTermResult",
    "extract_key_terms",
    "get_keyterm_extractor",
    # Enrichment
    "enrich_chunk",
    "enrich_chunks_batch",
    "get_enrichment_stats",
]
