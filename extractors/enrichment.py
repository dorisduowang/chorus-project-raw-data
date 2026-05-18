#!/usr/bin/env python3
"""
Chunk Enrichment Pipeline

Runs all extractors on chunks to populate rich metadata fields.
Provides both individual and batch enrichment functions.

Usage:
    from extractors.enrichment import enrich_chunk, enrich_chunks_batch

    # Single chunk
    chunk = ChunkWithMetadata(text="...")
    enriched = enrich_chunk(chunk)

    # Batch processing
    chunks = [chunk1, chunk2, ...]
    enriched = enrich_chunks_batch(chunks, use_ner=True)
"""

import sys
from typing import List, Dict, Any, Optional
from dataclasses import fields

# Import extractors
from .date_extractor import extract_dates
from .topic_extractor import classify_topics
from .keyterm_extractor import extract_key_terms

# NER is optional (requires spaCy)
try:
    from .ner_extractor import extract_entities
    HAS_NER = True
except ImportError:
    HAS_NER = False


def enrich_chunk(
    chunk: Any,
    use_ner: bool = True,
    use_dates: bool = True,
    use_topics: bool = True,
    use_keyterms: bool = True,
    top_k_topics: int = 3,
    top_k_terms: int = 10,
) -> Any:
    """
    Enrich a single chunk with extracted metadata.

    Args:
        chunk: ChunkWithMetadata object or dict with 'text' field
        use_ner: Run NER extraction (requires spaCy)
        use_dates: Run date extraction
        use_topics: Run topic classification
        use_keyterms: Run key term extraction
        top_k_topics: Max topics to extract
        top_k_terms: Max key terms to extract

    Returns:
        Same chunk object with enriched fields
    """
    # Handle both dataclass and dict
    is_dict = isinstance(chunk, dict)
    text = chunk.get('text', '') if is_dict else getattr(chunk, 'text', '')

    if not text:
        return chunk

    # NER extraction
    if use_ner and HAS_NER:
        try:
            entities = extract_entities(text)
            if is_dict:
                chunk['entities_people'] = entities.people
                chunk['entities_orgs'] = entities.organizations
                chunk['entities_projects'] = entities.projects
                chunk['entities_grants'] = entities.grant_ids
            else:
                chunk.entities_people = entities.people
                chunk.entities_orgs = entities.organizations
                chunk.entities_projects = entities.projects
                chunk.entities_grants = entities.grant_ids
        except Exception:
            pass  # NER failed, continue without

    # Date extraction
    if use_dates:
        try:
            dates = extract_dates(text)
            date_strs = [d.normalized for d in dates.dates]

            if is_dict:
                chunk['dates_mentioned'] = date_strs
                # Also update primary date if not set
                if dates.primary_date and not chunk.get('date_year'):
                    chunk['date_year'] = dates.primary_date.year
                    if dates.primary_date.month:
                        chunk['date_month'] = dates.primary_date.month
            else:
                chunk.dates_mentioned = date_strs
                if dates.primary_date and not chunk.date_year:
                    chunk.date_year = dates.primary_date.year
                    if dates.primary_date.month:
                        chunk.date_month = dates.primary_date.month
        except Exception:
            pass

    # Topic classification
    if use_topics:
        try:
            topics = classify_topics(text, top_k=top_k_topics)
            topic_labels = topics.topic_labels()

            if is_dict:
                chunk['topics'] = topic_labels
            else:
                chunk.topics = topic_labels
        except Exception:
            pass

    # Key term extraction
    if use_keyterms:
        try:
            terms = extract_key_terms(text, top_k=top_k_terms)
            term_list = terms.term_list()

            if is_dict:
                chunk['key_terms'] = term_list
            else:
                chunk.key_terms = term_list
        except Exception:
            pass

    return chunk


def enrich_chunks_batch(
    chunks: List[Any],
    use_ner: bool = True,
    use_dates: bool = True,
    use_topics: bool = True,
    use_keyterms: bool = True,
    top_k_topics: int = 3,
    top_k_terms: int = 10,
    verbose: bool = False,
) -> List[Any]:
    """
    Enrich multiple chunks with extracted metadata.

    Args:
        chunks: List of ChunkWithMetadata objects or dicts
        use_ner: Run NER extraction
        use_dates: Run date extraction
        use_topics: Run topic classification
        use_keyterms: Run key term extraction
        top_k_topics: Max topics per chunk
        top_k_terms: Max key terms per chunk
        verbose: Print progress

    Returns:
        Same chunks with enriched fields
    """
    total = len(chunks)

    for i, chunk in enumerate(chunks):
        enrich_chunk(
            chunk,
            use_ner=use_ner,
            use_dates=use_dates,
            use_topics=use_topics,
            use_keyterms=use_keyterms,
            top_k_topics=top_k_topics,
            top_k_terms=top_k_terms,
        )

        if verbose and (i + 1) % 100 == 0:
            print(f"  Enriched {i + 1}/{total} chunks")

    if verbose:
        print(f"  Enriched {total} chunks complete")

    return chunks


def get_enrichment_stats(chunks: List[Any]) -> Dict[str, Any]:
    """
    Get statistics about enriched chunks.

    Args:
        chunks: List of enriched chunks

    Returns:
        Dict with enrichment statistics
    """
    stats = {
        'total_chunks': len(chunks),
        'chunks_with_entities': 0,
        'chunks_with_dates': 0,
        'chunks_with_topics': 0,
        'chunks_with_keyterms': 0,
        'unique_people': set(),
        'unique_orgs': set(),
        'unique_projects': set(),
        'unique_grants': set(),
        'unique_topics': set(),
        'date_coverage': 0,
    }

    for chunk in chunks:
        is_dict = isinstance(chunk, dict)

        # Count entities
        people = chunk.get('entities_people', []) if is_dict else getattr(chunk, 'entities_people', None) or []
        orgs = chunk.get('entities_orgs', []) if is_dict else getattr(chunk, 'entities_orgs', None) or []
        projects = chunk.get('entities_projects', []) if is_dict else getattr(chunk, 'entities_projects', None) or []
        grants = chunk.get('entities_grants', []) if is_dict else getattr(chunk, 'entities_grants', None) or []

        if people or orgs or projects or grants:
            stats['chunks_with_entities'] += 1
        stats['unique_people'].update(people)
        stats['unique_orgs'].update(orgs)
        stats['unique_projects'].update(projects)
        stats['unique_grants'].update(grants)

        # Count dates
        dates = chunk.get('dates_mentioned', []) if is_dict else getattr(chunk, 'dates_mentioned', None) or []
        if dates:
            stats['chunks_with_dates'] += 1

        date_year = chunk.get('date_year') if is_dict else getattr(chunk, 'date_year', None)
        if date_year:
            stats['date_coverage'] += 1

        # Count topics
        topics = chunk.get('topics', []) if is_dict else getattr(chunk, 'topics', None) or []
        if topics:
            stats['chunks_with_topics'] += 1
        stats['unique_topics'].update(topics)

        # Count key terms
        terms = chunk.get('key_terms', []) if is_dict else getattr(chunk, 'key_terms', None) or []
        if terms:
            stats['chunks_with_keyterms'] += 1

    # Convert sets to counts
    stats['unique_people'] = len(stats['unique_people'])
    stats['unique_orgs'] = len(stats['unique_orgs'])
    stats['unique_projects'] = len(stats['unique_projects'])
    stats['unique_grants'] = len(stats['unique_grants'])
    stats['unique_topics'] = len(stats['unique_topics'])

    # Calculate percentages
    total = stats['total_chunks']
    if total > 0:
        stats['entity_coverage_pct'] = round(100 * stats['chunks_with_entities'] / total, 1)
        stats['date_mention_coverage_pct'] = round(100 * stats['chunks_with_dates'] / total, 1)
        stats['date_year_coverage_pct'] = round(100 * stats['date_coverage'] / total, 1)
        stats['topic_coverage_pct'] = round(100 * stats['chunks_with_topics'] / total, 1)
        stats['keyterm_coverage_pct'] = round(100 * stats['chunks_with_keyterms'] / total, 1)

    return stats


if __name__ == "__main__":
    # Test enrichment
    test_chunks = [
        {"text": "Dr. James Evans received NSF grant 2033345 for the MURI project. Deadline is March 15, 2024."},
        {"text": "Network analysis and machine learning methods for citation patterns."},
        {"text": "The FY24 budget meeting discussed resource allocation for the team."},
    ]

    print("Testing Chunk Enrichment\n" + "=" * 60)

    enriched = enrich_chunks_batch(test_chunks, use_ner=HAS_NER, verbose=True)

    for i, chunk in enumerate(enriched, 1):
        print(f"\n--- Chunk {i} ---")
        print(f"Text: {chunk['text'][:60]}...")
        print(f"Dates: {chunk.get('dates_mentioned', [])}")
        print(f"Topics: {chunk.get('topics', [])}")
        print(f"Key terms: {chunk.get('key_terms', [])[:5]}")
        if HAS_NER:
            print(f"People: {chunk.get('entities_people', [])}")
            print(f"Grants: {chunk.get('entities_grants', [])}")

    print("\n" + "=" * 60)
    stats = get_enrichment_stats(enriched)
    print("Enrichment Stats:")
    for key, val in stats.items():
        print(f"  {key}: {val}")
