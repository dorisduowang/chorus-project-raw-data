#!/usr/bin/env python3
"""
Rebuild RAG indexes with proper per-file metadata and smart chunking.

This version:
- Re-parses stored file data to properly attribute chunks to source files
- Uses smart chunking strategies (time-based for transcripts, sentence-based for docs)
- Extracts rich metadata (doc_type, dates, timestamps, speakers)
"""

import json
import pickle
import re
import string
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from rank_bm25 import BM25Okapi

# Import smart chunking strategies
from chunking_strategies import (
    ChunkWithMetadata,
    smart_chunk_document,
    detect_doc_type,
    extract_date_from_filename,
    chunk_text_with_overlap
)

# Import enrichment pipeline
from extractors.enrichment import enrich_chunks_batch, get_enrichment_stats


def load_and_chunk_by_file(
    manifest_path: str = "Data/parsed/manifest.json",
    full_text_path: str = "Data/parsed/full_text.txt",
    chunk_size: int = 500,
    overlap: int = 100,
    transcript_duration: float = 180  # 3 minutes for transcripts
) -> List[ChunkWithMetadata]:
    """
    Load parsed data and create chunks with proper file attribution.

    Strategy:
    1. Split full_text by double newlines (element separators)
    2. Assign segments to files proportionally based on num_elements
    3. Use smart chunking based on document type:
       - Transcripts: chunk by time (~3 min segments with overlap)
       - Documents: chunk by sentences with overlap
    """

    # Load manifest
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    files = manifest.get('files', [])
    total_elements = manifest.get('total_elements', 0)

    # Load full text
    with open(full_text_path, 'r') as f:
        full_text = f.read()

    print(f"Loaded {len(files)} files, {total_elements} total elements")
    print(f"Full text: {len(full_text)} characters")

    # Split full text into segments (separated by double newlines)
    segments = [s.strip() for s in full_text.split('\n\n') if s.strip()]
    print(f"Split into {len(segments)} segments")

    # Track stats by document type
    doc_type_stats = {}

    # Calculate cumulative element counts to map segments to files
    all_chunks = []
    segment_idx = 0

    for file_info in files:
        file_name = file_info.get('name', 'Unknown')
        file_path = file_info.get('path', file_name)
        num_elements = file_info.get('num_elements', 0)

        if num_elements == 0:
            continue

        # Collect segments for this file
        file_segments = []
        for _ in range(num_elements):
            if segment_idx < len(segments):
                file_segments.append(segments[segment_idx])
                segment_idx += 1

        if not file_segments:
            continue

        # Join segments for this file
        file_text = '\n\n'.join(file_segments)

        # Use smart chunking based on document type
        file_chunks = smart_chunk_document(
            content=file_text,
            filename=file_name,
            filepath=file_path,
            chunk_size=chunk_size,
            overlap=overlap,
            transcript_duration=transcript_duration
        )

        # Track stats
        doc_type = detect_doc_type(file_name)
        if doc_type not in doc_type_stats:
            doc_type_stats[doc_type] = {"files": 0, "chunks": 0}
        doc_type_stats[doc_type]["files"] += 1
        doc_type_stats[doc_type]["chunks"] += len(file_chunks)

        all_chunks.extend(file_chunks)

    print(f"\nCreated {len(all_chunks)} chunks with smart file attribution")

    # Print stats by document type
    print("\nChunks by document type:")
    for doc_type, stats in sorted(doc_type_stats.items()):
        avg_chunks = stats["chunks"] / stats["files"] if stats["files"] > 0 else 0
        print(f"  {doc_type}: {stats['files']} files, {stats['chunks']} chunks (avg {avg_chunks:.1f}/file)")

    # Count unique files
    files_with_chunks = len(set(c.source_file for c in all_chunks))
    print(f"\nTotal: {files_with_chunks} unique files")

    return all_chunks


def enrich_all_chunks(
    chunks: list[ChunkWithMetadata],
    use_ner: bool = False,
    use_dates: bool = True,
    use_topics: bool = True,
    use_keyterms: bool = True,
) -> list[ChunkWithMetadata]:
    """
    Run the enrichment pipeline on all chunks.

    Args:
        chunks: List of ChunkWithMetadata to enrich
        use_ner: Enable NER (requires spaCy)
        use_dates: Enable date extraction
        use_topics: Enable topic classification
        use_keyterms: Enable key term extraction

    Returns:
        Enriched chunks (modified in place)
    """
    print(f"\n{'=' * 60}")
    print("Running Enrichment Pipeline")
    print(f"{'=' * 60}")
    print(f"  NER: {'enabled' if use_ner else 'disabled (spaCy not available)'}")
    print(f"  Dates: {'enabled' if use_dates else 'disabled'}")
    print(f"  Topics: {'enabled' if use_topics else 'disabled'}")
    print(f"  Key Terms: {'enabled' if use_keyterms else 'disabled'}")
    print()

    # Run batch enrichment
    enriched = enrich_chunks_batch(
        chunks,
        use_ner=use_ner,
        use_dates=use_dates,
        use_topics=use_topics,
        use_keyterms=use_keyterms,
        verbose=True
    )

    # Get and print statistics
    stats = get_enrichment_stats(enriched)

    print(f"\n{'=' * 60}")
    print("Enrichment Statistics")
    print(f"{'=' * 60}")
    print(f"  Total chunks: {stats['total_chunks']}")
    print(f"\nCoverage:")
    print(f"  Chunks with dates mentioned: {stats['chunks_with_dates']} ({stats.get('date_mention_coverage_pct', 0)}%)")
    print(f"  Chunks with date_year set: {stats['date_coverage']} ({stats.get('date_year_coverage_pct', 0)}%)")
    print(f"  Chunks with topics: {stats['chunks_with_topics']} ({stats.get('topic_coverage_pct', 0)}%)")
    print(f"  Chunks with key terms: {stats['chunks_with_keyterms']} ({stats.get('keyterm_coverage_pct', 0)}%)")
    print(f"  Chunks with entities: {stats['chunks_with_entities']} ({stats.get('entity_coverage_pct', 0)}%)")
    print(f"\nUnique extractions:")
    print(f"  Unique topics: {stats['unique_topics']}")
    print(f"  Unique people: {stats['unique_people']}")
    print(f"  Unique organizations: {stats['unique_orgs']}")
    print(f"  Unique projects: {stats['unique_projects']}")
    print(f"  Unique grants: {stats['unique_grants']}")

    return enriched


def build_indexes(chunks: list[ChunkWithMetadata],
                  output_dir: str = "Data/rag_indexes",
                  model_name: str = "BAAI/bge-large-en-v1.5"):
    """Build all RAG indexes."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    texts = [c.text for c in chunks]

    print(f"Creating embeddings for {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]
    faiss_index = faiss.IndexFlatIP(dimension)
    faiss_index.add(embeddings)
    print(f"FAISS index: {faiss_index.ntotal} vectors, dim {dimension}")

    # BM25
    print("Building BM25 index...")
    tokenized = []
    for text in texts:
        tokens = text.lower().translate(str.maketrans('', '', string.punctuation)).split()
        tokens = [t for t in tokens if len(t) > 2]
        tokenized.append(tokens)

    # Save basic
    basic_path = output_path / "basic"
    basic_path.mkdir(exist_ok=True)
    faiss.write_index(faiss_index, str(basic_path / "index.faiss"))
    with open(basic_path / "chunks.pkl", 'wb') as f:
        pickle.dump({'chunks': texts, 'model_name': model_name}, f)
    print(f"Saved: {basic_path}")

    # Save with metadata
    metadata_path = output_path / "with_metadata"
    metadata_path.mkdir(exist_ok=True)
    faiss.write_index(faiss_index, str(metadata_path / "index.faiss"))
    with open(metadata_path / "chunks_metadata.json", 'w') as f:
        json.dump({
            'model_name': model_name,
            'chunks': [c.to_dict() for c in chunks]
        }, f)
    print(f"Saved: {metadata_path}")

    # Save hybrid
    hybrid_path = output_path / "hybrid"
    hybrid_path.mkdir(exist_ok=True)
    faiss.write_index(faiss_index, str(hybrid_path / "faiss_index.faiss"))
    with open(hybrid_path / "hybrid_data.pkl", 'wb') as f:
        pickle.dump({
            'chunks': [c.to_dict() for c in chunks],
            'tokenized_chunks': tokenized,
            'model_name': model_name,
            'semantic_weight': 0.7
        }, f)
    print(f"Saved: {hybrid_path}")

    # Manifest
    with open(output_path / "manifest.json", 'w') as f:
        json.dump({
            'created_at': datetime.now().isoformat(),
            'model_name': model_name,
            'total_chunks': len(chunks),
            'unique_files': len(set(c.source_file for c in chunks))
        }, f, indent=2)

    print(f"\nAll indexes saved to {output_path}")


def main(
    use_enrichment: bool = True,
    use_ner: bool = False,
    use_dates: bool = True,
    use_topics: bool = True,
    use_keyterms: bool = True,
):
    """
    Main rebuild function with optional enrichment pipeline.

    Args:
        use_enrichment: Enable the enrichment pipeline
        use_ner: Enable NER extraction (requires spaCy)
        use_dates: Enable date extraction
        use_topics: Enable topic classification
        use_keyterms: Enable key term extraction
    """
    print("=" * 60)
    print("Rebuilding RAG with Per-File Attribution")
    print("=" * 60)

    chunks = load_and_chunk_by_file(
        manifest_path="Data/parsed/manifest.json",
        full_text_path="Data/parsed/full_text.txt",
        chunk_size=500,
        overlap=100
    )

    # Show sample (before enrichment)
    print("\nSample chunks (before enrichment):")
    for c in chunks[:3]:
        print(f"  [{c.source_file}] {c.text[:60]}...")

    # Run enrichment pipeline if enabled
    if use_enrichment:
        chunks = enrich_all_chunks(
            chunks,
            use_ner=use_ner,
            use_dates=use_dates,
            use_topics=use_topics,
            use_keyterms=use_keyterms,
        )

        # Show sample after enrichment
        print("\nSample enriched chunks:")
        for c in chunks[:3]:
            print(f"  [{c.source_file}]")
            print(f"    Topics: {c.topics}")
            print(f"    Key terms: {c.key_terms[:5] if c.key_terms else []}")
            print(f"    Dates mentioned: {c.dates_mentioned}")

    build_indexes(chunks, "Data/rag_indexes")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Rebuild RAG indexes with enrichment pipeline")
    parser.add_argument("--no-enrichment", action="store_true", help="Disable enrichment pipeline")
    parser.add_argument("--use-ner", action="store_true", help="Enable NER extraction (requires spaCy)")
    parser.add_argument("--no-dates", action="store_true", help="Disable date extraction")
    parser.add_argument("--no-topics", action="store_true", help="Disable topic classification")
    parser.add_argument("--no-keyterms", action="store_true", help="Disable key term extraction")

    args = parser.parse_args()

    main(
        use_enrichment=not args.no_enrichment,
        use_ner=args.use_ner,
        use_dates=not args.no_dates,
        use_topics=not args.no_topics,
        use_keyterms=not args.no_keyterms,
    )
