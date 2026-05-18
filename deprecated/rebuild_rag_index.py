#!/usr/bin/env python3
"""
Rebuild RAG indexes with proper per-file metadata.

This script:
1. Loads the parsed files from Data/parsed/
2. Re-chunks with overlap and proper source file metadata
3. Rebuilds all RAG indexes (basic, with_metadata, hybrid)
4. Saves everything to Data/rag_indexes/

Run inside Docker container:
    python /app/rebuild_rag_index.py
"""

import json
import pickle
import re
import string
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from rank_bm25 import BM25Okapi


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class ChunkWithMetadata:
    """A text chunk with associated metadata for RAG retrieval."""
    text: str
    source_file: str = ""
    source_path: str = ""
    element_type: str = ""
    chunk_index: int = 0
    total_chunks: int = 0
    page_number: Optional[int] = None
    section: str = ""
    char_start: int = 0
    char_end: int = 0

    def to_dict(self) -> dict:
        return {
            'text': self.text,
            'source_file': self.source_file,
            'source_path': self.source_path,
            'element_type': self.element_type,
            'chunk_index': self.chunk_index,
            'total_chunks': self.total_chunks,
            'page_number': self.page_number,
            'section': self.section,
            'char_start': self.char_start,
            'char_end': self.char_end
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'ChunkWithMetadata':
        return cls(**d)

    def citation(self) -> str:
        parts = []
        if self.source_file:
            parts.append(self.source_file)
        if self.page_number:
            parts.append(f"p.{self.page_number}")
        if self.section:
            parts.append(f"§{self.section}")
        return " | ".join(parts) if parts else "Unknown source"


# ============================================================================
# Chunking Functions
# ============================================================================

def smart_chunk_with_overlap(text: str,
                              chunk_size: int = 500,
                              overlap: int = 100,
                              min_chunk_size: int = 50) -> list[tuple[str, int, int]]:
    """
    Split text into chunks with overlap, respecting sentence boundaries.

    Returns:
        List of tuples: (chunk_text, char_start, char_end)
    """
    if not text or len(text) < min_chunk_size:
        return [(text, 0, len(text))] if text else []

    # Split into sentences
    sentence_pattern = r'(?<=[.!?])\s+'
    sentences = re.split(sentence_pattern, text)

    chunks = []
    current_chunk = ""
    current_start = 0
    char_pos = 0

    for sentence in sentences:
        sentence_len = len(sentence)

        if len(current_chunk) + sentence_len > chunk_size and current_chunk:
            chunk_end = char_pos
            chunks.append((current_chunk.strip(), current_start, chunk_end))

            if overlap > 0 and len(current_chunk) > overlap:
                overlap_start = len(current_chunk) - overlap
                overlap_text = current_chunk[overlap_start:]
                current_chunk = overlap_text + " " + sentence
                current_start = chunk_end - len(overlap_text)
            else:
                current_chunk = sentence
                current_start = char_pos
        else:
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
                current_start = char_pos

        char_pos += sentence_len + 1

    if current_chunk.strip():
        if len(current_chunk.strip()) >= min_chunk_size or not chunks:
            chunks.append((current_chunk.strip(), current_start, len(text)))
        elif chunks:
            prev_text, prev_start, _ = chunks[-1]
            chunks[-1] = (prev_text + " " + current_chunk.strip(), prev_start, len(text))

    return chunks


def load_manifest_and_chunks(parsed_dir: str = "Data/parsed") -> tuple[dict, list[str]]:
    """Load the manifest and chunks from parsed data."""
    parsed_path = Path(parsed_dir)

    # Load manifest
    with open(parsed_path / "manifest.json", 'r') as f:
        manifest = json.load(f)

    # Load chunks
    with open(parsed_path / "chunks.json", 'r') as f:
        data = json.load(f)
        chunks = data.get('chunks', [])

    return manifest, chunks


def create_chunks_with_file_metadata(manifest: dict,
                                      full_text_path: str = "Data/parsed/full_text.txt",
                                      chunk_size: int = 500,
                                      overlap: int = 100) -> list[ChunkWithMetadata]:
    """
    Create chunks with proper per-file metadata.

    Since we don't have the original elements anymore, we'll:
    1. Use the file list from manifest
    2. Re-chunk the full text with overlap
    3. Try to attribute chunks to files based on content matching
    """
    # Load full text
    with open(full_text_path, 'r') as f:
        full_text = f.read()

    # Get file list from manifest
    files = manifest.get('files', [])
    file_names = [f.get('name', 'Unknown') for f in files]
    file_paths = [f.get('path', f.get('name', 'Unknown')) for f in files]

    print(f"Found {len(files)} source files in manifest")
    print(f"Full text length: {len(full_text)} characters")

    # Create chunks with overlap
    raw_chunks = smart_chunk_with_overlap(full_text, chunk_size, overlap)
    print(f"Created {len(raw_chunks)} chunks with {overlap} char overlap")

    # Create ChunkWithMetadata objects
    # We'll try to match chunks to files by looking for file names in the text
    metadata_chunks = []

    for i, (chunk_text, char_start, char_end) in enumerate(raw_chunks):
        # Try to find which file this chunk belongs to
        source_file = "Unknown"
        source_path = "Unknown"

        # Simple heuristic: check if any file name appears in the chunk
        for fname, fpath in zip(file_names, file_paths):
            # Clean up file name for matching
            clean_name = fname.replace('.pdf', '').replace('.docx', '').replace('.txt', '')
            if len(clean_name) > 5 and clean_name.lower() in chunk_text.lower():
                source_file = fname
                source_path = fpath
                break

        chunk = ChunkWithMetadata(
            text=chunk_text,
            source_file=source_file,
            source_path=source_path,
            element_type="Text",
            chunk_index=i,
            total_chunks=len(raw_chunks),
            char_start=char_start,
            char_end=char_end
        )
        metadata_chunks.append(chunk)

    # Count attribution stats
    attributed = sum(1 for c in metadata_chunks if c.source_file != "Unknown")
    print(f"Attributed {attributed}/{len(metadata_chunks)} chunks to source files")

    return metadata_chunks


# ============================================================================
# RAG Index Classes
# ============================================================================

class QueryExpander:
    """Expands queries for better retrieval."""

    ACRONYMS = {
        "ai": "artificial intelligence",
        "ml": "machine learning",
        "nlp": "natural language processing",
        "llm": "large language model",
        "nsf": "national science foundation",
        "nih": "national institutes of health",
    }

    SYNONYMS = {
        'research': ['study', 'investigation', 'paper'],
        'workshop': ['seminar', 'conference', 'meeting'],
        'grant': ['funding', 'award', 'proposal'],
    }

    def expand_query(self, query: str) -> list[str]:
        queries = [query]
        q_lower = query.lower()
        words = q_lower.split()

        for acr, full in self.ACRONYMS.items():
            if acr in words:
                queries.append(q_lower.replace(acr, full))

        for word in words:
            if word in self.SYNONYMS:
                for syn in self.SYNONYMS[word][:2]:
                    new_q = q_lower.replace(word, syn)
                    if new_q not in queries:
                        queries.append(new_q)

        return queries


def build_all_indexes(chunks: list[ChunkWithMetadata],
                      output_dir: str = "Data/rag_indexes",
                      model_name: str = "all-MiniLM-L6-v2"):
    """Build all RAG indexes from chunks."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    # Extract texts
    texts = [c.text for c in chunks]

    print(f"Creating embeddings for {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    faiss.normalize_L2(embeddings)

    # Build FAISS index
    dimension = embeddings.shape[1]
    faiss_index = faiss.IndexFlatIP(dimension)
    faiss_index.add(embeddings)
    print(f"FAISS index built: {faiss_index.ntotal} vectors, dimension {dimension}")

    # Build BM25 index
    print("Building BM25 index...")
    tokenized = []
    for text in texts:
        tokens = text.lower().translate(str.maketrans('', '', string.punctuation)).split()
        tokens = [t for t in tokens if len(t) > 2]
        tokenized.append(tokens)
    bm25_index = BM25Okapi(tokenized)

    # Save basic index
    basic_path = output_path / "basic"
    basic_path.mkdir(exist_ok=True)
    faiss.write_index(faiss_index, str(basic_path / "index.faiss"))
    with open(basic_path / "chunks.pkl", 'wb') as f:
        pickle.dump({'chunks': texts, 'model_name': model_name}, f)
    print(f"Saved basic index to {basic_path}")

    # Save index with metadata
    metadata_path = output_path / "with_metadata"
    metadata_path.mkdir(exist_ok=True)
    faiss.write_index(faiss_index, str(metadata_path / "index.faiss"))
    with open(metadata_path / "chunks_metadata.json", 'w') as f:
        json.dump({
            'model_name': model_name,
            'chunks': [c.to_dict() for c in chunks]
        }, f, indent=2)
    print(f"Saved metadata index to {metadata_path}")

    # Save hybrid index
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
    print(f"Saved hybrid index to {hybrid_path}")

    # Save manifest
    manifest = {
        'created_at': datetime.now().isoformat(),
        'model_name': model_name,
        'total_chunks': len(chunks),
        'dimension': dimension,
        'indexes': ['basic', 'with_metadata', 'hybrid']
    }
    with open(output_path / "manifest.json", 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\nAll indexes saved to {output_path}")
    return output_path


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 60)
    print("Rebuilding RAG Indexes with Per-File Metadata")
    print("=" * 60)

    # Load existing parsed data
    print("\n1. Loading parsed data...")
    manifest, old_chunks = load_manifest_and_chunks("Data/parsed")
    print(f"   Manifest: {len(manifest.get('files', []))} files")
    print(f"   Old chunks: {len(old_chunks)}")

    # Create new chunks with metadata
    print("\n2. Creating chunks with file metadata...")
    new_chunks = create_chunks_with_file_metadata(
        manifest,
        full_text_path="Data/parsed/full_text.txt",
        chunk_size=500,
        overlap=100
    )

    # Build all indexes
    print("\n3. Building RAG indexes...")
    build_all_indexes(new_chunks, "Data/rag_indexes")

    print("\n" + "=" * 60)
    print("Done! RAG indexes rebuilt with per-file metadata.")
    print("=" * 60)


if __name__ == "__main__":
    main()
