#!/usr/bin/env python3
"""
Incremental RAG Indexing System

Supports:
- Add new files without full rebuild
- Remove files and their chunks
- Update modified files (remove old + add new)
- Track file hashes for change detection
- Hot-reload capability for running servers

Usage:
    # Add new files
    python incremental_index.py add file1.pdf file2.docx

    # Add all files from a directory
    python incremental_index.py add-dir /path/to/documents

    # Remove files
    python incremental_index.py remove file1.pdf

    # Sync with a directory (add new, remove deleted, update modified)
    python incremental_index.py sync /path/to/documents

    # Show index status
    python incremental_index.py status
"""

import argparse
import hashlib
import json
import os
import pickle
import string
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

from chunking_strategies import (
    ChunkWithMetadata,
    smart_chunk_document,
    detect_doc_type,
)


# =============================================================================
# Configuration
# =============================================================================

INDEX_DIR = Path("Data/rag_indexes")
INCREMENTAL_DIR = INDEX_DIR / "incremental"
STATE_FILE = INCREMENTAL_DIR / "index_state.json"
MODEL_NAME = "BAAI/bge-large-en-v1.5"


@dataclass
class FileState:
    """Track state of an indexed file."""
    path: str
    name: str
    content_hash: str
    indexed_at: str
    chunk_ids: List[int]  # IDs in the FAISS index
    num_chunks: int


@dataclass
class IndexState:
    """Track overall index state."""
    created_at: str
    updated_at: str
    model_name: str
    dimension: int
    total_chunks: int
    next_id: int  # Next available ID for new chunks
    files: Dict[str, dict]  # path -> FileState as dict

    def get_file(self, path: str) -> Optional[FileState]:
        if path in self.files:
            d = self.files[path]
            return FileState(**d)
        return None

    def set_file(self, file_state: FileState):
        self.files[file_state.path] = asdict(file_state)

    def remove_file(self, path: str) -> Optional[FileState]:
        if path in self.files:
            d = self.files.pop(path)
            return FileState(**d)
        return None


def compute_file_hash(filepath: str) -> str:
    """Compute SHA256 hash of file contents."""
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            hasher.update(chunk)
    return hasher.hexdigest()[:16]  # First 16 chars is enough


def compute_content_hash(content: str) -> str:
    """Compute hash of text content."""
    return hashlib.sha256(content.encode()).hexdigest()[:16]


class IncrementalIndexer:
    """
    Incremental indexer using FAISS IndexIDMap for add/remove support.

    Architecture:
    - Uses IndexIDMap2 wrapper around IndexFlatIP for ID-based operations
    - Each chunk gets a unique integer ID
    - Files track which chunk IDs belong to them
    - Removal: look up chunk IDs for file, remove from index
    - BM25 is rebuilt on changes (fast enough for incremental updates)
    """

    def __init__(self, index_dir: Path = INDEX_DIR):
        self.index_dir = index_dir
        self.incremental_dir = index_dir / "incremental"
        self.incremental_dir.mkdir(parents=True, exist_ok=True)

        self.model: Optional[SentenceTransformer] = None
        self.faiss_index: Optional[faiss.IndexIDMap2] = None
        self.state: Optional[IndexState] = None
        self.chunks: Dict[int, dict] = {}  # id -> chunk metadata

    def _load_model(self):
        """Load embedding model (lazy loading)."""
        if self.model is None:
            print(f"Loading embedding model: {MODEL_NAME}")
            self.model = SentenceTransformer(MODEL_NAME)

    def _get_dimension(self) -> int:
        """Get embedding dimension."""
        self._load_model()
        return self.model.get_sentence_embedding_dimension()

    def load(self) -> bool:
        """Load existing index state. Returns True if index exists."""
        state_file = self.incremental_dir / "index_state.json"

        if not state_file.exists():
            return False

        # Load state
        with open(state_file, 'r') as f:
            data = json.load(f)
        self.state = IndexState(**data)

        # Load FAISS index
        faiss_path = self.incremental_dir / "index.faiss"
        if faiss_path.exists():
            self.faiss_index = faiss.read_index(str(faiss_path))
        else:
            # Create new index
            dim = self.state.dimension
            base_index = faiss.IndexFlatIP(dim)
            self.faiss_index = faiss.IndexIDMap2(base_index)

        # Load chunks
        chunks_path = self.incremental_dir / "chunks.json"
        if chunks_path.exists():
            with open(chunks_path, 'r') as f:
                # JSON keys are strings, convert to int
                self.chunks = {int(k): v for k, v in json.load(f).items()}

        print(f"Loaded index: {self.state.total_chunks} chunks from {len(self.state.files)} files")
        return True

    def initialize(self):
        """Initialize a new empty index."""
        dim = self._get_dimension()

        # Create FAISS index with ID support
        base_index = faiss.IndexFlatIP(dim)
        self.faiss_index = faiss.IndexIDMap2(base_index)

        # Initialize state
        self.state = IndexState(
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            model_name=MODEL_NAME,
            dimension=dim,
            total_chunks=0,
            next_id=0,
            files={}
        )
        self.chunks = {}

        print(f"Initialized new index (dimension: {dim})")

    def save(self):
        """Save index state to disk."""
        # Save state
        self.state.updated_at = datetime.now().isoformat()
        with open(self.incremental_dir / "index_state.json", 'w') as f:
            json.dump(asdict(self.state), f, indent=2)

        # Save FAISS index
        faiss.write_index(self.faiss_index, str(self.incremental_dir / "index.faiss"))

        # Save chunks
        with open(self.incremental_dir / "chunks.json", 'w') as f:
            json.dump(self.chunks, f)

        # Also export to hybrid format for compatibility with RAG server
        self._export_hybrid()

        print(f"Saved index: {self.state.total_chunks} chunks")

    def _export_hybrid(self):
        """Export to hybrid format for RAG server compatibility."""
        hybrid_dir = self.index_dir / "hybrid"
        hybrid_dir.mkdir(exist_ok=True)

        # Get all chunk data in order
        chunk_list = []
        tokenized = []

        # Sort by ID to maintain consistent ordering
        for chunk_id in sorted(self.chunks.keys()):
            chunk = self.chunks[chunk_id]
            chunk_list.append(chunk)

            # Tokenize for BM25
            text = chunk.get("text", "")
            tokens = text.lower().translate(str.maketrans('', '', string.punctuation)).split()
            tokens = [t for t in tokens if len(t) > 2]
            tokenized.append(tokens)

        # We need to rebuild FAISS without ID mapping for compatibility
        # Extract embeddings from current index
        if self.chunks:
            ids = np.array(sorted(self.chunks.keys()), dtype=np.int64)
            embeddings = np.zeros((len(ids), self.state.dimension), dtype=np.float32)
            for i, chunk_id in enumerate(ids):
                embeddings[i] = self.faiss_index.reconstruct(int(chunk_id))

            # Create flat index
            flat_index = faiss.IndexFlatIP(self.state.dimension)
            flat_index.add(embeddings)
            faiss.write_index(flat_index, str(hybrid_dir / "faiss_index.faiss"))

        # Save hybrid data
        with open(hybrid_dir / "hybrid_data.pkl", 'wb') as f:
            pickle.dump({
                'chunks': chunk_list,
                'tokenized_chunks': tokenized,
                'model_name': self.state.model_name,
                'semantic_weight': 0.7
            }, f)

        print(f"Exported hybrid index: {len(chunk_list)} chunks")

    def add_file(self, filepath: str, content: Optional[str] = None) -> int:
        """
        Add a file to the index.

        Args:
            filepath: Path to the file
            content: Optional pre-extracted content. If None, will read from file.

        Returns:
            Number of chunks added
        """
        self._load_model()

        filepath = os.path.abspath(filepath)
        filename = os.path.basename(filepath)

        # Check if already indexed
        existing = self.state.get_file(filepath)
        if existing:
            # Check if content changed
            if content:
                new_hash = compute_content_hash(content)
            else:
                new_hash = compute_file_hash(filepath)

            if existing.content_hash == new_hash:
                print(f"  Skipping {filename} (unchanged)")
                return 0

            # Content changed - remove old and re-add
            print(f"  Updating {filename} (content changed)")
            self.remove_file(filepath)

        # Read content if not provided
        if content is None:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

        # Compute hash
        content_hash = compute_content_hash(content)

        # Chunk the document
        chunks = smart_chunk_document(
            content=content,
            filename=filename,
            filepath=filepath,
            chunk_size=500,
            overlap=100,
            transcript_duration=180
        )

        if not chunks:
            print(f"  Skipping {filename} (no chunks generated)")
            return 0

        # Create embeddings
        texts = [c.text for c in chunks]
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        faiss.normalize_L2(embeddings)

        # Assign IDs and add to index
        chunk_ids = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_id = self.state.next_id
            self.state.next_id += 1
            chunk_ids.append(chunk_id)

            # Add to FAISS
            self.faiss_index.add_with_ids(
                embedding.reshape(1, -1),
                np.array([chunk_id], dtype=np.int64)
            )

            # Store chunk metadata
            self.chunks[chunk_id] = chunk.to_dict()

        # Update state
        file_state = FileState(
            path=filepath,
            name=filename,
            content_hash=content_hash,
            indexed_at=datetime.now().isoformat(),
            chunk_ids=chunk_ids,
            num_chunks=len(chunks)
        )
        self.state.set_file(file_state)
        self.state.total_chunks += len(chunks)

        print(f"  Added {filename}: {len(chunks)} chunks")
        return len(chunks)

    def remove_file(self, filepath: str) -> int:
        """
        Remove a file from the index.

        Returns:
            Number of chunks removed
        """
        filepath = os.path.abspath(filepath)

        file_state = self.state.remove_file(filepath)
        if not file_state:
            print(f"  File not in index: {filepath}")
            return 0

        # Remove chunks from FAISS
        ids_to_remove = np.array(file_state.chunk_ids, dtype=np.int64)
        self.faiss_index.remove_ids(ids_to_remove)

        # Remove chunk metadata
        for chunk_id in file_state.chunk_ids:
            self.chunks.pop(chunk_id, None)

        self.state.total_chunks -= file_state.num_chunks

        print(f"  Removed {file_state.name}: {file_state.num_chunks} chunks")
        return file_state.num_chunks

    def add_directory(self, dirpath: str, extensions: Set[str] = None) -> Tuple[int, int]:
        """
        Add all files from a directory.

        Args:
            dirpath: Directory path
            extensions: Set of extensions to include (e.g., {'.pdf', '.docx'})
                       If None, includes common document types.

        Returns:
            Tuple of (files_added, chunks_added)
        """
        if extensions is None:
            extensions = {'.pdf', '.docx', '.doc', '.txt', '.md', '.vtt', '.json'}

        dirpath = Path(dirpath)
        files_added = 0
        chunks_added = 0

        for filepath in dirpath.rglob('*'):
            if filepath.is_file() and filepath.suffix.lower() in extensions:
                n = self.add_file(str(filepath))
                if n > 0:
                    files_added += 1
                    chunks_added += n

        return files_added, chunks_added

    def sync_directory(self, dirpath: str, extensions: Set[str] = None) -> Dict[str, int]:
        """
        Sync index with a directory:
        - Add new files
        - Remove deleted files
        - Update modified files

        Returns:
            Dict with counts: added, removed, updated, unchanged
        """
        if extensions is None:
            extensions = {'.pdf', '.docx', '.doc', '.txt', '.md', '.vtt', '.json'}

        dirpath = Path(dirpath).absolute()

        # Get current files in directory
        current_files = set()
        for filepath in dirpath.rglob('*'):
            if filepath.is_file() and filepath.suffix.lower() in extensions:
                current_files.add(str(filepath.absolute()))

        # Get indexed files in this directory
        indexed_files = set()
        for path in self.state.files.keys():
            if path.startswith(str(dirpath)):
                indexed_files.add(path)

        stats = {"added": 0, "removed": 0, "updated": 0, "unchanged": 0}

        # Remove deleted files
        for path in indexed_files - current_files:
            self.remove_file(path)
            stats["removed"] += 1

        # Add/update files
        for path in current_files:
            existing = self.state.get_file(path)
            if existing:
                # Check if modified
                current_hash = compute_file_hash(path)
                if existing.content_hash != current_hash:
                    self.remove_file(path)
                    self.add_file(path)
                    stats["updated"] += 1
                else:
                    stats["unchanged"] += 1
            else:
                # New file
                if self.add_file(path) > 0:
                    stats["added"] += 1

        return stats

    def status(self) -> Dict:
        """Get index status."""
        doc_types = {}
        for chunk in self.chunks.values():
            dt = chunk.get("doc_type", "other")
            doc_types[dt] = doc_types.get(dt, 0) + 1

        return {
            "created_at": self.state.created_at,
            "updated_at": self.state.updated_at,
            "model_name": self.state.model_name,
            "dimension": self.state.dimension,
            "total_files": len(self.state.files),
            "total_chunks": self.state.total_chunks,
            "doc_types": doc_types,
            "files": [
                {
                    "name": fs["name"],
                    "chunks": fs["num_chunks"],
                    "indexed_at": fs["indexed_at"]
                }
                for fs in sorted(
                    self.state.files.values(),
                    key=lambda x: x["indexed_at"],
                    reverse=True
                )[:10]  # Last 10 files
            ]
        }


def main():
    parser = argparse.ArgumentParser(
        description="Incremental RAG indexing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Add command
    add_parser = subparsers.add_parser("add", help="Add files to index")
    add_parser.add_argument("files", nargs="+", help="Files to add")

    # Add directory command
    adddir_parser = subparsers.add_parser("add-dir", help="Add all files from directory")
    adddir_parser.add_argument("directory", help="Directory to add")

    # Remove command
    remove_parser = subparsers.add_parser("remove", help="Remove files from index")
    remove_parser.add_argument("files", nargs="+", help="Files to remove")

    # Sync command
    sync_parser = subparsers.add_parser("sync", help="Sync with directory")
    sync_parser.add_argument("directory", help="Directory to sync")

    # Status command
    subparsers.add_parser("status", help="Show index status")

    # Init command
    subparsers.add_parser("init", help="Initialize new index")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    indexer = IncrementalIndexer()

    if args.command == "init":
        indexer.initialize()
        indexer.save()
        print("Index initialized successfully")

    elif args.command == "status":
        if not indexer.load():
            print("No index found. Run 'init' first.")
            return

        status = indexer.status()
        print(f"\n{'='*50}")
        print("Index Status")
        print(f"{'='*50}")
        print(f"Created: {status['created_at']}")
        print(f"Updated: {status['updated_at']}")
        print(f"Model: {status['model_name']}")
        print(f"Dimension: {status['dimension']}")
        print(f"Total files: {status['total_files']}")
        print(f"Total chunks: {status['total_chunks']}")
        print(f"\nDocument types:")
        for dt, count in sorted(status['doc_types'].items()):
            print(f"  {dt}: {count}")
        print(f"\nRecent files:")
        for f in status['files']:
            print(f"  {f['name']}: {f['chunks']} chunks ({f['indexed_at'][:10]})")

    elif args.command == "add":
        if not indexer.load():
            print("No index found. Initializing...")
            indexer.initialize()

        total = 0
        for filepath in args.files:
            if os.path.exists(filepath):
                total += indexer.add_file(filepath)
            else:
                print(f"  File not found: {filepath}")

        indexer.save()
        print(f"\nAdded {total} chunks")

    elif args.command == "add-dir":
        if not indexer.load():
            print("No index found. Initializing...")
            indexer.initialize()

        files, chunks = indexer.add_directory(args.directory)
        indexer.save()
        print(f"\nAdded {files} files, {chunks} chunks")

    elif args.command == "remove":
        if not indexer.load():
            print("No index found.")
            return

        total = 0
        for filepath in args.files:
            total += indexer.remove_file(os.path.abspath(filepath))

        indexer.save()
        print(f"\nRemoved {total} chunks")

    elif args.command == "sync":
        if not indexer.load():
            print("No index found. Initializing...")
            indexer.initialize()

        stats = indexer.sync_directory(args.directory)
        indexer.save()
        print(f"\nSync complete:")
        print(f"  Added: {stats['added']} files")
        print(f"  Updated: {stats['updated']} files")
        print(f"  Removed: {stats['removed']} files")
        print(f"  Unchanged: {stats['unchanged']} files")


if __name__ == "__main__":
    main()
