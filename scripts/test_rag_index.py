#!/usr/bin/env python3
"""Test RAG indexing with the new venv."""
import sys
sys.path.insert(0, '/project/jevans/robbie/chorus')

from pathlib import Path
from ingest.snapshot_indexer import index_snapshot

snapshot = Path('/project/jevans/robbie/chorus/Data/snapshots/snapshot_2026-01-22.json')
print(f"Testing RAG indexing with {snapshot}")
result = index_snapshot(snapshot, verbose=True)
print(f"\nResult: {result}")
