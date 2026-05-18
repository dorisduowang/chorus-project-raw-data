"""
Ingest module for Chorus - extracts data from external sources into the hypergraph.

Components:
- jevans_scanner: Scans /project/jevans directory
- catalog_parser: Parses lab_directory_catalog.txt
- daily_scan: Daily automated scanning
- diff: Change detection between snapshots
- panel: Time-series query interface
- snapshot_indexer: Indexes snapshots for RAG search
- gdrive_sync: Sync snapshots to Google Drive
"""

from .jevans_scanner import JevansScanner, scan_jevans_directory
from .catalog_parser import CatalogParser, parse_catalog
from .diff import compute_diff, find_previous_snapshot, load_snapshot
from .panel import SnapshotPanel
from .snapshot_indexer import (
    index_snapshot,
    index_all_new_snapshots,
    get_unindexed_snapshots,
    reindex_all_snapshots,
)

# Optional: Google Drive sync (requires google-api-python-client)
try:
    from .gdrive_sync import sync_snapshot
    _has_gdrive = True
except ImportError:
    sync_snapshot = None
    _has_gdrive = False

__all__ = [
    'JevansScanner',
    'scan_jevans_directory',
    'CatalogParser',
    'parse_catalog',
    'compute_diff',
    'find_previous_snapshot',
    'load_snapshot',
    'SnapshotPanel',
    'index_snapshot',
    'index_all_new_snapshots',
    'get_unindexed_snapshots',
    'reindex_all_snapshots',
    'sync_snapshot',
]
