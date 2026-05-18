# Jevans Directory Integration Report

**Date:** 2026-01-13
**Author:** Claude (with rbward)

---

## Summary

Built a pipeline to ingest the `/project/jevans` directory structure into Chorus's knowledge hypergraph. The system extracts metadata from 127 top-level directories, enriches it with curated catalog data, and provides a queryable interface for resource discovery.

---

## What Was Built

### 1. Ingest Module (`chorus/ingest/`)

| File | Purpose | Lines |
|------|---------|-------|
| `__init__.py` | Package exports | 13 |
| `jevans_scanner.py` | Scans /project/jevans, extracts metadata | 295 |
| `catalog_parser.py` | Parses lab_directory_catalog.txt | 197 |

### 2. Query Interface (`chorus/registry/resources.py`)

| Function | Description |
|----------|-------------|
| `get_datasets()` | Returns all 17 dataset resources |
| `get_models()` | Returns all 4 model directories |
| `get_valuable()` | Returns 43 high-value resources |
| `get_by_owner(username)` | Filter by owner |
| `search(query)` | Full-text search across name/description |

### 3. Data Files

| File | Contents |
|------|----------|
| `Data/jevans_resources.json` | 127 resources with metadata |
| `Data/snapshots/snapshot_2026-01-13.json` | First temporal snapshot |

---

## Resource Statistics

```
Total resources:     127
├── personal:         95  (researcher directories)
├── dataset:          17  (data snapshots)
├── project:          10  (shared projects)
├── model:             4  (LLM/embedding caches)
└── system:            1  (infrastructure)

Valuable resources:   43  (marked in catalog)
Active resources:     43  (recent activity)
```

---

## Key Resources Discovered

### Datasets (17 total)
- **Bibliographic:** OpenAlex, Semantic Scholar, MAG, DBLP, S2AG
- **Patent:** PATSTAT (64GB EPO data)
- **Biomedical:** PKG v2.0, iCite, DrugBank, PubChem
- **Other:** Dimensions, Papers with Code

### Models (4 directories)
- **LLMs/**: Gemma 2/3, Llama 3.1/3.2, Qwen 2.5
- **models/**: SPECTER2 citation embeddings
- **hf-cache/**: Hugging Face model cache
- **ollama_models/**: Local inference cache

### High-Value Personal Directories
- **nadav**: 38+ projects, LLM research, economics
- **akozlo**: Digital doubles, Common Crawl, interpretability
- **esposito**: 695GB patent similarity matrices (1995-2010)
- **haiziyu**: 122GB PMI counters (1800-present)
- **kangd (donghyun)**: Data engineering, bibliometric snapshots

---

## Data Model

### Resource Node
```json
{
  "id": "jevans:openalex-snapshot",
  "name": "openalex-snapshot",
  "path": "/project/jevans/openalex-snapshot",
  "resource_type": "dataset",
  "owner_username": "beichenlu",
  "mtime": "2025-12-19T14:56:00",
  "is_valuable": true,
  "description": "OpenAlex bibliographic database snapshot..."
}
```

### Ownership Hyperedge
```json
{
  "id": "owns:beichenlu:jevans:openalex-snapshot",
  "person_username": "beichenlu",
  "resource_id": "jevans:openalex-snapshot",
  "timestamp": "2026-01-13T...",
  "confidence": 1.0,
  "evidence": "filesystem"
}
```

---

## Usage Examples

### Python
```python
from registry.resources import ResourceRegistry

registry = ResourceRegistry()

# Get all datasets
datasets = registry.get_datasets()

# Search for patent resources
patent_resources = registry.search('patent')

# Get resources by owner
nadav_resources = registry.get_by_owner('nadavkunievsky')

# Get valuable resources
valuable = registry.get_valuable()
```

### Command Line
```bash
cd /project/jevans/robbie/chorus
python3 registry/resources.py
```

---

## How Temporal Tracking Works

1. **Initial scan** creates `jevans_resources.json`
2. **Snapshots** saved to `Data/snapshots/snapshot_YYYY-MM-DD.json`
3. **Future diffs** compare snapshots to detect:
   - New directories → new nodes
   - Size changes → activity signal
   - mtime changes → modification hyperedges

To create a new snapshot:
```python
from ingest import scan_jevans_directory
scan_jevans_directory(output_path='Data/snapshots/snapshot_2026-01-20.json')
```

---

## Integration with Chorus

This data feeds into Chorus's existing systems:

| Chorus Component | Integration Point |
|------------------|-------------------|
| **Registry** | `resources.py` added alongside `lookup.py` |
| **RAG Search** | Resource descriptions can be indexed |
| **Hypergraph** | Ownership edges ready for graph storage |
| **Chat Interface** | "What datasets exist?" queries now answerable |

---

## Automated Daily Scanning (NEW)

### Components Added

| File | Purpose |
|------|---------|
| `ingest/daily_scan.py` | Daily runner with diff detection |
| `ingest/diff.py` | Change detection between snapshots |
| `ingest/panel.py` | Time-series query interface |
| `scripts/daily_jevans_scan.sh` | Shell wrapper for cron |
| `scripts/setup_cron.sh` | Cron setup helper |

### Panel Query Examples

```python
from ingest.panel import SnapshotPanel

panel = SnapshotPanel()

# List all snapshots
panel.list_snapshots()  # ['2026-01-13', ...]

# Get current stats
panel.get_current_stats()  # {'total_resources': 127, ...}

# Activity summary for last 7 days
summary = panel.activity_summary(days=7)

# Resource history
timeline = panel.resource_timeline('openalex-snapshot')
```

### Cron Setup

```bash
# Run setup script to add cron entry
/project/jevans/robbie/chorus/scripts/setup_cron.sh

# Or manually add to crontab:
# 0 2 * * * /project/jevans/robbie/chorus/scripts/daily_jevans_scan.sh
```

### Data Files

| File | Purpose |
|------|---------|
| `Data/snapshots/snapshot_YYYY-MM-DD.json` | Daily snapshots |
| `Data/change_log.json` | Append-only log of changes |
| `logs/scan_YYYY-MM-DD.log` | Daily scan logs |
| `logs/cron.log` | Cron execution log |

---

## Files Created

```
chorus/
├── ingest/
│   ├── __init__.py
│   ├── jevans_scanner.py
│   ├── catalog_parser.py
│   ├── daily_scan.py        (new)
│   ├── diff.py              (new)
│   └── panel.py             (new)
├── registry/
│   └── resources.py
├── scripts/
│   ├── daily_jevans_scan.sh (new)
│   └── setup_cron.sh        (new)
├── logs/                     (new)
├── Data/
│   ├── jevans_resources.json
│   ├── change_log.json      (created on first diff)
│   └── snapshots/
│       └── snapshot_2026-01-13.json
└── JEVANS_INTEGRATION_REPORT.md
```

---

## Technical Notes

- **Python compatibility**: Works with Python 3.6+ (no dataclasses dependency)
- **Performance**: Full scan completes in < 1 second
- **Permissions**: Reads any directory accessible to current user
- **Catalog enrichment**: Parses existing lab_directory_catalog.txt for descriptions
- **Storage**: ~150 KB/day, ~56 MB/year for snapshots
- **Log retention**: Automatically cleans logs older than 30 days
