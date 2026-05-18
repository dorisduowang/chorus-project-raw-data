# Chorus Auto-Report
**Generated:** 21:29:09 on 01/13/26

---

## Summary

Added automated daily profiling of `/project/jevans` directory to create a living "digital twin" of the lab's computational resources.

---

## New Files Added

### Ingest Module (`ingest/`)

| File | Lines | Purpose |
|------|-------|---------|
| `__init__.py` | 26 | Package exports |
| `jevans_scanner.py` | 295 | Scans /project/jevans, extracts metadata |
| `catalog_parser.py` | 197 | Parses lab_directory_catalog.txt for descriptions |
| `daily_scan.py` | 142 | Daily automated runner with diff detection |
| `diff.py` | 156 | Change detection between snapshots |
| `panel.py` | 208 | Time-series panel query interface |

### Registry Extension (`registry/`)

| File | Lines | Purpose |
|------|-------|---------|
| `resources.py` | 140 | Query interface for jevans resources |

### Scripts (`scripts/`)

| File | Purpose |
|------|---------|
| `daily_jevans_scan.sh` | Shell wrapper for cron execution |
| `setup_cron.sh` | One-command cron setup helper |

### Data Files (`Data/`)

| File | Size | Purpose |
|------|------|---------|
| `jevans_resources.json` | 156 KB | Current resource inventory |
| `snapshots/snapshot_2026-01-13.json` | 156 KB | First daily snapshot |

### Documentation

| File | Purpose |
|------|---------|
| `JEVANS_INTEGRATION_REPORT.md` | Full technical documentation |

---

## Features Implemented

### 1. Directory Scanner
- Scans 127 top-level directories in `/project/jevans`
- Extracts: name, path, owner, size, mtime, type
- Classifies resources: dataset (17), model (4), project (10), personal (95)

### 2. Catalog Enrichment
- Parses existing `lab_directory_catalog.txt`
- Adds descriptions and value assessments
- Flags 43 valuable and 43 active resources

### 3. Daily Automation
- Cron job runs at 2:00 AM daily
- Creates timestamped snapshots
- Computes diffs between consecutive days
- Logs changes to append-only change log

### 4. Panel Query Interface
```python
from ingest.panel import SnapshotPanel
panel = SnapshotPanel()
panel.list_snapshots()        # All available dates
panel.activity_summary(7)     # Week's changes
panel.resource_timeline(name) # Resource history
```

### 5. Resource Query Interface
```python
from registry.resources import ResourceRegistry
registry = ResourceRegistry()
registry.get_datasets()       # 17 datasets
registry.search('patent')     # Full-text search
registry.get_valuable()       # 43 high-value resources
```

---

## Resource Statistics

```
Total resources:     127
├── personal:         95
├── dataset:          17
├── project:          10
├── model:             4
└── system:            1

Valuable:            43
Active:              43
```

---

## Key Datasets Discovered

- OpenAlex (Dec 2024)
- Semantic Scholar (Jan 2025)
- PATSTAT (64 GB patent data)
- PKG v2.0 (144 GB biomedical knowledge graph)
- MAG (Dec 2021 - historical)
- DBLP (Jan 2025)

---

## Key Models Available

- LLMs/: Llama 3.1/3.2, Gemma 2/3, Qwen 2.5
- models/: SPECTER2 citation embeddings
- hf-cache/: HuggingFace model cache

---

## Storage Estimates

| Item | Per Day | Per Year |
|------|---------|----------|
| Snapshot | ~150 KB | ~55 MB |
| Change log | ~1 KB | ~365 KB |

---

## To Enable Daily Scans

```bash
/project/jevans/robbie/chorus/scripts/setup_cron.sh
```

---

## Technical Notes

- Python 3.6+ compatible (no dataclasses)
- Scan completes in < 1 second
- Logs auto-clean after 30 days
