"""
Snapshot Indexer - Converts jevans snapshots to searchable markdown and indexes them.

This module:
1. Converts JSON snapshots to markdown format with search-optimized keywords
2. Adds them to the RAG incremental index
3. Tracks which snapshots have been indexed

The markdown generation includes keyword enrichment for better RAG retrieval:
- Dataset aliases (e.g., "Semantic Scholar" -> "S2", "S2AG")
- Category labels (e.g., "bibliographic database", "patent data")
- Common search terms that should match each resource type
"""

import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Set

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


# Configuration
DATA_DIR = Path(__file__).parent.parent / "Data"
SNAPSHOTS_DIR = DATA_DIR / "snapshots"
INDEX_STATE_FILE = DATA_DIR / "snapshot_index_state.json"


# ============================================================================
# DATASET KNOWLEDGE BASE - Keywords, aliases, and categories for better search
# ============================================================================

# Maps dataset name patterns to search keywords and aliases
DATASET_KEYWORDS: Dict[str, Dict[str, Any]] = {
    # Bibliographic databases
    r'semantic_scholar|S2AG': {
        'aliases': ['S2', 'S2AG', 'Semantic Scholar', 'SemanticScholar'],
        'category': 'Bibliographic Database',
        'keywords': ['publications', 'papers', 'citations', 'abstracts', 'academic literature',
                     'research papers', 'scientific papers', 'bibliography', 'citation network'],
        'data_types': ['paper metadata', 'author data', 'citation links'],
    },
    r'openalex': {
        'aliases': ['OpenAlex', 'OA'],
        'category': 'Bibliographic Database',
        'keywords': ['publications', 'papers', 'citations', 'academic literature', 'open access',
                     'research papers', 'scientific papers', 'bibliography', 'MAG replacement'],
        'data_types': ['paper metadata', 'author data', 'institution data', 'citation links'],
    },
    r'MAG|microsoft.?academic': {
        'aliases': ['MAG', 'Microsoft Academic Graph', 'Microsoft Academic'],
        'category': 'Bibliographic Database (Historical)',
        'keywords': ['publications', 'papers', 'citations', 'academic literature',
                     'research papers', 'scientific papers', 'bibliography', 'discontinued 2022'],
        'data_types': ['paper metadata', 'author data', 'citation links'],
    },
    r'DBLP': {
        'aliases': ['DBLP', 'dblp'],
        'category': 'Computer Science Bibliography',
        'keywords': ['publications', 'papers', 'computer science', 'CS papers', 'venues',
                     'conferences', 'journals', 'bibliography', 'author collaboration'],
        'data_types': ['paper metadata', 'venue data', 'author data'],
    },
    r'dimensions': {
        'aliases': ['Dimensions'],
        'category': 'Bibliographic Database',
        'keywords': ['publications', 'papers', 'citations', 'grants', 'patents', 'clinical trials',
                     'policy documents', 'commercial database', 'comprehensive bibliometrics'],
        'data_types': ['paper metadata', 'grant data', 'patent links', 'citation data'],
    },
    # Patent databases
    r'PATSTAT': {
        'aliases': ['PATSTAT', 'EPO PATSTAT'],
        'category': 'Patent Database',
        'keywords': ['patents', 'patent data', 'intellectual property', 'IP data', 'EPO',
                     'European Patent Office', 'patent citations', 'patent classifications',
                     'inventors', 'assignees', 'patent families'],
        'data_types': ['patent bibliographic data', 'patent citations', 'legal status'],
    },
    r'USPTO|patent.*novelty': {
        'aliases': ['USPTO', 'US Patents'],
        'category': 'Patent Database',
        'keywords': ['patents', 'patent data', 'US patents', 'intellectual property',
                     'patent embeddings', 'patent novelty', 'inventors'],
        'data_types': ['patent metadata', 'patent embeddings'],
    },
    # Preprint servers
    r'arxiv|arXiv': {
        'aliases': ['arXiv', 'arxiv', 'ArXiv'],
        'category': 'Preprint Server Data',
        'keywords': ['preprints', 'pre-prints', 'arxiv papers', 'physics', 'math', 'CS',
                     'computer science preprints', 'scientific preprints', 'open access papers'],
        'data_types': ['preprint metadata', 'abstracts', 'categories'],
    },
    r'biorxiv|bioRxiv': {
        'aliases': ['bioRxiv', 'biorxiv', 'BioRxiv'],
        'category': 'Preprint Server Data',
        'keywords': ['preprints', 'pre-prints', 'biology preprints', 'life sciences',
                     'biomedical preprints', 'open access papers'],
        'data_types': ['preprint metadata', 'abstracts'],
    },
    r'medrxiv|medRxiv': {
        'aliases': ['medRxiv', 'medrxiv', 'MedRxiv'],
        'category': 'Preprint Server Data',
        'keywords': ['preprints', 'pre-prints', 'medical preprints', 'clinical preprints',
                     'health sciences', 'open access papers'],
        'data_types': ['preprint metadata', 'abstracts'],
    },
    r'ssrn': {
        'aliases': ['SSRN', 'ssrn'],
        'category': 'Preprint Server Data',
        'keywords': ['preprints', 'pre-prints', 'social science preprints', 'economics papers',
                     'law papers', 'working papers', 'SSRN papers'],
        'data_types': ['preprint metadata', 'abstracts', 'HTML data'],
    },
    # Biomedical databases
    r'iCite|icite': {
        'aliases': ['iCite', 'NIH iCite'],
        'category': 'Biomedical Citation Metrics',
        'keywords': ['NIH', 'citations', 'biomedical', 'RCR', 'relative citation ratio',
                     'NIH funding', 'grant linkages', 'impact metrics', 'bibliometrics'],
        'data_types': ['citation metrics', 'grant linkages', 'impact scores'],
    },
    r'PKG|pubmed.*knowledge': {
        'aliases': ['PKG', 'PubMed Knowledge Graph'],
        'category': 'Biomedical Knowledge Graph',
        'keywords': ['PubMed', 'NIH', 'biomedical', 'knowledge graph', 'MeSH', 'grants',
                     'clinical trials', 'bioentities', 'gene-paper links', 'drug-paper links'],
        'data_types': ['knowledge graph', 'entity linkages', 'grant-paper links'],
    },
    r'pubchem': {
        'aliases': ['PubChem'],
        'category': 'Chemical Database',
        'keywords': ['chemicals', 'compounds', 'molecules', 'chemical structures',
                     'drug compounds', 'bioassays', 'chemical data'],
        'data_types': ['compound data', 'chemical structures'],
    },
    r'drugbank': {
        'aliases': ['DrugBank'],
        'category': 'Drug Database',
        'keywords': ['drugs', 'pharmaceuticals', 'drug targets', 'drug interactions',
                     'medications', 'pharmacology', 'drug data'],
        'data_types': ['drug metadata', 'drug-target interactions'],
    },
    # Social media data
    r'reddit': {
        'aliases': ['Reddit'],
        'category': 'Social Media Data',
        'keywords': ['social media', 'Reddit', 'subreddits', 'comments', 'posts',
                     'user discussions', 'online communities', 'text data'],
        'data_types': ['posts', 'comments', 'user data'],
    },
    r'telegram': {
        'aliases': ['Telegram'],
        'category': 'Social Media Data',
        'keywords': ['social media', 'Telegram', 'messaging', 'channels', 'groups',
                     'chat data', 'text data'],
        'data_types': ['messages', 'channel data'],
    },
    r'common.?crawl': {
        'aliases': ['Common Crawl', 'CommonCrawl'],
        'category': 'Web Crawl Data',
        'keywords': ['web data', 'web crawl', 'internet archive', 'web pages',
                     'HTML data', 'web text', 'web corpus'],
        'data_types': ['web pages', 'HTML', 'text'],
    },
    # ML/AI datasets
    r'model|huggingface|transformers': {
        'aliases': [],
        'category': 'ML Models',
        'keywords': ['machine learning', 'ML models', 'pretrained models', 'transformers',
                     'neural networks', 'embeddings', 'model weights'],
        'data_types': ['model weights', 'model configs'],
    },
    # Papers with code
    r'PwC|papers.?with.?code': {
        'aliases': ['PwC', 'Papers with Code', 'PapersWithCode'],
        'category': 'ML Benchmark Data',
        'keywords': ['machine learning', 'benchmarks', 'datasets', 'models', 'code',
                     'research papers', 'ML papers', 'AI papers', 'leaderboards'],
        'data_types': ['benchmark data', 'paper-code links'],
    },
}

# Resource type enrichment for general categories
RESOURCE_TYPE_KEYWORDS: Dict[str, List[str]] = {
    'dataset': ['data', 'dataset', 'research data', 'shared data', 'lab data'],
    'project': ['project', 'collaborative', 'shared work', 'team project'],
    'personal': ['researcher directory', 'personal data', 'individual work'],
    'model': ['ML models', 'pretrained', 'model weights', 'neural network'],
}


def get_dataset_enrichment(name: str, description: str = '') -> Dict[str, Any]:
    """
    Get search enrichment keywords for a dataset based on its name and description.

    Returns dict with:
        - aliases: Alternative names for the dataset
        - category: High-level category
        - keywords: Search terms that should match this dataset
    """
    combined_text = f"{name} {description}".lower()

    enrichment = {
        'aliases': [],
        'category': None,
        'keywords': set(),
        'data_types': [],
    }

    for pattern, info in DATASET_KEYWORDS.items():
        if re.search(pattern, combined_text, re.IGNORECASE):
            enrichment['aliases'].extend(info.get('aliases', []))
            if info.get('category') and not enrichment['category']:
                enrichment['category'] = info['category']
            enrichment['keywords'].update(info.get('keywords', []))
            enrichment['data_types'].extend(info.get('data_types', []))

    # Deduplicate
    enrichment['aliases'] = list(set(enrichment['aliases']))
    enrichment['keywords'] = list(enrichment['keywords'])
    enrichment['data_types'] = list(set(enrichment['data_types']))

    return enrichment


def load_index_state() -> Dict[str, Any]:
    """Load the snapshot index state (which snapshots have been indexed)."""
    if INDEX_STATE_FILE.exists():
        with open(INDEX_STATE_FILE) as f:
            return json.load(f)
    return {"indexed_snapshots": [], "last_indexed": None}


def save_index_state(state: Dict[str, Any]) -> None:
    """Save the snapshot index state."""
    with open(INDEX_STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)


def snapshot_to_markdown(snapshot_data: Dict[str, Any], snapshot_date: str) -> str:
    """
    Convert a jevans snapshot JSON to searchable markdown.

    Args:
        snapshot_data: The snapshot JSON data
        snapshot_date: The date string (YYYY-MM-DD)

    Returns:
        Markdown string optimized for RAG indexing with keyword enrichment
    """
    resources = snapshot_data.get('resources', [])

    # Analyze datasets to build category summary
    categories: Dict[str, List[str]] = {}
    for r in resources:
        if r.get('resource_type') == 'dataset':
            enrichment = get_dataset_enrichment(r['name'], r.get('description', '') or '')
            cat = enrichment['category'] or 'Other Data'
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(r['name'])

    md = f"""# KLab Midway Directory Inventory

**Location:** /project/jevans on Midway3
**Scan Date:** {snapshot_date}
**Last Updated:** {snapshot_data.get('scan_timestamp', 'Unknown')}
**Total Resources:** {snapshot_data.get('resource_count', 0)} directories

This document contains an inventory of all directories and datasets available
on Midway3 in the /project/jevans space. Use this to find data, identify owners,
and understand what resources are available.

## Quick Reference - What Data is Available?

The following types of data are available on Midway:

"""

    # Add category summary for quick searching
    if categories:
        for cat, datasets in sorted(categories.items()):
            md += f"**{cat}:** {', '.join(datasets)}\n\n"

    # Add searchable keyword block
    md += """### Common Search Terms

This inventory includes data related to: publications, papers, citations, academic literature,
bibliographic databases, preprints, pre-prints, arxiv, biorxiv, patents, patent data, USPTO,
EPO, PATSTAT, social media, Reddit, web crawl, machine learning, ML models, biomedical,
PubMed, NIH, grants, clinical trials, knowledge graphs, embeddings, research data, datasets.

---

"""

    # Group by type
    by_type: Dict[str, List[Dict]] = {}
    for r in resources:
        rtype = r.get('resource_type', 'unknown')
        if rtype not in by_type:
            by_type[rtype] = []
        by_type[rtype].append(r)

    # Write datasets first, grouped by category for better organization
    if 'dataset' in by_type:
        datasets = by_type['dataset']

        # Group datasets by their enriched category
        by_category: Dict[str, List[Dict]] = {}
        for r in datasets:
            enrichment = get_dataset_enrichment(r['name'], r.get('description', '') or '')
            cat = enrichment['category'] or 'Other Datasets'
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(r)

        # Define category order (most important first)
        category_order = [
            'Bibliographic Database',
            'Bibliographic Database (Historical)',
            'Computer Science Bibliography',
            'Patent Database',
            'Preprint Server Data',
            'Biomedical Citation Metrics',
            'Biomedical Knowledge Graph',
            'Chemical Database',
            'Drug Database',
            'Social Media Data',
            'Web Crawl Data',
            'ML Benchmark Data',
            'ML Models',
            'Other Datasets',
        ]

        md += "## Datasets on Midway\n\n"
        md += "Shared datasets available for research. Organized by category.\n\n"

        # Write in category order
        written_cats = set()
        for cat in category_order:
            if cat in by_category:
                md += f"### {cat}\n\n"
                for r in sorted(by_category[cat], key=lambda x: x.get('size_bytes', 0), reverse=True):
                    md += format_resource(r)
                written_cats.add(cat)

        # Write any remaining categories
        for cat, items in by_category.items():
            if cat not in written_cats:
                md += f"### {cat}\n\n"
                for r in sorted(items, key=lambda x: x.get('size_bytes', 0), reverse=True):
                    md += format_resource(r)

    # Write projects
    if 'project' in by_type:
        md += "## Project Directories\n\n"
        md += "Shared project directories with collaborative research work.\n\n"
        for r in sorted(by_type['project'], key=lambda x: x.get('is_valuable', False), reverse=True):
            md += format_resource(r)

    # Write personal directories with notable data
    if 'personal' in by_type:
        md += "## Researcher Directories\n\n"
        md += "Individual researcher directories. Those marked VALUABLE contain significant data.\n\n"

        # First the valuable ones
        valuable = [r for r in by_type['personal'] if r.get('is_valuable') or r.get('description')]
        other = [r for r in by_type['personal'] if not r.get('is_valuable') and not r.get('description')]

        for r in sorted(valuable, key=lambda x: x.get('is_valuable', False), reverse=True):
            md += format_resource(r)

        # Brief listing of others
        if other:
            md += "### Other Researcher Directories\n\n"
            md += "These directories exist but have no detailed catalog information:\n\n"
            for r in sorted(other, key=lambda x: x.get('name', '')):
                owner = r.get('owner_username', 'unknown')
                md += f"- `{r['name']}` (owner: {owner}, path: `{r['path']}`)\n"
            md += "\n"

    # Write models if present
    if 'model' in by_type:
        md += "## Model Directories\n\n"
        md += "Pre-trained models and model caches for machine learning.\n\n"
        for r in by_type['model']:
            md += format_resource(r)

    return md


def format_resource(r: Dict[str, Any]) -> str:
    """Format a single resource as markdown with search keyword enrichment."""
    name = r['name']
    description = r.get('description', '') or ''

    # Get enrichment for this resource
    enrichment = get_dataset_enrichment(name, description)

    md = f"### {name}\n"

    # Add aliases if available (helps with search)
    if enrichment['aliases']:
        md += f"- **Also known as:** {', '.join(enrichment['aliases'])}\n"

    md += f"- **Path:** `{r['path']}`\n"
    md += f"- **Owner:** {r.get('owner_username', 'unknown')}\n"

    # Size from catalog or computed
    if r.get('catalog_size'):
        size_line = r['catalog_size'].split('\n')[0].strip()
        md += f"- **Size:** {size_line}\n"
    elif r.get('size_bytes'):
        size_gb = r['size_bytes'] / (1024**3)
        if size_gb > 1:
            md += f"- **Size:** {size_gb:.1f} GB\n"
        else:
            size_mb = r['size_bytes'] / (1024**2)
            md += f"- **Size:** {size_mb:.1f} MB\n"

    # Type - prefer enriched category, fall back to catalog type
    if enrichment['category']:
        md += f"- **Category:** {enrichment['category']}\n"
    if r.get('catalog_type'):
        md += f"- **Type:** {r['catalog_type']}\n"

    # Data types available
    if enrichment['data_types']:
        md += f"- **Contains:** {', '.join(enrichment['data_types'])}\n"

    # Status indicators
    status = []
    if r.get('is_valuable'):
        status.append("VALUABLE")
    if r.get('is_active'):
        status.append("ACTIVE")
    if status:
        md += f"- **Status:** {', '.join(status)}\n"

    # Description
    if description:
        desc = description.replace('\n', ' ').strip()[:600]
        md += f"- **Description:** {desc}\n"

    # Search keywords (helps RAG retrieval)
    if enrichment['keywords']:
        # Add as a searchable block
        keywords_str = ', '.join(sorted(enrichment['keywords'])[:15])
        md += f"- **Related terms:** {keywords_str}\n"

    # Last modified
    if r.get('mtime'):
        md += f"- **Last Modified:** {r['mtime'][:10]}\n"

    md += "\n"
    return md


def get_unindexed_snapshots() -> List[Path]:
    """Get list of snapshot files that haven't been indexed yet."""
    state = load_index_state()
    indexed = set(state.get('indexed_snapshots', []))

    unindexed = []
    if SNAPSHOTS_DIR.exists():
        for f in SNAPSHOTS_DIR.glob('snapshot_*.json'):
            if f.name not in indexed:
                unindexed.append(f)

    return sorted(unindexed)


def index_snapshot(snapshot_path: Path, verbose: bool = True) -> Optional[int]:
    """
    Index a single snapshot file.

    Args:
        snapshot_path: Path to the snapshot JSON file
        verbose: Print progress messages

    Returns:
        Number of chunks added, or None if failed
    """
    try:
        # Extract date from filename
        date_str = snapshot_path.stem.replace('snapshot_', '')

        if verbose:
            print(f"Processing {snapshot_path.name}...")

        # Load snapshot
        with open(snapshot_path) as f:
            snapshot_data = json.load(f)

        # Convert to markdown
        md_content = snapshot_to_markdown(snapshot_data, date_str)

        # Save markdown file
        md_path = DATA_DIR / f"midway_inventory_{date_str}.md"
        with open(md_path, 'w') as f:
            f.write(md_content)

        if verbose:
            print(f"  Created {md_path.name} ({len(md_content)} chars)")

        # Index using incremental indexer
        try:
            from incremental_index import IncrementalIndexer

            indexer = IncrementalIndexer()
            if not indexer.load():
                if verbose:
                    print("  No index found, initializing...")
                indexer.initialize()

            chunks_added = indexer.add_file(str(md_path))
            indexer.save()

            # Update state
            state = load_index_state()
            state['indexed_snapshots'].append(snapshot_path.name)
            state['last_indexed'] = datetime.now().isoformat()
            save_index_state(state)

            if verbose:
                print(f"  Indexed: {chunks_added} chunks")

            return chunks_added

        except ImportError as e:
            if verbose:
                print(f"  Warning: incremental_index not available ({e}), markdown saved but not indexed")
            return 0

    except Exception as e:
        if verbose:
            print(f"  Error: {e}")
        return None


def index_all_new_snapshots(verbose: bool = True) -> Dict[str, Any]:
    """
    Index all snapshots that haven't been indexed yet.

    Returns:
        Dict with results summary
    """
    unindexed = get_unindexed_snapshots()

    if not unindexed:
        if verbose:
            print("No new snapshots to index")
        return {"indexed": 0, "chunks": 0, "errors": 0}

    if verbose:
        print(f"Found {len(unindexed)} unindexed snapshot(s)")

    results = {"indexed": 0, "chunks": 0, "errors": 0, "files": []}

    for snapshot_path in unindexed:
        chunks = index_snapshot(snapshot_path, verbose=verbose)
        if chunks is not None:
            results["indexed"] += 1
            results["chunks"] += chunks
            results["files"].append(snapshot_path.name)
        else:
            results["errors"] += 1

    return results


def reindex_all_snapshots(verbose: bool = True) -> Dict[str, Any]:
    """
    Re-index ALL snapshots (including already indexed ones).

    This regenerates markdown with current enrichment settings and re-indexes.
    Useful when the markdown generation logic has been updated.

    Returns:
        Dict with results summary
    """
    if not SNAPSHOTS_DIR.exists():
        if verbose:
            print("No snapshots directory found")
        return {"indexed": 0, "chunks": 0, "errors": 0}

    all_snapshots = sorted(SNAPSHOTS_DIR.glob('snapshot_*.json'))

    if not all_snapshots:
        if verbose:
            print("No snapshots found")
        return {"indexed": 0, "chunks": 0, "errors": 0}

    if verbose:
        print(f"Re-indexing {len(all_snapshots)} snapshot(s) with enhanced keywords...")

    # Clear the index state to force re-indexing
    state = {"indexed_snapshots": [], "last_indexed": None}
    save_index_state(state)

    results = {"indexed": 0, "chunks": 0, "errors": 0, "files": []}

    for snapshot_path in all_snapshots:
        chunks = index_snapshot(snapshot_path, verbose=verbose)
        if chunks is not None:
            results["indexed"] += 1
            results["chunks"] += chunks
            results["files"].append(snapshot_path.name)
        else:
            results["errors"] += 1

    return results


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Index jevans directory snapshots for RAG')
    parser.add_argument('--all', '-a', action='store_true', help='Index all unindexed snapshots')
    parser.add_argument('--file', '-f', type=str, help='Index a specific snapshot file')
    parser.add_argument('--reindex', '-r', action='store_true',
                        help='Re-index ALL snapshots (regenerate markdown with current enrichment)')
    parser.add_argument('--status', '-s', action='store_true', help='Show indexing status')
    parser.add_argument('--quiet', '-q', action='store_true', help='Suppress output')
    args = parser.parse_args()

    verbose = not args.quiet

    if args.status:
        state = load_index_state()
        unindexed = get_unindexed_snapshots()
        print(f"Indexed snapshots: {len(state.get('indexed_snapshots', []))}")
        print(f"Unindexed snapshots: {len(unindexed)}")
        if state.get('last_indexed'):
            print(f"Last indexed: {state['last_indexed']}")
        if unindexed:
            print(f"Pending: {', '.join(f.name for f in unindexed)}")
        return

    if args.reindex:
        results = reindex_all_snapshots(verbose=verbose)
        if verbose:
            print(f"\nRe-index complete: {results['indexed']} indexed, {results['chunks']} chunks, {results['errors']} errors")
        return

    if args.file:
        path = Path(args.file)
        if not path.exists():
            path = SNAPSHOTS_DIR / args.file
        if path.exists():
            index_snapshot(path, verbose=verbose)
        else:
            print(f"File not found: {args.file}")
            sys.exit(1)
    else:
        # Default: index all new snapshots
        results = index_all_new_snapshots(verbose=verbose)
        if verbose:
            print(f"\nSummary: {results['indexed']} indexed, {results['chunks']} chunks, {results['errors']} errors")


if __name__ == "__main__":
    main()
