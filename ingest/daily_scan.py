"""
Daily Scan Runner - Runs the jevans scanner daily and tracks changes.

This script:
1. Scans /project/jevans directory
2. Enriches with catalog data
3. Saves timestamped snapshot
4. Computes diff from previous snapshot
5. Appends changes to change log
6. Updates current state file
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ingest.jevans_scanner import scan_jevans_directory
from ingest.catalog_parser import CatalogParser
from ingest.diff import compute_diff, find_previous_snapshot, load_snapshot, format_changes


# Configuration
DATA_DIR = Path(__file__).parent.parent / "Data"
SNAPSHOTS_DIR = DATA_DIR / "snapshots"
CURRENT_STATE_FILE = DATA_DIR / "jevans_resources.json"
CHANGE_LOG_FILE = DATA_DIR / "change_log.json"


def ensure_directories():
    """Ensure required directories exist."""
    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_change_log():
    """Load existing change log or create empty one."""
    if CHANGE_LOG_FILE.exists():
        with open(CHANGE_LOG_FILE) as f:
            return json.load(f)
    return {"changes": []}


def save_change_log(log):
    """Save change log to file."""
    with open(CHANGE_LOG_FILE, 'w') as f:
        json.dump(log, f, indent=2)


def run_daily_scan(verbose=True):
    """
    Run the daily scan pipeline.

    Args:
        verbose: Print progress messages

    Returns:
        Dict with scan results and changes
    """
    ensure_directories()

    today = datetime.now().strftime('%Y-%m-%d')
    timestamp = datetime.now().isoformat()

    if verbose:
        print("=" * 50)
        print("Jevans Daily Scan - {}".format(today))
        print("=" * 50)

    # Step 1: Scan directory
    if verbose:
        print("\n[1/6] Scanning /project/jevans...")
    results = scan_jevans_directory()
    if verbose:
        print("      Found {} resources".format(results['resource_count']))

    # Step 2: Enrich with catalog
    if verbose:
        print("\n[2/6] Enriching with catalog data...")
    parser = CatalogParser()
    parser.parse()
    results['resources'] = parser.enrich_all(results['resources'])

    # Update summary with enrichment counts
    valuable_count = sum(1 for r in results['resources'] if r.get('is_valuable'))
    active_count = sum(1 for r in results['resources'] if r.get('is_active'))
    results['summary']['valuable_resources'] = valuable_count
    results['summary']['active_resources'] = active_count

    if verbose:
        print("      Valuable: {}, Active: {}".format(valuable_count, active_count))

    # Step 3: Save today's snapshot
    snapshot_path = SNAPSHOTS_DIR / "snapshot_{}.json".format(today)
    if verbose:
        print("\n[3/6] Saving snapshot to {}".format(snapshot_path.name))
    with open(snapshot_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Step 4: Find previous snapshot
    if verbose:
        print("\n[4/6] Finding previous snapshot...")
    prev_snapshot_path = find_previous_snapshot(SNAPSHOTS_DIR, today)
    changes = None

    if prev_snapshot_path:
        if verbose:
            print("      Found: {}".format(prev_snapshot_path.name))

        # Step 5: Compute diff
        if verbose:
            print("\n[5/6] Computing changes...")
        prev_snapshot = load_snapshot(prev_snapshot_path)
        changes = compute_diff(prev_snapshot, results)

        if changes['summary']['has_changes']:
            if verbose:
                print("      Added: {}".format(changes['summary']['added_count']))
                print("      Removed: {}".format(changes['summary']['removed_count']))
                print("      Modified: {}".format(changes['summary']['modified_count']))
                print("      Grown: {}".format(changes['summary']['grown_count']))

            # Append to change log
            change_log = load_change_log()
            change_log['changes'].append(changes)
            change_log['last_updated'] = timestamp
            save_change_log(change_log)
            if verbose:
                print("      Appended to change log")
        else:
            if verbose:
                print("      No changes detected")
    else:
        if verbose:
            print("      No previous snapshot found (first run)")

    # Step 6: Update current state
    if verbose:
        print("\n[6/7] Updating current state file...")
    with open(CURRENT_STATE_FILE, 'w') as f:
        json.dump(results, f, indent=2)

    # Step 7: Auto-index for RAG
    indexed_chunks = 0
    if verbose:
        print("\n[7/7] Indexing snapshot for RAG search...")
    try:
        from ingest.snapshot_indexer import index_snapshot
        chunks = index_snapshot(snapshot_path, verbose=verbose)
        if chunks:
            indexed_chunks = chunks
            if verbose:
                print(f"      Indexed {chunks} chunks for RAG")
    except ImportError as e:
        if verbose:
            print(f"      Skipped (indexer not available): {e}")
    except Exception as e:
        if verbose:
            print(f"      Warning: Indexing failed: {e}")

    if verbose:
        print("\n" + "=" * 50)
        print("Scan complete!")
        print("=" * 50)

    return {
        'results': results,
        'changes': changes,
        'snapshot_path': str(snapshot_path),
        'timestamp': timestamp,
        'indexed_chunks': indexed_chunks
    }


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Run daily jevans directory scan')
    parser.add_argument('--quiet', '-q', action='store_true', help='Suppress output')
    parser.add_argument('--show-changes', action='store_true', help='Print detailed changes')
    args = parser.parse_args()

    result = run_daily_scan(verbose=not args.quiet)

    if args.show_changes and result['changes']:
        print("\n" + format_changes(result['changes']))


if __name__ == "__main__":
    main()
