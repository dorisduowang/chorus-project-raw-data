"""
Diff module - Computes changes between snapshots.

Detects:
- New directories (added)
- Removed directories (deleted)
- Modified directories (mtime changed)
- Size changes (grown/shrunk)
"""

import json
from datetime import datetime
from pathlib import Path


def compute_diff(old_snapshot, new_snapshot):
    """
    Compute differences between two snapshots.

    Args:
        old_snapshot: Previous snapshot dict
        new_snapshot: Current snapshot dict

    Returns:
        Dict with change categories
    """
    today = datetime.now().strftime('%Y-%m-%d')

    changes = {
        'date': today,
        'old_timestamp': old_snapshot.get('scan_timestamp', ''),
        'new_timestamp': new_snapshot.get('scan_timestamp', ''),
        'added': [],
        'removed': [],
        'modified': [],
        'grown': [],
        'shrunk': [],
        'owner_changed': []
    }

    # Build lookup dicts
    old_resources = {r['name']: r for r in old_snapshot.get('resources', [])}
    new_resources = {r['name']: r for r in new_snapshot.get('resources', [])}

    old_names = set(old_resources.keys())
    new_names = set(new_resources.keys())

    # Detect additions and removals
    changes['added'] = sorted(list(new_names - old_names))
    changes['removed'] = sorted(list(old_names - new_names))

    # Detect modifications in common resources
    common = old_names & new_names
    for name in common:
        old_r = old_resources[name]
        new_r = new_resources[name]

        # Check mtime change
        if new_r.get('mtime', '') != old_r.get('mtime', ''):
            changes['modified'].append(name)

        # Check size changes
        old_size = old_r.get('size_bytes', 0)
        new_size = new_r.get('size_bytes', 0)
        if new_size > old_size:
            changes['grown'].append({
                'name': name,
                'old_size': old_size,
                'new_size': new_size,
                'delta': new_size - old_size
            })
        elif new_size < old_size:
            changes['shrunk'].append({
                'name': name,
                'old_size': old_size,
                'new_size': new_size,
                'delta': old_size - new_size
            })

        # Check owner changes
        if new_r.get('owner_username') != old_r.get('owner_username'):
            changes['owner_changed'].append({
                'name': name,
                'old_owner': old_r.get('owner_username'),
                'new_owner': new_r.get('owner_username')
            })

    # Add summary
    changes['summary'] = {
        'added_count': len(changes['added']),
        'removed_count': len(changes['removed']),
        'modified_count': len(changes['modified']),
        'grown_count': len(changes['grown']),
        'shrunk_count': len(changes['shrunk']),
        'has_changes': any([
            changes['added'], changes['removed'],
            changes['modified'], changes['grown'], changes['shrunk']
        ])
    }

    return changes


def load_snapshot(path):
    """Load a snapshot from JSON file."""
    with open(path) as f:
        return json.load(f)


def find_previous_snapshot(snapshots_dir, current_date=None):
    """
    Find the most recent snapshot before current_date.

    Args:
        snapshots_dir: Path to snapshots directory
        current_date: Date string (YYYY-MM-DD) or None for today

    Returns:
        Path to previous snapshot or None
    """
    snapshots_path = Path(snapshots_dir)
    if not snapshots_path.exists():
        return None

    if current_date is None:
        current_date = datetime.now().strftime('%Y-%m-%d')

    # Find all snapshots
    snapshot_files = sorted(snapshots_path.glob('snapshot_*.json'), reverse=True)

    for snapshot_file in snapshot_files:
        # Extract date from filename
        date_str = snapshot_file.stem.replace('snapshot_', '')
        if date_str < current_date:
            return snapshot_file

    return None


def format_changes(changes):
    """Format changes for human-readable output."""
    lines = []
    lines.append("=== Jevans Directory Changes ===")
    lines.append("From: {}".format(changes.get('old_timestamp', 'N/A')))
    lines.append("To:   {}".format(changes.get('new_timestamp', 'N/A')))
    lines.append("")

    if changes['added']:
        lines.append("ADDED ({})".format(len(changes['added'])))
        for name in changes['added']:
            lines.append("  + {}".format(name))
        lines.append("")

    if changes['removed']:
        lines.append("REMOVED ({})".format(len(changes['removed'])))
        for name in changes['removed']:
            lines.append("  - {}".format(name))
        lines.append("")

    if changes['modified']:
        lines.append("MODIFIED ({})".format(len(changes['modified'])))
        for name in changes['modified']:
            lines.append("  ~ {}".format(name))
        lines.append("")

    if changes['grown']:
        lines.append("GROWN ({})".format(len(changes['grown'])))
        for item in changes['grown']:
            delta_mb = item['delta'] / (1024 * 1024)
            lines.append("  ^ {} (+{:.1f} MB)".format(item['name'], delta_mb))
        lines.append("")

    if not changes['summary']['has_changes']:
        lines.append("No changes detected.")

    return "\n".join(lines)


if __name__ == "__main__":
    # Test with existing snapshots
    import sys

    snapshots_dir = "Data/snapshots"

    if len(sys.argv) == 3:
        old_path = sys.argv[1]
        new_path = sys.argv[2]
    else:
        # Find most recent two snapshots
        snapshot_files = sorted(Path(snapshots_dir).glob('snapshot_*.json'))
        if len(snapshot_files) < 2:
            print("Need at least 2 snapshots to compute diff")
            sys.exit(1)
        old_path = snapshot_files[-2]
        new_path = snapshot_files[-1]

    old_snap = load_snapshot(old_path)
    new_snap = load_snapshot(new_path)

    changes = compute_diff(old_snap, new_snap)
    print(format_changes(changes))
