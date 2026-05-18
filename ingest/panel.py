"""
Panel Query Interface - Query across multiple snapshots over time.

Provides time-series analysis of the jevans directory:
- Load snapshots in date range
- Track resource changes over time
- Summarize activity patterns
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict


class SnapshotPanel:
    """Query interface for multiple snapshots over time."""

    def __init__(self, snapshots_dir="Data/snapshots", change_log_path="Data/change_log.json"):
        self.snapshots_dir = Path(snapshots_dir)
        self.change_log_path = Path(change_log_path)
        self._snapshots = {}  # Cache: date -> snapshot
        self._change_log = None

    def list_snapshots(self):
        """List all available snapshot dates."""
        if not self.snapshots_dir.exists():
            return []

        dates = []
        for f in sorted(self.snapshots_dir.glob('snapshot_*.json')):
            date_str = f.stem.replace('snapshot_', '')
            dates.append(date_str)
        return dates

    def load_snapshot(self, date):
        """Load a snapshot by date (YYYY-MM-DD)."""
        if date in self._snapshots:
            return self._snapshots[date]

        path = self.snapshots_dir / "snapshot_{}.json".format(date)
        if not path.exists():
            return None

        with open(path) as f:
            snapshot = json.load(f)
        self._snapshots[date] = snapshot
        return snapshot

    def load_range(self, start_date, end_date):
        """
        Load all snapshots in date range.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            Dict mapping dates to snapshots
        """
        snapshots = {}
        for date in self.list_snapshots():
            if start_date <= date <= end_date:
                snapshot = self.load_snapshot(date)
                if snapshot:
                    snapshots[date] = snapshot
        return snapshots

    def load_change_log(self):
        """Load the change log."""
        if self._change_log is not None:
            return self._change_log

        if not self.change_log_path.exists():
            self._change_log = {"changes": []}
            return self._change_log

        with open(self.change_log_path) as f:
            self._change_log = json.load(f)
        return self._change_log

    def get_changes_since(self, date):
        """
        Get all changes since a specific date.

        Args:
            date: Start date (YYYY-MM-DD)

        Returns:
            List of change records
        """
        log = self.load_change_log()
        return [c for c in log.get('changes', []) if c.get('date', '') >= date]

    def resource_timeline(self, resource_name):
        """
        Get history of a specific resource across all snapshots.

        Args:
            resource_name: Name of the resource

        Returns:
            List of (date, resource_dict) tuples
        """
        timeline = []
        for date in self.list_snapshots():
            snapshot = self.load_snapshot(date)
            if snapshot:
                for r in snapshot.get('resources', []):
                    if r.get('name') == resource_name:
                        timeline.append((date, r))
                        break
        return timeline

    def activity_summary(self, days=7):
        """
        Summarize activity over last N days.

        Args:
            days: Number of days to look back

        Returns:
            Summary dict with aggregated changes
        """
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

        changes = self.get_changes_since(start_date)

        summary = {
            'period': {'start': start_date, 'end': end_date, 'days': days},
            'total_added': [],
            'total_removed': [],
            'total_modified': set(),
            'growth_by_resource': defaultdict(int),
            'most_active_resources': []
        }

        for change in changes:
            summary['total_added'].extend(change.get('added', []))
            summary['total_removed'].extend(change.get('removed', []))
            summary['total_modified'].update(change.get('modified', []))

            for grown in change.get('grown', []):
                summary['growth_by_resource'][grown['name']] += grown.get('delta', 0)

        # Convert set to list for JSON serialization
        summary['total_modified'] = list(summary['total_modified'])

        # Find most active resources (modified most often)
        modification_counts = defaultdict(int)
        for change in changes:
            for name in change.get('modified', []):
                modification_counts[name] += 1

        summary['most_active_resources'] = sorted(
            modification_counts.items(),
            key=lambda x: -x[1]
        )[:10]

        # Convert growth dict to sorted list
        summary['growth_by_resource'] = sorted(
            summary['growth_by_resource'].items(),
            key=lambda x: -x[1]
        )[:10]

        return summary

    def format_summary(self, summary):
        """Format activity summary for display."""
        lines = []
        lines.append("=== Activity Summary ===")
        lines.append("Period: {} to {} ({} days)".format(
            summary['period']['start'],
            summary['period']['end'],
            summary['period']['days']
        ))
        lines.append("")

        if summary['total_added']:
            lines.append("New resources ({}):".format(len(summary['total_added'])))
            for name in summary['total_added'][:5]:
                lines.append("  + {}".format(name))
            if len(summary['total_added']) > 5:
                lines.append("  ... and {} more".format(len(summary['total_added']) - 5))
            lines.append("")

        if summary['total_removed']:
            lines.append("Removed resources ({}):".format(len(summary['total_removed'])))
            for name in summary['total_removed'][:5]:
                lines.append("  - {}".format(name))
            lines.append("")

        if summary['most_active_resources']:
            lines.append("Most active resources:")
            for name, count in summary['most_active_resources'][:5]:
                lines.append("  {} ({} modifications)".format(name, count))
            lines.append("")

        if summary['growth_by_resource']:
            lines.append("Largest growth:")
            for name, delta in summary['growth_by_resource'][:5]:
                delta_mb = delta / (1024 * 1024)
                lines.append("  {} (+{:.1f} MB)".format(name, delta_mb))
            lines.append("")

        if not any([summary['total_added'], summary['total_removed'],
                    summary['most_active_resources']]):
            lines.append("No activity recorded in this period.")

        return "\n".join(lines)

    def get_current_stats(self):
        """Get stats from most recent snapshot."""
        dates = self.list_snapshots()
        if not dates:
            return None

        latest = self.load_snapshot(dates[-1])
        if not latest:
            return None

        return {
            'date': dates[-1],
            'total_resources': latest.get('resource_count', 0),
            'by_type': latest.get('summary', {}).get('by_type', {}),
            'valuable': latest.get('summary', {}).get('valuable_resources', 0),
            'active': latest.get('summary', {}).get('active_resources', 0)
        }


def main():
    """Demo the panel interface."""
    panel = SnapshotPanel()

    print("Available snapshots:")
    for date in panel.list_snapshots():
        print("  - {}".format(date))
    print("")

    # Current stats
    stats = panel.get_current_stats()
    if stats:
        print("Current stats ({}):".format(stats['date']))
        print("  Total resources: {}".format(stats['total_resources']))
        print("  By type: {}".format(stats['by_type']))
        print("  Valuable: {}".format(stats['valuable']))
        print("")

    # Activity summary
    summary = panel.activity_summary(days=7)
    print(panel.format_summary(summary))


if __name__ == "__main__":
    main()
