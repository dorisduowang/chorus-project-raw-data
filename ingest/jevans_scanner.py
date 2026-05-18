"""
Jevans Directory Scanner - Extracts resource metadata from /project/jevans

Scans the lab's shared directory to extract:
- Directory names and paths
- Ownership information
- File sizes and modification times
- Resource type classification

Outputs structured data for the Chorus hypergraph.
"""

import os
import json
import re
from datetime import datetime
from pathlib import Path
import pwd


class ResourceNode:
    """Represents a resource (dataset, model, code, etc.) in the lab directory."""

    def __init__(self, id, name, path, resource_type, owner_username, owner_uid,
                 size_bytes, mtime, description=None, assessment=None,
                 is_valuable=False, subdirectory_count=0, file_count=0):
        self.id = id
        self.name = name
        self.path = path
        self.resource_type = resource_type
        self.owner_username = owner_username
        self.owner_uid = owner_uid
        self.size_bytes = size_bytes
        self.mtime = mtime
        self.description = description
        self.assessment = assessment
        self.is_valuable = is_valuable
        self.subdirectory_count = subdirectory_count
        self.file_count = file_count

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'path': self.path,
            'resource_type': self.resource_type,
            'owner_username': self.owner_username,
            'owner_uid': self.owner_uid,
            'size_bytes': self.size_bytes,
            'mtime': self.mtime,
            'description': self.description,
            'assessment': self.assessment,
            'is_valuable': self.is_valuable,
            'subdirectory_count': self.subdirectory_count,
            'file_count': self.file_count
        }


class OwnershipHyperedge:
    """Represents ownership relationship between person and resource."""

    def __init__(self, id, person_username, resource_id, timestamp, confidence=1.0, evidence="filesystem"):
        self.id = id
        self.person_username = person_username
        self.resource_id = resource_id
        self.timestamp = timestamp
        self.confidence = confidence
        self.evidence = evidence

    def to_dict(self):
        return {
            'id': self.id,
            'person_username': self.person_username,
            'resource_id': self.resource_id,
            'timestamp': self.timestamp,
            'confidence': self.confidence,
            'evidence': self.evidence
        }


class JevansScanner:
    """Scans /project/jevans and extracts resource metadata."""

    # Known data snapshot directories
    DATA_SNAPSHOTS = {
        'openalex-snapshot', 'openalex-snapshot_old',
        'semantic_scholar_Jan31_2025', 'S2AG_Dec_2024_snapshot',
        'MAG_Dec_2021_snapshot', 'DBLP_2025_Jan', 'dimensions',
        'PATSTAT', 'PKG_v2_2023', 'iCite_bulk_snapshot_Sep_2023',
        'drugbank_5.1.13', 'pubchem_compound_2025_May',
        'PwC_Mar_04_2025', 'PwC_Nov_13_2024'
    }

    # Known model directories
    MODEL_DIRS = {'LLMs', 'models', 'hf-cache', 'ollama_models'}

    # Known project directories
    PROJECT_DIRS = {
        'gender_surprise', 'interp_group', 'tip', 'apto_data_engineering',
        'agent_42_pilot', 'crystal_embedding', 'llm_social_simul',
        'QuestionAnswerApproach', 'Wikipedia_rp', 'subspace_sos'
    }

    # System/utility directories to skip
    SKIP_DIRS = {'.', '..', '.claude', '.ipynb_checkpoints', '.vscode',
                 '.virtual_documents', 'Mysql', 'database'}

    def __init__(self, base_path="/project/jevans"):
        self.base_path = Path(base_path)
        self.resources = []
        self.ownership_edges = []
        self.scan_timestamp = datetime.now().isoformat()

    def classify_resource_type(self, name, path):
        """Classify a directory into a resource type."""
        if name in self.DATA_SNAPSHOTS:
            return "dataset"
        if name in self.MODEL_DIRS:
            return "model"
        if name in self.PROJECT_DIRS:
            return "project"

        # Check for common patterns
        name_lower = name.lower()
        if any(x in name_lower for x in ['data', 'snapshot', 'corpus', 'corpora']):
            return "dataset"
        if any(x in name_lower for x in ['model', 'llm', 'embed']):
            return "model"
        if '_' not in name and len(name) > 0 and name[0].isupper():
            # CamelCase names are often personal directories
            return "personal"
        if name.startswith('.'):
            return "system"

        # Default to personal directory
        return "personal"

    def get_owner_info(self, path):
        """Get owner username and uid for a path."""
        try:
            stat_info = path.stat()
            uid = stat_info.st_uid
            try:
                username = pwd.getpwuid(uid).pw_name
            except KeyError:
                username = str(uid)
            return username, uid
        except (OSError, PermissionError):
            return "unknown", -1

    def get_size_and_mtime(self, path):
        """Get size in bytes and modification time for a path."""
        try:
            stat_info = path.stat()
            size = stat_info.st_size
            mtime = datetime.fromtimestamp(stat_info.st_mtime).isoformat()
            return size, mtime
        except (OSError, PermissionError):
            return 0, ""

    def count_contents(self, path):
        """Count subdirectories and files in a directory."""
        subdirs = 0
        files = 0
        try:
            for entry in path.iterdir():
                if entry.is_dir():
                    subdirs += 1
                else:
                    files += 1
        except (OSError, PermissionError):
            pass
        return subdirs, files

    def scan_directory(self, path):
        """Scan a single directory and create a ResourceNode."""
        name = path.name

        if name in self.SKIP_DIRS:
            return None

        owner_username, owner_uid = self.get_owner_info(path)
        size_bytes, mtime = self.get_size_and_mtime(path)
        resource_type = self.classify_resource_type(name, path)
        subdirs, files = self.count_contents(path)

        # Generate a stable ID
        resource_id = "jevans:{}".format(name)

        return ResourceNode(
            id=resource_id,
            name=name,
            path=str(path),
            resource_type=resource_type,
            owner_username=owner_username,
            owner_uid=owner_uid,
            size_bytes=size_bytes,
            mtime=mtime,
            subdirectory_count=subdirs,
            file_count=files
        )

    def scan(self):
        """Scan the entire /project/jevans directory."""
        self.resources = []
        self.ownership_edges = []
        self.scan_timestamp = datetime.now().isoformat()

        try:
            for entry in self.base_path.iterdir():
                if entry.is_dir():
                    resource = self.scan_directory(entry)
                    if resource:
                        self.resources.append(resource)

                        # Create ownership hyperedge
                        edge = OwnershipHyperedge(
                            id="owns:{}:{}".format(resource.owner_username, resource.id),
                            person_username=resource.owner_username,
                            resource_id=resource.id,
                            timestamp=self.scan_timestamp
                        )
                        self.ownership_edges.append(edge)
        except (OSError, PermissionError) as e:
            print("Error scanning {}: {}".format(self.base_path, e))

        return self.to_dict()

    def to_dict(self):
        """Convert scan results to dictionary."""
        return {
            "scan_timestamp": self.scan_timestamp,
            "base_path": str(self.base_path),
            "resource_count": len(self.resources),
            "resources": [r.to_dict() for r in self.resources],
            "ownership_hyperedges": [e.to_dict() for e in self.ownership_edges],
            "summary": self.get_summary()
        }

    def get_summary(self):
        """Get summary statistics of the scan."""
        type_counts = {}
        for r in self.resources:
            type_counts[r.resource_type] = type_counts.get(r.resource_type, 0) + 1

        owner_counts = {}
        for r in self.resources:
            owner_counts[r.owner_username] = owner_counts.get(r.owner_username, 0) + 1

        return {
            "total_resources": len(self.resources),
            "by_type": type_counts,
            "top_owners": dict(sorted(owner_counts.items(), key=lambda x: -x[1])[:10])
        }

    def save(self, output_path):
        """Save scan results to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print("Saved {} resources to {}".format(len(self.resources), output_path))


def scan_jevans_directory(base_path="/project/jevans", output_path=None):
    """
    Convenience function to scan jevans directory.

    Args:
        base_path: Path to scan (default: /project/jevans)
        output_path: Optional path to save JSON output

    Returns:
        Dictionary with scan results
    """
    scanner = JevansScanner(base_path)
    results = scanner.scan()

    if output_path:
        scanner.save(output_path)

    return results


if __name__ == "__main__":
    import sys

    output_file = sys.argv[1] if len(sys.argv) > 1 else "Data/jevans_resources.json"

    print("Scanning /project/jevans...")
    results = scan_jevans_directory(output_path=output_file)

    print("\nScan complete!")
    print("  Resources found: {}".format(results['resource_count']))
    print("  By type: {}".format(results['summary']['by_type']))
    print("  Top owners: {}".format(list(results['summary']['top_owners'].keys())[:5]))
