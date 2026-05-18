"""
Catalog Parser - Parses lab_directory_catalog.txt to extract enrichment data.

The catalog contains curated descriptions, assessments, and value judgments
for each directory in /project/jevans. This parser extracts that information
to enrich the basic metadata from the scanner.
"""

import re
from pathlib import Path


class CatalogEntry:
    """Parsed entry from the lab directory catalog."""

    def __init__(self, index, name, size, owner, entry_type, contents,
                 assessment, is_valuable=False, is_active=False):
        self.index = index
        self.name = name
        self.size = size
        self.owner = owner
        self.entry_type = entry_type
        self.contents = contents
        self.assessment = assessment
        self.is_valuable = is_valuable
        self.is_active = is_active


class CatalogParser:
    """Parses the lab_directory_catalog.txt file."""

    # Patterns for parsing catalog entries
    ENTRY_HEADER = re.compile(
        r'\[(\d+)\]\s+Directory:\s+(\S+)'
    )
    SIZE_PATTERN = re.compile(r'Size:\s*~?([^(]+?)(?:\s*\(|$)')
    OWNER_PATTERN = re.compile(r'Owner:\s*(\S+)')
    TYPE_PATTERN = re.compile(r'Type:\s*(.+?)(?:\n|$)')
    ASSESSMENT_PATTERN = re.compile(r'Assessment:\s*(.+?)(?:\n\n|\Z)', re.DOTALL)

    # Value indicators
    VALUE_KEYWORDS = [
        'VALUABLE DATA', 'ACTIVE RESEARCH', 'MAJOR PROJECT',
        'HIGH VALUE', 'ESSENTIAL', 'COMPREHENSIVE'
    ]

    ACTIVITY_KEYWORDS = [
        'ACTIVE', 'RECENT', 'ONGOING', '2025', '2026'
    ]

    def __init__(self, catalog_path="/project/jevans/lab_directory_catalog.txt"):
        self.catalog_path = Path(catalog_path)
        self.entries = {}  # keyed by directory name

    def parse(self):
        """Parse the catalog file and return entries by directory name."""
        if not self.catalog_path.exists():
            print("Warning: Catalog not found at {}".format(self.catalog_path))
            return {}

        content = self.catalog_path.read_text()
        self.entries = self._parse_content(content)
        return self.entries

    def _parse_content(self, content):
        """Parse the catalog content into entries."""
        entries = {}

        # Split by entry markers
        entry_blocks = re.split(r'\n-{70,}\n', content)

        for block in entry_blocks:
            entry = self._parse_entry(block)
            if entry:
                entries[entry.name] = entry

        return entries

    def _parse_entry(self, block):
        """Parse a single catalog entry block."""
        # Find entry header
        header_match = self.ENTRY_HEADER.search(block)
        if not header_match:
            return None

        index = int(header_match.group(1))
        name = header_match.group(2)

        # Extract size
        size_match = self.SIZE_PATTERN.search(block)
        size = size_match.group(1).strip() if size_match else "Unknown"

        # Extract owner
        owner_match = self.OWNER_PATTERN.search(block)
        owner = owner_match.group(1) if owner_match else "Unknown"

        # Extract type
        type_match = self.TYPE_PATTERN.search(block)
        entry_type = type_match.group(1).strip() if type_match else "Unknown"

        # Extract contents (bullet points)
        contents = re.findall(r'^\s*[-*]\s*(.+)$', block, re.MULTILINE)

        # Extract assessment
        assessment_match = self.ASSESSMENT_PATTERN.search(block)
        assessment = assessment_match.group(1).strip() if assessment_match else ""

        # Determine value and activity
        block_upper = block.upper()
        is_valuable = any(kw in block_upper for kw in self.VALUE_KEYWORDS)
        is_active = any(kw in block_upper for kw in self.ACTIVITY_KEYWORDS)

        return CatalogEntry(
            index=index,
            name=name,
            size=size,
            owner=owner,
            entry_type=entry_type,
            contents=contents,
            assessment=assessment,
            is_valuable=is_valuable,
            is_active=is_active
        )

    def get_entry(self, name):
        """Get catalog entry by directory name."""
        if not self.entries:
            self.parse()
        return self.entries.get(name)

    def enrich_resource(self, resource):
        """Enrich a resource dict with catalog data."""
        name = resource.get('name', '')
        entry = self.get_entry(name)

        if entry:
            resource['description'] = entry.assessment[:500] if entry.assessment else None
            resource['assessment'] = entry.assessment
            resource['is_valuable'] = entry.is_valuable
            resource['catalog_type'] = entry.entry_type
            resource['catalog_size'] = entry.size
            resource['is_active'] = entry.is_active

        return resource

    def enrich_all(self, resources):
        """Enrich a list of resource dicts with catalog data."""
        if not self.entries:
            self.parse()

        return [self.enrich_resource(r) for r in resources]

    def get_valuable_resources(self):
        """Get list of directory names marked as valuable."""
        if not self.entries:
            self.parse()
        return [name for name, entry in self.entries.items() if entry.is_valuable]

    def get_active_resources(self):
        """Get list of directory names marked as active."""
        if not self.entries:
            self.parse()
        return [name for name, entry in self.entries.items() if entry.is_active]


def parse_catalog(catalog_path="/project/jevans/lab_directory_catalog.txt"):
    """
    Convenience function to parse the catalog.

    Args:
        catalog_path: Path to catalog file

    Returns:
        Dictionary mapping directory names to CatalogEntry objects
    """
    parser = CatalogParser(catalog_path)
    return parser.parse()


if __name__ == "__main__":
    print("Parsing lab_directory_catalog.txt...")

    parser = CatalogParser()
    entries = parser.parse()

    print("\nParsed {} entries".format(len(entries)))

    valuable = parser.get_valuable_resources()
    print("\nValuable resources ({}):".format(len(valuable)))
    for name in valuable[:10]:
        print("  - {}".format(name))

    active = parser.get_active_resources()
    print("\nActive resources ({}):".format(len(active)))
    for name in active[:10]:
        print("  - {}".format(name))
