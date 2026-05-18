"""
Resource Query Interface - Query jevans directory resources.

Provides search and filtering capabilities for lab resources
extracted from the /project/jevans directory.
"""

import json
from pathlib import Path


class ResourceRegistry:
    """Query interface for jevans directory resources."""

    def __init__(self, data_path="Data/jevans_resources.json"):
        self.data_path = Path(data_path)
        self.data = None
        self.resources = []
        self._load()

    def _load(self):
        """Load resource data from JSON file."""
        if self.data_path.exists():
            with open(self.data_path) as f:
                self.data = json.load(f)
                self.resources = self.data.get('resources', [])
        else:
            print("Warning: Resource data not found at {}".format(self.data_path))
            self.data = {}
            self.resources = []

    def reload(self):
        """Reload data from disk."""
        self._load()

    def get_all(self):
        """Get all resources."""
        return self.resources

    def get_by_type(self, resource_type):
        """Get resources by type (dataset, model, project, personal, etc.)."""
        return [r for r in self.resources if r.get('resource_type') == resource_type]

    def get_by_owner(self, username):
        """Get resources owned by a specific user."""
        return [r for r in self.resources if r.get('owner_username') == username]

    def get_valuable(self):
        """Get resources marked as valuable in the catalog."""
        return [r for r in self.resources if r.get('is_valuable')]

    def get_active(self):
        """Get resources marked as active in the catalog."""
        return [r for r in self.resources if r.get('is_active')]

    def search(self, query):
        """Search resources by name or description."""
        query_lower = query.lower()
        results = []
        for r in self.resources:
            name = r.get('name', '').lower()
            desc = (r.get('description') or '').lower()
            assessment = (r.get('assessment') or '').lower()
            if query_lower in name or query_lower in desc or query_lower in assessment:
                results.append(r)
        return results

    def get_datasets(self):
        """Get all datasets."""
        return self.get_by_type('dataset')

    def get_models(self):
        """Get all models."""
        return self.get_by_type('model')

    def get_projects(self):
        """Get all project directories."""
        return self.get_by_type('project')

    def get_summary(self):
        """Get summary statistics."""
        return self.data.get('summary', {})

    def get_ownership_edges(self):
        """Get ownership hyperedges."""
        return self.data.get('ownership_hyperedges', [])

    def format_resource(self, resource):
        """Format a resource for display."""
        lines = []
        lines.append("Name: {}".format(resource.get('name')))
        lines.append("Type: {}".format(resource.get('resource_type')))
        lines.append("Owner: {}".format(resource.get('owner_username')))
        lines.append("Path: {}".format(resource.get('path')))

        if resource.get('is_valuable'):
            lines.append("Status: VALUABLE")
        if resource.get('is_active'):
            lines.append("Activity: ACTIVE")

        desc = resource.get('description')
        if desc:
            lines.append("Description: {}...".format(desc[:200]))

        return "\n".join(lines)

    def format_list(self, resources, max_items=10):
        """Format a list of resources for display."""
        lines = []
        for i, r in enumerate(resources[:max_items]):
            valuable = " [VALUABLE]" if r.get('is_valuable') else ""
            lines.append("{:3}. {} ({}){}".format(
                i+1, r.get('name'), r.get('resource_type'), valuable))

        if len(resources) > max_items:
            lines.append("... and {} more".format(len(resources) - max_items))

        return "\n".join(lines)


# Convenience functions
_registry = None


def get_registry():
    """Get singleton registry instance."""
    global _registry
    if _registry is None:
        _registry = ResourceRegistry()
    return _registry


def search_resources(query):
    """Search resources by query."""
    return get_registry().search(query)


def get_datasets():
    """Get all datasets."""
    return get_registry().get_datasets()


def get_models():
    """Get all models."""
    return get_registry().get_models()


def get_valuable_resources():
    """Get valuable resources."""
    return get_registry().get_valuable()


if __name__ == "__main__":
    registry = ResourceRegistry()

    print("=== Resource Registry ===\n")

    summary = registry.get_summary()
    print("Total resources: {}".format(summary.get('total_resources', 0)))
    print("By type: {}".format(summary.get('by_type', {})))
    print("")

    print("=== Datasets ===")
    datasets = registry.get_datasets()
    print(registry.format_list(datasets))
    print("")

    print("=== Models ===")
    models = registry.get_models()
    print(registry.format_list(models))
    print("")

    print("=== Valuable Resources ===")
    valuable = registry.get_valuable()
    print(registry.format_list(valuable, max_items=15))
    print("")

    print("=== Search: 'patent' ===")
    patent_results = registry.search('patent')
    print(registry.format_list(patent_results))
