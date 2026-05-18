"""
Registry Text Generator Module

Generates natural language summaries from registry entries.
These can be embedded for semantic search.
"""

from typing import Dict, List, Any


class RegistryTextGenerator:
    """
    Generate natural language summaries from registry entries.
    These can be embedded for semantic search.
    """

    def generate_person_summary(self, person: Dict[str, Any]) -> str:
        """Generate a searchable text summary for a person."""
        parts = []

        name = person.get("name", "Unknown")
        role = person.get("role", "")
        institution = person.get("institution", "")

        parts.append(f"{name} is a {role}" + (f" at {institution}" if institution else "") + ".")

        # Add bibliometric info
        if person.get("openalex"):
            oa = person["openalex"]
            parts.append(f"Publications: {oa.get('works_count', 0)} works, {oa.get('cited_by_count', 0)} citations, h-index {oa.get('h_index', 0)}.")
        elif person.get("google_scholar"):
            gs = person["google_scholar"]
            if gs.get("citations"):
                parts.append(f"Google Scholar citations: {gs['citations']}.")

        if person.get("email"):
            parts.append(f"Contact: {person['email']}")

        return " ".join(parts)

    def generate_project_summary(self, project: Dict[str, Any]) -> str:
        """Generate a searchable text summary for a project."""
        parts = []

        name = project.get("name", "Unknown Project")
        status = project.get("status", "")
        description = project.get("description", "")

        parts.append(f"{name} ({status}).")
        if description:
            parts.append(description)

        if project.get("leads"):
            parts.append(f"Led by: {', '.join(project['leads'])}.")

        if project.get("tags"):
            parts.append(f"Topics: {', '.join(project['tags'])}.")

        return " ".join(parts)

    def generate_dataset_summary(self, dataset: Dict[str, Any]) -> str:
        """Generate a searchable text summary for a dataset."""
        parts = []

        name = dataset.get("name", "Unknown Dataset")
        description = dataset.get("description", "")
        version = dataset.get("version", "")

        header = name
        if version:
            header += f" (v{version})"
        parts.append(f"{header}.")
        if description:
            parts.append(description)

        # Provenance
        prov = dataset.get("provenance", {})
        if prov:
            if prov.get("source"):
                parts.append(f"Source: {prov['source']}.")
            if prov.get("license"):
                parts.append(f"License: {prov['license']}.")

        # Technical
        tech = dataset.get("technical", {})
        if tech:
            details = []
            if tech.get("format"):
                details.append(tech["format"])
            if tech.get("size_gb"):
                details.append(f"{tech['size_gb']} GB")
            if tech.get("num_records"):
                details.append(f"{tech['num_records']:,} records")
            if details:
                parts.append(f"Technical: {', '.join(details)}.")

        # Access
        access = dataset.get("access", {})
        if access and access.get("level"):
            parts.append(f"Access: {access['level']}.")

        # Sensitivity
        sensitivity = dataset.get("sensitivity", {}) or {}
        if sensitivity.get("contains_pii"):
            parts.append("Contains PII.")
        if sensitivity.get("risk_level"):
            parts.append(f"Risk: {sensitivity['risk_level']}.")

        # Tags
        if dataset.get("tags"):
            parts.append(f"Topics: {', '.join(dataset['tags'])}.")

        return " ".join(parts)

    def generate_funding_summary(self, funding: Dict[str, Any]) -> str:
        """Generate a searchable text summary for a funding source."""
        parts = []

        name = funding.get("name", "Unknown Grant")
        source = funding.get("source", "")
        status = funding.get("status", "")

        parts.append(f"{name} from {source} ({status}).")

        if funding.get("amount"):
            parts.append(f"Amount: ${funding['amount']:,}.")

        if funding.get("pi"):
            parts.append(f"PI: {funding['pi']}.")

        return " ".join(parts)

    def generate_all_summaries(self, registry: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate text summaries for all registry entries."""
        summaries = []

        # People
        if "people" in registry and "members" in registry["people"]:
            for person in registry["people"]["members"]:
                summaries.append({
                    "entity_type": "people",
                    "entity_id": person.get("id", ""),
                    "text": self.generate_person_summary(person),
                    "source": "lab_registry.json"
                })

        # Projects
        for project in registry.get("projects", []):
            summaries.append({
                "entity_type": "projects",
                "entity_id": project.get("id", ""),
                "text": self.generate_project_summary(project),
                "source": "lab_registry.json"
            })

        # Datasets
        for dataset in registry.get("datasets", []):
            summaries.append({
                "entity_type": "datasets",
                "entity_id": dataset.get("id", ""),
                "text": self.generate_dataset_summary(dataset),
                "source": "lab_registry.json"
            })

        # Funding
        for funding in registry.get("funding", []):
            summaries.append({
                "entity_type": "funding",
                "entity_id": funding.get("id", ""),
                "text": self.generate_funding_summary(funding),
                "source": "lab_registry.json"
            })

        return summaries
