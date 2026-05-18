# Lab Registry Guide

The lab registry (`lab_registry.json`) is a structured database of everything in the lab. It enables Chorus to answer questions like:
- "Who works on project X?"
- "What datasets do we have for Y?"
- "What compute resources are available?"
- "Why did we decide to do Z?"

## Quick Start: Adding Data

### Add a Project

```json
{
  "id": "my-project",
  "name": "My New Project",
  "status": "active",
  "description": "Brief description of the project",
  "start_date": "2025-01",
  "funding": ["grant-id"],
  "leads": ["Person Name"],
  "members": ["Member 1", "Member 2"],
  "papers": [],
  "code_repos": ["repo-id"],
  "datasets": ["dataset-id"],
  "tags": ["topic1", "topic2"]
}
```

### Add a Dataset

```json
{
  "id": "my-dataset",
  "name": "Dataset Name",
  "description": "What this data contains",
  "format": "parquet",
  "size_gb": 50,
  "access_level": "internal",
  "location": "/path/to/data or URL",
  "created_by": "Person Name",
  "projects": ["project-id"],
  "papers": [],
  "tags": ["patent data", "NLP"]
}
```

### Add a Code Repository

```json
{
  "id": "my-repo",
  "name": "Repository Name",
  "description": "What this code does",
  "url": "https://github.com/org/repo",
  "language": "Python",
  "status": "active",
  "maintainers": ["Person Name"],
  "projects": ["project-id"],
  "tags": ["ML", "data processing"]
}
```

### Add a Funding Source

```json
{
  "id": "my-grant",
  "name": "Grant Name",
  "source": "NSF",
  "type": "research_grant",
  "amount": 500000,
  "currency": "USD",
  "pi": "PI Name",
  "co_pis": ["Co-PI 1"],
  "start_date": "2025-01",
  "end_date": "2028-01",
  "status": "active",
  "projects": ["project-id"],
  "documents": ["path/to/proposal.pdf"]
}
```

### Add a Compute Resource

```json
{
  "id": "my-cluster",
  "name": "Cluster Name",
  "type": "hpc_cluster",
  "provider": "Provider Name",
  "description": "What this resource is for",
  "access": "how to get access",
  "specs": {
    "gpu_nodes": 10,
    "gpu_type": "A100",
    "storage_tb": 100
  },
  "documentation": "https://docs.example.com",
  "contact": "admin@example.com"
}
```

### Document a Decision (Institutional Memory)

```json
{
  "id": "decision-2025-01-embeddings",
  "date": "2025-01-15",
  "title": "Switched from OpenAI to BGE embeddings",
  "description": "Choice of embedding model for Chorus RAG",
  "context": "OpenAI embeddings were expensive and had rate limits",
  "options_considered": [
    "OpenAI text-embedding-3-large",
    "BGE-large-en-v1.5",
    "Cohere embeddings"
  ],
  "decision": "Use BGE-large-en-v1.5",
  "rationale": "Free, fast, comparable quality, runs locally",
  "made_by": ["Person Name"],
  "related_projects": ["chorus"],
  "source_docs": ["slack thread link or meeting notes"]
}
```

## Entity Types

| Entity | Purpose | Key Fields |
|--------|---------|------------|
| `projects` | Research initiatives | name, status, leads, funding |
| `funding` | Grants and contracts | source, amount, PI, dates |
| `papers` | Publications (drafts/published) | title, authors, status, venue |
| `datasets` | Data assets | format, size, access_level, location |
| `code_repos` | GitHub repositories | url, language, maintainers |
| `compute` | Computing resources | type, specs, access |
| `events` | Meetings, seminars, conferences | type, date, recordings |
| `tools` | External services/databases | type, url, access |
| `decisions` | Why we chose X over Y | context, options, rationale |

## Linking Entities

Use IDs to link entities together:

```
Project "apto"
  → funded by Funding "nsf-apto"
  → uses Dataset "patent-abstracts"
  → code in Repo "apto-models"
  → led by Person "James Evans"
```

This creates a queryable knowledge graph.

## Access Levels

For datasets and sensitive info:
- `public` - Anyone can access
- `internal` - Lab members only
- `restricted` - Specific people only
- `confidential` - Requires explicit approval

## Status Values

Projects: `active`, `completed`, `paused`, `proposed`
Funding: `active`, `completed`, `submitted`, `planned`, `rejected`
Repos: `active`, `archived`, `deprecated`

## Validation

The schema is defined in `lab_registry_schema.json`. Validate with:

```bash
python3 -c "
import json
from jsonschema import validate
schema = json.load(open('Data/lab_registry_schema.json'))
data = json.load(open('Data/lab_registry.json'))
validate(data, schema)
print('Valid!')
"
```

## Integration with Chorus

The registry is used by Chorus for:
1. **Structured queries** - "List all active projects" → direct lookup
2. **Context enrichment** - When discussing a paper, pull related project/funding info
3. **People profiles** - Publication data is stored directly in `people.members[].openalex`
4. **Institutional memory** - Surface relevant decisions when similar topics arise

## Updating the Registry

1. Edit `lab_registry.json` directly
2. Run validation to check for errors
3. Re-index if needed for Chorus to pick up changes

For bulk updates or imports, contact the Chorus maintainer.
