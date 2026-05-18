# Chorus Project Progress Summary

*Last Updated: January 13, 2026 (v2 - with improvements)*

## Overview

Chorus is a knowledge management and retrieval system for Knowledge Lab at the University of Chicago. It provides structured data access, semantic search, and a directed hypergraph model for understanding relationships between people, publications, datasets, and computational resources.

---

## Architecture

```
chorus/
├── registry/                 # Structured data layer
│   ├── classifier.py         # Query classification (70+ patterns)
│   ├── lookup.py             # RegistryLookup + MidwayLookup classes
│   ├── rag.py                # RegistryRAG with hybrid search
│   ├── cross_links.py        # NEW: Cross-registry linking
│   ├── query_suggest.py      # NEW: Query suggestions & typo correction
│   └── topic_aliases.py      # Research topic normalization
├── hypergraph/               # Directed hypergraph model
│   ├── models.py             # HyperNode, HyperEdge, HyperGraph
│   ├── storage.py            # JSON persistence with hot-reload
│   ├── build.py              # Build script
│   └── builders/             # Data source builders
│       ├── openalex.py       # Publications → hyperedges
│       ├── registry.py       # Projects, funding → hyperedges
│       └── relationships.py  # NEW: Co-authorship, topic, usage edges
├── Data/
│   ├── lab_registry.json     # 71 lab members, 482 publications
│   ├── midway_registry.json  # Midway storage catalog
│   └── hypergraph.json       # 1,312 nodes, 594 edges
├── rag_http_server.py        # HTTP API server (port 8765) + hypergraph API
└── chat.py                   # Interactive chat interface
```

---

## Data Sources

### Lab Registry (`lab_registry.json`)
- **71 lab members** with profiles including:
  - Role, institution, email
  - OpenAlex data (publications, citations, h-index, topics)
  - Google Scholar data (where available)
  - Midway directory information (16 members)
- **482 publications** with DOIs, venues, citation counts
- **3 active projects** (APTO, C3S2, Socio-Cognitive AI)
- **3 funding sources** (NSF, MURI, NNF)
- **10 datasets** with access levels and documentation
- **6 compute resources** (Midway3, Knowledge Garden, APIs)
- **8 tools** (SLURM, PySpark, Jupyter, etc.)

### Midway Registry (`midway_registry.json`)
- **12 data snapshots**: OpenAlex (Dec 2024), Semantic Scholar (Jan 2025), PATSTAT, USPTO, MAG, DBLP, PubMed, etc.
- **7 precomputed resources**: Patent similarity matrices (695GB), PMI statistics (122GB), concept co-occurrence data
- **9 embedding collections**: SPECTER2, Word2Vec, mat2vec, hyperbolic models, patent/paper/MeSH embeddings
- **6 social media datasets**: Reddit (multiple), 4chan, Weibo, Telegram
- **25 researcher directories** with storage estimates and research focus
- **6 infrastructure items**: LLM caches, shared models, Ollama

### Hypergraph (`hypergraph.json`)
- **1,312 nodes** by type:
  - Person: 71
  - Document: 467
  - Concept: 470 (+52 shared research topics)
  - Venue: 242
  - Dataset: 29 (+19 from Midway)
  - Project: 6
  - Code: 1
  - Institution: 26
- **594 edges** by type (+121 from relationships builder):
  - Publication: 467
  - Collaboration: 86 (co-authorship strength + topic + affiliation)
  - Funding: 3
  - Usage: 38 (person-to-dataset links)
- **Time range**: 1986 to 2026

---

## Search Capabilities

### Query Classification
The classifier routes queries to appropriate handlers using:
- **50+ regex patterns** for structured queries
- **Entity keyword matching** across 15+ categories
- **Compound query detection** (e.g., "Stanford researchers working on NLP")

### Supported Query Types

| Category | Example Queries |
|----------|----------------|
| People | "Who works on network science?", "Tell me about James Evans" |
| Publications | "Papers about machine learning", "What has Jake published?" |
| Topics | "Who researches computational social science?" |
| Datasets | "Where is the OpenAlex data?", "Do we have Weibo data?" |
| Compute | "What GPUs are available?", "How do I use SLURM?" |
| Midway Snapshots | "Latest Semantic Scholar snapshot", "Where is PATSTAT?" |
| Embeddings | "What word2vec embeddings exist?", "Patent embeddings?" |
| Social Media | "Reddit toxicity data", "4chan datasets" |
| Infrastructure | "LLM cache location", "Shared models" |

### Search Features
- **Hot-reload**: Both registries automatically reload when files change (10s check interval)
- **Fallback search**: When primary classification fails, searches all Midway categories for keyword matches
- **Hybrid routing**: Queries can match multiple entity types simultaneously

---

## Performance

### RAG Server Stress Test (50 queries, post-improvements)
```
Total queries:     50
Successes:         50 (100%)
Failures:          0
Throughput:        0.6 queries/second (includes reranking)
```

### Query Response Times
- Structured lookup: < 5ms
- Classification: < 1ms
- Fallback search: < 10ms
- Hypergraph queries: < 10ms

### Server Stability
- Single-threaded but stable for interactive use
- All platform-specific queries (weibo, reddit, twitter) working correctly

---

## Hypergraph Analytics

### Terminal Nodes Analysis
- **467 terminal nodes** identified (outputs never reused as inputs)
- All are Document nodes (publications)
- Indicates papers are being produced but not yet feeding into subsequent work

### Available Queries
```python
graph.what_produced(person_id)      # Documents authored
graph.collaborators_of(person_id)   # Co-authors
graph.who_worked_with(node_id)      # People connected to any node
graph.terminal_nodes()              # Dead-end outputs
graph.edges_in_range(start, end)    # Temporal filtering
```

### HTTP API Endpoints (NEW)
```bash
# Get hypergraph statistics
curl "http://localhost:8765/hypergraph/stats"

# Find collaborators of a person
curl "http://localhost:8765/hypergraph/collaborators?person=james-evans"

# Find outputs of a person
curl "http://localhost:8765/hypergraph/outputs?person=james-evans"

# Find terminal nodes (dead ends)
curl "http://localhost:8765/hypergraph/terminal"

# Search nodes by name
curl "http://localhost:8765/hypergraph/search?q=network"
```

---

## What's Working Well

1. **Structured Query Routing**: Specific queries like "Do we have Weibo data?" correctly route to `midway_social` and return precise results

2. **Hot-Reload**: Changes to registry files are automatically detected and loaded without server restart

3. **Comprehensive Midway Catalog**: 60+ data assets cataloged with paths, sizes, owners, and descriptions

4. **Hypergraph Foundation**: Core model built with 1,312 nodes and 594 edges from structured sources

5. **Multi-Category Search**: Queries can span people, publications, datasets, and compute resources

6. **Fallback Search**: When classification misses, keyword-based fallback finds relevant Midway resources

7. **Cross-Registry Linking** (NEW): Automatically links people to their Midway resources

8. **Query Suggestions** (NEW): Provides helpful suggestions when no results found

9. **Hypergraph API** (NEW): HTTP endpoints for collaborator lookup, output tracking, and analytics

---

## Weak Points

### 1. ~~Query Classification Gaps~~ RESOLVED
- **Status**: Added 20+ new patterns for social media, toxicity, platform-specific queries
- **Now works**: "weibo data", "toxicity data", "reddit datasets" all route correctly

### 2. HTTP Server Limitations
- **Problem**: Python's `http.server` is single-threaded
- **Status**: Now achieves 100% success rate in stress tests (50 queries)
- **Workaround**: Works well for typical usage

### 3. ~~Hypergraph Sparsity~~ IMPROVED
- **Status**: Added relationships builder (+121 edges, now 594 total)
- **New edges**: Co-authorship strength, topic collaboration, institution affiliation, dataset usage
- **Still missing**: LLM-extracted discussion/computation edges

### 4. ~~No Semantic Search Integration~~ IMPROVED
- **Status**: Hypergraph now integrated with RAG server
- **New endpoints**: `/hypergraph/stats`, `/hypergraph/collaborators`, `/hypergraph/outputs`, `/hypergraph/terminal`, `/hypergraph/search`
- **Future**: Could add hypergraph-based relevance boosting

### 5. ~~Limited Cross-Registry Linking~~ RESOLVED
- **Status**: Created `cross_links.py` module
- **Now works**: "What data does Jake use?" returns linked resources
- **Features**: Automatic linking based on directory ownership and research focus

### 6. ~~No Query Refinement~~ RESOLVED
- **Status**: Created `query_suggest.py` module
- **Features**: Typo detection, alternative query suggestions, helpful "no results" messages

---

## Potential Improvements

### Short-Term (Low Effort)

1. **Add More Classifier Patterns**
   - Cover more phrasings: "Is there any X?", "Can I access X?", "Who has X?"
   - Add specific dataset names as keywords

2. **Upgrade to WSGI Server**
   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:8765 rag_http_server:app
   ```
   - Eliminates connection drops, enables parallel requests

3. **Add Query Suggestions**
   - When no results, suggest: "Did you mean: [similar queries]?"
   - Use edit distance or keyword overlap

4. **Expand Fallback Keywords**
   - Add all dataset names, researcher usernames, tool names
   - Make fallback more aggressive

### Medium-Term (Moderate Effort)

5. **Cross-Registry Queries**
   - Link people to their Midway directories automatically
   - Enable: "What data does [person] work with?"

6. **Hypergraph LLM Extraction**
   - Extract relationships from document chunks
   - Add discussion, computation, derivation edges
   - Target: 5,000+ edges

7. **Semantic Midway Search**
   - Embed dataset descriptions
   - Enable: "Datasets similar to OpenAlex", "Resources for bibliometrics"

8. **Query History & Caching**
   - Cache frequent queries
   - Track popular searches for optimization

### Long-Term (High Effort)

9. **Interactive Hypergraph Visualization**
   - D3.js or vis.js browser-based explorer
   - Filter by node type, time range, person
   - Click to see connected entities

10. **Proactive Recommendations**
    - "Based on your research, you might want to use [dataset]"
    - "Other researchers working on [topic] use [resources]"

11. **Natural Language Query Understanding**
    - Replace regex patterns with LLM classification
    - Handle ambiguous queries gracefully

12. **Automated Registry Updates**
    - Sync OpenAlex data weekly
    - Auto-detect new Midway directories
    - Track dataset freshness

---

## File Locations

| Resource | Path |
|----------|------|
| Lab Registry | `Data/lab_registry.json` |
| Midway Registry | `Data/midway_registry.json` |
| Hypergraph | `Data/hypergraph.json` |
| RAG Index | `Data/rag_indexes/hybrid/` |
| Server | `rag_http_server.py` (port 8765) |
| Chat Interface | `chat.py` |

---

## Quick Start

```bash
# Start the RAG server
venv_rag/bin/python rag_http_server.py

# Test a query
curl "http://localhost:8765/registry?q=What+embeddings+are+on+Midway"

# Rebuild hypergraph
python -m hypergraph.build

# Run chat interface
python chat.py
```

---

## Contributors

- Knowledge Lab, University of Chicago
- Built with Claude Code assistance

---

*This document reflects the state of the project as of January 13, 2026.*
