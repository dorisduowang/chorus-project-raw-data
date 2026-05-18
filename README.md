# CHORUS - Knowledge Lab RAG System

CHORUS (Comprehensive Hybrid Orchestrated Retrieval and Understanding System) is an AI-powered knowledge assistant for the Knowledge Lab at the University of Chicago. It combines structured registry data with semantic document search to answer questions about lab members, projects, publications, and resources.

## Features

- **Hybrid Search**: Combines semantic embeddings (FAISS) with keyword search (BM25) for robust retrieval
- **Structured Registry**: Direct lookup of people, projects, funding, compute resources, and publications
- **Topic Aliases**: Searches "NLP" to find "natural language processing" researchers
- **Compound Queries**: Supports complex queries like "UChicago researchers in network science"
- **Publication Search**: Search 482+ publications by author, topic, venue, or title
- **LLM Query Reformulation**: Expands queries for better semantic matching
- **Cross-Encoder Reranking**: Improves result quality using neural reranking
- **Hot Reload**: Registry updates without server restart

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Chat Interface                          │
│                        (chat.py)                             │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    RAG HTTP Server                           │
│                  (rag_http_server.py)                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │ LLM Reform.  │  │   Reranker   │  │       MMR        │   │
│  │   (Claude)   │  │ (MiniLM-L6)  │  │  Diversification │   │
│  └──────────────┘  └──────────────┘  └──────────────────┘   │
└─────────────────────────┬───────────────────────────────────┘
                          │
          ┌───────────────┴───────────────┐
          ▼                               ▼
┌─────────────────────┐       ┌─────────────────────────────┐
│   Semantic Search   │       │      Registry Module        │
│  (FAISS + BM25)     │       │     (registry/*.py)         │
│                     │       │  ┌─────────────────────┐    │
│  11,188 chunks      │       │  │  QueryClassifier    │    │
│  all-MiniLM-L6-v2   │       │  │  RegistryLookup     │    │
│                     │       │  │  TopicAliases       │    │
└─────────────────────┘       │  │  CompoundSearch     │    │
                              │  └─────────────────────┘    │
                              └─────────────────────────────┘
                                            │
                                            ▼
                              ┌─────────────────────────────┐
                              │   Data/lab_registry.json    │
                              │  71 members, 482 pubs       │
                              │  3 projects, 6 compute      │
                              └─────────────────────────────┘
```

## Project Structure

```
chorus/
├── chat.py                 # CLI chat interface with Claude
├── rag_http_server.py      # Main HTTP API server (1,026 lines)
├── core_agent.py           # Core agent logic
├── registry/               # Modular registry package
│   ├── __init__.py         # Package exports
│   ├── classifier.py       # QueryClassifier (738 lines)
│   ├── lookup.py           # RegistryLookup (465 lines)
│   ├── rag.py              # RegistryRAG main class (504 lines)
│   ├── models.py           # Data classes
│   ├── text_generator.py   # Natural language summaries
│   └── topic_aliases.py    # NLP→"natural language processing"
├── agents/                 # Multi-agent orchestrator
│   ├── orchestrator.py     # Agent coordination
│   └── specialists/        # Specialist agents
├── tests/                  # Test suite (14 files)
├── Data/
│   ├── lab_registry.json   # Primary structured data
│   └── rag_indexes/        # FAISS and BM25 indexes
├── ingest/                 # Data ingestion (Google Drive, daily scans)
└── requirements.txt        # Python dependencies
```

## Data Coverage

| Entity | Count | Description |
|--------|-------|-------------|
| Lab Members | 71 | Researchers, faculty, students, affiliates |
| OpenAlex Data | 66 | Members with bibliometric profiles |
| Publications | 482 | Recent papers with citations |
| Projects | 3 | APTO, C3S2, Socio-Cognitive AI |
| Funding Sources | 3 | NSF, AFOSR grants |
| Compute Resources | 6 | Midway3, Knowledge Garden, GPUs |
| Datasets | 10 | MAG, OpenAlex, WoS, Reddit, etc. |
| Tools | 8 | SLURM, PySpark, Jupyter, etc. |

## Quick Start

### 1. Install Dependencies

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY (and Slack tokens if using the bot)
```

See `docs/GETTING_STARTED.md` for full credential setup, Google Drive sync, and multi-machine deployment instructions.

### 3. Start the RAG Server

```bash
python rag_http_server.py
```

The server starts on `http://localhost:8765` with endpoints:
- `GET /health` - Server status
- `GET /hybrid?q=query` - Hybrid search (recommended)
- `GET /semantic?q=query` - Semantic-only search
- `GET /bm25?q=query` - Keyword-only search
- `POST /reload` - Hot-reload registry

### 4. Start the Chat Interface

In a separate terminal:
```bash
python chat.py
```

## Usage Examples

### Chat Interface

```
You: Who is James Evans?

CHORUS: James Evans is the Director of the Knowledge Lab at the University
of Chicago. He holds the position of Max Palevsky Professor of Sociology
and Data Science. According to his bibliometric profile, he has 775
published works, over 15,000 citations, and an h-index of 60.

You: Who works on NLP?

CHORUS: Several people work on NLP (Natural Language Processing):
- Doug Downey (Affiliate Faculty, Allen Institute for AI)
- Chenhao Tan (Affiliate Faculty, University of Chicago)
- Dan Weld (Affiliate Faculty, Allen Institute)
...

You: UChicago researchers in network science

CHORUS: Found 1 researcher at University of Chicago working on network science:
- Haizi Yu (Affiliate Researcher)
  Topics: neural networks, complex network analysis
```

### API Queries

```bash
# Person lookup
curl "http://localhost:8765/hybrid?q=Who+is+James+Evans"

# Topic search with alias
curl "http://localhost:8765/hybrid?q=Who+works+on+NLP"

# Compound query
curl "http://localhost:8765/hybrid?q=UChicago+researchers+in+machine+learning"

# Publication search
curl "http://localhost:8765/hybrid?q=Papers+by+James+Evans"

# Compute resources
curl "http://localhost:8765/hybrid?q=What+compute+resources+are+available"
```

## Query Types Supported

| Query Type | Example | Description |
|------------|---------|-------------|
| Person lookup | "Who is James Evans?" | Direct name lookup |
| Role query | "List the PhD students" | Filter by role |
| Topic search | "Who works on machine learning?" | Find by research topic |
| Topic alias | "Researchers in NLP" | Expands abbreviations |
| Compound | "UChicago faculty in AI" | Multiple filters |
| Publication | "Papers by James Evans" | Search publications |
| Compute | "What GPUs are available?" | List resources |
| Project | "Tell me about APTO" | Project details |
| Funding | "What grants does the lab have?" | Funding info |

### Topic Aliases

The system automatically expands common abbreviations:

| Abbreviation | Expands To |
|--------------|------------|
| NLP | natural language processing, computational linguistics |
| ML | machine learning, statistical learning |
| AI | artificial intelligence |
| DL | deep learning, neural networks |
| CSS | computational social science |
| HCI | human-computer interaction |
| CV | computer vision |

## Performance

### RAG-Only (no Claude)

| Metric | Value |
|--------|-------|
| Average latency | ~1.7s |
| p95 latency | ~2.0s |
| Throughput | ~0.6 queries/sec |
| Query accuracy | 91.7% |

### Full Pipeline (RAG + Claude)

| Phase | Latency |
|-------|---------|
| RAG search + ranking | ~1.7s |
| Claude response | ~4-5s |
| **Total end-to-end** | **~6-7s** |

### Latency Breakdown

| Component | Time |
|-----------|------|
| FAISS semantic search | <1ms |
| BM25 keyword search | <1ms |
| LLM query reformulation | ~500ms |
| Cross-encoder reranking | ~800ms |
| Registry structured search | ~200ms |

## Testing

### Run Performance Tests

```bash
# RAG performance test
python perf_test.py

# Chat interface test
python test_chat.py

# Registry integration tests
python -m tests.test_registry_integration
```

### Test Categories

| Category | Tests | Pass Rate |
|----------|-------|-----------|
| Person lookup | 3 | 100% |
| Role queries | 3 | 67% |
| Topic search | 3 | 100% |
| Topic aliases | 3 | 100% |
| Publication search | 3 | 100% |
| Compute/tools | 3 | 67% |
| Compound queries | 3 | 100% |
| Project/funding | 3 | 100% |

## Configuration

### Server Options

Edit `rag_http_server.py` to configure:

```python
# Reranking (improves quality, adds latency)
RERANKING_ENABLED = True
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# MMR diversification
MMR_ENABLED = True
MMR_LAMBDA = 0.7  # 0=diversity, 1=relevance

# LLM query reformulation
LLM_REFORMULATION_ENABLED = True
```

### Registry Schema

The registry follows the schema in `Data/lab_registry_schema.json`:

```json
{
  "people": {
    "members": [{
      "id": "unique_id",
      "name": "Full Name",
      "role": "Role Title",
      "institution": "University",
      "email": "email@example.com",
      "openalex": {
        "works_count": 100,
        "cited_by_count": 5000,
        "h_index": 30,
        "topics": ["Topic 1", "Topic 2"],
        "recent_publications": [...]
      }
    }]
  },
  "projects": [...],
  "funding": [...],
  "datasets": [...],
  "compute": [...],
  "tools": [...]
}
```

## Development

### Adding New Entity Types

1. Add data to `Data/lab_registry.json`
2. Add entity keywords in `registry/classifier.py`
3. Add lookup handler in `registry/rag.py`
4. Add formatting in `registry/rag.py:_format_structured_answer()`

### Adding Topic Aliases

Edit `registry/topic_aliases.py`:

```python
TOPIC_ALIASES = {
    "your_abbrev": ["full form 1", "full form 2"],
    ...
}
```

### Hot Reload Registry

```bash
# Trigger reload via API
curl -X POST http://localhost:8765/reload

# Or modify lab_registry.json - auto-detected on next query
```

## Key Files

| File | Lines | Purpose |
|------|-------|---------|
| `rag_http_server.py` | 1,026 | HTTP API server with search endpoints |
| `registry/classifier.py` | 738 | Query classification and routing |
| `registry/lookup.py` | 465 | Registry data access and search |
| `registry/rag.py` | 504 | Main RAG integration class |
| `chat.py` | 311 | CLI chat interface |
| `core_agent.py` | 804 | Core agent orchestration |

## Dependencies

- Python 3.10+
- sentence-transformers
- faiss-cpu
- rank-bm25
- anthropic
- torch

## License

Internal use only - Knowledge Lab, University of Chicago.

---

*Last updated: January 2026*
