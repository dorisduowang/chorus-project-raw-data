# Search Architecture Improvements

## Current Architecture Analysis

### Data Flow
```
                    ┌─────────────────┐
                    │  User Query     │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ Regex Classifier│  ← Brittle, pattern-dependent
                    │  (75+ patterns) │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
     ┌────────▼───────┐ ┌────▼────┐ ┌───────▼───────┐
     │ Structured     │ │ Semantic│ │  Hypergraph   │
     │ Lookup         │ │ Search  │ │  Traversal    │
     │ (string match) │ │ (FAISS) │ │  (in-memory)  │
     └────────┬───────┘ └────┬────┘ └───────┬───────┘
              │              │              │
              └──────────────┼──────────────┘
                             │
                    ┌────────▼────────┐
                    │  Merged Results │
                    └─────────────────┘
```

### Current Weaknesses

| Layer | Problem | Impact |
|-------|---------|--------|
| Classification | Regex patterns miss variations | 8% of queries misrouted |
| Structured | String matching only | No semantic similarity |
| Semantic | Documents only, not registry | Can't find "similar datasets" |
| Hypergraph | Disconnected from search | No relevance boosting |
| Ingestion | Manual JSON files | Stale data, no entity linking |

---

## Proposed Improvements

### 1. Semantic Registry Index

**Problem**: Registry entries (people, datasets, projects) aren't searchable by semantic similarity.

**Current behavior**:
```
Query: "datasets similar to OpenAlex"
→ Fails: No pattern for "similar to X"
→ Falls back to keyword search, finds nothing useful
```

**Proposed solution**: Embed registry entries and add to vector index.

```python
# During ingestion, create embeddings for registry items
registry_embeddings = []
for person in registry["people"]["members"]:
    text = f"""
    {person["name"]} is a {person["role"]} at {person["institution"]}.
    Research interests: {', '.join(person.get('openalex', {}).get('topics', []))}.
    Recent work: {person.get('openalex', {}).get('recent_publications', [{}])[0].get('title', '')}
    """
    embedding = model.encode(text)
    registry_embeddings.append({
        "id": person["id"],
        "type": "person",
        "text": text,
        "embedding": embedding,
        "metadata": person
    })

# Same for datasets, projects, etc.
for dataset in midway_registry["social_media_datasets"]["items"]:
    text = f"""
    {dataset["name"]} - {dataset.get("content_description", "")}
    Platform: {dataset.get("platform")}
    Path: {dataset.get("path")}
    """
    # ... embed and store
```

**Benefits**:
- "Find researchers like James Evans" works via embedding similarity
- "Datasets similar to OpenAlex" finds bibliometric datasets
- Handles paraphrases without explicit patterns

---

### 2. Query Understanding Layer (Replace Regex)

**Problem**: Regex patterns are brittle and require manual maintenance.

**Current**:
```python
# 75+ patterns like:
(r"who\s+works\s+on\s+(.+)", ["people", "topics"], "topic_search")
```

**Option A: Embedding-based classification**
```python
# Pre-compute embeddings for query templates
QUERY_TEMPLATES = {
    "person_lookup": ["who is X", "tell me about X", "find X"],
    "topic_search": ["who works on X", "researchers studying X"],
    "dataset_lookup": ["where is X data", "find X dataset"],
    # ...
}

template_embeddings = {
    category: [model.encode(t) for t in templates]
    for category, templates in QUERY_TEMPLATES.items()
}

def classify_query(query: str) -> str:
    query_emb = model.encode(query)
    best_category = None
    best_score = -1

    for category, embeddings in template_embeddings.items():
        scores = [cosine_similarity(query_emb, emb) for emb in embeddings]
        max_score = max(scores)
        if max_score > best_score:
            best_score = max_score
            best_category = category

    return best_category
```

**Option B: Small LLM for query parsing**
```python
def parse_query(query: str) -> dict:
    """Use a small model to extract intent and entities."""
    prompt = f"""
    Parse this query into structured form:
    Query: "{query}"

    Output JSON with:
    - intent: person_lookup|topic_search|dataset_lookup|publication_search|...
    - entities: list of named entities mentioned
    - filters: any constraints (role, institution, topic, etc.)
    """
    # Use Claude Haiku or similar small model
    response = llm.complete(prompt)
    return json.loads(response)
```

**Benefits**:
- Handles typos naturally ("reseachers" → researchers)
- Understands paraphrases without explicit patterns
- Self-improving with few-shot examples

---

### 3. Unified Search Index

**Problem**: Structured lookup and semantic search are disconnected.

**Current architecture**:
```
Registry JSON → String matching (separate)
Documents    → Vector index (separate)
```

**Proposed architecture**:
```
┌─────────────────────────────────────────────┐
│           Unified Vector Index              │
├─────────────────────────────────────────────┤
│ Document chunks     (doc_type: "document")  │
│ Person profiles     (doc_type: "person")    │
│ Dataset descriptions(doc_type: "dataset")   │
│ Project summaries   (doc_type: "project")   │
│ Publication abstracts(doc_type: "publication")│
└─────────────────────────────────────────────┘
```

**Implementation**:
```python
class UnifiedIndex:
    def __init__(self):
        self.index = faiss.IndexFlatIP(768)
        self.metadata = []

    def add_document(self, text: str, doc_type: str, entity_id: str, metadata: dict):
        embedding = self.model.encode(text)
        self.index.add(embedding)
        self.metadata.append({
            "doc_type": doc_type,
            "entity_id": entity_id,
            "text": text,
            **metadata
        })

    def search(self, query: str, doc_types: list = None, top_k: int = 10):
        query_emb = self.model.encode(query)
        scores, indices = self.index.search(query_emb, top_k * 3)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            meta = self.metadata[idx]
            if doc_types is None or meta["doc_type"] in doc_types:
                results.append({"score": score, **meta})

        return results[:top_k]
```

**Query routing becomes filtering**:
```python
# Instead of pattern matching to route, use query intent for filtering
intent = classify_query(query)

if intent == "person_lookup":
    results = unified_index.search(query, doc_types=["person"])
elif intent == "dataset_lookup":
    results = unified_index.search(query, doc_types=["dataset"])
else:
    # Search everything, let relevance sort it out
    results = unified_index.search(query)
```

---

### 4. Entity Extraction During Ingestion

**Problem**: Document chunks don't link to registry entities.

**Current**: Chunks are indexed with minimal metadata (source file, page).

**Proposed**: Extract and link entities during ingestion.

```python
def ingest_document(doc_path: str, registry: dict):
    """Ingest document with entity extraction."""
    chunks = chunk_document(doc_path)

    # Build entity matchers from registry
    person_names = {p["name"].lower(): p["id"] for p in registry["people"]["members"]}
    dataset_names = {d["name"].lower(): d["id"] for d in all_datasets}
    project_names = {"apto": "apto", "c3s2": "c3s2", ...}

    for chunk in chunks:
        # Extract entities
        mentioned_people = []
        mentioned_datasets = []
        mentioned_projects = []

        chunk_lower = chunk.text.lower()
        for name, id in person_names.items():
            if name in chunk_lower:
                mentioned_people.append(id)

        for name, id in dataset_names.items():
            if name in chunk_lower:
                mentioned_datasets.append(id)

        # Store with entity links
        index.add_document(
            text=chunk.text,
            doc_type="document",
            entity_id=None,
            metadata={
                "source": doc_path,
                "mentioned_people": mentioned_people,
                "mentioned_datasets": mentioned_datasets,
                "mentioned_projects": mentioned_projects,
            }
        )
```

**Benefits**:
- "Documents mentioning Jake" returns relevant chunks
- Hypergraph edges can be created from co-mentions
- Better context for question answering

---

### 5. Hypergraph-Boosted Retrieval

**Problem**: Hypergraph relationships don't influence search ranking.

**Proposed**: Boost results connected to query entities in hypergraph.

```python
def hypergraph_boosted_search(query: str, top_k: int = 10):
    # Step 1: Get base results
    results = unified_index.search(query, top_k=top_k * 3)

    # Step 2: Extract entities from query
    query_entities = extract_entities(query)  # ["james-evans", "network-science"]

    # Step 3: Get related entities from hypergraph
    related_entities = set()
    for entity_id in query_entities:
        # 1-hop neighbors
        related_entities.update(hypergraph.neighbors(entity_id))

    # Step 4: Boost results that mention related entities
    for result in results:
        boost = 0
        mentioned = (
            result.get("mentioned_people", []) +
            result.get("mentioned_datasets", []) +
            result.get("mentioned_projects", [])
        )
        for entity in mentioned:
            if entity in query_entities:
                boost += 0.3  # Direct mention
            elif entity in related_entities:
                boost += 0.1  # Related entity

        result["score"] += boost

    # Re-sort by boosted scores
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]
```

---

### 6. Query Expansion via Hypergraph

**Problem**: Queries about one entity don't find related entities.

**Example**:
```
Query: "What data does James Evans use?"
Current: Only finds direct matches
Better: Also find data used by James's collaborators on similar projects
```

**Proposed**:
```python
def expand_query_entities(query: str, hypergraph: HyperGraph) -> list:
    """Expand query with related entities from hypergraph."""
    entities = extract_entities(query)
    expanded = set(entities)

    for entity in entities:
        node = hypergraph.get_node(entity)
        if not node:
            continue

        # Add collaborators for people
        if node.type == NodeType.PERSON:
            collabs = hypergraph.collaborators_of(entity)
            expanded.update([c.id for c in collabs[:5]])

        # Add co-occurring datasets/topics
        edges = hypergraph.edges_involving(entity)
        for edge in edges:
            expanded.update(edge.inputs[:3])
            expanded.update(edge.outputs[:3])

    return list(expanded)
```

---

### 7. Feedback & Learning Loop

**Problem**: System doesn't learn from failures.

**Proposed**: Log queries and implement feedback mechanisms.

```python
# query_log.jsonl - append-only log
{"timestamp": "2026-01-13T10:00:00", "query": "reseachers studying NLP",
 "classification": "topics", "results": 0, "latency_ms": 150}

# Analysis script
def analyze_failures():
    """Find patterns in zero-result queries."""
    failures = [q for q in load_log() if q["results"] == 0]

    # Cluster by edit distance to find typo patterns
    typo_clusters = cluster_by_similarity(failures)

    # Find common unmatched keywords
    unmatched_keywords = extract_keywords(failures)

    return {
        "typo_suggestions": typo_clusters,
        "new_keywords": unmatched_keywords,
        "new_patterns": suggest_patterns(failures)
    }
```

**Auto-correction**:
```python
def maybe_correct_query(query: str) -> str:
    """Attempt typo correction using registry vocabulary."""
    words = query.lower().split()
    corrected = []

    # Build vocabulary from registry
    vocab = set()
    vocab.update(p["name"].lower().split() for p in people)
    vocab.update(known_topics)
    vocab.update(["researcher", "researchers", "dataset", "data", ...])

    for word in words:
        if word in vocab:
            corrected.append(word)
        else:
            # Find closest match
            closest = min(vocab, key=lambda v: edit_distance(word, v))
            if edit_distance(word, closest) <= 2:
                corrected.append(closest)
            else:
                corrected.append(word)

    return " ".join(corrected)
```

---

## Implementation Priority

### Phase 1: Quick Wins (1-2 days)
1. **Typo correction** - Edit distance matching against registry vocabulary
2. **Query logging** - Start collecting data for analysis
3. **More entity keywords** - Add all dataset names, researcher names to classifier

### Phase 2: Semantic Registry (3-5 days)
4. **Embed registry entries** - Create embeddings for people, datasets, projects
5. **Unified index** - Merge document and registry embeddings
6. **Entity extraction** - Link chunks to registry entities during ingestion

### Phase 3: Smart Classification (1 week)
7. **Embedding-based classifier** - Replace regex with semantic matching
8. **Query expansion** - Use hypergraph for related entity discovery
9. **Hypergraph boosting** - Incorporate graph structure into ranking

### Phase 4: Learning System (ongoing)
10. **Feedback analysis** - Identify patterns in failures
11. **Auto-suggestions** - "Did you mean..." for ambiguous queries
12. **Pattern generation** - Semi-automated pattern discovery

---

## Data Structure Changes

### Current Schema
```
Data/
├── lab_registry.json      # People, projects, funding (manual)
├── midway_registry.json   # Datasets, embeddings (manual)
├── hypergraph.json        # Nodes + edges (built from registry)
└── rag_indexes/
    └── hybrid/
        ├── index.faiss    # Document embeddings only
        └── chunks.pkl     # Document chunks only
```

### Proposed Schema
```
Data/
├── registries/
│   ├── lab_registry.json
│   ├── midway_registry.json
│   └── entity_links.json       # NEW: chunk → entity mappings
├── hypergraph.json
├── indexes/
│   └── unified/
│       ├── index.faiss         # Documents + registry entries
│       ├── metadata.pkl        # Unified metadata with entity links
│       └── vocabulary.json     # NEW: Known terms for typo correction
└── logs/
    └── queries.jsonl           # NEW: Query log for analysis
```

### New Entity Links Schema
```json
{
  "chunk_id": "doc123_chunk_45",
  "source_file": "proposals/MURI_2024.pdf",
  "mentioned_entities": {
    "people": ["james-evans", "jake-burchard"],
    "datasets": ["openalex-dec-2024"],
    "projects": ["apto"],
    "topics": ["network-science", "science-of-science"]
  },
  "extracted_at": "2026-01-13T10:00:00"
}
```

---

## Expected Impact

| Improvement | Edge Cases Fixed | Effort |
|-------------|------------------|--------|
| Typo correction | "reseachers" → "researchers" | Low |
| Semantic registry | "datasets similar to X" | Medium |
| Unified index | Removes routing brittleness | Medium |
| Entity extraction | "docs mentioning Jake" | Medium |
| Hypergraph boosting | Better relevance for connected entities | Medium |
| Query expansion | Multi-hop discovery | High |
| Learning loop | Continuous improvement | High |

**Projected success rate improvement**: 92% → 97%+

---

## Questions to Consider

1. **Latency budget**: How much latency is acceptable for smarter classification?
2. **Index rebuild frequency**: How often should we re-embed registry changes?
3. **Entity extraction accuracy**: Use simple string matching or NER model?
4. **Feedback mechanism**: Explicit thumbs up/down or implicit (click-through)?
5. **LLM usage**: Acceptable to use LLM for query parsing? Cost implications?

---

*Document created: January 13, 2026*
