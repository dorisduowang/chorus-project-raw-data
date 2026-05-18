# RAG System Enrichment: Performance Report & Simplification Recommendations

**Date:** January 2026
**Scope:** New extractors module for richer metadata extraction

---

## Executive Summary

We implemented four new metadata extractors to enhance the RAG system's ingestion pipeline:

| Extractor | Purpose | Performance | Coverage |
|-----------|---------|-------------|----------|
| **Date** | Extract dates from content | 0.2ms/chunk | 6.3% of chunks |
| **Topic** | Classify research themes | 0.3ms/chunk | 69.8% of chunks |
| **Key Terms** | Extract salient terms | 0.3ms/chunk | 83.3% of chunks |
| **NER** | People, orgs, grants | ~2ms/chunk* | Requires spaCy |

*NER requires spaCy installation (`pip install spacy && python -m spacy download en_core_web_sm`)

**Combined throughput: ~1,450 chunks/second** (without NER)
**Full corpus enrichment time: ~8 seconds** for 11,188 chunks

---

## 1. New Extractors Overview

### 1.1 Date Extractor (`extractors/date_extractor.py`)

Extracts dates from text content, improving on filename-only extraction.

**Formats Supported:**
- ISO: `2024-03-15`, `2024-03`
- US: `03/15/2024`, `3/15/24`
- Text: `March 15, 2024`, `15 March 2024`
- Partial: `March 2024`, `Q1 2024`, `FY24`, `Spring 2024`
- Ranges: `2024-2025`, `March 15-20, 2024`

**Context Detection:**
- Deadlines: "due March 15", "deadline March 15"
- Events: "conference on March 15"
- Publications: "published March 2024"
- Projects: "started in 2023"

### 1.2 Topic Extractor (`extractors/topic_extractor.py`)

Classifies chunks into 11 research topics using keyword/phrase matching.

**Topics:**
1. Network Analysis / Graph Theory
2. Natural Language Processing
3. Science of Science / Scientometrics
4. Machine Learning / AI
5. Social Networks / Social Media
6. Knowledge Discovery / Information Retrieval
7. Computational Methods
8. Policy / Ethics / Governance
9. Collaboration / Team Science
10. Education / Training
11. Administrative / Operational

**Distribution in Corpus:**
```
Machine Learning           22.1%
Science of Science         19.7%
Administrative             13.3%
Knowledge Discovery        13.1%
Natural Language Processing 13.0%
Education                  10.3%
Social Media                9.0%
Collaboration               8.0%
Computational Methods       7.3%
Network Analysis            6.8%
Policy and Ethics           5.6%
```

### 1.3 Key Term Extractor (`extractors/keyterm_extractor.py`)

Extracts important terms using TF-IDF with RAKE-inspired scoring.

**Features:**
- Single words and phrases (1-3 words)
- Academic term detection (acronyms, hyphenated terms)
- Stopword filtering
- Optional corpus-level IDF fitting

### 1.4 NER Extractor (`extractors/ner_extractor.py`)

Extracts named entities using spaCy.

**Entity Types:**
- People (PERSON entities)
- Organizations (ORG entities + funding agency keywords)
- Projects (MURI, APTO, CAREER, etc.)
- Grant IDs (NSF, NIH, DARPA, DOE, NASA patterns)

---

## 2. Performance Benchmarks

### 2.1 Individual Extractor Performance

| Extractor | Time/Chunk | Throughput | Notes |
|-----------|------------|------------|-------|
| Date | 0.15-0.25ms | 4,000-6,500/s | Pure regex |
| Topic | 0.25-0.35ms | 2,800-4,000/s | Keyword matching |
| Key Terms | 0.25-0.35ms | 2,800-4,000/s | TF-IDF scoring |
| NER | 1.5-2.5ms | 400-650/s | spaCy model |

### 2.2 Combined Pipeline Performance

**Without NER:**
- Total time for 1000 chunks: 0.69s
- Average per chunk: 0.69ms
- Throughput: 1,458 chunks/second
- Full corpus (11,188 chunks): ~8 seconds

**With NER:**
- Estimated per chunk: ~3ms
- Throughput: ~330 chunks/second
- Full corpus: ~34 seconds

### 2.3 Test Results

```
41/41 unit tests passing (excluding NER without spaCy)
- Date extraction: 15 tests
- Topic classification: 13 tests
- Key term extraction: 10 tests
- Performance benchmarks: 3 tests
```

---

## 3. Enrichment Coverage Analysis

Based on 1,000-chunk sample:

| Metric | Count | Percentage |
|--------|-------|------------|
| Chunks with dates | 63 | 6.3% |
| Chunks with topics | 698 | 69.8% |
| Chunks with key terms | 833 | 83.3% |
| Unique topics found | 11 | 100% |

**Date Year Distribution:**
- 2025: 26 chunks (most common)
- 2023: 9 chunks
- 2021: 8 chunks
- 2024: 4 chunks
- Others: 16 chunks

---

## 4. Codebase Simplification Recommendations

### 4.1 High Priority: Consolidate HTTP Servers

**Problem:** Two separate HTTP server implementations
- `rag_http_server.py` (1,303 lines)
- `rag_server_fastapi.py` (1,222 lines)

**Recommendation:** Consolidate into single FastAPI server
- FastAPI is more maintainable and async-native
- Remove `rag_http_server.py` after migration
- **Estimated reduction:** ~1,000 lines

### 4.2 Medium Priority: Simplify Registry Module

**Problem:** Multiple overlapping files in `registry/`:
- `registry/rag.py` (910 lines)
- `registry/classifier.py` (957 lines)
- `registry/lookup.py` (710 lines)
- `registry/semantic_index.py` (861 lines)
- `registry/unified_index.py` (506 lines)

**Recommendation:**
- Merge `classifier.py` and `query_router.py` - both handle query routing
- Consolidate `semantic_index.py` and `unified_index.py`
- **Estimated reduction:** ~500 lines

### 4.3 Medium Priority: Merge Cache Implementations

**Problem:** Two cache modules with similar patterns
- `cache/query_cache.py` (863 lines)
- `cache/reformulation_cache.py` (699 lines)

**Recommendation:**
- Create base cache class with common functionality
- Inherit for specific cache types
- **Estimated reduction:** ~400 lines

### 4.4 Low Priority: Remove Deprecated Code

**Current:** `deprecated/` directory exists

**Recommendation:**
- Audit deprecated files
- Remove if no longer used
- Add migration notes if needed

### 4.5 Low Priority: Standardize Extractor Interface

**Current:** Each extractor has slightly different interfaces

**Recommendation:**
- Create base `Extractor` class
- Standardize `extract()` method signature
- Unify result dataclasses

---

## 5. Suggested Architecture Changes

### 5.1 Unified Enrichment Pipeline

```python
# Current: Manual extraction calls
dates = extract_dates(text)
topics = classify_topics(text)
terms = extract_key_terms(text)
entities = extract_entities(text)

# Proposed: Single enrichment call
enriched_chunk = enrich_chunk(chunk, config={
    'use_ner': True,
    'use_dates': True,
    'use_topics': True,
    'use_keyterms': True,
})
```

**Already implemented in:** `extractors/enrichment.py`

### 5.2 Lazy Loading Pattern

```python
# Current: All extractors loaded at import
from extractors import extract_dates, classify_topics

# Better: Lazy loading (already implemented)
def get_date_extractor():
    global _extractor
    if _extractor is None:
        _extractor = DateExtractor()
    return _extractor
```

### 5.3 Configuration-Driven Pipeline

```yaml
# config/enrichment.yaml
extractors:
  ner:
    enabled: true
    model: en_core_web_sm
  dates:
    enabled: true
    context_detection: true
  topics:
    enabled: true
    top_k: 3
    min_confidence: 0.1
  keyterms:
    enabled: true
    top_k: 10
    use_idf: true
```

---

## 6. Integration Guide

### 6.1 During Index Rebuild

```python
from extractors.enrichment import enrich_chunks_batch

# In rebuild_rag_with_files.py
chunks = load_and_chunk_by_file(...)

# Add enrichment step
print("Enriching chunks with metadata...")
enrich_chunks_batch(
    chunks,
    use_ner=True,  # Set False if spaCy not installed
    use_dates=True,
    use_topics=True,
    use_keyterms=True,
    verbose=True
)

# Continue with embedding and indexing
build_indexes(chunks, ...)
```

### 6.2 For Incremental Updates

```python
from extractors.enrichment import enrich_chunk

# In incremental_index.py
def add_file(self, filepath):
    chunks = self._chunk_file(filepath)
    for chunk in chunks:
        enrich_chunk(chunk, use_ner=False)  # Fast path
    self._add_chunks_to_index(chunks)
```

### 6.3 Query-Time Enrichment

```python
# Optional: Enrich queries for better matching
from extractors import extract_key_terms, classify_topics

def enhance_query(query: str):
    terms = extract_key_terms(query, top_k=5)
    topics = classify_topics(query, top_k=2)
    return {
        'original': query,
        'key_terms': terms.term_list(),
        'topics': topics.topic_labels(),
    }
```

---

## 7. Files Created/Modified

### New Files
- `extractors/__init__.py` - Module exports
- `extractors/ner_extractor.py` - NER extraction
- `extractors/date_extractor.py` - Date extraction
- `extractors/topic_extractor.py` - Topic classification
- `extractors/keyterm_extractor.py` - Key term extraction
- `extractors/enrichment.py` - Unified enrichment pipeline
- `tests/test_extractors.py` - Comprehensive test suite

### Modified Files
- `chunking_strategies.py` - Added new metadata fields to `ChunkWithMetadata`

---

## 8. Dependencies

### Required (already installed)
- Python 3.8+
- Standard library only for dates, topics, key terms

### Optional
- `spacy>=3.7.0` - For NER extraction
- `en_core_web_sm` - spaCy model

**Installation:**
```bash
pip install spacy
python -m spacy download en_core_web_sm
```

---

## 9. Next Steps

1. **Immediate:** Re-run index rebuild with enrichment enabled
2. **Short-term:** Add faceted search by topic in UI
3. **Medium-term:** Implement simplification recommendations
4. **Long-term:** Explore ML-based topic classification for better accuracy

---

## 10. Conclusion

The new extractors provide significant metadata enrichment capability with minimal performance overhead. The ~0.7ms per chunk processing time means full corpus enrichment takes under 10 seconds, making it practical to run during every index rebuild.

Key wins:
- **70% of chunks** now have topic classifications
- **83% of chunks** have extracted key terms
- **6% additional date coverage** from content (vs filename-only)
- **Throughput of 1,450 chunks/second** allows real-time enrichment

The codebase could benefit from consolidation, with potential to reduce ~2,000 lines through server unification and module merging.
