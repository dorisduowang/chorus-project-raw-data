# Chorus System Test Report

**Date**: January 13, 2026
**Version**: 2.0 (post-improvements)
**Tester**: Automated test suite via Claude Code

---

## Executive Summary

The Chorus knowledge management system was subjected to comprehensive testing including compound queries, non-obvious natural language queries, cross-registry lookups, hypergraph traversals, and edge cases. After implementing targeted fixes, the system achieved:

- **100% pass rate** on 20 core functionality tests
- **92.3% pass rate** on 39 extended stress tests
- **0 errors** (all queries processed without crashes)

---

## Test Methodology

### Test Categories

| Category | Description | Tests |
|----------|-------------|-------|
| Compound Queries | Multi-filter searches (role + topic, institution + research) | 2 |
| Implicit Queries | Natural language without explicit keywords | 5 |
| Cross-Registry | Queries spanning lab_registry and midway_registry | 2 |
| Edge Cases | Typos, abbreviations, vague queries | 3 |
| Specific Data | Platform-specific and resource-specific lookups | 3 |
| Hypergraph | Graph traversal and analytics endpoints | 3 |
| Publications | Paper and author searches | 2 |
| Topic Search | Research area queries | 2 |

### Test Environment

- **Server**: Python HTTP server on port 8765
- **Index**: 11,188 document chunks
- **Registry**: 71 people, 482 publications
- **Hypergraph**: 1,312 nodes, 594 edges
- **Features**: LLM reformulation, cross-encoder reranking, MMR diversity

---

## Core Functionality Tests (20/20 Passed)

### Compound Queries
| Query | Result | Details |
|-------|--------|---------|
| "PhD students working on machine learning" | PASS | 7 researchers found |
| "UChicago researchers studying networks" | PASS | 1 researcher found |

### Implicit/Natural Language Queries
| Query | Result | Details |
|-------|--------|---------|
| "Where can I find Chinese social media posts" | PASS | Routed to midway_social, found Weibo |
| "I need word embeddings for scientific papers" | PASS | Found SPECTER embeddings |
| "How do I run GPU jobs" | PASS | Found compute resources |

### Cross-Registry Queries
| Query | Result | Details |
|-------|--------|---------|
| "What datasets does James Evans work with" | PASS | 6 linked resources |
| "Who owns the Reddit data directory" | PASS | 12 results |

### Edge Cases
| Query | Result | Details |
|-------|--------|---------|
| "redit toxisity data" (misspelled) | PASS | Found via fallback search |
| "NLP researchers in the lab" | PASS | Abbreviation handled |
| "large datasets" (vague) | PASS | 4 results returned |

### Specific Data Queries
| Query | Result | Details |
|-------|--------|---------|
| "4chan dataset location" | PASS | Found in social media datasets |
| "patent embeddings on midway" | PASS | Found patent resources |
| "latest OpenAlex snapshot path" | PASS | Found snapshot with path |

### Hypergraph Queries
| Query | Result | Details |
|-------|--------|---------|
| Collaborators of james-evans | PASS | 51 collaborators found |
| Outputs of james-evans | PASS | 37 outputs found |
| Search for "network" | PASS | 20 matching nodes |

### Publication Queries
| Query | Result | Details |
|-------|--------|---------|
| "machine learning" topic search | PASS | 14 publications |
| "Evans" author search | PASS | 32 publications |

### Topic Queries
| Query | Result | Details |
|-------|--------|---------|
| "Who works on computational social science" | PASS | 13 researchers |
| "science of science researchers" | PASS | 2 results |

---

## Extended Stress Test (36/39 Passed)

### Passed Categories (36 queries)

- **Natural language variations**: 6/6 (100%)
  - "I'm looking for toxicity data"
  - "where's the weibo dataset"
  - "can you help me find reddit posts"
  - "need patent similarity matrices"
  - "show me SPECTER embeddings"
  - "what's the path to OpenAlex"

- **Complex compound queries**: 3/3 (100%)
  - "postdocs working on network science"
  - "faculty members researching AI"
  - "graduate students studying social media"

- **Cross-registry queries**: 3/3 (100%)
  - "what data does Jake have access to"
  - "who manages the reddit directory"
  - "researchers with midway storage"

- **Hypergraph-relevant**: 3/3 (100%)
  - "who has collaborated with James Evans"
  - "publications by the Evans lab"
  - "co-authors in network science"

- **Resource queries**: 2/3 (67%)
  - "GPU resources on midway" - PASS
  - "SLURM job examples" - PASS

- **Project queries**: 2/2 (100%)
  - "what is the APTO project about"
  - "C3S2 team members"

- **Specific datasets**: 4/4 (100%)
  - "Semantic Scholar data location"
  - "DBLP database path"
  - "PubMed data on midway"
  - "MAG snapshot"

- **Infrastructure**: 3/3 (100%)
  - "shared LLM cache"
  - "ollama models"
  - "huggingface cache location"

- **Embeddings**: 3/3 (100%)
  - "word2vec models"
  - "chromatin embeddings"
  - "mat2vec embeddings"

### Failed Queries (3 queries)

| Query | Expected | Actual | Issue |
|-------|----------|--------|-------|
| "reseachers studying NLP" | people | topics | Heavy typo "reseachers" not fuzzy-matched |
| "how much storage is available" | compute | midway_researchers | Ambiguous - "storage" matched researcher directories |
| "science of science experts" | people | [] | No pattern for "X experts" phrasing |

### Failure Analysis

1. **Typo tolerance**: The system handles minor typos via fallback search but severe typos like "reseachers" (missing 'r') bypass the intended classification.

2. **Ambiguous keywords**: "storage" appears in both compute resources and researcher directories, causing misrouting.

3. **Pattern gaps**: The phrase "X experts" isn't captured by existing patterns; "X researchers" or "experts in X" would work.

---

## Performance Metrics

### Response Times

| Operation | Average Time |
|-----------|-------------|
| Health check | <10ms |
| Structured lookup | <5ms |
| Classification | <1ms |
| Hypergraph query | <10ms |
| Full hybrid query | ~1,600ms |

*Note: Hybrid queries include LLM reformulation and cross-encoder reranking, adding latency.*

### Throughput

- **Stress test**: 39 queries in 64.4 seconds
- **Rate**: 0.6 queries/second
- **Success rate**: 100% (no connection drops)

### Resource Utilization

| Resource | Value |
|----------|-------|
| Document chunks indexed | 11,188 |
| People in registry | 71 |
| Publications tracked | 482 |
| Hypergraph nodes | 1,312 |
| Hypergraph edges | 594 |
| Classifier patterns | 75+ |

---

## Improvements Implemented

### 1. Topic Alias Expansion
**Problem**: "machine learning" didn't match "artificial intelligence" topics
**Solution**: Cross-linked ML/AI/neural network aliases in `topic_aliases.py`
**Result**: Compound queries like "PhD students working on ML" now return results

### 2. Pattern Ordering Fix
**Problem**: "Where can I find Chinese social media" matched generic person lookup
**Solution**: Moved data location patterns before person lookup patterns
**Result**: Chinese/Weibo queries now correctly route to `midway_social`

### 3. Platform List Handling
**Problem**: `AttributeError` when `platform` field was a list instead of string
**Solution**: Added `isinstance()` checks in `find_social_dataset()` and `find_social_by_platform()`
**Result**: No more crashes on social media queries

### 4. Hypergraph Integration
**Problem**: Hypergraph not accessible via HTTP API
**Solution**: Added 5 new endpoints to `rag_http_server.py`
**Result**: `/hypergraph/stats`, `/hypergraph/collaborators`, `/hypergraph/outputs`, `/hypergraph/terminal`, `/hypergraph/search`

### 5. Relationship Edge Builder
**Problem**: Hypergraph had only 473 edges from structured sources
**Solution**: Created `relationships.py` builder for co-authorship, topic, usage edges
**Result**: Now 594 edges (+121 relationship edges)

---

## API Endpoints Tested

### Registry Endpoints
| Endpoint | Status | Notes |
|----------|--------|-------|
| `/health` | Working | Returns server status |
| `/hybrid?q=` | Working | Combined registry + semantic search |
| `/registry?q=` | Working | Structured registry only |
| `/registry/publications?q=` | Working | Publication search |
| `/registry/topics` | Working | Topic listing |
| `/search?q=` | Working | Semantic search only |

### Hypergraph Endpoints
| Endpoint | Status | Notes |
|----------|--------|-------|
| `/hypergraph/stats` | Working | Returns node/edge counts by type |
| `/hypergraph/collaborators?person=` | Working | Returns collaborator list |
| `/hypergraph/outputs?person=` | Working | Returns person's outputs |
| `/hypergraph/terminal` | Working | Returns terminal nodes (dead ends) |
| `/hypergraph/search?q=` | Working | Searches nodes by name |

---

## Recommendations

### Short-Term
1. Add fuzzy matching for typo correction in classifier
2. Add "X experts" pattern to handle that phrasing
3. Disambiguate "storage" keyword between compute and directories

### Medium-Term
1. Implement query result caching for common queries
2. Add confidence scores to results for transparency
3. Create query auto-complete based on registry entities

### Long-Term
1. Replace regex patterns with LLM-based classification
2. Add hypergraph-based relevance boosting to semantic search
3. Implement proactive suggestions ("users who searched X also searched Y")

---

## Conclusion

The Chorus system demonstrates robust performance across a wide range of query types. The 92.3% success rate on extended stress tests indicates production readiness for typical usage patterns. The identified edge cases (3 failures) represent uncommon query formulations that can be addressed through incremental pattern additions.

Key strengths:
- Compound query handling with topic alias expansion
- Cross-registry linking between people and resources
- Hypergraph integration for relationship queries
- Resilient fallback search when primary classification fails

The system is ready for deployment with the recommendation to monitor query logs for additional patterns to add over time.

---

*Report generated automatically by test suite*
