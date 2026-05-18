# Session Report: Conversation Context & Semantic Search Improvements

**Date:** 2025-01-17
**Duration:** ~30 minutes
**Focus:** Implementing conversational context tracking and improving semantic search reliability

---

## Overview

This session implemented a comprehensive plan to add conversation context tracking to `chat.py` and improve semantic search with adaptive weighting. The implementation extracts existing patterns from `slack_bot.py` into reusable modules.

## Problem Statement

**Issues Addressed:**
- `chat.py` used a simple list with no entity tracking or pronoun resolution
- Follow-up questions like "What's his h-index?" failed (no context)
- Fixed 70/30 semantic/BM25 weighting regardless of query type
- `/semantic` and `/bm25` endpoints documented but not implemented

---

## Implementation Summary

### Phase 1: Conversation Context Module ✅

**Created:** `context/` directory with reusable context tracking

| File | Purpose |
|------|---------|
| `context/__init__.py` | Module exports |
| `context/conversation.py` | EntityMention and ConversationContext classes |

**Key Features:**
- `EntityMention` dataclass with recency decay scoring (24-hour window)
- `ConversationContext` with entity extraction, message summarization
- Automatic entity tracking via `registry/entity_extractor.py`
- `get_context_for_reformulation()` for query preprocessing
- Max 15 entities tracked, 10 recent messages, summarization at 15+ messages

### Phase 2: Query Preprocessor ✅

**Created:** `context/preprocessor.py`

**Key Features:**
- Pronoun detection regex: `his, her, their, it, that, this, these, those, etc.`
- Uses Claude 3.5 Haiku for fast, cheap reformulation (~$0.001/call)
- Async version available for FastAPI integration
- Falls back to original query on API failure

**Example Reformulations:**
```
"What's his h-index?" → "What is James Evans' h-index?"
"Tell me about their research" → "Tell me about James Evans' research"
```

### Phase 3: New RAG Endpoints ✅

**Modified:** `rag_server_fastapi.py` (lines 569-636)

| Endpoint | Description |
|----------|-------------|
| `GET /semantic` | Pure embedding-based search (no BM25) |
| `GET /bm25` | Pure keyword search (no semantic) |

**Parameters:**
- `q` - Search query (required)
- `top_k` - Number of results (1-50, default 5)
- `rerank` - Use cross-encoder reranking (semantic only)
- `mmr` - Use MMR diversity (semantic only)
- `doc_type`, `year` - Filters

**Modified:** `rag_core.py` (lines 791-933)

Added methods to `RAGServer` class:
- `search_semantic_only()` - FAISS search with reranking and MMR
- `search_bm25_only()` - BM25 keyword search with filters

### Phase 4: Adaptive Search Weighting ✅

**Created:** `rag_weighting.py`

**Query Classification:**
| Query Type | Semantic | BM25 | Example |
|------------|----------|------|---------|
| `name_lookup` | 50% | 50% | "Who is James Evans?" |
| `conceptual` | 85% | 15% | "Explain computational social science" |
| `exact_match` | 30% | 70% | `"exact phrase"` queries |
| `mixed` | 70% | 30% | Default fallback |

**Integration:** Modified `rag_core.py` line 735 to use adaptive weighting:
```python
if ADAPTIVE_WEIGHTING_AVAILABLE:
    weights = get_adaptive_weighting().calculate_weights(original_query)
    combined = weights.semantic * float(sem) + weights.bm25 * (kw_scores[idx] / max_kw)
```

### Phase 5: chat.py Integration ✅

**Modified:** `chat.py`

**Changes:**
1. Import context module (lines 25-31)
2. Initialize `ConversationContext` and `QueryPreprocessor` (lines 367-374)
3. Preprocess queries before RAG search (lines 394-397)
4. Track user messages and assistant responses (lines 417-419, 433-435)

---

## Files Changed

### New Files (4)
```
context/__init__.py          # 17 lines
context/conversation.py      # 168 lines
context/preprocessor.py      # 148 lines
rag_weighting.py             # 103 lines
```

### Modified Files (3)
```
chat.py                      # +25 lines (context integration)
rag_core.py                  # +155 lines (new methods, adaptive weighting)
rag_server_fastapi.py        # +70 lines (new endpoints)
```

---

## Verification Results

### Module Import Tests ✅
```
EntityExtractor loaded: 137 people, 64 datasets, 6 projects, 218 topics
ConversationContext: OK (2 messages, 1 entity tracked)
QueryPreprocessor: OK
AdaptiveWeighting: OK
```

### Pronoun Detection Tests ✅
```
✓ "What is his h-index?" -> True
✓ "Tell me about their research" -> True
✓ "What does she work on?" -> True
✓ "What is APTO?" -> False
✓ "Explain machine learning" -> False
✓ "How is that related to the lab?" -> True
```

### Syntax Validation ✅
All 7 files passed `python -m py_compile`

---

## Usage Examples

### Test Conversational Context
```bash
python chat.py
> Who is James Evans?
> What's his h-index?        # Resolves "his" → James Evans
> Tell me about his research  # Maintains context
```

### Test New Endpoints
```bash
# Pure semantic search
curl "http://localhost:8765/semantic?q=machine+learning&top_k=5"

# Pure BM25 keyword search
curl "http://localhost:8765/bm25?q=James+Evans&top_k=5"
```

### Test Adaptive Weighting
```python
from rag_weighting import get_adaptive_weighting
aw = get_adaptive_weighting()

# Name lookup: 50/50
aw.calculate_weights("Who is James Evans?")
# → SearchWeights(semantic=0.50, bm25=0.50, query_type='name_lookup')

# Conceptual: 85/15
aw.calculate_weights("Explain computational social science")
# → SearchWeights(semantic=0.85, bm25=0.15, query_type='conceptual')
```

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                        chat.py                               │
│  ┌──────────────────┐    ┌─────────────────────────────┐    │
│  │ ConversationCtx  │    │     QueryPreprocessor       │    │
│  │  - entities      │───>│  - needs_reformulation()    │    │
│  │  - messages      │    │  - reformulate() → Haiku    │    │
│  └──────────────────┘    └─────────────────────────────┘    │
│           │                          │                       │
│           v                          v                       │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                    query_rag()                       │    │
│  │           (with reformulated query)                  │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                              │
                              v
┌─────────────────────────────────────────────────────────────┐
│                    rag_server_fastapi.py                     │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐                │
│  │ /search   │  │ /semantic │  │  /bm25    │                │
│  │ (hybrid)  │  │ (pure)    │  │  (pure)   │                │
│  └───────────┘  └───────────┘  └───────────┘                │
└─────────────────────────────────────────────────────────────┘
                              │
                              v
┌─────────────────────────────────────────────────────────────┐
│                       rag_core.py                            │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                  AdaptiveWeighting                   │    │
│  │  - name_lookup: 50/50   - conceptual: 85/15         │    │
│  │  - exact_match: 30/70   - mixed: 70/30              │    │
│  └─────────────────────────────────────────────────────┘    │
│  ┌─────────────┐  ┌──────────────────┐  ┌─────────────┐    │
│  │ FAISS Index │  │ _search_pipeline │  │ BM25 Index  │    │
│  │ (semantic)  │──│ (hybrid scoring) │──│ (keyword)   │    │
│  └─────────────┘  └──────────────────┘  └─────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

---

## Next Steps

1. **Test with RAG server running** - Verify endpoints work in production
2. **Run test_conversation.py** - Validate multi-turn context handling
3. **Monitor Haiku costs** - Track reformulation API usage
4. **Optional: Refactor slack_bot.py** - Use shared context module

---

## Session Metrics

- **Files created:** 4
- **Files modified:** 3
- **Lines added:** ~550
- **Tests passed:** All syntax and import tests
- **Dependencies:** None added (uses existing anthropic SDK)
