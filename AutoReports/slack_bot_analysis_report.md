# CHORUS Slack Bot Query Analysis Report

**Date:** January 10, 2026
**Analyst:** Claude Opus 4.5
**RAG Server Status:** Online (11,188 chunks indexed)
**Registry Status:** Active (71 people, 3 projects, 3 funding sources, 10 datasets)

---

## Executive Summary

The CHORUS Slack bot is a multi-agent system designed to help Knowledge Lab members with information lookup, technical support, research collaboration, and intellectual exploration. This analysis evaluates query handling across five categories and identifies issues requiring attention.

**Overall Assessment:** The system is **largely functional** but has several areas requiring improvement:
- **Strengths:** Excellent structured data retrieval, good topic alias expansion, strong technical query handling
- **Weaknesses:** Agent classifier accuracy issues (36%), some misrouted queries, Chorus self-awareness queries unhandled
- **Critical Issue:** LLM fallback for classifier fails due to API key accessibility in subprocess

---

## 1. System Architecture Overview

### Query Flow
```
User Query (Slack)
       |
       v
[ChorusOrchestrator]
       |
       v
[QueryClassifier] -----> Regex patterns (fast path)
       |                       |
       |                       v
       |              LLM fallback (Haiku) if ambiguous
       |
       v
[Specialist Agent Routing]
       |
   +---+---+---+---+
   |       |       |       |
   v       v       v       v
Memory  Mechanic  Muse  Matchmaker
(Haiku) (Haiku) (Sonnet) (Haiku)
       |
       v
[RAG Server - /hybrid endpoint]
       |
   +---+---+
   |       |
   v       v
Registry  Semantic
Lookup    Search
```

### Agent Responsibilities
| Agent | Model | Purpose | RAG Endpoint |
|-------|-------|---------|--------------|
| Memory | Haiku | Fast factual retrieval | `/hybrid` |
| Mechanic | Haiku | Technical troubleshooting | `/search` |
| Muse | Sonnet | Deep analysis, exploration | `/search` |
| Matchmaker | Haiku | Finding connections | `/search` |

---

## 2. Test Results by Category

### Category 1: Information Lookup (Memory Agent)

#### Person Queries
| Query | RAG Classification | Structured Results | Status |
|-------|-------------------|-------------------|--------|
| "Who is James Evans?" | structured (people) | Yes - full profile with OpenAlex data | PASS |
| "Tell me about Julia Koschinsky" | hybrid (people) | Yes - full profile | PASS |
| "Who is Jake?" | structured (people) | Yes - matched Jake Burchard | PASS |

**Issues Found:**
- The OpenAlex data for "James Evans" appears to match a different person (wrong publications shown - chemistry/materials science instead of sociology of science)
- This is a data quality issue in the registry, not the bot

#### Topic Queries
| Query | Classification | Results Found | Status |
|-------|---------------|---------------|--------|
| "Who works on NLP?" | structured (people, topics) | 1+ people (Doug Downey, etc.) | PASS |
| "Who works on machine learning?" | structured (people, topics) | Multiple researchers | PASS |
| "Researchers in network science" | structured (people, topics) | Compound query detected | PASS |

**Topic Alias Expansion Working:**
- "NLP" correctly expands to "natural language processing", "computational linguistics", etc.
- "ML" expands to "machine learning"
- Network science topic found 9 people with "Complex Network Analysis Techniques"

#### Publication Queries
| Query | Classification | Results | Status |
|-------|---------------|---------|--------|
| "Papers by James Evans" | structured (publications) | Found via co-author matching | PASS |
| "Recent work on embeddings" | hybrid (publications) | Semantic search fallback | PARTIAL |

#### Resource Queries
| Query | Classification | Results | Status |
|-------|---------------|---------|--------|
| "What compute resources are available?" | hybrid (compute) | 6 compute resources | PASS |
| "What datasets do we have?" | structured (datasets) | 10 datasets | PASS |
| "What grants does the lab have?" | hybrid (funding) | 3 funding sources | PASS |

#### Project Queries
| Query | Classification | Results | Status |
|-------|---------------|---------|--------|
| "Tell me about APTO" | structured (people) | **MISCLASSIFIED** - No results | FAIL |
| "What projects do we have?" | structured (projects) | 3 projects found | PASS |

**Issue:** "Tell me about APTO" is classified as a person lookup, not a project lookup. The regex pattern matches "Tell me about X" as a person query. APTO information is only found via semantic search.

---

### Category 2: Questions About Chorus Itself

| Query | Agent Classification | Expected | Status |
|-------|---------------------|----------|--------|
| "What can you help me with?" | ambiguous -> Muse | Works via fallback | PARTIAL |
| "How do I use you?" | ambiguous -> Muse | Works via fallback | PARTIAL |
| "What do you know about the lab?" | ambiguous -> Muse | Works via fallback | PARTIAL |

**Issue:** These queries fall through to LLM classification but the LLM fails due to API key issues in the test environment. In production with proper environment variables, these would route to Muse (explore) agent.

**Recommendation:** Add explicit regex patterns for Chorus self-awareness queries:
- Pattern: `what can you (help|do|assist)` -> explore
- Pattern: `how (do i|can i|to) use you` -> explore
- Pattern: `what (do you know|can you tell me) about` -> lookup OR explore

---

### Category 3: Project Collaboration (Muse/Matchmaker)

| Query | Classification | Routing | Status |
|-------|---------------|---------|--------|
| "I'm working on a paper about social networks - who should I talk to?" | connect | Matchmaker | PARTIAL |
| "Help me think through a research question about AI ethics" | explore | Muse | PASS |
| "Can you help me brainstorm ideas for a grant proposal?" | ambiguous | Muse (fallback) | PARTIAL |

**Issues:**
- "I'm working on a paper about social networks - who should I talk to?" gets classified as `publications` query by the registry classifier, missing the collaboration intent
- The agent classifier (regex) correctly identifies this as "connect" but the RAG response focuses on publications, not people

**Recommendation:** For collaboration queries, ensure the response includes:
1. People who work on the topic
2. Their contact information
3. Suggested next steps for connecting

---

### Category 4: Practical/Technical (Mechanic Agent)

| Query | Classification | Confidence | Status |
|-------|---------------|-----------|--------|
| "How do I access Midway3?" | technical | 0.85 | PASS |
| "What's the process for getting GPU access?" | technical | 0.90 | PASS |
| "How do I use SLURM?" | technical | 0.90 | PASS |

**Excellent Results:** All technical queries are correctly classified and routed to the Mechanic agent with high confidence. The regex patterns work well:
- `\b(midway|cluster|slurm|conda|pip|docker)\b`
- `\b(access|permission|login|credential|ssh|vpn)\b`

**Note:** The Mechanic agent uses `/search` endpoint, not `/hybrid`. For Midway queries, the structured registry data (from `/hybrid`) would provide better information. Consider updating Mechanic to use `/hybrid` as well.

---

### Category 5: Follow-up Queries (Thread Memory)

#### Thread Memory Architecture
```python
class ThreadConversation:
    messages: List[Dict[str, str]]  # Last 20 messages
    last_activity: float
    bot_active: bool  # Auto-respond after first @mention

class ThreadMemoryManager:
    threads: Dict[str, ThreadConversation]
    ttl_hours: int = 24
```

#### Context Injection Flow
1. User query arrives
2. Thread conversation history retrieved (last 10 messages)
3. Context string built: "Previous conversation in this thread:\nUser: ...\nChorus: ..."
4. Augmented query sent to agent

#### Test Scenarios

| Sequence | Query | Expected Behavior | Status |
|----------|-------|-------------------|--------|
| 1 | "Who is James Evans?" | Fresh lookup | PASS |
| 2 | "What's his h-index?" | Should use pronoun resolution from context | DESIGNED |
| 3 | "Who works on similar topics?" | Should reference previous person | DESIGNED |

**Design Assessment:** The thread memory system is well-designed:
- Context string includes last 10 messages
- Pronoun resolution depends on LLM capability
- 24-hour TTL prevents stale context
- 20-message limit keeps context manageable

**Potential Issues:**
- No explicit entity tracking (relies on LLM understanding "his" = "James Evans")
- Context truncation (500 chars per message) could lose important details
- No mechanism to reset context if user switches topics mid-thread

---

## 3. Classifier Analysis

### Agent Classifier (agents/classifier.py)

**Accuracy:** 36.4% (8/22 queries matched expected classification)

**Root Cause:** The regex-based classifier has good patterns but:
1. Many queries fall through to LLM fallback
2. LLM fallback fails if API key not properly set
3. Returns "ambiguous" with 0.5 confidence for many legitimate queries

**Pattern Coverage Analysis:**
| Intent | Pattern Examples | Coverage |
|--------|-----------------|----------|
| LOOKUP | "who is", "what is the", "list", "show me" | Good |
| TECHNICAL | "access", "setup", "error", "midway", "slurm" | Excellent |
| CONNECT | "who's working on", "connect me", "collaborate" | Good |
| EXPLORE | "synthesize", "analyze", "think through", "what if" | Moderate |

**Missing Patterns:**
- "Tell me about X" (currently no match, falls through)
- "Papers by X" (no match)
- Chorus self-reference ("what can you help me with")

### Registry Classifier (registry/classifier.py)

**Accuracy:** 82.6% query type, 91.3% entity type

**Strengths:**
- Extensive regex patterns for structured queries
- Good compound query detection (institution + role + topic)
- Person name extraction
- Topic alias expansion

**Issues Found:**
- "Tell me about APTO" -> classified as person lookup (matches person pattern before project pattern)
- "Recent work on embeddings" -> hybrid instead of structured
- "Stanford faculty working on AI" -> topic not detected

---

## 4. RAG Server Analysis

### Hybrid Endpoint Performance

The `/hybrid` endpoint combines:
1. **Structured Registry Lookup** - Fast, deterministic queries against lab_registry.json
2. **Semantic Search** - FAISS + BM25 hybrid with cross-encoder reranking

**Features Working:**
- LLM query reformulation (Haiku)
- Cross-encoder reranking (ms-marco-MiniLM)
- MMR diversity (lambda=0.7)
- Topic alias expansion
- Compound query handling

**Data Quality Issues:**
| Issue | Example | Impact |
|-------|---------|--------|
| Wrong OpenAlex match | James Evans shows chemistry papers | High - misleading data |
| Missing project patterns | APTO not recognized | Medium |
| Sparse documentation | Semantic search returns chunks.json | Low |

---

## 5. Identified Issues Summary

### Critical Issues
1. **OpenAlex Data Mismatch** - James Evans profile shows wrong publications
2. **Agent Classifier LLM Fallback** - API key not accessible in test subprocess
3. **APTO Project Misclassification** - "Tell me about APTO" routes to person lookup

### High Priority Issues
4. **Collaboration Query Routing** - "Who should I talk to about X" classified as publication query
5. **Chorus Self-Awareness** - No patterns for "what can you help me with"
6. **Mechanic Using Wrong Endpoint** - Should use `/hybrid` for structured resource data

### Medium Priority Issues
7. **Pattern Order in Registry Classifier** - Person patterns match before project patterns
8. **Thread Context Truncation** - 500 char limit may lose important details
9. **Topic Extraction Gaps** - Some compound queries miss topic component

### Low Priority Issues
10. **Entity Tracking in Threads** - No explicit entity resolution for pronouns
11. **Cache Key Collision Risk** - Simple MD5 hash could collide on similar queries

---

## 6. Recommendations

### Immediate Fixes

1. **Update Registry Classifier Pattern Order**
```python
# Move project patterns BEFORE generic person lookup
STRUCTURED_PATTERNS = [
    # ...existing specific patterns...

    # === Project patterns (before person lookup) ===
    (r"(?:tell\s+me\s+about|describe|what\s+is)\s+(?:the\s+)?(apto|c3s2|...)", ["projects"], "project_lookup"),

    # === Person lookup (after projects) ===
    (r"(?:tell\s+me\s+about|describe)\s+(.+?)(?:\?|$)", ["people"], "person_lookup"),
]
```

2. **Add Chorus Self-Awareness Patterns to Agent Classifier**
```python
EXPLORE_PATTERNS = [
    # Existing patterns...
    r"\bwhat can you\b",
    r"\bhow (do|can|should) i use you\b",
    r"\bwhat do you know\b",
    r"\bwhat are you\b",
]
```

3. **Fix OpenAlex Data** - Re-run OpenAlex matching for James Evans with ORCID verification

### Short-term Improvements

4. **Update Mechanic to Use Hybrid Endpoint**
```python
async def _handle_tool(self, name: str, inputs: dict) -> str:
    if name == "search_knowledge_base":
        # Use /hybrid for better structured data
        result = await call_rag_async("/hybrid", {
            "q": inputs["query"],
            "top_k": inputs.get("num_results", 5)
        })
```

5. **Enhance Collaboration Query Handling**
   - When query contains "who should I talk to" + topic, force include people results
   - Add contact info extraction to Matchmaker responses

### Long-term Improvements

6. **Entity Tracking in Thread Memory**
   - Maintain list of mentioned entities per thread
   - Use entity list to resolve pronouns before sending to agent

7. **Query Intent Refinement**
   - Add confidence calibration to classifier
   - Use ensemble of regex + keyword + LLM for ambiguous queries

8. **Response Quality Monitoring**
   - Log query -> classification -> response for analysis
   - Track user follow-ups to identify missed intents

---

## 7. Test Suite

### Recommended Automated Tests

```python
# tests/test_slack_bot_queries.py

import pytest
from agents.classifier import QueryClassifier
from registry import RegistryRAG

class TestQueryClassification:

    @pytest.fixture
    def classifier(self):
        return QueryClassifier()

    @pytest.fixture
    def registry_rag(self):
        return RegistryRAG()

    # Category 1: Information Lookup
    @pytest.mark.parametrize("query,expected_intent", [
        ("Who is James Evans?", "lookup"),
        ("Tell me about Julia Koschinsky", "lookup"),
        ("Who works on NLP?", "lookup"),
        ("Papers by James Evans", "lookup"),
        ("What datasets do we have?", "lookup"),
        ("Tell me about APTO", "lookup"),  # Currently FAILING
    ])
    def test_lookup_classification(self, classifier, query, expected_intent):
        result = classifier.classify(query)
        assert result.intent == expected_intent

    # Category 2: Chorus Self-Awareness
    @pytest.mark.parametrize("query,expected_intent", [
        ("What can you help me with?", "explore"),
        ("How do I use you?", "explore"),
        ("What do you know about the lab?", "explore"),
    ])
    def test_self_awareness_classification(self, classifier, query, expected_intent):
        result = classifier.classify(query)
        assert result.intent == expected_intent

    # Category 3: Collaboration
    @pytest.mark.parametrize("query,expected_intent", [
        ("I'm working on a paper about social networks - who should I talk to?", "connect"),
        ("Help me think through a research question about AI ethics", "explore"),
    ])
    def test_collaboration_classification(self, classifier, query, expected_intent):
        result = classifier.classify(query)
        assert result.intent == expected_intent

    # Category 4: Technical
    @pytest.mark.parametrize("query,expected_intent", [
        ("How do I access Midway3?", "technical"),
        ("What's the process for getting GPU access?", "technical"),
        ("How do I use SLURM?", "technical"),
    ])
    def test_technical_classification(self, classifier, query, expected_intent):
        result = classifier.classify(query)
        assert result.intent == expected_intent

class TestRegistryRAG:

    @pytest.fixture
    def rag(self):
        return RegistryRAG()

    def test_person_lookup(self, rag):
        result = rag.query("Who is James Evans?")
        assert result["classification"]["type"] == "structured"
        assert len(result["structured_results"]) > 0
        assert result["structured_results"][0]["type"] == "person"

    def test_topic_search_with_alias(self, rag):
        result = rag.query("Who works on NLP?")
        assert "people" in result["classification"]["entity_types"]
        # Should find people with natural language processing topics
        assert len(result["structured_results"]) > 0

    def test_compound_query(self, rag):
        result = rag.query("UChicago researchers in network science")
        assert result["classification"]["is_compound"] == True
        assert "institution" in result["classification"]["compound_filters"]

    def test_resource_query(self, rag):
        result = rag.query("What compute resources are available?")
        assert "compute" in result["classification"]["entity_types"]
        assert len(result["structured_results"]) > 0
```

---

## 8. Conclusion

The CHORUS Slack bot demonstrates solid foundational architecture with:
- Well-designed multi-agent routing
- Comprehensive RAG system with hybrid search
- Good thread memory for conversation context
- Extensive topic alias support

**Key areas requiring attention:**
1. Agent classifier accuracy (add missing patterns, fix LLM fallback)
2. Registry classifier pattern ordering (projects before persons)
3. OpenAlex data quality verification
4. Collaboration query handling improvements

With the recommended fixes, the system should handle the expected query types from Knowledge Lab members effectively.

---

*Report generated by Claude Opus 4.5 analyzing the CHORUS codebase and testing against the live RAG server.*
