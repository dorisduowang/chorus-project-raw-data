# Chorus Memory Architecture: Current State & Improvements

This document summarizes the current memory implementation in Chorus and outlines potential improvements for both in-session and cross-session memory.

---

## Current State

### In-Session Memory (Conversation Context)

| Interface | Storage | Window Size | Format | Persistence |
|-----------|---------|-------------|--------|-------------|
| **Slack Bot** | `ThreadConversation` class | 20 msgs stored, 10 used for context | String (formatted) | Runtime only, 24hr TTL |
| **Core Agent** | None (stateless) | N/A | N/A | N/A |
| **Memory Agent** | Passed via `context` param | External (from caller) | String | None |
| **Chat CLI** | Python list | 20 msgs, truncates to 10 | Message objects | Runtime only |

#### How It Works Today

**Slack Bot** (`slack_bot.py`):
```python
class ThreadConversation:
    messages: List[Dict]      # Last 20 messages
    last_activity: datetime   # For TTL cleanup
    bot_active: bool          # Respond without @mention after activation

    def get_context_string(self) -> str:
        # Returns last 10 messages as "User: ...\nChorus: ..."
        # Each message truncated to 500 chars
```

**Memory Agent** (`agents/specialists/memory.py`):
```python
async def _query_fast(self, message: str, context: Dict):
    thread_context = context.get("thread_context", "")

    # Reformulate pronouns: "his h-index" → "James Evans's h-index"
    search_query = await self._reformulate_query(message, thread_context)

    # Inject context into system prompt
    system_prompt = MEMORY_PROMPT_FAST.format(
        conversation_context=thread_context,
        rag_context=rag_results
    )
```

**Chat CLI** (`chat.py`):
```python
conversation = []
# ... after each turn ...
if len(conversation) > 20:
    conversation = conversation[-10:]  # Hard truncation
```

#### Current Limitations

1. **Abrupt Context Loss**: When window fills, old messages are dropped entirely
2. **No Semantic Compression**: No summarization of "what we discussed"
3. **Cache Ignores Context**: `core_agent.py` caches by query hash only - "What's their h-index?" returns same result regardless of who "they" refers to
4. **String-Based Context**: Opaque format, can't prioritize important vs. filler messages
5. **No Topic Tracking**: Can't detect conversation shifts or maintain topic-specific context
6. **Stateless Core**: `core_agent.py` treats each query independently

---

### Cross-Session Memory

**Current state: None implemented**

- Conversations don't persist across bot restarts
- No way to reference "what we discussed yesterday"
- No user preference learning
- No conversation history search

---

## Proposed Improvements

### Phase 1: Enhanced In-Session Memory

#### 1.1 Structured Context Format

Replace string-based context with structured format:

```python
@dataclass
class ConversationContext:
    topic: Optional[str]                    # "James Evans's publications"
    entities_mentioned: List[str]           # ["james-evans", "apto-project"]
    recent_turns: List[Dict]                # Last 5-10 exchanges
    summary: Optional[str]                  # Compressed older context

    def to_prompt_context(self) -> str:
        """Generate context for system prompt."""
        parts = []
        if self.topic:
            parts.append(f"Current topic: {self.topic}")
        if self.entities_mentioned:
            parts.append(f"Entities discussed: {', '.join(self.entities_mentioned)}")
        if self.summary:
            parts.append(f"Earlier in conversation: {self.summary}")
        parts.append("Recent exchanges:")
        for turn in self.recent_turns[-5:]:
            parts.append(f"  {turn['role']}: {turn['content'][:200]}")
        return "\n".join(parts)
```

**Benefits:**
- Enables intelligent context selection
- Topic-aware caching
- Better pronoun resolution

#### 1.2 Context Summarization

When window fills, compress old messages instead of dropping:

```python
async def summarize_context(self, messages: List[Dict]) -> str:
    """Compress old messages into a summary."""
    prompt = f"""Summarize this conversation excerpt in 2-3 sentences.
    Focus on: topics discussed, decisions made, questions asked.

    {format_messages(messages)}

    Summary:"""

    response = await self.client.messages.create(
        model="claude-3-5-haiku-20241022",
        max_tokens=150,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text
```

**Window management strategy:**
```
[Summary of msgs 1-10] + [Full msgs 11-20] + [Current query]
     ↓ when msgs 21+ arrive ↓
[Summary of msgs 1-20] + [Full msgs 21-30] + [Current query]
```

#### 1.3 Entity Tracking

Track entities mentioned to improve context resolution:

```python
class EntityTracker:
    """Track entities mentioned in conversation."""

    def __init__(self):
        self.entities: Dict[str, EntityMention] = {}

    def update(self, message: str, response: str):
        """Extract and track entities from exchange."""
        # Use existing EntityExtractor from registry module
        from registry import EntityExtractor
        extractor = EntityExtractor()

        for text in [message, response]:
            entities = extractor.extract(text)
            for person_id in entities.people:
                self._track(person_id, "person", text)
            for project_id in entities.projects:
                self._track(project_id, "project", text)

    def resolve_pronoun(self, pronoun: str) -> Optional[str]:
        """Resolve 'he/she/they/it' to most recent entity."""
        # Return most recently mentioned entity of appropriate type
```

#### 1.4 Context-Aware Caching

Modify cache key to include conversation context:

```python
def _get_cache_key(self, query: str, context: ConversationContext) -> str:
    """Generate cache key that includes context."""
    # Include dominant entity in cache key
    entity_key = context.entities_mentioned[0] if context.entities_mentioned else ""
    topic_key = context.topic or ""

    combined = f"{query}|{entity_key}|{topic_key}"
    return "chorus:" + hashlib.md5(combined.lower().encode()).hexdigest()
```

---

### Phase 2: Cross-Session Memory

#### 2.1 Conversation Storage

Persist conversations for later reference:

```python
# Schema for SQLite storage
CREATE TABLE conversations (
    id TEXT PRIMARY KEY,
    channel_id TEXT,           -- Slack channel or "cli"
    thread_id TEXT,            -- Slack thread_ts or session UUID
    user_id TEXT,
    started_at TIMESTAMP,
    last_activity TIMESTAMP,
    topic TEXT,
    summary TEXT,
    message_count INTEGER
);

CREATE TABLE messages (
    id INTEGER PRIMARY KEY,
    conversation_id TEXT REFERENCES conversations(id),
    role TEXT,                 -- "user" or "assistant"
    content TEXT,
    timestamp TIMESTAMP,
    entities_mentioned TEXT,   -- JSON array
    tool_calls TEXT            -- JSON array of tools used
);

CREATE INDEX idx_conv_user ON conversations(user_id);
CREATE INDEX idx_conv_topic ON conversations(topic);
CREATE INDEX idx_msg_conv ON messages(conversation_id);
```

#### 2.2 Conversation Search

Enable searching past conversations:

```python
class ConversationMemory:
    """Long-term conversation storage and retrieval."""

    async def search(
        self,
        query: str,
        user_id: Optional[str] = None,
        days_back: int = 30
    ) -> List[ConversationSummary]:
        """Search past conversations semantically."""
        # Embed query
        query_embedding = self.model.encode(query)

        # Search conversation summaries
        results = self.index.search(
            query_embedding,
            filter={"user_id": user_id} if user_id else None
        )

        return [
            ConversationSummary(
                id=r.id,
                topic=r.metadata["topic"],
                summary=r.metadata["summary"],
                date=r.metadata["date"],
                relevance=r.score
            )
            for r in results
        ]

    async def recall(self, conversation_id: str) -> List[Dict]:
        """Retrieve full conversation history."""
        return self.db.query(
            "SELECT role, content, timestamp FROM messages "
            "WHERE conversation_id = ? ORDER BY timestamp",
            [conversation_id]
        )
```

#### 2.3 User Preference Learning

Track user preferences over time:

```python
class UserPreferences:
    """Learn and apply user preferences."""

    preferences: Dict[str, Any] = {
        "response_length": "medium",      # short/medium/long
        "formality": "casual",            # casual/formal
        "detail_level": "high",           # low/medium/high
        "preferred_topics": [],           # frequently asked topics
        "expertise_areas": [],            # areas user knows well
    }

    def update_from_conversation(self, messages: List[Dict]):
        """Infer preferences from conversation patterns."""
        # Track question types, follow-up patterns, feedback signals

    def apply_to_prompt(self, base_prompt: str) -> str:
        """Customize system prompt based on preferences."""
        customizations = []
        if self.preferences["response_length"] == "short":
            customizations.append("Be concise. Aim for 2-3 sentences unless detail is requested.")
        if self.preferences["formality"] == "formal":
            customizations.append("Use formal language appropriate for professional communication.")

        return base_prompt + "\n\nUser preferences:\n" + "\n".join(customizations)
```

#### 2.4 Conversation Continuity

Enable resuming past conversations:

```python
# New tool for agents
{
    "name": "recall_conversation",
    "description": "Search and recall past conversations with the user",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to search for in past conversations"
            },
            "days_back": {
                "type": "integer",
                "description": "How many days back to search (default 30)"
            }
        },
        "required": ["query"]
    }
}
```

Usage in conversation:
```
User: "What did we decide about the APTO data pipeline last week?"

Chorus: [uses recall_conversation tool]
        "Last Tuesday we discussed the APTO pipeline. You decided to:
         1. Use incremental indexing instead of full rebuilds
         2. Add entity extraction during ingestion
         3. Target 5-minute refresh cycles

         Want me to pull up the specific technical details?"
```

---

## Implementation Priority

### High Impact, Lower Effort
1. **Structured Context Format** - Foundation for other improvements
2. **Entity Tracking** - Improves pronoun resolution immediately
3. **Context-Aware Cache Keys** - Fixes incorrect cached responses

### High Impact, Higher Effort
4. **Context Summarization** - Requires LLM calls, needs latency optimization
5. **Conversation Storage** - Database schema, migration, maintenance
6. **Conversation Search** - Embedding + indexing infrastructure

### Medium Impact
7. **User Preference Learning** - Nice-to-have, improves UX over time
8. **Cross-Interface Memory** - Reference Slack from CLI (complex)

---

## Architecture Considerations

### Token Efficiency

Current cost per conversation turn:
- System prompt: ~1,500 tokens
- Context (10 messages): ~500-1,000 tokens
- RAG results: ~500-2,000 tokens
- User query: ~50-200 tokens
- **Total input: ~2,500-4,700 tokens**

With improvements:
- Cached system prompt: 0 tokens (prompt caching)
- Compressed context: ~200-400 tokens (summarization)
- Selective RAG: ~300-800 tokens (topic-aware retrieval)
- **Total input: ~550-1,400 tokens (60-70% reduction)**

### Latency Considerations

| Operation | Current | With Improvements |
|-----------|---------|-------------------|
| Context preparation | ~0ms | ~50-100ms (entity tracking) |
| Summarization | N/A | ~500-800ms (when triggered) |
| Cache lookup | ~1ms | ~2-3ms (context-aware key) |
| Conversation storage | N/A | ~5-10ms (async write) |

### Storage Requirements

For cross-session memory (estimated):
- 100 users, 10 conversations/week, 20 messages/conversation
- ~200KB/week raw storage
- ~50KB/week with summarization
- Embeddings: ~4KB per conversation summary
- **Monthly: ~1-2MB storage, ~200KB embeddings**

---

## Industry Best Practices

This section summarizes memory architecture patterns from leading LLM frameworks and applications.

### LangChain Memory Types

[LangChain's memory documentation](https://docs.langchain.com/oss/python/langgraph/memory) defines a hierarchy of memory strategies:

| Memory Type | Mechanism | Token Cost | Best For |
|-------------|-----------|------------|----------|
| **ConversationBufferMemory** | Store all messages raw | High (linear growth) | Short conversations (<10 turns) |
| **ConversationBufferWindowMemory** | Sliding window of N messages | Medium (fixed) | Most applications |
| **ConversationSummaryMemory** | LLM summarizes progressively | Low (compressed) | Long conversations |
| **ConversationSummaryBufferMemory** | Hybrid: summary + recent buffer | Medium | Balance of context + detail |
| **EntityMemory** | Track entities separately | Low | Entity-heavy domains |
| **VectorStoreMemory** | Embed messages, retrieve relevant | Variable | Very long histories |

**Key insight from [Pinecone](https://www.pinecone.io/learn/series/langchain/langchain-conversational-memory/):** "When conversations hit 30+ messages, the model forgets names, contradicts itself, or starts hallucinating details. The problem is rarely 'I need a bigger model.' It's almost always 'my memory strategy is weak.'"

### ChatGPT's Four-Layer Architecture

[Reverse engineering of ChatGPT's memory](https://medium.com/aimonks/inside-chatgpts-memory-how-the-most-sophisticated-memory-system-in-ai-really-works-f2b3f32d86b3) reveals a sophisticated multi-layer system:

| Layer | Content | Persistence | Token Budget |
|-------|---------|-------------|--------------|
| **Session Metadata** | Device info, usage patterns, preferences | Ephemeral | ~100 tokens |
| **Saved Memories** | Explicit facts ("I work at X", "I prefer Y") | Persistent | ~200-500 tokens |
| **Conversation Summaries** | `<timestamp>: <title> \|\| key points` | Persistent | ~100-300 tokens |
| **Current Conversation** | Full recent messages (sliding window) | Session only | Variable |

**Critical insight:** ChatGPT does **not** use RAG for conversation memory. No vector database, no embedding search. Instead:
- Facts are extracted explicitly and stored as structured data
- Summaries are lightweight text digests
- Entity extraction identifies Person, Organization, Event, Location
- Memory updates happen in background, not blocking responses

### Memory Update Strategies

From [LangChain's agent memory research](https://blog.langchain.com/memory-for-agents/):

| Strategy | Latency Impact | Freshness | Complexity |
|----------|---------------|-----------|------------|
| **Inline (synchronous)** | +200-500ms per turn | Immediate | Low |
| **Background (async)** | None | Delayed 1-2 turns | Medium |
| **Triggered (on-demand)** | Occasional spikes | Configurable | High |
| **Batched** | None | Delayed N turns | Medium |

**Recommended pattern:** Background updates with periodic sync. Extract entities and update summaries asynchronously after each turn without blocking the response.

### Long-Term Memory Patterns

From [LangChain's LangMem framework](https://langchain-ai.github.io/langmem/concepts/conceptual_guide/):

**Semantic Memory** (facts about the world):
- Collections: Unbounded, searchable knowledge stores
- Profiles: Strict schema for task-specific info (user preferences, project details)
- Example: "User works at Knowledge Lab on computational social science"

**Episodic Memory** (past experiences):
- Records of past actions and outcomes
- Implemented as few-shot examples in prompts
- Example: "Last time user asked about X, they wanted technical detail not overview"

**Procedural Memory** (how to do things):
- Stored as tool definitions or action templates
- Updated based on success/failure of past actions

### Entity Extraction Patterns

From [analysis of ChatGPT's entity layer](https://wordlift.io/blog/en/chatgpt-named-entities/):

ChatGPT uses lightweight Named Entity Recognition (NER) with categories:
- **Person**: Names of people discussed
- **Organization**: Companies, labs, institutions
- **Location**: Places mentioned
- **Event**: Meetings, deadlines, conferences
- **Product**: Tools, datasets, systems

Entities are tracked with recency weighting—most recently mentioned entities are prioritized for pronoun resolution.

### Anti-Patterns to Avoid

1. **Full transcript storage**: Storing every message verbatim wastes tokens and creates noise
2. **RAG for conversation memory**: Overkill for most use cases; simple entity/keyword lookup is faster
3. **Synchronous memory updates**: Blocks responses, degrades UX
4. **Single-layer memory**: Missing the hierarchy (facts vs. context vs. history)
5. **No decay/forgetting**: Old irrelevant context pollutes current responses

---

## Chorus vs. Industry Best Practices

### Comparison Matrix

| Capability | ChatGPT | LangChain Best Practice | Chorus Current | Gap |
|------------|---------|------------------------|----------------|-----|
| **Memory Layers** | 4 layers (metadata, facts, summaries, current) | 2-3 layers typical | 1 layer (raw messages) | Major |
| **Entity Tracking** | Yes, with NER | EntityMemory class | None (has EntityExtractor unused) | Major |
| **Summarization** | Automatic, background | ConversationSummaryMemory | None (hard truncation) | Major |
| **Cross-Session** | Yes, persistent facts | VectorStoreMemory | None | Major |
| **Update Strategy** | Background async | Configurable | Synchronous (implicit) | Moderate |
| **Token Efficiency** | Highly optimized | Configurable | Unoptimized | Moderate |
| **Pronoun Resolution** | Via entity tracking | Via EntityMemory | Via LLM reformulation | Minor |
| **Context Format** | Structured + templated | Flexible | String-based | Minor |

### Detailed Gap Analysis

#### 1. Memory Architecture

**Industry standard (ChatGPT model):**
```
┌─────────────────────────────────────────────────┐
│ Layer 4: Current Conversation (sliding window)  │ ← Session
├─────────────────────────────────────────────────┤
│ Layer 3: Conversation Summaries (compressed)    │ ← Persistent
├─────────────────────────────────────────────────┤
│ Layer 2: Saved Facts ("user prefers X")         │ ← Persistent
├─────────────────────────────────────────────────┤
│ Layer 1: Session Metadata (device, patterns)    │ ← Ephemeral
└─────────────────────────────────────────────────┘
```

**Chorus current:**
```
┌─────────────────────────────────────────────────┐
│ Single Layer: Raw messages (truncated to 10)    │ ← Session only
└─────────────────────────────────────────────────┘
```

**Gap:** Chorus has no separation between facts, summaries, and current context. Everything is treated equally and truncated uniformly.

#### 2. Entity Handling

**Industry standard:**
- Dedicated entity memory with recency tracking
- Automatic entity extraction on every turn
- Pronoun resolution via entity lookup (fast, deterministic)

**Chorus current:**
- `EntityExtractor` exists but unused in conversation flow
- Pronoun resolution via LLM reformulation (slow, costs API call)
- No entity persistence across turns

**Gap:** Chorus has the building blocks (`EntityExtractor`) but doesn't use them for conversation memory.

#### 3. Context Compression

**Industry standard:**
- Progressive summarization as conversation grows
- Summary + buffer hybrid (old summarized, recent full)
- Background compression to avoid latency

**Chorus current:**
- Hard truncation: 20 messages → keep last 10
- No summarization
- Complete loss of older context

**Gap:** Major. Users lose all context from early conversation with no summary.

#### 4. Cross-Session Memory

**Industry standard:**
- Persistent fact storage (user preferences, known entities)
- Conversation summaries indexed for search
- "What did we discuss last week?" capability

**Chorus current:**
- No persistence (runtime only, 24hr TTL)
- No conversation search
- Every session starts fresh

**Gap:** Major for power users who have ongoing projects.

#### 5. Update Strategy

**Industry standard:**
- Background/async memory updates
- No added latency to responses
- Eventual consistency acceptable

**Chorus current:**
- Implicit synchronous (context prepared before each query)
- No background processing
- No memory updates (just windowing)

**Gap:** Moderate. Current approach works but doesn't scale.

### Strengths of Chorus

Despite gaps, Chorus has some advantages:

| Strength | Description |
|----------|-------------|
| **Simplicity** | Single-layer model is easy to understand and debug |
| **Low latency baseline** | No memory overhead currently |
| **Domain-specific RAG** | Strong lab knowledge base integration |
| **Query reformulation** | LLM-based pronoun resolution (flexible if slow) |
| **Existing EntityExtractor** | Building block ready to use |
| **Thread-based isolation** | Slack threads naturally scope conversations |

### Recommended Alignment Path

**Phase 1: Close critical gaps (1-2 weeks)**
1. Enable entity tracking using existing `EntityExtractor`
2. Add lightweight summarization when window fills
3. Structure context format (entities + summary + recent)

**Phase 2: Add persistence (2-3 weeks)**
4. SQLite storage for conversation summaries
5. Fact extraction for persistent user/project knowledge
6. Background update worker

**Phase 3: Advanced features (optional)**
7. Semantic search over past conversations
8. User preference learning
9. Cross-interface memory (Slack ↔ CLI)

### Migration Complexity

| Change | Code Impact | Risk | Rollback |
|--------|-------------|------|----------|
| Entity tracking | Add ~100 lines to `slack_bot.py` | Low | Easy (remove tracking) |
| Structured context | Refactor context passing | Medium | Medium (compatibility layer) |
| Summarization | Add async worker | Medium | Easy (disable summarization) |
| Persistence | New database, schema | High | Medium (delete DB) |

---

## Next Steps

1. **Prototype structured context** in `memory.py`
2. **Add entity tracking** using existing `EntityExtractor`
3. **Test context summarization** latency with Haiku
4. **Design database schema** for conversation storage
5. **Evaluate embedding models** for conversation search (reuse existing?)

---

## References

- [LangChain Memory Documentation](https://docs.langchain.com/oss/python/langgraph/memory)
- [Pinecone: Conversational Memory for LLMs](https://www.pinecone.io/learn/series/langchain/langchain-conversational-memory/)
- [LangChain: Memory for Agents](https://blog.langchain.com/memory-for-agents/)
- [Inside ChatGPT's Memory System](https://medium.com/aimonks/inside-chatgpts-memory-how-the-most-sophisticated-memory-system-in-ai-really-works-f2b3f32d86b3)
- [LangMem: Long-term Memory Concepts](https://langchain-ai.github.io/langmem/concepts/conceptual_guide/)
- [ChatGPT's Entity Layer Analysis](https://wordlift.io/blog/en/chatgpt-named-entities/)

---

*Document created: January 2025*
*Last updated: January 2025 (added industry best practices and gap analysis)*
*Related files: `slack_bot.py`, `core_agent.py`, `agents/specialists/memory.py`, `chat.py`, `registry/entity_extractor.py`*
