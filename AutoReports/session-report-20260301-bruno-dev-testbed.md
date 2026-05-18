# Session Report: Bruno Notebook Evolved into Development Testbed
**Generated:** 2026-03-01

---

## Objective

Expand `agents/bruno_dev.ipynb` from a 2-cell prototype (LangGraph ReAct agent with MCP RAG + calculator) into a 6-cell development testbed that wires in the project's voice profile system, adds conversation memory and streaming, and provides demo/debugging tools. Import from existing modules — don't reimplement.

This work closes the documented integration gap (VOICE_SYSTEM.md:289): voice profiles were never wired into any agent at runtime.

---

## Phase 1: Cell 1 Improvements (Setup)

**Cell ID:** `9f365808`

| Change | Before | After |
|---|---|---|
| `sys.path` | Not set — project imports impossible | `sys.path.insert(0, str(REPO_DIR))` |
| Model | Hardcoded `"claude-sonnet-4-5-20250514"` | Env-configurable `MODEL_NAME`, default `"claude-sonnet-4-20250514"` (matches `MuseAgent.DEFAULT_MODEL` in `agents/specialists/muse.py:95`) |
| Memory | None | `MemorySaver` checkpointer instantiated |
| Imports | `create_react_agent`, `ChatAnthropic`, `tool` | Added `MemorySaver` from `langgraph.checkpoint.memory` |

Calculator tool and MCP config preserved unchanged. The AST-based safe eval fix from the prior session was already in place.

---

## Phase 2: Cell 2 — Voice Profile Integration

**Cell ID:** `f8113c29` (replaced)

This is the core integration work. The cell:

1. Imports `VoiceProfileStore` from `voice_profiles/store.py` (project module, now accessible via `sys.path`)
2. Loads all 6 compound voice types from `Data/voice_profiles.json`
3. Prints them for visibility on cell execution
4. Defines `CHORUS_BASE_PROMPT` — a cleaned-up system prompt that:
   - Removes hardcoded tool names (ReAct agent discovers tools from MCP)
   - Removes false claim about "subagents"
   - Describes the 4 modes (Memory, Mechanic, Matchmaker, Muse) and declares Muse active
5. Defines `build_system_prompt(query, n_voices=3)` that dynamically selects voices per query

**Key integration path:**
```
build_system_prompt(query)
  → VoiceProfileStore.select_voices(query, n=3)      # store.py:88
  → VoiceProfileStore.generate_voice_prompts(voices)  # store.py:227
  → CHORUS_BASE_PROMPT + voice_section
```

Each query gets a different set of 3 voices selected with productive tension consideration. This means the agent's personality shifts subtly depending on the topic — a formalization question might get the Formal Unifier, while an interdisciplinary question gets the Playful Bridger.

---

## Phase 3: Cell 3 — Agent with Conversation Memory

**Cell ID:** `07s7cquorryq` (new)

Defines `run_chorus(query, thread_id="dev-session")` which:

- Calls `build_system_prompt(query)` for per-query voice selection
- Opens MCP client session, gets RAG tools, adds calculator
- Creates ReAct agent with `checkpointer=checkpointer`
- Invokes with `config={"configurable": {"thread_id": thread_id}}`

The `thread_id` parameter enables multiple independent conversation threads. Queries on the same `thread_id` share conversation history via LangGraph's `MemorySaver`, enabling follow-up questions with pronoun resolution ("they", "that paper", etc.).

---

## Phase 4: Cell 4 — Streaming Output

**Cell ID:** `4q5iutmro07` (new)

Defines `run_chorus_streaming(query, thread_id)` using `agent.astream_events(version="v2")`:

- `on_chat_model_stream` events: accumulates text, updates a `display(Markdown(...), display_id=True)` handle in-place for live rendering
- `on_tool_start` events: prints tool name to stdout for visibility into the ReAct loop
- Returns the full collected text

This provides a Jupyter-native streaming experience where markdown renders progressively as tokens arrive.

---

## Phase 5: Cell 5 — Multi-Query Demo Battery

**Cell ID:** `q8xtyvkivp` (new)

Five queries exercising different capability types:

| Query | Type | Tests |
|---|---|---|
| "Who is James Evans and what does he research?" | lookup | RAG retrieval |
| "How do I access the computing cluster?" | technical | Practical knowledge |
| "What are the epistemological implications of using LLMs as research tools?" | explore | Deep multi-perspective analysis (Muse mode) |
| "Who in the lab works on computational social science?" | connect | Matchmaker-style social knowledge |
| "What specific papers have they published recently?" | follow-up | **MemorySaver context** — "they" must resolve from prior conversation |

All 5 queries run in a single MCP session (avoiding 5x subprocess startup overhead of 5-15s each) with a shared `thread_id="demo-battery"`. The follow-up query is the critical memory test.

---

## Phase 6: Cell 6 — Voice Profile Explorer

**Cell ID:** `5hkj4tlp0jk` (new)

A standalone development/debugging tool for the voice system:

1. **Full inventory** — prints all 6 compound types with description, exemplars, disposition description, and productive tensions
2. **Topic-based selection** — demonstrates `select_voices_by_topic()` with 3 topic sets:
   - `["interdisciplinary", "analogy"]` → Playful Bridger (correct)
   - `["math", "proof"]` → Formal Unifier (correct)
   - `["paradigm", "assumption"]` → Quiet Revolutionary (correct)
3. **Query-based selection** — demonstrates `select_voices()` with 3 sample queries
4. Seeded with `random.seed(42)` for reproducible output

---

## Cell Dependencies

```
Cell 1 (setup) → Cell 2 (voices + prompt) → Cell 3 (agent + memory) → Cell 5 (demo battery)
                                           → Cell 4 (streaming)
                      Cell 2 → Cell 6 (voice explorer, standalone)
```

---

## Test Results

Testing was performed using Python 3.11 (`/software/python-3.11.9-el8-x86_64/bin/python3`) since the system Python (3.6) and Jupyter kernel Python (3.9) are too old for `langchain-mcp-adapters` (requires >=3.10).

| Cell | Test Method | Result |
|---|---|---|
| 1 (Setup) | All imports, calculator assertions, MemorySaver instantiation | **PASSED** |
| 2 (Voices) | VoiceProfileStore loads 6 types, `build_system_prompt` produces 3 voices + tensions section | **PASSED** |
| 3 (Agent) | AST parse — syntax valid, async function with correct params | **PASSED** |
| 4 (Streaming) | AST parse — syntax valid, `astream_events` v2, display_id pattern, tool visibility | **PASSED** |
| 5 (Demo battery) | AST parse — syntax valid, 5 queries, single MCP session, shared thread | **PASSED** |
| 6 (Voice explorer) | **Full end-to-end execution** — all 6 types printed, topic selection topically correct, query selection works | **PASSED** |

Cells 3-5 require an `ANTHROPIC_API_KEY` for full execution. Their structure, syntax, and integration points are verified; live testing requires running the notebook in Jupyter.

---

## Files Modified

| File | Change |
|---|---|
| `agents/bruno_dev.ipynb` | Expanded from 2 cells to 6: voice integration, memory, streaming, demos |

## Files Read (Not Modified)

| File | What's Imported |
|---|---|
| `voice_profiles/store.py` | `VoiceProfileStore` class |
| `voice_profiles/models.py` | `CompoundType`, `DispositionVector` (via store) |
| `voice_profiles/__init__.py` | Package exports |
| `Data/voice_profiles.json` | Loaded by VoiceProfileStore at runtime |
| `agents/specialists/muse.py` | Referenced for `DEFAULT_MODEL` value |

---

## What This Does NOT Do

- **Does not replicate the orchestrator's routing.** The orchestrator uses classification → specialist routing. The notebook uses a single ReAct agent where the LLM decides tool use. These are complementary architectures.
- **Does not add web search.** Natural next step but out of scope.
- **Does not wrap specialist agents as LangGraph tools.** The specialists use raw Anthropic SDK; wrapping them is a production integration task.
- **Does not persist memory across kernel restarts.** `MemorySaver` is in-memory. `SqliteSaver` would be the upgrade path.

---

## Lessons Learned

1. **The voice integration gap was trivial to close.** The `VoiceProfileStore` API was clean and well-designed — `select_voices()` + `generate_voice_prompts()` is the entire integration surface. The gap existed because nobody had added `sys.path` setup and a 5-line `build_system_prompt()` function. Sometimes the biggest architectural gaps are plumbing, not design.

2. **Model version strings matter.** The original notebook used `claude-sonnet-4-5-20250514` (Sonnet 4.5) while the production `MuseAgent` uses `claude-sonnet-4-20250514` (Sonnet 4). Making this env-configurable avoids the notebook silently using a different model than production.

3. **Single MCP session for batches is important.** The MCP RAG server starts as a subprocess over stdio. Each `MultiServerMCPClient` context manager spawns a new process (5-15s startup). The demo battery cell runs all 5 queries inside one session to avoid 25-75s of overhead.

4. **`MemorySaver` thread_id is the key UX feature.** Different `thread_id` values create independent conversation threads. This makes the notebook useful for A/B testing different conversation flows without cross-contamination.

---

## Next Steps

1. **Run full notebook in Jupyter** with API key to validate Cells 3-5 end-to-end
2. **Add web search tool** (Tavily or similar) as a second MCP server
3. **Upgrade to `SqliteSaver`** for memory persistence across kernel restarts
4. **Add mode switching** — let the user toggle between Memory/Mechanic/Matchmaker/Muse modes, each with a different base prompt and voice selection strategy
5. **Wrap specialist agents as LangGraph tools** — the ReAct agent could delegate to the existing MuseAgent, MemoryAgent, etc.
