# Getting Started with CHORUS

A practical guide for contributors. Read this first, then explore the code.

## What is CHORUS?

CHORUS (Comprehensive Hybrid Orchestrated Retrieval and Understanding System) is a knowledge assistant for the [Knowledge Lab](https://knowledgelab.org) at the University of Chicago. Ask it about lab members, research, publications, compute resources -- it retrieves answers from structured data and documents using a combination of semantic search, keyword search, and LLM reasoning.

The interesting part: CHORUS doesn't just retrieve facts. It can respond through different intellectual *voices* -- simulated researcher dispositions that bring diverse perspectives to questions.

## Setup

```bash
# Clone and enter the repo
git clone <repo-url> && cd chorus

# Create a virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Configure credentials

CHORUS needs API keys and (optionally) Google Drive credentials. **None of these should ever be committed to git** — the `.gitignore` is configured to exclude them.

#### Step 1: Create your `.env` file

```bash
cp .env.example .env
```

This creates a local `.env` file (gitignored) from the template.

#### Step 2: Get an Anthropic API key

1. Go to [console.anthropic.com](https://console.anthropic.com) and sign in (or create an account).
2. Navigate to **Settings → API Keys** ([direct link](https://console.anthropic.com/settings/keys)).
3. Click **Create Key**, give it a name like "chorus", and copy the key.
4. Open your `.env` file in any text editor and replace the placeholder:

```bash
# Before:
ANTHROPIC_API_KEY=your-anthropic-api-key-here

# After:
ANTHROPIC_API_KEY=your-real-anthropic-api-key
```

Save the file. This is the only credential required to run the chat interface and RAG server.

#### Step 3: Slack tokens (only if running the Slack bot)

If you're running the Slack bot (`slack_bot.py`), you'll also need:

1. **Bot Token**: Go to your Slack app's **OAuth & Permissions** page → copy the **Bot User OAuth Token** (starts with `xoxb-`).
2. **App Token**: Go to **Basic Information → App-Level Tokens** → create a token with `connections:write` scope (starts with `xapp-`).

Add both to your `.env`:

```bash
SLACK_BOT_TOKEN=xoxb-your-real-bot-token
SLACK_APP_TOKEN=xapp-your-real-app-token
```

If you're only using the CLI chat or API, you can skip these.

#### Step 4: Verify your setup

```bash
# Quick check that your API key works
python -c "import anthropic; c = anthropic.Anthropic(); print('API key is valid')"
```

If this prints "API key is valid", you're good to go.

**Summary of credentials:**

| Credential | Required for | Where to get it |
|---|---|---|
| `ANTHROPIC_API_KEY` | Everything (chat, API, Slack) | [console.anthropic.com/settings/keys](https://console.anthropic.com/settings/keys) |
| `SLACK_BOT_TOKEN` | Slack bot only | Slack app → OAuth & Permissions |
| `SLACK_APP_TOKEN` | Slack bot only | Slack app → Basic Information → App-Level Tokens |
| Google OAuth client secret | Google Drive sync only | [Google Cloud Console](https://console.cloud.google.com/apis/credentials) (project: `chorus-483718`) |

#### Google Drive sync (optional)

If you need Google Drive sync for data snapshots:

1. Get the `client_secret_*.json` file from a team member or download it from Google Cloud Console.
2. Place it in the repo root (it's gitignored by the `client_secret*.json` pattern).
3. Run the one-time authentication flow:
   ```bash
   python -c "from ingest.gdrive_sync import authenticate; authenticate()"
   ```
   This opens a browser for Google OAuth consent and saves a token locally to `gdrive_sync_token.json` (also gitignored).

4. After that, sync works automatically:
   ```bash
   python -m ingest.gdrive_sync
   ```

### Setting up on a new machine

When deploying chorus to additional hardware (another laptop, a server, a cluster node):

1. **Clone the repo** — the repo contains no secrets, so this is safe.
2. **Copy your `.env` file** from an existing machine (via `scp`, a password manager, or manually).
3. **Copy the `client_secret_*.json`** if you need Google Drive sync.
4. **Re-run the Google OAuth flow** — each machine needs its own `gdrive_sync_token.json`:
   ```bash
   python -c "from ingest.gdrive_sync import authenticate; authenticate()"
   ```
5. **Install dependencies** — `pip install -r requirements.txt`

The Anthropic API key and Slack tokens work across machines. Only the Google OAuth token is machine-specific.

### Security rules

- **Never commit credentials** — `.env`, `client_secret*.json`, and `*token*.json` are all gitignored.
- **Never put real keys in `.env.example`** — use placeholders only.
- **Rotate keys immediately** if they are ever accidentally committed. See `SECURITY_AUDIT_REPORT.md` for the rotation procedure.
- **Use environment variables** for any new secrets — load them with `os.environ.get()` in code, document them in `.env.example` with placeholders.

### Start the RAG server

```bash
./scripts/run_rag_server.sh           # development mode (auto-reload)
./scripts/run_rag_server.sh production # production mode (4 workers)
```

The server runs at `http://localhost:8765`. Verify with:

```bash
curl http://localhost:8765/health
```

### Try a query

```bash
# From the API
curl "http://localhost:8765/hybrid?q=Who+works+on+NLP"

# Or start the chat interface
python scripts/chat.py
```

API docs are at `http://localhost:8765/docs` once the server is running.

## How it works

Understanding the architecture will help you know where to make changes.

```
          Your question
               |
               v
    +---------------------+
    |    Orchestrator      |  Classifies intent, picks a specialist
    |  agents/orchestrator |
    +---------------------+
          |         |
          v         v
    +---------+  +---------+
    | Memory  |  | Mechanic|  ...other specialists
    | (RAG)   |  | (tech)  |
    +---------+  +---------+
          |
    +-----+----------+
    |     |          |
    v     v          v
  FAISS  BM25    Registry       Three search strategies
  (semantic) (keyword) (structured)    run in parallel
    |     |          |
    +-----+----------+
          |
          v
    +-----------+
    | Reranker  |   Cross-encoder scores and re-orders results
    +-----------+
          |
          v
    +-----------+
    |  Claude   |   LLM generates a natural language answer
    +-----------+
          |
          v
       Response
```

### Key concepts

**Orchestrator** (`agents/orchestrator.py`): The router. It classifies your query's intent (lookup, technical, exploratory, connective) and dispatches to the right specialist agent.

**Specialist agents** (`agents/specialists/`):
- **MemoryAgent** -- Fast factual retrieval. Uses RAG (semantic + keyword search) and the structured registry. This handles most questions.
- **MechanicAgent** -- Technical problem-solving (documentation, how-to).
- **MuseAgent** -- Deep, multi-perspective analysis using voice profiles.
- **MatchmakerAgent** -- Finds connections between people, topics, projects.

**Registry** (`registry/`): Structured data access layer for people, projects, publications, compute resources. Backed by `Data/lab_registry.json`.

**Voice profiles** (`voice_profiles/`): A disposition model that gives CHORUS different intellectual personalities. Each voice has a 6-dimensional profile (epistemic, cognitive, methodological, social, temporal, aesthetic). More on this below.

**RAG pipeline** (`scripts/rag_server_fastapi.py`): The search server. Combines FAISS vector search, BM25 keyword search, LLM query reformulation, and cross-encoder reranking.

## Project layout

```
chorus/
├── agents/                 # Multi-agent system
│   ├── orchestrator.py     #   Query router and dispatcher
│   ├── base.py             #   Base classes for all agents
│   ├── classifier.py       #   Intent classification (regex)
│   ├── llm_classifier.py   #   Intent classification (LLM fallback)
│   ├── query_planner.py    #   Multi-step query planning
│   ├── plan_executor.py    #   Executes query plans
│   ├── confidence.py       #   Confidence scoring
│   └── specialists/        #   Specialist agent implementations
│       ├── memory.py       #     RAG-based retrieval
│       ├── mechanic.py     #     Technical problem-solving
│       ├── muse.py         #     Multi-perspective analysis
│       └── matchmaker.py   #     Connection finding
│
├── voice_profiles/         # Voice/disposition system
│   ├── models.py           #   Data models (6 dimensions)
│   ├── store.py            #   Profile storage and selection
│   ├── collectors/         #   Data ingestion from Slack, transcripts, papers
│   └── processors/         #   LLM-based disposition extraction
│
├── registry/               # Structured data access
│   ├── rag.py              #   Main RAG integration
│   ├── classifier.py       #   Query type classification
│   ├── lookup.py           #   Data lookup handlers
│   ├── semantic_index.py   #   FAISS + BM25 indexes
│   ├── query_router.py     #   Semantic vs. structured routing
│   ├── topic_aliases.py    #   Abbreviation expansion (NLP -> natural language processing)
│   └── text_generator.py   #   Natural language formatting
│
├── scripts/                # Servers and utilities
│   ├── rag_server_fastapi.py  # Main FastAPI server
│   ├── mcp_rag_server.py     # MCP server (for Claude Desktop)
│   ├── run_rag_server.sh     # Server launcher script
│   ├── chat.py               # CLI chat interface
│   └── rebuild_rag_with_files.py  # Rebuild search indexes
│
├── Data/                   # All data lives here
│   ├── lab_registry.json   #   People, projects, publications, etc.
│   ├── voice_profiles.json #   Voice disposition profiles
│   ├── lab_registry_schema.json    # JSON Schema for registry
│   ├── voice_profiles_schema.json  # JSON Schema for voices
│   └── hybrid_rag_index/  #   FAISS and BM25 index files
│
├── ingest/                 # Data ingestion pipeline
├── extractors/             # Entity and topic extraction
├── tests/                  # Test suite
├── config/                 # Configuration management
└── requirements.txt        # Python dependencies
```

## How to extend CHORUS

This section covers the most common ways to plug your own work into the system — starting with the easiest.

---

### The simplest integration: use CHORUS as a tool

Most people don't need to touch CHORUS internals. If you're building your own agent, analysis pipeline, or Claude-based project for the lab, the fastest path is to run the CHORUS RAG server locally and connect to it as an MCP tool. You get the full knowledge base — people, publications, projects, compute resources, ingested documents — without writing any retrieval code yourself. CHORUS handles the search; your project handles the reasoning.

This is the recommended starting point for masters students building on top of lab infrastructure.

#### What this gives you

| You bring | CHORUS provides |
|---|---|
| Your own Anthropic API key | Hybrid search (semantic + keyword) over lab documents |
| Claude Desktop, Claude Code, or any MCP client | Structured registry lookup (people, projects, publications) |
| Your own project and prompts | Cross-encoder reranking for result quality |
| | Topic alias expansion (NLP → "natural language processing") |

CHORUS is just the retrieval layer. Claude uses your API key to reason over the results and generate answers. You control your own costs and usage.

#### Step 1: Start the RAG server

```bash
cd chorus
source venv/bin/activate
./scripts/run_rag_server.sh
```

Verify it's running:
```bash
curl http://localhost:8765/health
```

The server needs an `ANTHROPIC_API_KEY` in your `.env` file (see [Configure credentials](#configure-credentials) above). It runs at `http://localhost:8765`.

#### Step 2: Connect via MCP

MCP (Model Context Protocol) lets Claude Desktop and Claude Code call CHORUS tools directly. There are two options:

**Option A: HTTP client (recommended)** — A thin wrapper that talks to the running RAG server. Lightweight, and the server can serve multiple clients at once.

**Option B: Standalone** — Loads the FAISS/BM25 indexes directly into the MCP process. One fewer thing to run, but uses more memory and is single-client.

For either option you need `fastmcp`:
```bash
pip install fastmcp
```

##### Claude Desktop

Open your config file:
- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`

Add under `mcpServers`:

```json
{
  "mcpServers": {
    "chorus": {
      "command": "/path/to/chorus/venv/bin/python",
      "args": ["/path/to/chorus/scripts/mcp_rag_client.py"]
    }
  }
}
```

Replace `/path/to/chorus` with your actual repo path. For the standalone option, swap `mcp_rag_client.py` for `mcp_rag_server.py`.

Restart Claude Desktop. You'll now have three CHORUS tools available:
- **`search_documents`** — search the knowledge base by query
- **`get_context_for_question`** — retrieve context with citations to answer a question
- **`list_available_sources`** — see what documents are indexed

##### Claude Code

Add this to your project's `.mcp.json`:

```json
{
  "mcpServers": {
    "chorus": {
      "command": "/path/to/chorus/venv/bin/python",
      "args": ["/path/to/chorus/scripts/mcp_rag_client.py"]
    }
  }
}
```

#### Step 3: Ask questions

Once connected, you (or your agent) can ask Claude things like:
- "Who in the lab works on NLP?"
- "What compute resources are available on Midway?"
- "Find publications about cultural evolution"

Claude will call the CHORUS tools behind the scenes, get search results, and synthesize an answer. You don't need to know anything about FAISS, BM25, or the registry internals.

#### When to go deeper

If you need to add new data to the knowledge base, create new specialist agents, build voice profiles, or modify the search pipeline itself — read on.

---

### Add a new voice profile

Voice profiles define intellectual dispositions along 6 dimensions, each grounded in academic frameworks:

| Dimension | What it captures | Scale endpoints |
|-----------|-----------------|-----------------|
| **Epistemic** | How you seek knowledge | Busybody / Hunter / Dancer; Fox / Hedgehog |
| **Cognitive** | How you think | Convergent / Divergent; Frog / Bird; Lumper / Splitter |
| **Methodological** | How you work | Engineer / Bricoleur; Narrative / Mathematical |
| **Social** | How you relate | Private / Open; Solitary / Collective |
| **Temporal** | Time orientation | Normal science / Revolutionary; Quick / Patient |
| **Aesthetic** | What you find beautiful | Completeness / Elegance; Cautious / Bold |

To add a voice, edit `Data/voice_profiles.json`. Add an entry under `individuals`:

```json
{
  "your_researcher_id": {
    "id": "your_researcher_id",
    "name": "Display Name",
    "compound_type": "playful_bridger",
    "disposition": {
      "epistemic": {
        "curiosity_style": "dancer",
        "scope": 0.3,
        "surprise_seeking": 0.8
      },
      "cognitive": {
        "thinking_mode": 0.7,
        "abstraction": 0.6,
        "classification": 0.4
      },
      "methodological": {
        "making_style": 0.8,
        "formalism": 0.3,
        "tacit_explicit": 0.4
      },
      "social": {
        "disclosure": 0.7,
        "inquiry": 0.6,
        "collaboration": 0.8,
        "competition": 0.3
      },
      "temporal": {
        "paradigm_orientation": 0.6,
        "patience": 0.5,
        "archival": 0.3
      },
      "aesthetic": {
        "elegance_completeness": 0.7,
        "mechanism_phenomenon": 0.4,
        "risk": 0.6
      }
    },
    "critique_behavior": {
      "question_types": {
        "clarifying": 0.1,
        "challenging": 0.2,
        "building": 0.15,
        "reframing": 0.2,
        "connecting": 0.15,
        "scaling": 0.05,
        "mechanistic": 0.1,
        "historical": 0.05
      },
      "signature_moves": [
        "Describes what this voice characteristically does in discussion"
      ]
    },
    "confidence": 0.7
  }
}
```

The existing compound types (in the same file) are good starting points:
- **Seductive Scientist** -- high disclosure, bold, divergent (think Feynman)
- **Quiet Revolutionary** -- private, patient, revolutionary (think McClintock)
- **Systematic Empire Builder** -- collective, methodical, convergent (think Rutherford)
- **Playful Bridger** -- fox-like, high surprise, bricoleur (think von Neumann)

You can also define new compound types under the `compound_types` section.

**Validate your profile** against the schema at `Data/voice_profiles_schema.json`.

#### Collecting data for new voices

If you have source material (Slack messages, meeting transcripts, papers), use the collectors in `voice_profiles/collectors/`:

```python
from voice_profiles.collectors import TranscriptCollector, PaperCollector, SlackCollector

# From meeting transcripts (.vtt files)
collector = TranscriptCollector()
samples = collector.collect("path/to/transcript.vtt", speaker="Name")

# From papers
collector = PaperCollector()
samples = collector.collect("path/to/paper.pdf")

# From Slack
collector = SlackCollector(token="xoxb-...")
samples = collector.collect(channel="C01234")
```

Then run the disposition extractor to generate a profile:

```python
from voice_profiles.processors import DispositionExtractor

extractor = DispositionExtractor()
profile = extractor.extract(samples)
```

---

### Add a new specialist agent

Create a new file in `agents/specialists/`:

```python
# agents/specialists/my_agent.py

from ..base import BaseAgent, AgentResponse

class MyAgent(BaseAgent):
    """One-line description of what this agent does."""

    async def _query(self, message: str, context: dict) -> AgentResponse:
        # Your logic here. You have access to:
        #   self.client     - Anthropic API client
        #   self.rag_url    - RAG server URL for search
        #   context         - Conversation context

        # Example: call the RAG server for search results
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.rag_url}/hybrid",
                params={"q": message, "top_k": 5}
            ) as resp:
                results = await resp.json()

        # Process results, call Claude, etc.
        return AgentResponse(
            content="Your response here",
            agent_type="my_agent",
            confidence=0.8
        )
```

Then wire it into the orchestrator:

1. **Export it** in `agents/specialists/__init__.py`:
   ```python
   from .my_agent import MyAgent
   ```

2. **Register it** in `agents/orchestrator.py`:
   ```python
   # Add to INTENT_AGENT_MAP
   INTENT_AGENT_MAP["my_intent"] = "my_agent"

   # Add to ALL_AGENTS
   ALL_AGENTS = {"memory", "mechanic", "muse", "matchmaker", "my_agent"}

   # Add to DEFAULT_ENABLED_AGENTS if it should be on by default
   DEFAULT_ENABLED_AGENTS = {"memory", "mechanic", "my_agent"}
   ```

3. **Add intent patterns** to `agents/classifier.py` so queries get routed to your agent.

---

### Add data to the registry

The registry (`Data/lab_registry.json`) stores structured data about the lab. To add a new person, project, dataset, or other entity:

1. **Edit `Data/lab_registry.json`** following the schema in `Data/lab_registry_schema.json`.

2. **Person example:**
   ```json
   {
     "id": "jane_doe",
     "name": "Jane Doe",
     "role": "PhD Student",
     "institution": "University of Chicago",
     "email": "janedoe@uchicago.edu",
     "openalex": {
       "works_count": 5,
       "cited_by_count": 42,
       "h_index": 3,
       "topics": ["computational social science", "network analysis"],
       "recent_publications": []
     }
   }
   ```

3. **Reload without restarting:**
   ```bash
   curl -X POST http://localhost:8765/reload
   ```

If you're adding a **new entity type** (not just a new entry in an existing type), you'll also need to update:
- `registry/classifier.py` -- add keyword patterns for query detection
- `registry/lookup.py` -- add a lookup handler
- `registry/rag.py` -- add formatting in `_format_structured_answer()`

---

### Add topic aliases

Topic aliases let users search using abbreviations. Edit `registry/topic_aliases.py`:

```python
TOPIC_ALIASES = {
    "NLP": ["natural language processing", "computational linguistics"],
    "ML": ["machine learning", "statistical learning"],
    # Add yours:
    "SNA": ["social network analysis", "network science"],
}
```

---

### Add documents to the search index

To make new documents searchable:

1. Place files in the appropriate data directory.
2. Rebuild the index:
   ```bash
   python scripts/rebuild_rag_with_files.py
   ```
   Or for incremental updates:
   ```bash
   python scripts/incremental_index.py
   ```

The system chunks documents and creates both FAISS (semantic) and BM25 (keyword) indexes.

---

### Add an ingestion source

The `ingest/` directory contains data pipelines. Look at existing ingesters for patterns:

- `ingest/gdrive_sync.py` -- Google Drive
- `ingest/daily_scan.py` -- Scheduled file scanning
- `ingest/snapshot_indexer.py` -- File snapshot indexing

Each ingester reads from a source, processes the data, and writes to `Data/` or directly to the index.

## Running tests

```bash
# All tests
pytest tests/

# Specific test file
pytest tests/test_orchestrator.py

# With output
pytest tests/ -v
```

## Useful endpoints

Once the server is running at `http://localhost:8765`:

| Endpoint | Method | What it does |
|----------|--------|-------------|
| `/health` | GET | Server status and config |
| `/hybrid?q=...` | GET | Hybrid search (recommended) |
| `/semantic?q=...` | GET | Semantic search only |
| `/bm25?q=...` | GET | Keyword search only |
| `/reload` | POST | Hot-reload registry data |
| `/docs` | GET | Interactive API documentation |

## Common workflows

**"I want to answer questions about a new dataset"**
1. Add the dataset info to `Data/lab_registry.json`
2. Place any documents in the data directory
3. Run `python scripts/rebuild_rag_with_files.py`
4. `curl -X POST http://localhost:8765/reload`

**"I want CHORUS to have a different personality when answering"**
1. Create a voice profile in `Data/voice_profiles.json`
2. The MuseAgent will pick it up automatically

**"I want to add a totally new capability"**
1. Create a specialist agent in `agents/specialists/`
2. Wire it into the orchestrator
3. Add intent classification patterns

**"I want to connect CHORUS to a new data source"**
1. Write an ingester in `ingest/`
2. Have it output to `Data/` in the expected format
3. Rebuild indexes

## Where to ask questions

- Check the existing docs in `docs/` and the main `README.md`
- Look at the test files in `tests/` -- they show how each component is used
- The code has docstrings explaining the "why" behind design decisions
