# Voice System Architecture

How CHORUS generates distinct intellectual perspectives, end to end.

## The big idea

Voices aren't just prompts with different names. They're multi-dimensional disposition models grounded in academic frameworks about how researchers think. Each voice is a point in an 18-dimensional space (plus one categorical dimension), and the system uses that to generate distinct intellectual perspectives that can productively disagree with each other.

## Layer 1: The Disposition Model

**File:** `voice_profiles/models.py`

Every voice is a `DispositionVector` with 6 sub-dimensions, each drawn from a specific academic framework:

```
DispositionVector
├── EpistemicAppetite        (Bassett & Zurn)
│   ├── curiosity_style      categorical: "busybody" | "hunter" | "dancer"
│   ├── scope                0=fox ←→ 1=hedgehog         (Berlin)
│   └── surprise_seeking     0=confirmation ←→ 1=novelty
│
├── CognitiveStyle           (Guilford, Dyson)
│   ├── thinking_mode        0=convergent ←→ 1=divergent  (Guilford)
│   ├── abstraction          0=frog ←→ 1=bird             (Dyson)
│   └── classification       0=lumper ←→ 1=splitter
│
├── Methodological           (Lévi-Strauss, Polanyi)
│   ├── making_style         0=engineer ←→ 1=bricoleur    (Lévi-Strauss)
│   ├── formalism            0=narrative ←→ 1=mathematical
│   └── tacit_explicit       0=intuitive ←→ 1=codified    (Polanyi)
│
├── Social
│   ├── disclosure           0=private ←→ 1=thinks aloud
│   ├── inquiry              0=heads-down ←→ 1=monitors others
│   ├── collaboration        0=solitary ←→ 1=collective
│   └── competition          0=communal ←→ 1=agonistic
│
├── Temporal                 (Kuhn)
│   ├── paradigm_orientation 0=normal science ←→ 1=revolutionary  (Kuhn)
│   ├── patience             0=quick iterations ←→ 1=long-term
│   └── archival             0=present ←→ 1=historical
│
└── Aesthetic
    ├── elegance_completeness  0=completeness ←→ 1=elegance
    ├── mechanism_phenomenon   0=phenomenological ←→ 1=mechanist
    └── risk                   0=cautious ←→ 1=bold
```

The `describe()` method (`models.py:163`) turns these numbers into natural language. For example, a vector with `curiosity_style=dancer`, `scope=0.2`, `abstraction=0.8`, `thinking_mode=0.9` produces:

> "leaper between unexpected domains, draws from many frameworks as needed, flies high to see patterns, generates many possibilities"

## Layer 2: Critique Behavior

**File:** `voice_profiles/models.py:222-274`

On top of the disposition, each voice has observable behavior patterns:

- **Question type distribution** -- 8 types with weights summing to ~1.0. Controls *what kind* of questions the voice asks:

  | Type | What it does |
  |------|-------------|
  | clarifying | Seeks definition, scope, precision |
  | challenging | Questions assumptions, validity, evidence |
  | building | Extends, adds to, develops the idea |
  | reframing | Offers an alternative lens |
  | connecting | Links to other domains, finds analogies |
  | scaling | Asks about scale effects |
  | mechanistic | Asks how it works, causal pathways |
  | historical | References past work, precedent |

- **Critique patterns** -- Actual phrases like *"But isn't the real question...?"* or *"What does the organism/system tell you?"*
- **Signature moves** -- Higher-level intellectual behaviors like *"Imports concepts across domains"* or *"Insists on precise taxonomy"*
- **Productive tensions** -- Which other voices this one will disagree with. This is how the system creates genuine intellectual friction, not just different flavors of agreement.

## Layer 3: Compound Types (archetypes)

**File:** `Data/voice_profiles.json` under `compound_types`

There are 6 predefined archetypes, each with a full disposition + critique behavior:

| Archetype | Core Profile | Exemplars | Tensions With |
|-----------|-------------|-----------|---------------|
| **Seductive Scientist** | dancer, bold, high disclosure, agonistic | Watson, Feynman | Quiet Revolutionary, Meticulous Naturalist |
| **Quiet Revolutionary** | hunter, patient, private, revolutionary | McClintock, Darwin | Seductive Scientist, Formal Unifier |
| **Systematic Empire Builder** | hunter, hedgehog, collective, engineer | Rutherford, Edison | Playful Bridger, Quiet Revolutionary |
| **Playful Bridger** | dancer, fox, bricoleur, high surprise | Feynman, von Neumann | Meticulous Naturalist, Systematic Empire Builder |
| **Formal Unifier** | hunter, hedgehog, bird, max formalism | Einstein, Hilbert | Meticulous Naturalist, Quiet Revolutionary |
| **Meticulous Naturalist** | hunter, splitter, frog, completeness | Linnaeus, Darwin (detail) | Formal Unifier, Playful Bridger |

The tension graph forms a network -- Playful Bridger clashes with Meticulous Naturalist (breadth vs. precision), Seductive Scientist clashes with Quiet Revolutionary (loud vs. private), and so on.

## Layer 4: Individual Profiles

**File:** `Data/voice_profiles.json` under `individuals`

Individual people get mapped onto the disposition space. Their profile includes:

- A `disposition` vector (the 18 numbers)
- A `compound_type` they're closest to (optional, matched by Euclidean distance)
- A `confidence` score (how much data backed the extraction)
- `data_sources` tracking what evidence was used

There's a `_template` entry in the JSON showing the default structure -- all 0.5s (neutral on every dimension), empty critique patterns, zero confidence.

## Layer 5: The Store

**File:** `voice_profiles/store.py`

`VoiceProfileStore` is the runtime manager. It loads from `Data/voice_profiles.json` on init and provides:

### Selection

When the MuseAgent needs voices for a query, there are two strategies:

**`select_voices(query, n=3)`** -- Random selection with tension awareness (`store.py:88`):
1. Picks a first voice randomly from all candidates
2. Tries to find a second voice that has `productive_tensions` with the first
3. Fills remaining slots randomly
4. Only includes individuals with `confidence >= 0.3`

**`select_voices_by_topic(topics, n=3)`** -- Topic-affinity scoring (`store.py:158`):
- Uses a hardcoded topic-to-archetype map
- Math/statistics topics pull in Formal Unifier
- Interdisciplinary/creativity topics pull in Playful Bridger
- Longitudinal/historical topics pull in Quiet Revolutionary and Meticulous Naturalist
- Scale/organization topics pull in Systematic Empire Builder

### Prompt generation

`generate_voice_prompts(voices)` (`store.py:227`) produces text for injection into the LLM prompt:

```
Channel these intellectual perspectives:

### Voice 1: Seductive Scientist
High disclosure, agonistic, divergent, low formalism, bold...
Disposition: leaper between unexpected domains, seeks anomalies...

Characteristic critique patterns:
- "But isn't the real question...?"
- "That's clever, but have you considered...?"

Signature intellectual moves:
- Reframes problem as game or puzzle
- Finds unexpected analogy from different field

### Voice 2: Meticulous Naturalist
...

### Productive Tensions
Let these voices engage each other directly--building on,
challenging, and reframing what others say. Key tensions:
- Seductive Scientist productively conflicts with: quiet_revolutionary, meticulous_naturalist
```

## Layer 6: The MuseAgent

**File:** `agents/specialists/muse.py`

The MuseAgent is the consumer. Its system prompt (`muse.py:15-40`) instructs the LLM to:

- Generate distinct voices approaching questions from different angles
- Let them "disagree substantively" without forcing premature consensus
- Prioritize structural thinking, scaling dynamics, emergence, isomorphisms
- Search the knowledge base when lab-specific context would help

The voice prompts from the store are designed to be appended to the Muse system prompt. The MuseAgent also has a tool (`search_knowledge_base`) to ground its exploration in actual lab data via the RAG server.

## Layer 7: The Pipeline (building new voices from data)

This is the automated voice-building pipeline:

```
Raw Data Sources
    │
    ├── SlackCollector       → parses Slack messages by user
    ├── TranscriptCollector  → parses VTT/text meeting transcripts by speaker
    └── PaperCollector       → pulls titles/abstracts from OpenAlex via registry
    │
    ▼
List[TextSample]             (text + source_type + source_id + metadata)
    │
    ▼
DispositionExtractor         (LLM-based analysis via Claude)
    │
    ├── extract_disposition()   → sends samples to Claude with the full taxonomy
    │                             → Claude returns JSON with all 6 dimensions,
    │                               question types, critique patterns, confidence
    │
    ├── match_compound_type()   → Euclidean distance across all 18 numeric dims
    │                             → matches to nearest archetype if similarity > 0.6
    │                             → copies productive_tensions from matched type
    │
    └── build_profile()         → assembles IndividualProfile from all of the above
    │
    ▼
VoiceProfileStore.add_individual() + save()  → writes back to voice_profiles.json
```

### Collectors

**File:** `voice_profiles/collectors/`

Three collectors turn raw data into `TextSample` objects:

**TranscriptCollector** (`transcript_collector.py`) -- Parses meeting transcripts:
- Supports VTT (WebVTT) files from Zoom and plain text with speaker labels
- Extracts utterances per speaker with timestamps
- `collect_samples_by_speaker("Name")` returns all utterances from one person
- `collect_all_speakers()` returns samples grouped by speaker (min 10 utterances to include)
- `extract_questions_from_transcript()` pulls out just the questions (lines containing `?`)

**PaperCollector** (`paper_collector.py`) -- Extracts from publications:
- Pulls titles and abstracts from the lab registry's OpenAlex data
- Can search by author name or by topic
- `get_coauthor_network()` finds related researchers by topic overlap
- `list_authors_by_topic()` finds who works on what

**SlackCollector** (`collectors/slack_collector.py`) -- Collects from Slack channels:
- Uses Slack API to pull messages by user
- Extracts communication patterns and question styles

### The Extractor

**File:** `voice_profiles/processors/disposition_extractor.py`

`DispositionExtractor` is the core of the pipeline. It:

1. **Formats samples** into a numbered list with source type labels
2. **Sends them to Claude** with `EXTRACTION_PROMPT` -- a detailed prompt that defines all 6 dimensions and asks for JSON output with disposition values, question type distribution, critique patterns, signature moves, confidence, and reasoning
3. **Parses the response** into `DispositionVector` and `CritiqueBehavior` objects
4. **Matches to archetype** using Euclidean distance on the 18 numeric dimensions, normalized to 0-1 similarity. A small boost is added if the categorical `curiosity_style` matches. Threshold is 0.6 for a "strong enough" match.
5. **Assembles the profile** with `build_profile()`, which combines everything into an `IndividualProfile` ready to save.

There's also `extract_questions()` which uses a separate prompt focused just on categorizing the types of questions a person asks.

## Building a voice: step by step

Here's the complete workflow to build a voice profile for a new person:

```python
from voice_profiles import VoiceProfileStore
from voice_profiles.collectors import TranscriptCollector, PaperCollector
from voice_profiles.processors.disposition_extractor import DispositionExtractor

# 1. Collect text samples
transcript_collector = TranscriptCollector()
paper_collector = PaperCollector()

samples = []
samples += transcript_collector.collect_samples_by_speaker("Jane Doe")
samples += paper_collector.collect_samples_for_author("Jane Doe")

# 2. Extract disposition (calls Claude)
extractor = DispositionExtractor()

# 3. Build complete profile with archetype matching
store = VoiceProfileStore()
profile = extractor.build_profile(
    person_id="jane_doe",
    person_name="Jane Doe",
    samples=samples,
    affiliation="University of Chicago",
    compound_types=store._compound_types,  # for matching
)

# 4. Save to store
store.add_individual(profile)
store.save()

# 5. Verify
print(f"Matched archetype: {profile.compound_type}")
print(f"Confidence: {profile.confidence}")
print(f"Description: {profile.disposition.describe()}")
```

Or skip the pipeline and write profiles by hand in `Data/voice_profiles.json` -- use the `_template` entry as your starting point.

## What's connected and what's not yet

**Working:**
- Data models, compound types, individual profiles, JSON storage
- VoiceProfileStore with selection and prompt generation
- All three collectors (Slack, transcripts, papers)
- DispositionExtractor with LLM-based profile building
- MuseAgent with multi-perspective system prompt

**The gap:** The MuseAgent's system prompt currently uses a generic multi-perspective approach rather than pulling from `VoiceProfileStore` dynamically. The store produces the prompts via `generate_voice_prompts()`, but the wiring to inject selected voice prompts into the Muse query flow at runtime is the integration point for someone to connect. This is a good first contribution.

## Key files

| File | What it does |
|------|-------------|
| `voice_profiles/models.py` | Data models: DispositionVector, CritiqueBehavior, CompoundType, IndividualProfile |
| `voice_profiles/store.py` | Runtime store: load, select, format, save profiles |
| `voice_profiles/processors/disposition_extractor.py` | LLM-based extraction from text samples |
| `voice_profiles/collectors/transcript_collector.py` | Parse VTT/text meeting transcripts |
| `voice_profiles/collectors/paper_collector.py` | Pull publication data from registry |
| `voice_profiles/collectors/slack_collector.py` | Collect Slack messages |
| `Data/voice_profiles.json` | All profile data (compound types + individuals) |
| `Data/voice_profiles_schema.json` | JSON Schema for validation |
| `agents/specialists/muse.py` | The agent that consumes voice profiles |
