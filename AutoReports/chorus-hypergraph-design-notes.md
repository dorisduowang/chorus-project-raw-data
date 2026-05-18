# CHORUS Hypergraph Design Notes

**Date:** January 12, 2026  
**Context:** Design conversation for extending CHORUS RAG with a temporal hypergraph representation

---

## The Core Idea

Build a **directed hypergraph** as a relational overlay on top of CHORUS's existing RAG chunks. The hypergraph serves as a "digital twin" of Knowledge Lab—both directly queryable for structural questions and used to enhance retrieval by surfacing chunks based on relational proximity, not just semantic similarity.

---

## Why Hypergraphs?

Standard graphs represent pairwise relations. But research activities are inherently multi-party: "James and Sarah used BERT on census data for the metaknowledge project" is one occasion, not five separate edges. A hyperedge captures the entire configuration as a unit.

**Benefits over standard graphs:**
- Captures jointness of collaboration/production events
- Enables queries like "what configurations of people and resources have produced outputs?"
- Natural fit for occasions/events as primary units

---

## Design Evolution

### Starting Point: Nodes and Hyperedges
- Nodes: people, datasets, models, ideas, documents, etc.
- Hyperedges: connect sets of nodes involved in the same occasion

### Considered: Whiteheadian Process Ontology
- Occasions (events) as primary; "entities" as emergent patterns across occasions
- Philosophically elegant; handles drift and temporal evolution naturally
- **Rejected for now:** Extraction too hard, debugging too complex, payoff uncertain
- **Takeaway:** Build with "Whiteheadian sympathies"—timestamps everywhere, explicit succession relations, chunks as first-class provenance—without full process ontology

### Considered: Function/Object Split
- Functions (things that act): people, models, methods
- Objects (things acted upon): data, concepts, papers
- Cleaner than pure substance ontology; captures causal asymmetry
- **Issue:** People aren't functions, they occupy functional *positions* in specific events

### Settled On: Directed Hypergraphs with Role Annotations

Direction encodes causal/productive flow. Nodes don't need to be pre-classified as "function" or "object"—the same node can be on the source side in one hyperedge and target side in another.

Richer than binary source/target: participants have **roles** that specify their function in each hyperedge.

---

## Current Schema

### Node Types

| Type | Description | Examples |
|------|-------------|----------|
| Person | Individual human agents | James Evans, Sarah Chen |
| Organization | Institutions, funders | NSF, Knowledge Lab |
| System | Models, tools, pipelines | BERT classifier, LightRAG |
| Method | Analytical approaches | Topic modeling, network analysis |
| Dataset | Named data resources | OpenAlex, census data |
| Document | Papers, grants, transcripts | NSF grant #1234 |
| Concept | Ideas, theories, keywords | Citation prediction, coherence theory |
| Code | Repositories, scripts | chorus-rag, parable/src |
| Project | Named initiatives | CHORUS, Parable |
| Event | Meetings, talks | Lab meeting 2024-03-15 |
| Chunk | RAG text chunks (evidence) | chunk:abc123 |

### Person Identity Features

Person nodes carry persistent features (not hyperedge participation):
- Role: PhD student, PI, postdoc
- Affiliation: Knowledge Lab, CS department
- Expertise: NLP, network analysis
- Status: active, alumni

Features have temporal validity intervals and confidence scores.

### Participation Roles

**Agent roles (Person, sometimes Organization):**
- `initiator` — started or requested the activity
- `executor` — did the primary work
- `advisor` — guided or supervised
- `reviewer` — evaluated or critiqued
- `collaborator` — contributed alongside others
- `recipient` — received training, mentorship, info

**Instrument roles (System, Method, Code):**
- `instrument` — tool used to accomplish work
- `infrastructure` — underlying resource (compute, storage)
- `method` — analytical approach applied

**Material roles (Dataset, Document, Concept, Code):**
- `input` — consumed or transformed
- `output` — produced or created
- `reference` — cited/consulted, not transformed
- `subject` — what the work is about

**Context roles (Project, Event, Organization):**
- `scope` — project this belongs to
- `setting` — where/when it happened
- `sponsor` — funding source

### Type → Allowed Roles

| Type | Allowed Roles |
|------|---------------|
| Person | initiator, executor, advisor, reviewer, collaborator, recipient, subject |
| Organization | collaborator, sponsor, scope |
| System | instrument, infrastructure, input, output |
| Method | method, input, output |
| Dataset | input, output, reference, subject |
| Document | input, output, reference |
| Concept | input, output, subject |
| Code | instrument, input, output |
| Project | scope |
| Event | setting |

### Hyperedge Structure

```
Hyperedge {
  id: string
  type: production | analysis | computation | writing | discussion | 
        presentation | mentorship | review | derivation | funding | 
        collaboration_formation
  participants: [
    {node: NodeRef, role: Role}
    ...
  ]
  timestamp: datetime | null
  interval: [start, end] | null
  confidence: float [0-1]
  evidence: [ChunkRef | ExternalSourceRef]
  created_by: "llm_extraction" | "structured_import" | "user_feedback"
  confirmed: boolean
}
```

### Example Hyperedge

```
Hyperedge {
  type: analysis
  participants: [
    {node: Person:sarah, role: executor},
    {node: Person:james, role: advisor},
    {node: System:bert_classifier, role: instrument},
    {node: Dataset:census, role: input},
    {node: Dataset:embeddings_v2, role: output},
    {node: Project:metaknowledge, role: scope}
  ]
  timestamp: 2024-03
  confidence: 0.8
  evidence: [chunk:abc123, midway_job:12345]
}
```

---

## Complexity Check

**The concern:** Are 10 node types, 15 roles, and 10 hyperedge types too much?

**The minimal alternative:**

```
Nodes: Person, Thing, Concept

Hyperedge {
  agents: [Person+]
  inputs: [Thing | Concept]*
  outputs: [Thing | Concept]*
  context: [Project | Event]*
  timestamp
  evidence
}
```

Four slots. Simpler extraction, fewer error modes, faster iteration. Lose some expressiveness (advisor vs executor) but can enrich later.

**Decision needed:** Start minimal or start with full role taxonomy?

---

## Data Sources

### Currently in CHORUS RAG (~98 sources, ~5400 chunks):
- Grant proposals
- CVs
- Newsletters
- Transcripts
- Meeting notes

### Not yet integrated:
- Slack (2015–present) — major gap, informal discourse
- Midway compute logs — structured, high confidence
- Publications via OpenAlex — structured, free hyperedges
- Calendars, email — potential future

---

## Extraction Strategy

### Structured sources (high confidence):

**OpenAlex:** Each paper gives authors, date, citations, concepts, affiliations. Direct mapping to hyperedges with confidence 1.0.

**Midway logs:** User, timestamp, resources, job name. Computation hyperedges even if outputs are fuzzy.

**Grant records:** Funder, PI, co-PIs, period. Funding hyperedges.

### Unstructured sources (LLM extraction):

For each chunk:
1. Extract entity mentions
2. Resolve against existing nodes + thesaurus
3. Extract participant-role pairs
4. Validate roles against type constraints
5. Infer hyperedge type
6. Store with confidence and evidence link

### Bootstrapping sequence:

1. **Phase 1:** Structured sources first (OpenAlex, Midway, grants) → skeleton with high-confidence anchors
2. **Phase 2:** Entity extraction over existing chunks, resolve against Phase 1 nodes
3. **Phase 3:** Thesaurus construction (parallel track)
4. **Phase 4:** Slack integration
5. **Phase 5:** Feedback loop for low-confidence hyperedges

---

## The Thesaurus Idea

Build a lab-specific thesaurus of distinctive terminology:
- Canonical terms with aliases ("metaknowledge" → {metascience, science of science})
- Lab-coined concepts
- Project codenames → formal names
- Contrastive definitions (how KL uses term vs. elsewhere)

**Purpose:** Entity resolution layer between raw text and hypergraph. Without it, "James," "Evans," and "JE" become separate nodes.

**Agent approach:**
1. Corpus frequency analysis vs. reference corpus (arXiv CS, science of science lit)
2. Contextual clustering of candidate terms
3. External contrast against OpenAlex definitions
4. Human-in-loop refinement

---

## Query Patterns

**Structural queries (traverse graph directly):**
- "Who has worked with OpenAlex data?"
- "What methods have been applied to the metaknowledge project?"
- "Show me the collaboration network for 2023"

**Retrieval-augmented queries:**
- User asks about "measuring scientific impact"
- Resolve to Concept:scientific_impact
- Find hyperedges containing that concept
- Boost chunks that are provenance for those hyperedges

**Simulation/counterfactual:**
- "If we hired someone with network analysis expertise, what would they connect to?"

---

## Storage Options

| Option | Pros | Cons |
|--------|------|------|
| Neo4j / TigerGraph | Native traversal | Hypergraph support awkward |
| Postgres + arrays | Familiar tooling | Less elegant traversal |
| Specialized hypergraph DB | High fidelity | Small ecosystem |
| In-memory (NetworkX) + JSON/Parquet | Fast iteration | Doesn't scale forever |

**Recommendation:** Start in-memory, persist to disk, migrate to Postgres or Neo4j once schema stabilizes.

---

## Open Questions

1. **Schema complexity:** Full role taxonomy or minimal 4-slot version?

2. **Extraction target:** If LLM must assign 1 of 15 roles, error rate is high. Simpler schema = more reliable extraction.

3. **Validation strictness:** Reject invalid hyperedges, or accept with low confidence and flag for review?

4. **Concept granularity:** Is "coherence theory" one node, or do sub-components (trading zones, authentication networks) get their own nodes?

5. **Thesaurus scope:** How much human effort to invest upfront vs. build incrementally through feedback?

6. **Slack modeling:** Model conversations as hyperedges? Or wait for evidence of completed actions?

---

## Recommendation

Start with the minimal schema:

```
Nodes: Person, Thing, Concept, Project, Event

Hyperedge {
  agents: [Person+]
  inputs: [Thing | Concept]*
  outputs: [Thing | Concept]*
  context: [Project | Event]*
  timestamp
  evidence: [Chunk | ExternalSource]
  confidence
}
```

Get extraction working, see what queries actually need finer distinctions, then enrich. The full role taxonomy is available when you need it, but you don't pay the extraction complexity cost until you know it's worth it.

Build OpenAlex ingest first—it's structured, high-value, and gives you a solid person/paper backbone for resolving other extractions against.

---

## Next Steps

1. Define minimal schema formally
2. Build OpenAlex → hyperedge pipeline
3. Prototype thesaurus extraction from existing chunks
4. Test retrieval augmentation with simple hypergraph
5. Iterate based on what queries reveal
