# Session Report: Data Cards Extension, Critique, and Quality Fixes
**Generated:** 2026-03-01

---

## Objective

Extend the 14 dataset entries in `Data/lab_registry.json` with structured sections aligned to the Data Cards framework (Pushkarna et al., arXiv 2204.01075), then systematically critique every card for errors, inconsistencies, and completeness gaps, and implement fixes.

---

## Phase 1: Schema and Pipeline Implementation

### 1.1 Schema Enum Fixes

**File:** `Data/lab_registry_schema.json`

The schema had never been validated against the actual data. Running `jsonschema.validate()` revealed four pre-existing enum mismatches:

| Field | Added Values | Reason |
|---|---|---|
| `datasets.provenance.update_frequency` | `"biannual"`, `"periodic"` | Used by dimensions, icite, papers-with-code, patstat |
| `compute.type` | `"bare_metal_server"` | Used by existing compute entry |
| `tools.type` | `"data_processing"`, `"development_environment"`, `"job_scheduler"`, `"monitoring"` | Used by existing tool entries |
| `datasets.technical.format` | Changed to `["string", "null"]` | peer-review-cs has `format: null` |

**Lesson:** The schema was aspirational, not enforced. The registry had drifted from it in multiple places. Any CI/CD pipeline should include schema validation as a gate.

### 1.2 Ten New Data Card Sections Added to Schema

Inserted after `maintainer`, before `tags`. All sections optional with nullable leaf fields:

| Section | Purpose |
|---|---|
| `sensitivity` | PII, consent, risk classification |
| `creators` | Dataset author/curator array |
| `preprocessing` | Cleaning steps, tools, raw data availability |
| `example_data_points` | Sample records (free-form objects) |
| `annotation` | Human labeling details (type, count, IAA, label schema) |
| `sampling` | Methodology, population, exclusions, temporal range |
| `known_models_and_benchmarks` | ML models, benchmarks, leaderboard URLs |
| `lifecycle` | Retention, deprecation, version history |
| `social_impact` | Beneficial uses, harms, affected groups |
| `data_card_completeness` | Documentation audit trail |

### 1.3 Initial Population of 14 Dataset Entries

All entries received the 10 new sections with `null`/`[]` defaults. Only fields with directly verifiable information were populated: reddit PII, IRIS PII, peer-review-cs annotations, MAG lifecycle/deprecation, bibliometric sensitivity flags, full-population sampling.

### 1.4 Pipeline Code Fixes (4 Files)

| File | Problem | Fix |
|---|---|---|
| `registry/semantic_index.py` | `_create_dataset_text()` used flat paths (`dataset.get("format")`) that never matched the nested schema — silently producing empty embedding text | Corrected to nested paths; added harvesting for new Data Card fields |
| `registry/unified_index.py` | `create_registry_text()` dataset branch read `data.get("access")` (returns dict, not string) and `data.get("documentation")` (nonexistent field) | Fixed to use nested `access.level`, `provenance.source`, `technical.format` |
| `registry/text_generator.py` | Had person, project, funding summaries but no dataset summary — datasets omitted from `generate_all_summaries()` | Added `generate_dataset_summary()`, wired into `generate_all_summaries()` |
| `registry/rag.py` | Single-dataset formatter only rendered provenance, technical, variables, intended_use, limitations, access | Added conditional rendering for sensitivity, preprocessing, annotation, models/benchmarks, lifecycle, data_card_completeness |

**Lesson:** The pipeline code had never been tested against the actual nested schema. `_create_dataset_text()` was silently generating near-empty text for every dataset in the semantic index, meaning dataset search quality was severely degraded without anyone noticing.

---

## Phase 2: Systematic Critique

After implementing the Data Cards framework, a card-by-card audit revealed issues at three severity levels.

### 2.1 Errors Found

**Proprietary data mislabeled as public (WoS, Dimensions).** Both had `consent_status: "public_data_no_consent_required"` despite being proprietary licensed databases. This conflated CC0/public-domain datasets (OpenAlex, DBLP) with institutional-license datasets, which have very different legal implications for derived work and redistribution.

**PII classification was inconsistent.** The initial pass treated all bibliometric datasets as `contains_pii: false`, but PATSTAT contains an explicit `person_name` field with inventor/applicant names — real people's names that qualify as personal data under GDPR. Meanwhile, peer-review-cs had all sensitivity fields left null despite containing reviewer identities and opinions, which are inherently sensitive. Reddit's PII was understated as just "usernames" when posts routinely contain self-disclosed location, health, age, and relationship information.

**IRIS UMETRICS consent was misleading.** `"collected_with_consent"` implied research participants opted in, when the data is actually university administrative records collected through employment. The distinction matters for IRB review and downstream use.

**Maintainer/contact information was fragmented.** Four entries had contact names in `access.contact` but `maintainer.name` was null, creating a situation where the registry knew who to ask about a dataset but didn't record who was responsible for it.

### 2.2 Completeness Gaps Found

The audit revealed that the original (pre-Data-Cards) entries were far more uneven in quality than expected:

| Dataset | Missing Core Sections |
|---|---|
| iris-umetrics | `technical`, `variables`, `limitations`, `usage` — **no technical metadata at all** |
| dblp | `variables`, `limitations`, `usage` |
| peer-review-cs | `limitations`, `usage` |
| 6 other entries | `usage` only |

IRIS UMETRICS was essentially a stub: it had a name, description, provenance source, intended use, and access requirements, but nothing about what the data actually looks like. Someone encountering this entry would know the dataset exists and is restricted, but would have zero information about its format, size, schema, or where to find it.

### 2.3 Richness Gaps Found

Three high-value fields were null across nearly all entries despite being derivable from existing information:

- **`sampling.population`** — null for all 14, yet every entry's description already implied its population
- **`sampling.temporal_range`** — null for all 14, yet most entries had version dates and collection dates
- **`sensitivity.data_subjects`** — null for 12 of 14, yet every dataset has identifiable subjects

Additionally, `sampling.methodology` was only set for the 4 full-population datasets, despite WoS (curated journal selection), DBLP (curated venue coverage), and Reddit (near-complete API archive) all having well-known, distinctive methodologies.

Semantic Scholar's `known_models_and_benchmarks` was empty even though its own `variables` section listed SPECTER embeddings — the model trained on the data was documented in one field but not cross-referenced in the obvious place.

**Lesson:** Fields that require cross-referencing information from multiple sections within the same entry (e.g., "the collection_method mentions medrxivr, so preprocessing.tools_used should list it") are consistently missed by manual entry. Automated consistency checks would catch these.

---

## Phase 3: Fixes Implemented

### 3.1 Error Corrections

| Fix | Datasets | Change |
|---|---|---|
| Consent reclassification | wos-2023, dimensions | `"public_data_no_consent_required"` → `"proprietary_institutional_license"` |
| PII flagged | patstat | `contains_pii: true`, categories: `["inventor names", "applicant names"]` |
| Sensitivity populated | peer-review-cs | `contains_pii: true`, `risk_level: "medium"`, categories: `["reviewer identities", "review opinions"]` |
| PII categories expanded | reddit | Added `"self-disclosed personal information"`, `ethical_review` set |
| Consent corrected | iris-umetrics | `"collected_with_consent"` → `"administrative_data_governance"`, `ethical_review: "IRB approval required"` |
| Maintainer synced | reddit, peer-review-cs, dimensions | Populated `maintainer.name`/`contact` from `access.contact` |

### 3.2 Missing Sections Filled

**IRIS UMETRICS:** Added `technical` (all null — needs human verification), `variables` (7 fields: person_id, institution, department, role, funding_source, expenditure, year), `limitations` (coverage, bias, gaps documented).

**DBLP:** Added `variables` (8 fields: key, title, author, year, venue, type, doi, url), `limitations` (CS-only scope, limited citations, venue coverage bias).

**Peer Review CS:** Added `limitations` (OpenReview venue bias, historical coverage gaps).

**9 entries:** Added missing `usage` sections with empty defaults.

### 3.3 Enrichment Across All 14 Entries

| Field | Entries Filled | Examples |
|---|---|---|
| `sampling.population` | 14/14 | "Global scholarly publications across all disciplines" (OpenAlex), "Reddit platform users and their posts/comments" (Reddit), "Research personnel at 30+ US research universities" (IRIS) |
| `sampling.temporal_range` | 12/14 | "1800s-2024 (strongest post-2000)" (OpenAlex), "1900-2023" (WoS), "2005-2025" (Reddit) |
| `sensitivity.data_subjects` | 14/14 | "Scholarly authors, institutions, and publishers" (OpenAlex), "Patent inventors and applicants worldwide" (PATSTAT), "Biomedical researchers, patent inventors, clinical trial investigators" (PKG) |
| `sampling.methodology` | 10/14 | "Curated journal selection (editorial board review)" (WoS), "Near-complete historical archive via API and bulk downloads" (Reddit) |
| `sampling.known_exclusions` | reddit | "Private subreddits", "Deleted posts and comments", "Quarantined subreddit content (partial)" |

| Field | Entry | Value |
|---|---|---|
| `creators` | openalex | Jason Priem, Heather Piwowar, Richard Orr (OurResearch) |
| `preprocessing.raw_data_available` | openalex | `true` (format field says "parquet (cleaned), json (raw)") |
| `preprocessing.tools_used` | arxiv | `["medrxivr (R package for medRxiv/bioRxiv)"]` |
| `known_models_and_benchmarks.models_trained` | semantic-scholar | SPECTER, SPECTER2 |
| `known_models_and_benchmarks.leaderboard_url` | papers-with-code | `https://paperswithcode.com/sota` |
| `lifecycle.update_plan` | openalex, s2ag, mag, pwc | "Monthly snapshot refreshes" / "Weekly bulk releases" / "None - discontinued" / "Periodic bulk downloads" |
| `lifecycle.version_history` | mag-dec-2021 | "Monthly releases through December 2021 (final)" |

### 3.4 Structural Normalization

All 14 entries reordered to canonical key sequence: `id` → `name` → `description` → `version` → `provenance` → `technical` → `variables` → `intended_use` → `limitations` → `access` → `usage` → `maintainer` → [10 Data Card sections] → `tags`. Previously 12 of 14 had `maintainer` placed after `tags`.

---

## Lessons Learned

### On schema governance
The schema had never been validated against the data. Four enum mismatches existed undetected. **Recommendation:** Add `jsonschema.validate()` to CI or pre-commit hooks.

### On silent pipeline failures
`_create_dataset_text()` in `semantic_index.py` was silently generating near-empty text for every dataset because it used flat field paths (`dataset.get("format")`) that never matched the nested schema. Dataset semantic search was effectively broken with no errors raised. **Recommendation:** Add a minimum-length assertion or log warning for generated embedding text.

### On PII classification
The binary `contains_pii: true/false` is insufficient for scholarly data. Author names are personal data under GDPR but pose minimal re-identification risk when they're already public. Patent inventor names are a step higher. Reddit usernames with post content are higher still. IRIS employment records are genuinely sensitive. The `risk_level` field captures this gradient, but the initial pass applied `contains_pii: false` too broadly because all the bibliometric data "feels" public. **Recommendation:** Default to `contains_pii: true` for any dataset containing human names, then use `risk_level` to express the gradient.

### On consent terminology
`"public_data_no_consent_required"` was applied to both CC0 public-domain data (OpenAlex) and proprietary licensed data (WoS, Dimensions). These have very different implications: CC0 data can be freely redistributed, while proprietary data cannot, even though neither requires individual consent. The consent vocabulary needs to distinguish data-access-model from individual-consent-model. **Recommendation:** Use distinct values: `"public_domain"`, `"open_license"`, `"proprietary_institutional_license"`, `"terms_of_service"`, `"administrative_data_governance"`.

### On documentation decay
Several entries show clear signs of "wrote it once, never updated": IRIS has no technical metadata at all, DBLP is missing variables and limitations, peer-review-cs has null version/date/format. The most-documented entries (OpenAlex, Reddit, S2AG) are also the most recently updated. **Recommendation:** Use `data_card_completeness.last_reviewed` and set up periodic review reminders. The `data_card_completeness.sections_documented` field was designed for exactly this purpose but is currently empty everywhere.

### On cross-referencing within entries
Information present in one field was consistently not propagated to the logical companion field: ArXiv's `collection_method` mentions "R medrxivr package" but `preprocessing.tools_used` was empty; S2AG's `variables` lists SPECTER embeddings but `known_models_and_benchmarks` was empty; OpenAlex's `format` says "parquet (cleaned), json (raw)" but `preprocessing.raw_data_available` was null. **Recommendation:** An automated lint pass that checks for cross-field consistency would catch these.

---

## Verification

| Check | Result |
|---|---|
| `jsonschema.validate()` | PASSED |
| All 14 entries have `sampling.population` | PASSED |
| All 14 entries have `sensitivity.data_subjects` | PASSED |
| All 14 entries have `usage` section | PASSED |
| All 14 entries have canonical key ordering | PASSED |
| WoS consent = `proprietary_institutional_license` | PASSED |
| Dimensions consent = `proprietary_institutional_license` | PASSED |
| PATSTAT `contains_pii = true` | PASSED |
| Peer-review-cs sensitivity fully populated | PASSED |
| Reddit PII includes self-disclosed info | PASSED |
| IRIS has technical + variables + limitations | PASSED |
| DBLP has variables + limitations | PASSED |
| S2AG has SPECTER in known_models | PASSED |
| OpenAlex has 3 creators | PASSED |
| MAG lifecycle.update_plan = "None - discontinued" | PASSED |
| `pytest tests/test_query_router.py` (86 tests) | 86 PASSED |
| `pytest tests/test_semantic_registry.py` | 9 failed (pre-existing: missing `sentence_transformers`) |

---

## Remaining Items for Human Review

These items could not be filled without direct verification on Midway or consultation with data owners:

| Dataset | Missing Field | Action Needed |
|---|---|---|
| openalex | `technical.size_gb` | `du -sh /project/jevans/openalex-snapshot` |
| mag-dec-2021 | `technical.size_gb` | `du -sh /project/jevans/MAG_Dec_2021_snapshot/` |
| wos-2023 | `technical.size_gb`, `num_records` | Check actual data on Midway |
| peer-review-cs | `version`, `collection_date`, `format` | Ask Honglin Bao |
| iris-umetrics | All `technical` fields (format, size, location) | Requires DUA access; ask data steward |
| dimensions | `technical.size_gb`, `num_records` | `du -sh /project/jevans/dimensions` |
| icite | `technical.num_records` | `wc -l` on CSV files |
| All bibliometric datasets | Reconsider `contains_pii` | Policy decision: should author names = PII? |
| All entries | `data_card_completeness` | Set `last_reviewed` and `sections_documented` after human review |
| All entries | `social_impact` | Requires domain expert input per dataset |
| All entries | `example_data_points` | Populate with 1-2 sample records per dataset |

---

## Files Modified

| File | Commit | Nature |
|---|---|---|
| `Data/lab_registry_schema.json` | `b77891d8` | Enum fixes + 10 new Data Card sections |
| `Data/lab_registry.json` | `b77891d8`, `7c3c5fe6` | 14 entries: new sections, then error fixes + enrichment |
| `registry/semantic_index.py` | `b77891d8` | Fixed `_create_dataset_text()` + metadata extraction |
| `registry/unified_index.py` | `b77891d8` | Fixed `create_registry_text()` dataset branch |
| `registry/text_generator.py` | `b77891d8` | Added `generate_dataset_summary()` |
| `registry/rag.py` | `b77891d8` | Added Data Card section rendering |
