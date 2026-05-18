# Data Cards Coverage Report
**Generated:** 2026-01-15
**Total Data Cards:** 14

---

## Executive Summary

This report analyzes the coverage and completeness of the 14 data cards in the Knowledge Lab registry. Coverage is calculated based on the presence of key documentation fields defined in the data card schema.

### Coverage Metrics

| Dataset | Version | Coverage | Confidence | Variables |
|---------|---------|----------|------------|-----------|
| Semantic Scholar Academic Graph (S2AG) | 2025-01 | 100% | HIGH | 8 |
| PATSTAT Global Patent Database | 2023-autumn | 100% | HIGH | 6 |
| DBLP Computer Science Bibliography | 2025-01 | 100% | HIGH | 0 |
| iCite NIH Citation Data | 2023-09 | 100% | HIGH | 7 |
| Papers with Code | 2025-03 | 100% | HIGH | 7 |
| OpenAlex | 2024-12 | 93% | HIGH | 10 |
| Microsoft Academic Graph (Dec 2021) | 2021-12 | 93% | HIGH | 7 |
| Reddit Complete Archive | 2025-04 | 93% | HIGH | 7 |
| PubMed Knowledge Graph v2.0 | 2023 | 93% | HIGH | 6 |
| ArXiv and Preprint Metadata | 2023 | 93% | HIGH | 6 |
| Dimensions Database | 2025-06 | 86% | HIGH | 6 |
| Web of Science | 2023 | 79% | MEDIUM | 8 |
| Peer Review Data (CS Conferences) | — | 50% | LOW | 4 |
| IRIS UMETRICS | — | 43% | LOW | 0 |

### Coverage Distribution

- **HIGH (≥80%):** 11 datasets
- **MEDIUM (60-79%):** 1 dataset
- **LOW (<60%):** 2 datasets

**Average Coverage:** 87%

---

## Coverage Criteria

Each data card is evaluated against 14 key fields:

| Section | Field | Description |
|---------|-------|-------------|
| Core | `version` | Dataset version identifier |
| Core | `description` | What the dataset contains |
| Provenance | `source` | Original data provider |
| Provenance | `collection_date` | When data was collected |
| Provenance | `license` | Usage terms |
| Provenance | `citation` | How to cite |
| Technical | `format` | File format |
| Technical | `size_gb` | Storage size |
| Technical | `location` | Path on Midway |
| Variables | `variables` | Key fields documented |
| Use | `primary_use` | Main intended purpose |
| Use | `use_cases` | Specific applications |
| Limitations | `known_issues` | Documented problems |
| Access | `level` | Public/internal/restricted |

---

## Individual Card Analysis

### 1. OpenAlex
**Coverage: 93% (13/14 fields) | Confidence: HIGH**

```
ID:          openalex
Version:     2024-12
Size:        Not specified
Records:     250,000,000
Format:      parquet (cleaned), json (raw)
Location:    /project/jevans/openalex-snapshot
Access:      internal
```

**Strengths:**
- Comprehensive provenance (CC0 license, clear citation)
- Well-documented variables (10 key fields)
- Detailed intended use and limitations
- Clear coverage gaps documented

**Missing Fields:**
- `size_gb` not specified

**Example Variables:**
| Field | Type | Description |
|-------|------|-------------|
| id | string | OpenAlex ID (e.g., W2741809807) |
| doi | string | Digital Object Identifier |
| title | string | Work title |
| cited_by_count | integer | Total citation count |
| authorships | array | Authors with positions and affiliations |

---

### 2. Semantic Scholar Academic Graph (S2AG)
**Coverage: 100% (14/14 fields) | Confidence: HIGH**

```
ID:          semantic-scholar
Version:     2025-01
Size:        89 GB
Records:     200,000,000
Format:      json
Location:    /project/jevans/semantic_scholar_Jan31_2025
Access:      internal
```

**Strengths:**
- Complete documentation across all sections
- SPECTER embeddings included (768-dim vectors)
- AI-generated TLDRs as unique feature
- Citation contexts available

**Notable Variables:**
| Field | Type | Description |
|-------|------|-------------|
| embedding | array | SPECTER paper embedding (768-dim) |
| tldr | string | AI-generated summary |
| influentialCitationCount | integer | Highly influential citations |

---

### 3. Microsoft Academic Graph (Dec 2021)
**Coverage: 93% (13/14 fields) | Confidence: HIGH**

```
ID:          mag-dec-2021
Version:     2021-12
Size:        Not specified
Records:     260,000,000
Format:      mixed (tsv, json)
Location:    /project/jevans/MAG_Dec_2021_snapshot/
Access:      internal
```

**Strengths:**
- Historical reference for pre-2022 analysis
- Rich field of study taxonomy
- Well-documented as discontinued dataset

**Important Limitation:**
- Static dataset - no longer updated (Microsoft discontinued Dec 2021)

---

### 4. Web of Science
**Coverage: 79% (11/14 fields) | Confidence: MEDIUM**

```
ID:          wos-2023
Version:     2023
Size:        Not specified
Records:     Not specified
Format:      xml, parquet
Location:    /project/jevans/tip/data/wos_2023/
Access:      restricted
```

**Strengths:**
- Gold standard for citation analysis
- Coverage back to 1900
- High-quality curated journals

**Missing Fields:**
- `size_gb` not specified
- `num_records` not specified
- `citation` not provided

**Access Requirements:**
- Midway account
- UChicago affiliation (proprietary license)

---

### 5. Reddit Complete Archive
**Coverage: 93% (13/14 fields) | Confidence: HIGH**

```
ID:          reddit
Version:     2025-04
Size:        126 GB
Records:     Not specified
Format:      json (zst compressed)
Location:    /project/jevans/reddit_data (2024-2025)
            /project/jevans/hongkai/reddit (to 2023-02)
Access:      internal
```

**Strengths:**
- Long temporal coverage (2005-present)
- Multiple snapshots with clear locations
- Detailed demographic bias documentation

**Contact:** Hongkai Mao, Ruining He

---

### 6. PubMed Knowledge Graph v2.0
**Coverage: 93% (13/14 fields) | Confidence: HIGH**

```
ID:          pubmed-kg-v2
Version:     2023
Size:        144 GB
Records:     36,000,000
Format:      mixed
Location:    /project/jevans/PKG_v2_2023
Access:      internal
```

**Unique Features:**
- Cross-links publications with patents, clinical trials, and grants
- Entity annotations (genes, drugs, diseases)
- USPTO PatentsView integration
- NIH RePORTER grant linkage

---

### 7. PATSTAT Global Patent Database
**Coverage: 100% (14/14 fields) | Confidence: HIGH**

```
ID:          patstat
Version:     2023-autumn
Size:        64 GB
Records:     100,000,000
Format:      relational (multiple tables)
Location:    /project/jevans/PATSTAT
Access:      restricted
```

**Strengths:**
- Complete global patent coverage (100+ offices)
- Biannual update cycle documented
- Clear citation for academic use

**Access Requirements:**
- PATSTAT license awareness required

---

### 8. DBLP Computer Science Bibliography
**Coverage: 100% (14/14 fields) | Confidence: HIGH**

```
ID:          dblp
Version:     2025-01
Size:        10 GB
Records:     6,000,000
Format:      xml
Location:    /project/jevans/DBLP_2025_Jan
Access:      internal
```

**Strengths:**
- High-quality venue disambiguation
- Monthly update cycle
- Open license (ODC-BY)

**Scope:** Computer science only

---

### 9. Peer Review Data (CS Conferences)
**Coverage: 50% (7/14 fields) | Confidence: LOW**

```
ID:          peer-review-cs
Version:     —
Size:        Not specified
Records:     Not specified
Format:      Not specified
Location:    /project/jevans/Honglin_Bao_share/peer_review
Access:      internal
```

**Documented Fields:**
- Basic variables (review_text, score, confidence, decision)
- Intended use cases
- Contact: Honglin Bao

**Missing Documentation:**
- Version information
- Data format
- Size and record count
- Collection date
- License terms
- Limitations/biases

**Recommendation:** HIGH PRIORITY for documentation improvement

---

### 10. IRIS UMETRICS
**Coverage: 43% (6/14 fields) | Confidence: LOW**

```
ID:          iris-umetrics
Version:     —
Size:        Not specified
Records:     Not specified
Format:      Not specified
Location:    Not specified
Access:      restricted
```

**Known Information:**
- Source: University of Michigan IRIS
- Covers 30+ research universities
- Requires DUA and IRB approval

**Missing Documentation:**
- Almost all technical fields
- Variables/schema
- Limitations
- Collection details

**Recommendation:** HIGH PRIORITY - requires contact with IRIS for documentation

---

### 11. ArXiv and Preprint Metadata
**Coverage: 93% (13/14 fields) | Confidence: HIGH**

```
ID:          arxiv
Version:     2023
Size:        4 GB
Records:     Not specified
Format:      json, tsv
Location:    /project/jevans/arxiv
Access:      internal
```

**Strengths:**
- Covers multiple preprint servers (ArXiv, bioRxiv, medRxiv)
- Open access metadata (CC0 for ArXiv)
- Detailed variable documentation

**Limitations:**
- May need updating (2023 vintage)
- Physics/CS dominant on ArXiv, biomedical for bioRxiv/medRxiv

**Contact:** Jamshid Sourati

---

### 12. Dimensions Database
**Coverage: 86% (12/14 fields) | Confidence: HIGH**

```
ID:          dimensions
Version:     2025-06
Size:        Not specified
Records:     Not specified
Format:      partitioned (14 subdirectories)
Location:    /project/jevans/dimensions
Access:      restricted
```

**Strengths:**
- Enterprise-grade commercial database
- Cross-links papers, grants, patents, clinical trials, policy documents
- Broader coverage than academic-only databases

**Access Requirements:**
- Requires Dimensions license
- Contact: akozlo

---

### 13. iCite NIH Citation Data
**Coverage: 100% (14/14 fields) | Confidence: HIGH**

```
ID:          icite
Version:     2023-09
Size:        41 GB
Records:     Not specified
Format:      csv
Location:    /project/jevans/iCite_bulk_snapshot_Sep_2023
Access:      internal
```

**Unique Features:**
- Relative Citation Ratio (RCR) field-normalized metric
- NIH percentile rankings
- Clinical vs basic research classification
- Approximate Potential to Translate (APT) scores

**Notable Variables:**
| Field | Type | Description |
|-------|------|-------------|
| relative_citation_ratio | float | Field-normalized citation metric |
| nih_percentile | float | Percentile within NIH-funded papers |
| apt | float | Approximate Potential to Translate |

---

### 14. Papers with Code
**Coverage: 100% (14/14 fields) | Confidence: HIGH**

```
ID:          papers-with-code
Version:     2025-03
Size:        0.5 GB
Records:     Not specified
Format:      json.gz
Location:    /project/jevans/PwC_Mar_04_2025
Access:      internal
```

**Unique Features:**
- Links papers to code implementations
- Benchmark leaderboard data
- Method and dataset usage tracking
- State-of-the-art tracking for ML/AI

**Notable Variables:**
| Field | Type | Description |
|-------|------|-------------|
| repo_url | string | GitHub repository URL |
| tasks | array | ML tasks addressed |
| methods | array | Methods/architectures used |

---

## Recommendations

### High Priority (Confidence: LOW)

1. **Peer Review Data (CS Conferences)**
   - Add version/date information
   - Document file format and size
   - Add collection methodology
   - Document known limitations and biases

2. **IRIS UMETRICS**
   - Document data location on Midway
   - Add schema/variables section
   - Document size and record counts
   - Add collection date

### Medium Priority (Confidence: MEDIUM)

3. **Web of Science**
   - Add record count estimate
   - Add size in GB
   - Document official citation format

### Enhancement Opportunities

4. **All Datasets:**
   - Add `publications` linking papers that used each dataset
   - Add `derived_datasets` to track preprocessing artifacts
   - Document `last_verified` dates for maintainers

---

## Field Completion Summary

| Field | Complete | Missing |
|-------|----------|---------|
| description | 14/14 | 0 |
| location | 13/14 | 1 |
| primary_use | 14/14 | 0 |
| access.level | 14/14 | 0 |
| provenance.source | 14/14 | 0 |
| variables | 12/14 | 2 |
| version | 12/14 | 2 |
| technical.format | 12/14 | 2 |
| technical.size_gb | 9/14 | 5 |
| technical.num_records | 6/14 | 8 |
| provenance.citation | 9/14 | 5 |
| limitations | 11/14 | 3 |

---

## Next Steps

1. **Immediate:** Complete documentation for LOW confidence cards
2. **Short-term:** Add missing fields for MEDIUM confidence cards
3. **Ongoing:** Establish annual verification process for all cards
4. **Future:** Create model cards for embedding models (8 models identified)

---

*Report generated by CHORUS data catalog analysis*
