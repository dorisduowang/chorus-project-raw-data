# Memory Agent Improvement Proposals v2

**Date:** January 11, 2026
**Based on:** Comprehensive study of 26 query types

---

## Current Performance Summary

| Metric | Value |
|--------|-------|
| Tests Passed | 26/26 |
| Apologies | 0 |
| Hedging | 0 |
| Avg Latency | 3-4s (fast mode) |
| Avg Lines | 10.6 |

**Mode Distribution:** 25 fast mode, 1 tool mode (external query correctly routed)

---

## Issues Identified

### 1. Person Profiles Still Too Long
**Current:** 6 lines for "Who is James Evans?"
**Target:** 3-4 lines per prompt

**Example current response:**
```
James Evans is:
- Director of Knowledge Lab
- Max Palevksy Professor of Sociology and Data Science
- At University of Chicago
- Founding Faculty Director of Chicago Center for Computational Social Science
- Research areas include scientometrics, scientific computing, network analysis, and computational social science
```

**Proposed ideal:**
```
James Evans - Director, Knowledge Lab
Max Palevksy Professor of Sociology & Data Science, UChicago
Research: scientometrics, scientific computing, network analysis
```

### 2. List Queries Don't Respect 5-Item Cap
**Current:** "Who studies science of science?" returns 7 people + commentary (15 lines)
**Target:** 5 names max, one line each

**Example current response:**
```
Based on the lab registry, key researchers in science of science include:

James Evans (Director, University of Chicago): Leading researcher in scientometrics
Dashun Wang (Northwestern University): Complex network analysis, scientometrics
Misha Teplitskiy (University of Michigan): Academic publishing, scientometrics
Eamon Duede (Purdue University): Scientific computing, data management
Yong-Yeol Ahn (Indiana University): Research data management practices

PhD Students in this area:
Honglin Bao (University of Chicago)
Renli Wu (Wuhan University)

The lab documents note science of science is a "transdisciplinary approach..."

Current challenge: The field lacks comprehensive forums...
```

**Proposed ideal:**
```
Science of science researchers:
- James Evans (Director, UChicago)
- Dashun Wang (Northwestern)
- Misha Teplitskiy (Michigan)
- Eamon Duede (Purdue)
- Yong-Yeol Ahn (Indiana)
...and 2 more. Ask for full list if needed.
```

### 3. Comparisons Too Verbose
**Current:** "Compare James Evans and Dashun Wang" returns 22 lines
**Target:** Compact table or short bullet format

**Proposed ideal:**
```
Evans vs Wang:
- Evans: Director, UChicago. Focus: scientometrics, text mining, network analysis
- Wang: Affiliate, Northwestern. Focus: scientometrics, network analysis, career dynamics
- Overlap: Both study science of science, collaborate on research methodology
```

### 4. Unnecessary Commentary/Quotes
**Issue:** Adding document quotes and meta-commentary like:
- "The lab documents note science of science is a 'transdisciplinary approach...'"
- "Current challenge: The field lacks comprehensive forums..."

**Rule:** Just answer the question. Don't add context unless asked.

### 5. Resource Lists Could Be Tighter
**Current:** 15 lines for compute resources
**Target:** 8-10 lines with essential info only

---

## Proposed Prompt Changes

### A. Add explicit length examples

```diff
Style:
- Matter-of-fact. State what you have, nothing more.
- Default to short. One fact = one line. Person profile = 3-4 lines max.
+
+ FORMAT EXAMPLES:
+ Person: "Name - Role, Institution. Research: area1, area2"
+ List: "Name1 (Role), Name2 (Role), Name3 (Role)... and X more"
+ Not found: "I don't have that. Try [wiki / #tech-support]."
```

### B. Add "no commentary" rule

```diff
- If you don't have it: "I don't have that. Try [lab wiki / #tech-support / web search]."
+ - No commentary, quotes, or analysis unless asked
+ - If you don't have it: "I don't have that. Try [wiki / #tech-support]."
```

### C. Strengthen list limit with explicit format

```diff
- For lists: 5 names max, comma-separated. No details. "Name (Role), Name (Role), ..."
+ For lists: 5 names only. Format: "Name (Institution)" one per line. End with "...and X more" if needed.
```

---

## Implementation Priority

| Priority | Change | Impact |
|----------|--------|--------|
| High | Add "no commentary" rule | Reduces 15-22 line responses |
| High | Strengthen list format | Enforces 5-item cap |
| Medium | Add format examples | Guides response structure |
| Low | Tighten person profiles | Minor improvement |

---

## Expected Outcomes

| Metric | Current | Target |
|--------|---------|--------|
| Person profile lines | 6 | 3-4 |
| Topic list lines | 15-21 | 6-8 |
| Comparison lines | 22 | 8-10 |
| Resource list lines | 15 | 8-10 |

---

## Alternative Approaches

### A. Temperature Tuning
- Lower temperature might reduce verbosity
- Current: default (~1.0)
- Try: 0.7 or 0.5 for more focused responses

### B. Post-processing
- Truncate responses that exceed line limits
- Risk: Could cut off important info

### C. Few-shot Examples in Prompt
- Add 2-3 concrete Q&A examples
- Risk: Increases prompt size and token usage

---

## Recommendation

Implement the prompt changes (A, B, C above) and re-test. If verbosity persists, try temperature=0.7.
