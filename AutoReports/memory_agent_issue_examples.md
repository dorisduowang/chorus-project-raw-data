# Memory Agent Issue Examples

**Date:** January 11, 2026

This document shows current vs. ideal responses for each identified issue.

---

## Issue 1: Person Profiles Too Long

**Target:** 3-4 lines

### Current Response (12 lines)
```
James Evans is:
- Director of Knowledge Lab
- Max Palevksy Professor of Sociology and Data Science at University of Chicago
- Founding Faculty Director of Chicago Center for Computational Social Science
- Also affiliated with Santa Fe Institute

Research areas include:
- Scientometrics
- Scientific Computing
- Biomedical Text Mining
- Topic Modeling
- Complex Network Analysis
```

### Ideal Response (3 lines)
```
James Evans - Director, Knowledge Lab
Max Palevksy Professor of Sociology & Data Science, UChicago
Research: scientometrics, scientific computing, network analysis
```

---

## Issue 2: Lists Exceed 5-Item Cap

**Target:** 5 names max, ~6 lines

### Current Response (18 lines)
```
Based on the lab registry, key researchers studying science of science include:

James Evans (Director, University of Chicago)
Dashun Wang (Affiliate Faculty, Northwestern University)
Misha Teplitskiy (Affiliate Faculty, University of Michigan)
Eamon Duede (Affiliate Faculty, Purdue University)
Yong-Yeol Ahn (Affiliate Faculty, Indiana University)
Honglin Bao (PhD Student, University of Chicago)
Renli Wu (PhD Student, Wuhan University)

Common research areas they explore include:
- Scientometrics and bibliometrics
- Complex network analysis
- Scientific computing and data management
- Scientific discovery processes
- Academic publishing dynamics

The search results note that while interest is growing, the field still lacks comprehensive institutional forums to bring all researchers together.
```

### Ideal Response (6 lines)
```
Science of science researchers:
- James Evans (Director, UChicago)
- Dashun Wang (Northwestern)
- Misha Teplitskiy (Michigan)
- Eamon Duede (Purdue)
- Yong-Yeol Ahn (Indiana)
...and 2 more.
```

---

## Issue 3: Comparisons Too Verbose

**Target:** 8-10 lines

### Current Response (23 lines)
```
James Evans and Dashun Wang:

**Similarities:**
- Both are researchers in scientometrics
- Both study complex network analysis techniques
- Both associated with Knowledge Lab

**James Evans**
- Role: Director
- Institution: University of Chicago
- Research areas: Scientific Computing, Biomedical Text Mining, Topic Modeling

**Dashun Wang**
- Role: Affiliate Faculty
- Institution: Northwestern University
- Research areas: Scientific Computing, Opinion Dynamics, Research Data Management

**Key Differences:**
- Evans leads the lab, Wang is an affiliate
- Different primary institutions
- Slightly different focus within scientific research methods

They have collaborated on multiple research publications, including work on scientific teams and science dynamics.
```

### Ideal Response (7 lines)
```
Evans vs Wang:
- Evans: Director, UChicago. Scientometrics, text mining, network analysis
- Wang: Affiliate, Northwestern. Scientometrics, network analysis, career dynamics

Overlap: Both study science of science, collaborate on methodology
Difference: Evans leads the lab; Wang is affiliate at different institution
```

---

## Issue 4: Unnecessary Commentary

**Target:** Just facts, no meta-analysis

### Current Response (20 lines)
```
Based on the available information:

The lab is focused on data projects related to technology and science. Specifically:

Key Goals:
- Produce and organize data on tech products, performance, and outcomes
- Link tech data to proposals, science, and technology information

Key People Mentioned:
- James Evans (likely lab lead)
- Alexis Puzon

Current Work:
- Developing projects like AutoInterp (an open-source, community-driven analysis project)
- Studying technology and expectation violations

Meeting Schedule:
- Weekly meetings on Mondays, 10-11am Central Time (hybrid format)

The lab seems to be doing complex work around data analysis, technology tracking, and understanding technological development and innovation.
```

**Problem:** Last line is unnecessary commentary/analysis.

### Ideal Response (8 lines)
```
Current lab activity from recent discussions:
- Data projects on tech products and outcomes
- AutoInterp: open-source analysis project
- Weekly meetings: Mondays 10-11am CT (hybrid)

Key people: James Evans (lead), Alexis Puzon
For more: check #general or lab wiki
```

---

## Issue 5: Resource Lists Could Be Tighter

**Target:** 8-10 lines

### Current Response (13 lines)
```
Based on the Lab Registry Data, Knowledge Lab has:

1. Microsoft Academic Graph (Dec 2021)
2. OpenAlex (October 2023)
3. Web of Science (2023)
4. IRIS Umetrics
5. Reddit Dump (two versions: to 2023-02 and 2024-04 to 2025-04)
6. PubMed Knowledge Graph Version 2 (2023)
7. Peer Review Data (Computer Science Conferences)
8. ProQuest TDM (newspapers and press releases)
9. Refinitiv Earnings Call Transcripts

These datasets cover academic publications, social media, research metrics, peer reviews, press materials, and corporate communications.
```

**Problem:** Last line is unnecessary summary/commentary.

### Ideal Response (10 lines)
```
Lab datasets:
- Microsoft Academic Graph (Dec 2021)
- OpenAlex (Oct 2023)
- Web of Science (2023)
- IRIS Umetrics
- Reddit Dump (2023-02, 2024-04)
- PubMed Knowledge Graph v2 (2023)
- Peer Review Data (CS conferences)
- ProQuest TDM
- Refinitiv Earnings Transcripts
```

---

## Summary of Changes Needed

| Issue | Problem | Fix |
|-------|---------|-----|
| Person profiles | 12 lines instead of 3-4 | Enforce compact format |
| List queries | 7+ names instead of 5 | Hard cap with "...and X more" |
| Comparisons | 23 lines with headers | Compact side-by-side format |
| Commentary | Adds analysis/summaries | "No commentary unless asked" rule |
| Resource lists | Ends with summary | Just list, no footer |

---

## Recommended Prompt Changes

1. Add explicit format examples to prompt
2. Add rule: "No commentary, summaries, or analysis unless asked"
3. Strengthen: "Lists: exactly 5 items max. End with '...and X more' if needed."
4. Add: "Person format: Name - Role, Institution. Research: area1, area2, area3"
