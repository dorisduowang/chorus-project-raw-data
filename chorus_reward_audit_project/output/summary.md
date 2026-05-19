# CHORUS Reward Audit Rebuild Summary

## Inputs

- Registry file: `Data/lab_registry.json`
- Hypergraph file: `Data/hypergraph.json`
- Analysis sample: 467 hypergraph document nodes
- Registry people: 71
- Profile-publication rows before consolidation: 482
- Hypergraph edges: 594

## What the script does

1. Builds a paper-level panel from recovered registry and hypergraph data.
2. Preserves document provenance and linked author/profile information.
3. Constructs transparent proxy scores for text surprise and representation contrast.
4. Uses citation counts only as an external audit outcome.
5. Writes a panel, summary files, and three diagnostic figures.

## Main rebuilt patterns

- Papers with six or fewer title words average 152 citations.
- Papers with twenty or more title words average 21 citations.
- Hidden Gem papers average 97 citations.
- False Positive papers average 43 citations.
- Hidden Gem papers have about 2.25 times the citation count of False Positive papers.
- Solo PhD linked papers average 117 citations, compared with 69 for solo faculty linked papers.

## Reconstruction note

An earlier draft used a smaller audit sample. The current recovered data snapshot contains 467 hypergraph document nodes. I keep the current recovered sample visible rather than forcing the earlier sample size.

The original perplexity and embedding artifacts were not preserved in the recovered folder. The scores in this rebuild are proxies, not the original model scores. The purpose of the package is to make the audit logic transparent and reproducible, not to claim exact recovery of the original analysis.
