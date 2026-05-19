# CHORUS Audit Rebuild Summary

## Inputs

- Analysis documents: 467
- Registry people: 71
- Profile-publication rows: 482
- Source registry: `Data/lab_registry.json`
- Source hypergraph: `Data/hypergraph.json`

## Main diagnostic comparison

- Hidden Gem vs False Positive citation ratio: 2.25x

## What the script does

1. Builds a paper-level panel from recovered registry and hypergraph data.
2. Preserves document provenance and linked author/profile information.
3. Constructs transparent proxy scores for text surprise and representation contrast.
4. Uses citation counts only as an external audit outcome.
5. Writes a panel, summary files, and diagnostic figures.

## Reconstruction note

The recovered repository currently contains 467 hypergraph document nodes. The code keeps the current recovered sample visible rather than forcing an earlier sample size.

The score columns are transparent proxies because the original perplexity and embedding artifacts were not present in the recovered folder. Citations are used only as an audit outcome, not as inputs to the two proxy scores.

The result should be read as a diagnostic audit of ranking risk, not as a final estimate of scientific value.
