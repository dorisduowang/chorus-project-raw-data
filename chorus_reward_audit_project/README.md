# CHORUS Reward Signal Audit

This folder contains a small reproducible audit of a text-based ranking signal in the CHORUS project. The question is whether a system that rewards textual surprise would surface papers with stronger external evidence of influence, or whether it would over-rank papers that simply look complex in title form.

The project rebuilds a paper-level audit panel from recovered CHORUS registry and hypergraph data. Because the original perplexity scores and embedding artifacts were not preserved in the recovered folder, the code uses transparent proxy scores rather than claiming to recover the original model outputs. Citation counts are used only as an external audit outcome, not as inputs to the reconstructed scores.

## What this code does

The script does five things:

1. Reads the recovered registry and hypergraph files.
2. Builds a paper-level panel with visible provenance.
3. Constructs transparent proxy scores for text surprise and representation contrast.
4. Assigns papers to audit groups based on those two proxy scores.
5. Writes a panel, summary files, and three diagnostic figures.

The purpose is to make the audit logic inspectable and easy to extend once true perplexity scores or embeddings are available.

## Project contents

- `code/rebuild_chorus_audit.py`  
  Rebuilds the audit panel, constructs proxy measures, generates summary statistics, and writes figures.

- `output/chorus_audit_panel.csv`  
  Generated paper-level analysis panel. Each row is one hypergraph document node.

- `output/summary_stats.json`  
  Machine-readable summary of the rebuild.

- `output/summary.md`  
  Human-readable summary of the rebuild.

- `output/figure1_title_gradient.png`  
  Diagnostic figure showing the relationship between title word count and citation uptake.

- `output/figure2_reward_quadrants.png`  
  Diagnostic figure comparing quiet high-contrast papers with verbose low-contrast papers.

- `output/figure3_author_composition.png`  
  Diagnostic figure showing how linked author roles change the interpretation of the ranking signal.

- `overleaf/chorus_reward_audit.tex` and `overleaf/references.bib`  
  Optional LaTeX source for the short writing sample.

## How to run

From the repository root:

```bash
pip install -r chorus_reward_audit_project/requirements.txt

python3 chorus_reward_audit_project/code/rebuild_chorus_audit.py \
  --registry Data/lab_registry.json \
  --hypergraph Data/hypergraph.json \
  --output chorus_reward_audit_project/output
```
