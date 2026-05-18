# CHORUS Reward Audit Project

This folder contains an independent research project built from the recovered
CHORUS raw data. The project asks whether an AI research assistant can
misallocate attention when its reward signal treats surface textual surprise as
a proxy for intellectual value.

The project is designed to stand on its own. It is not framed as a package for
one lab. Its connection to AI advice, entrepreneurial learning, and heterogeneous
AI effects appears in the research memo as a substantive extension of the
mechanism studied here.

## Project Contents

1. `code/rebuild_chorus_audit.py`

   Rebuilds the paper level audit panel from the raw CHORUS registry and
   hypergraph data. The script reconstructs transparent proxy measures because
   the original perplexity and embedding artifacts were not present in the
   recovered folder.

2. `output/chorus_audit_panel.csv`

   Generated analysis panel with one row per hypergraph document node. The file
   includes title metadata, citation counts, author links, role composition,
   topic counts, reconstructed scores, and quadrant labels.

3. `output/summary_stats.json` and `output/summary.md`

   Machine readable and human readable summaries of the rebuild.

4. `output/figure1_title_gradient.png`

   Diagnostic figure showing the relationship between title word count and
   citation impact.

5. `output/figure2_reward_quadrants.png`

   Diagnostic figure comparing quiet high contrast papers with verbose false
   positives.

6. `output/figure3_author_composition.png`

   Diagnostic figure showing how role composition changes the interpretation of
   the ranking signal.

7. `overleaf/chorus_reward_audit.tex` and `overleaf/references.bib`

   Overleaf ready research memo and bibliography.

## How to Run the Project

From the workspace root, run:

```bash
python3 chorus_reward_audit_project/code/rebuild_chorus_audit.py
```

The script reads the registry and hypergraph files from the recovered CHORUS raw
data folder and writes all outputs into `chorus_reward_audit_project/output/`.

## Reconstruction Note

Memo 9 described a 443 paper audit sample. The current raw folder contains 467
hypergraph document nodes and 482 profile publication rows. The rebuilt code
keeps the current raw data sample visible rather than forcing the old sample
size.

The original text perplexity and embedding files were not present. The script
therefore rebuilds transparent proxy scores and keeps citations as an audit
outcome. Citations are not used as inputs to the proxy scores.

## Overleaf Note

Upload `overleaf/chorus_reward_audit.tex`, `overleaf/references.bib`, and the
PNG files from `output/` to Overleaf. The TeX file currently assumes the figures
remain in an output folder. If Overleaf places every file in one flat folder,
change the graphics path in the TeX source to the current folder.
