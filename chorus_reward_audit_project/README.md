# CHORUS Reward Signal Audit

This folder contains a small reproducible audit of a text-based ranking signal
in the CHORUS project. The question is whether a system that rewards textual
surprise would surface papers with stronger external evidence of influence, or
whether it would over-rank papers that simply look complex in title form.

The project rebuilds a paper-level audit panel from recovered CHORUS registry
and hypergraph data. Because the original perplexity scores and embedding
artifacts were not preserved in the recovered folder, the code uses transparent
proxy scores rather than claiming to recover the original model outputs.
Citation counts are used only as an external audit outcome, not as inputs to the
reconstructed scores.

## What this code does

The script builds a paper-level panel, reconstructs simple proxy measures,
assigns papers to audit groups, and writes summary files and diagnostic figures.
The purpose is to make the audit logic inspectable and easy to extend once true
perplexity scores or embeddings are available.

## How to run

From the repository root:

```bash
python3 chorus_reward_audit_project/code/rebuild_chorus_audit.py \
  --registry Data/lab_registry.json \
  --hypergraph Data/hypergraph.json \
  --output chorus_reward_audit_project/output
```

Install dependencies with:

```bash
pip install -r chorus_reward_audit_project/requirements.txt
```

Before sharing or extending the package, I check that the script parses:

```bash
python3 -m py_compile chorus_reward_audit_project/code/rebuild_chorus_audit.py
```

## Reconstruction note

Memo 9 used an earlier audit sample. The recovered repository currently
contains 467 hypergraph document nodes and 482 profile-publication rows. This
rebuild keeps the current recovered sample visible rather than forcing the older
sample size.

The score columns are transparent proxies. They should not be read as the
original CHORUS perplexity or embedding outputs. The purpose of the rebuild is
to show the audit logic clearly, not to claim exact recovery of the original
analysis.
