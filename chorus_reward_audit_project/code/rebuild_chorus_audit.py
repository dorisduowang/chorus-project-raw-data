#!/usr/bin/env python3
"""
Rebuild the CHORUS reward-signal audit from recovered project data.

The original Memo 9 analysis referred to language-model perplexity and
representation-contrast measures, but the corresponding model artifacts were
not present in the recovered folder. This script therefore builds a transparent
reconstruction: it creates a paper-level panel, defines clearly labeled proxy
scores, and uses citation counts only as an external audit outcome.

The goal is not to reproduce the original model scores exactly. The goal is to
make the audit logic inspectable and easy to extend once true perplexity scores
or embeddings become available.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import textwrap
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

WORKSPACE = Path(__file__).resolve().parents[2]
PACKAGE_DIR = WORKSPACE / "chorus_reward_audit_project"
DEFAULT_REGISTRY = WORKSPACE / "Data" / "lab_registry.json"
DEFAULT_HYPERGRAPH = WORKSPACE / "Data" / "hypergraph.json"
DEFAULT_OUTPUT = PACKAGE_DIR / "output"

# Set the Matplotlib cache inside the project folder so the script can run
# cleanly in temporary or sandboxed environments.
os.environ.setdefault("MPLCONFIGDIR", str(PACKAGE_DIR / ".matplotlib"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt  # noqa: E402


TITLE_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?")

FACULTY_ROLES = {"Director", "Affiliate Faculty"}
PHD_ROLES = {"PhD Student"}
CORE_ROLES = FACULTY_ROLES | PHD_ROLES

HIGH_SIGNAL_VENUES = (
    "science",
    "nature",
    "pnas",
    "proceedings of the national academy",
    "cell",
    "management science",
    "american sociological review",
    "administrative science quarterly",
)


def stable_float(text: str, low: float = -0.02, high: float = 0.02) -> float:
    """Return deterministic jitter for plotting points with similar positions."""
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    unit = int(digest[:8], 16) / 0xFFFFFFFF
    return low + (high - low) * unit


def clean_key(doi: str | None, title: str) -> str:
    """Use DOI when available; otherwise fall back to a normalized title."""
    if doi:
        return "doi:" + doi.lower().strip()
    title_key = re.sub(r"\s+", " ", title.lower()).strip()
    title_key = re.sub(r"[^a-z0-9 ]", "", title_key)
    return "title:" + title_key


def count_title_words(title: str) -> int:
    return len(TITLE_WORD_RE.findall(title or ""))


def compact_list(values: list[str]) -> str:
    return "; ".join(v for v in values if v)


def safe_relative_path(path: Path, base: Path = WORKSPACE) -> str:
    """Return a project-relative path when possible."""
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except ValueError:
        return path.name


def validate_inputs(registry_path: Path, hypergraph_path: Path) -> None:
    """Fail early with a readable message if input files are missing."""
    missing = [str(p) for p in (registry_path, hypergraph_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required input file(s):\n"
            + "\n".join(missing)
            + "\nRun from the repository root or pass --registry and --hypergraph."
        )


def load_people(registry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    people: dict[str, dict[str, Any]] = {}
    for person in registry.get("people", {}).get("members", []):
        person_id = person.get("id")
        if not person_id:
            continue
        people[person_id] = {
            "name": person.get("name", ""),
            "role": person.get("role", ""),
            "institution": person.get("institution", ""),
            "topics": person.get("openalex", {}).get("topics", []) or [],
            "works_count": person.get("openalex", {}).get("works_count"),
            "cited_by_count": person.get("openalex", {}).get("cited_by_count"),
        }
    return people


def profile_publication_owners(registry: dict[str, Any]) -> dict[str, set[str]]:
    """Map each publication key to the lab profiles where it appeared."""
    owners: dict[str, set[str]] = {}
    for person in registry.get("people", {}).get("members", []):
        person_id = person.get("id")
        if not person_id:
            continue
        for pub in person.get("openalex", {}).get("recent_publications", []) or []:
            title = pub.get("title") or ""
            if not title:
                continue
            key = clean_key(pub.get("doi"), title)
            owners.setdefault(key, set()).add(person_id)
    return owners


def load_document_panel(
    registry_path: Path = DEFAULT_REGISTRY,
    hypergraph_path: Path = DEFAULT_HYPERGRAPH,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build one row per document node, enriched with lab author/profile links."""
    registry = json.loads(registry_path.read_text())
    people = load_people(registry)
    owner_lookup = profile_publication_owners(registry)

    graph = json.loads(hypergraph_path.read_text())
    edge_by_doc: dict[str, dict[str, Any]] = {}
    for edge in graph.get("edges", []):
        if edge.get("edge_type") != "publication":
            continue
        for doc_id in edge.get("outputs", []) or []:
            edge_by_doc[doc_id] = edge

    rows: list[dict[str, Any]] = []
    for node in graph.get("nodes", []):
        if node.get("type") != "Document":
            continue

        attrs = node.get("attributes", {}) or {}
        title = node.get("name") or ""
        if not title:
            continue

        key = clean_key(attrs.get("doi"), title)
        edge = edge_by_doc.get(node.get("id"), {})
        edge_author_ids = list(dict.fromkeys(edge.get("agents", []) or []))
        profile_owner_ids = sorted(owner_lookup.get(key, set()))

        # Keep both links. Edge authors come from the hypergraph builder; profile
        # owners preserve the raw OpenAlex profile source. The union is useful
        # for role-aware diagnostics, while both components stay visible.
        lab_link_ids = list(dict.fromkeys(edge_author_ids + profile_owner_ids))
        linked_people = [people[p] for p in lab_link_ids if p in people]
        roles = [p.get("role", "") for p in linked_people]
        names = [p.get("name", "") for p in linked_people]
        topics = sorted({t for p in linked_people for t in p.get("topics", [])})

        citation_count = attrs.get("citation_count")
        citation_count = 0 if citation_count in (None, "") else int(citation_count)

        rows.append(
            {
                "doc_id": node.get("id"),
                "paper_key": key,
                "title": title,
                "doi": attrs.get("doi") or "",
                "year": attrs.get("year"),
                "venue": attrs.get("venue") or "",
                "citation_count": citation_count,
                "log1p_citations": math.log1p(citation_count),
                "title_word_count": count_title_words(title),
                "edge_author_ids": compact_list(edge_author_ids),
                "profile_owner_ids": compact_list(profile_owner_ids),
                "lab_link_ids": compact_list(lab_link_ids),
                "lab_author_names": compact_list(names),
                "lab_author_roles": compact_list(roles),
                "topic_count": len(topics),
                "profile_link_count": len(profile_owner_ids),
                "lab_link_count": len(lab_link_ids),
                "topics": compact_list(topics[:12]),
            }
        )

    df = pd.DataFrame(rows).sort_values(["year", "title"], ascending=[False, True])
    meta = {
        "registry_people": len(people),
        "profile_publication_rows": sum(
            len(m.get("openalex", {}).get("recent_publications", []) or [])
            for m in registry.get("people", {}).get("members", [])
        ),
        "hypergraph_documents": len(df),
        "hypergraph_edges": len(graph.get("edges", [])),
        "source_registry": safe_relative_path(registry_path),
        "source_hypergraph": safe_relative_path(hypergraph_path),
    }
    return df.reset_index(drop=True), meta


def add_title_bins(df: pd.DataFrame) -> pd.DataFrame:
    bins = [
        (df["title_word_count"] <= 6, "<= 6 words"),
        (df["title_word_count"].between(7, 10), "7-10 words"),
        (df["title_word_count"].between(11, 14), "11-14 words"),
        (df["title_word_count"].between(15, 19), "15-19 words"),
        (df["title_word_count"] >= 20, ">= 20 words"),
    ]
    df = df.copy()
    df["title_bin"] = np.select([b[0] for b in bins], [b[1] for b in bins], default="other")
    return df


def add_proxy_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create transparent stand-ins for the missing model artifacts.

    The contrast proxy avoids citation counts and does not mechanically use
    the inverse of the text-surprise score, so the quadrant comparison is not
    generated by title length alone.
    """
    df = df.copy()
    word_count = df["title_word_count"].astype(float)
    wc_low = max(1, float(word_count.quantile(0.05)))
    wc_high = max(wc_low + 1, float(word_count.quantile(0.95)))
    verbosity = ((word_count - wc_low) / (wc_high - wc_low)).clip(0, 1)

    title = df["title"].fillna("")
    punctuation = (
        title.str.contains(":", regex=False).astype(float) * 0.40
        + title.str.contains("?", regex=False).astype(float) * 0.20
        + title.str.count(",").clip(0, 3) / 3 * 0.25
        + title.str.contains(" and ", case=False, regex=False).astype(float) * 0.15
    ).clip(0, 1)

    df["text_surprise_score"] = (
        0.84 * verbosity
        + 0.16 * punctuation
    ).clip(0, 1)

    topic_breadth = (df["topic_count"].astype(float) / 12).clip(0, 1)
    role_diversity = df["lab_author_roles"].fillna("").map(
        lambda text: len({r.strip() for r in text.split(";") if r.strip()}) / 4
    ).clip(0, 1)
    venue_signal = df["venue"].fillna("").str.lower().map(
        lambda venue: 1.0 if any(marker in venue for marker in HIGH_SIGNAL_VENUES) else 0.0
    )
    linked_profile_signal = (df["profile_link_count"].astype(float) / 3).clip(0, 1)

    df["representation_contrast_proxy"] = (
        0.45 * topic_breadth
        + 0.25 * role_diversity
        + 0.20 * venue_signal
        + 0.10 * linked_profile_signal
    ).clip(0, 1)

    df["role_aware_reward"] = (
        0.60 * df["representation_contrast_proxy"]
        + 0.25 * role_diversity
        + 0.15 * linked_profile_signal
    ).clip(0, 1)
    return df


def author_group(role_text: str) -> str:
    roles = [r.strip() for r in (role_text or "").split(";") if r.strip()]
    core_roles = [r for r in roles if r in CORE_ROLES]
    faculty = sum(1 for r in core_roles if r in FACULTY_ROLES)
    phd = sum(1 for r in core_roles if r in PHD_ROLES)

    if not core_roles:
        return "No KLab core author"
    if len(core_roles) == 1 and phd == 1:
        return "Solo PhD student"
    if len(core_roles) == 1 and faculty == 1:
        return "Solo faculty"
    if faculty >= 2 and phd == 0:
        return "Faculty + Faculty"
    if faculty >= 1 and phd >= 1:
        return "Mixed (faculty + PhD)"
    return "Other KLab author config"


def add_quadrants_and_groups(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    text_cut = float(df["text_surprise_score"].median())
    contrast_cut = float(df["representation_contrast_proxy"].median())
    conditions = [
        (df["text_surprise_score"] <= text_cut) & (df["representation_contrast_proxy"] > contrast_cut),
        (df["text_surprise_score"] > text_cut) & (df["representation_contrast_proxy"] > contrast_cut),
        (df["text_surprise_score"] <= text_cut) & (df["representation_contrast_proxy"] <= contrast_cut),
        (df["text_surprise_score"] > text_cut) & (df["representation_contrast_proxy"] <= contrast_cut),
    ]
    labels = ["Hidden Gem", "Aligned Surprise", "Routine", "False Positive"]
    df["reward_quadrant"] = np.select(conditions, labels, default="Unclassified")
    df["author_composition"] = df["lab_author_roles"].map(author_group)
    return df


def summarize(df: pd.DataFrame, meta: dict[str, Any]) -> dict[str, Any]:
    title_bins = (
        df.groupby("title_bin", observed=True)
        .agg(n=("title", "size"), mean_citations=("citation_count", "mean"))
        .reset_index()
    )
    quadrants = (
        df.groupby("reward_quadrant", observed=True)
        .agg(
            n=("title", "size"),
            mean_citations=("citation_count", "mean"),
            median_citations=("citation_count", "median"),
            mean_text_surprise=("text_surprise_score", "mean"),
            mean_contrast=("representation_contrast_proxy", "mean"),
        )
        .reset_index()
    )
    author_groups = (
        df.groupby("author_composition", observed=True)
        .agg(
            n=("title", "size"),
            mean_citations=("citation_count", "mean"),
            mean_contrast=("representation_contrast_proxy", "mean"),
        )
        .reset_index()
    )

    hidden = quadrants.loc[quadrants["reward_quadrant"] == "Hidden Gem", "mean_citations"]
    false = quadrants.loc[quadrants["reward_quadrant"] == "False Positive", "mean_citations"]
    ratio = None
    if not hidden.empty and not false.empty and float(false.iloc[0]) > 0:
        ratio = float(hidden.iloc[0] / false.iloc[0])

    return {
        "meta": meta,
        "analysis_documents": int(len(df)),
        "memo9_note": (
            "Memo 9 used an earlier audit sample. The recovered repository "
            f"currently contains {len(df)} hypergraph document nodes; the code "
            "keeps the current raw-data sample visible rather than forcing the old N."
        ),
        "title_bins": title_bins.to_dict(orient="records"),
        "reward_quadrants": quadrants.to_dict(orient="records"),
        "author_composition": author_groups.to_dict(orient="records"),
        "hidden_vs_false_positive_citation_ratio": ratio,
    }


def set_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 220,
            "font.family": "DejaVu Sans",
            "axes.edgecolor": "#d9e2ec",
            "axes.labelcolor": "#334155",
            "xtick.color": "#334155",
            "ytick.color": "#334155",
            "text.color": "#111827",
            "axes.titleweight": "bold",
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "grid.color": "#e5e7eb",
            "grid.linewidth": 0.8,
        }
    )


def plot_title_gradient(df: pd.DataFrame, output_dir: Path) -> None:
    set_style()
    order = ["<= 6 words", "7-10 words", "11-14 words", "15-19 words", ">= 20 words"]
    colors = {
        "<= 6 words": "#1f4fbf",
        "7-10 words": "#3b82f6",
        "11-14 words": "#64748b",
        "15-19 words": "#94a3b8",
        ">= 20 words": "#cbd5e1",
    }

    fig, ax = plt.subplots(figsize=(11.5, 6.8))
    for i, label in enumerate(order):
        subset = df[df["title_bin"] == label]
        x = np.full(len(subset), i, dtype=float)
        jitter = subset["paper_key"].map(lambda key: stable_float(key, -0.22, 0.22)).to_numpy()
        y = np.log1p(subset["citation_count"].to_numpy())
        size = 22 + 18 * np.sqrt(np.clip(subset["citation_count"].to_numpy(), 0, None) + 1) / 10
        ax.scatter(x + jitter, y, s=size, color=colors[label], alpha=0.72, edgecolor="white", linewidth=0.35)

        if len(subset):
            mean_citations = subset["citation_count"].mean()
            ax.hlines(math.log1p(mean_citations), i - 0.33, i + 0.33, color="#111827", linewidth=2.3)
            ax.text(
                i - 0.16,
                math.log1p(mean_citations) + 0.12,
                f"mean {mean_citations:.0f}",
                fontsize=9,
                weight="bold",
            )

    first_mean = df.loc[df["title_bin"] == "<= 6 words", "citation_count"].mean()
    last_mean = df.loc[df["title_bin"] == ">= 20 words", "citation_count"].mean()
    ratio = first_mean / last_mean if last_mean else np.nan
    ax.annotate(
        f"{ratio:.1f}x citation gap",
        xy=(4, math.log1p(last_mean)),
        xytext=(1.8, math.log1p(max(first_mean, 1)) - 0.6),
        arrowprops={"arrowstyle": "->", "color": "#475569", "lw": 1.8},
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#cbd5e1"},
        fontsize=11,
        weight="bold",
        color="#475569",
    )

    ticks = [0, 1, 5, 20, 50, 150, 400, 1100, 2200]
    ax.set_yticks([math.log1p(t) for t in ticks])
    ax.set_yticklabels([str(t) for t in ticks])
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order)
    ax.set_ylabel("log-scaled citation count")
    ax.set_xlabel("Title word-count bin")
    ax.grid(axis="y")
    ax.set_title(
        "Figure 1. Title-length bins and external uptake in the CHORUS audit\n"
        f"Current rebuild uses {len(df)} hypergraph document nodes"
    )
    fig.tight_layout()
    fig.savefig(output_dir / "figure1_title_gradient.png")
    plt.close(fig)


def plot_quadrants(df: pd.DataFrame, output_dir: Path) -> None:
    set_style()
    text_cut = float(df["text_surprise_score"].median())
    contrast_cut = float(df["representation_contrast_proxy"].median())
    fig, ax = plt.subplots(figsize=(10.8, 7.2))

    ax.axvspan(0, text_cut, ymin=contrast_cut, ymax=1, color="#dbeafe", alpha=0.35)
    ax.axvspan(text_cut, 1, ymin=0, ymax=contrast_cut, color="#fee2e2", alpha=0.32)
    ax.axvspan(text_cut, 1, ymin=contrast_cut, ymax=1, color="#ffedd5", alpha=0.25)
    ax.axvspan(0, text_cut, ymin=0, ymax=contrast_cut, color="#f8fafc", alpha=0.65)

    sc = ax.scatter(
        df["text_surprise_score"],
        df["representation_contrast_proxy"],
        c=df["log1p_citations"],
        cmap="YlOrRd",
        s=42,
        alpha=0.78,
        edgecolor="white",
        linewidth=0.35,
    )
    ax.axvline(text_cut, color="#94a3b8", linestyle="--", linewidth=1.5)
    ax.axhline(contrast_cut, color="#94a3b8", linestyle="--", linewidth=1.5)

    labels = {
        "Hidden Gem": (0.09, 0.88, "#1d4ed8"),
        "Aligned Surprise": (0.66, 0.88, "#9a3412"),
        "Routine": (0.09, 0.08, "#475569"),
        "False Positive": (0.66, 0.08, "#991b1b"),
    }
    qstats = (
        df.groupby("reward_quadrant")
        .agg(n=("title", "size"), mean_citations=("citation_count", "mean"))
        .to_dict(orient="index")
    )
    for label, (x, y, color) in labels.items():
        stats = qstats.get(label, {"n": 0, "mean_citations": 0})
        ax.text(
            x,
            y,
            f"{label.upper()}\nn = {stats['n']:.0f}\nmean cit = {stats['mean_citations']:.0f}",
            transform=ax.transAxes,
            color=color,
            fontsize=11,
            weight="bold",
            bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": color, "alpha": 0.18},
        )

    hidden_mean = qstats.get("Hidden Gem", {}).get("mean_citations", np.nan)
    false_mean = qstats.get("False Positive", {}).get("mean_citations", np.nan)
    if false_mean and not np.isnan(hidden_mean):
        ax.annotate(
            f"{hidden_mean / false_mean:.1f}x citation gap\nHidden Gems vs False Positives",
            xy=(text_cut, contrast_cut + 0.08),
            xytext=(text_cut + 0.05, contrast_cut + 0.18),
            arrowprops={"arrowstyle": "<->", "color": "#475569", "lw": 1.8},
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#cbd5e1"},
            fontsize=10,
            weight="bold",
            color="#475569",
        )

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("log(1 + citations)")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("Text surprise score (title-verbosity proxy)")
    ax.set_ylabel("Representation contrast proxy")
    ax.set_title(
        "Figure 2. Audit groups from text-surprise and contrast proxies"
    )
    ax.grid(alpha=0.6)
    fig.tight_layout()
    fig.savefig(output_dir / "figure2_reward_quadrants.png")
    plt.close(fig)


def plot_author_composition(df: pd.DataFrame, output_dir: Path) -> None:
    set_style()
    order = [
        "Solo PhD student",
        "Solo faculty",
        "Faculty + Faculty",
        "Mixed (faculty + PhD)",
        "No KLab core author",
        "Other KLab author config",
    ]
    agg = (
        df.groupby("author_composition", observed=True)
        .agg(
            n=("title", "size"),
            mean_citations=("citation_count", "mean"),
            mean_contrast=("representation_contrast_proxy", "mean"),
        )
        .reindex(order)
        .dropna(subset=["n"])
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 6.4), sharey=True)
    y = np.arange(len(agg))
    colors = ["#1d4ed8", "#3b82f6", "#64748b", "#93a4b8", "#cbd5e1", "#e2e8f0"][: len(agg)]

    axes[0].hlines(y, 0, agg["mean_citations"], color="#cbd5e1", linewidth=2.2)
    axes[0].scatter(agg["mean_citations"], y, s=170, color=colors, edgecolor="white", linewidth=1.2, zorder=3)
    for yy, (_, row) in zip(y, agg.iterrows()):
        axes[0].text(row["mean_citations"] + 3, yy, f"{row['mean_citations']:.0f} cit   (n={int(row['n'])})", va="center", fontsize=10, weight="bold")
    axes[0].set_xlabel("Mean citation count")
    axes[0].set_title("Mean citations by author composition")
    axes[0].grid(axis="x")

    axes[1].barh(y, agg["mean_contrast"], color=colors, alpha=0.72)
    for yy, (_, row) in zip(y, agg.iterrows()):
        axes[1].text(row["mean_contrast"] + 0.006, yy, f"{row['mean_contrast']:.3f}", va="center", fontsize=10, weight="bold")
    axes[1].axvline(df["representation_contrast_proxy"].median(), color="#ef4444", linestyle="--", linewidth=1.4, label="median")
    axes[1].set_xlabel("Mean representation contrast proxy")
    axes[1].set_title("Mean contrast by author composition")
    axes[1].grid(axis="x")
    axes[1].legend(loc="lower right", frameon=True)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(agg.index)
    axes[0].invert_yaxis()
    fig.suptitle("Figure 3. Citation and contrast patterns by linked author role", fontsize=14, weight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "figure3_author_composition.png")
    plt.close(fig)


def write_plain_language_summary(summary: dict[str, Any], output_dir: Path) -> None:
    ratio = summary.get("hidden_vs_false_positive_citation_ratio")
    ratio_text = "not available" if ratio is None else f"{ratio:.2f}x"
    lines = [
        "# CHORUS Audit Rebuild Summary",
        "",
        f"- Analysis documents: {summary['analysis_documents']}",
        f"- Registry people: {summary['meta']['registry_people']}",
        f"- Profile-publication rows: {summary['meta']['profile_publication_rows']}",
        f"- Hidden Gem vs False Positive citation ratio: {ratio_text}",
        "",
        "## Reconstruction note",
        "",
        textwrap.fill(summary["memo9_note"], width=92),
        "",
        "The score columns are transparent proxies because the original Memo 9 embedding and perplexity",
        "artifacts were not present in the recovered folder. Citations are used only as an audit",
        "outcome, not as an input to the two proxy scores.",
        "",
        "The result should be read as a diagnostic audit of ranking risk, not as a final estimate",
        "of scientific value.",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild the CHORUS Memo 9 audit package.")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--hypergraph", type=Path, default=DEFAULT_HYPERGRAPH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_inputs(args.registry, args.hypergraph)
    args.output.mkdir(parents=True, exist_ok=True)
    df, meta = load_document_panel(args.registry, args.hypergraph)
    df = add_title_bins(df)
    df = add_proxy_scores(df)
    df = add_quadrants_and_groups(df)

    df.to_csv(args.output / "chorus_audit_panel.csv", index=False)
    summary = summarize(df, meta)
    (args.output / "summary_stats.json").write_text(json.dumps(summary, indent=2))
    write_plain_language_summary(summary, args.output)

    plot_title_gradient(df, args.output)
    plot_quadrants(df, args.output)
    plot_author_composition(df, args.output)

    print(f"Wrote audit panel and figures to {args.output}")
    print(f"Documents analyzed: {len(df)}")
    if summary.get("hidden_vs_false_positive_citation_ratio") is not None:
        print(
            "Hidden Gem vs False Positive citation ratio: "
            f"{summary['hidden_vs_false_positive_citation_ratio']:.2f}x"
        )


if __name__ == "__main__":
    main()
