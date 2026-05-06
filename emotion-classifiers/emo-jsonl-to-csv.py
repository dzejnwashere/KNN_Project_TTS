"""
Emotion Classifier Comparison Analysis
Compares pipe1, pipe2, pipe3 against ground truth emotion labels.

Usage:
    python analyze_classifiers.py --input your_data.jsonl
    python analyze_classifiers.py --input your_data.jsonl --output results/
"""

import json
import argparse
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report
)

# ─────────────────────────────────────────────
# LABEL MAPPING
# ─────────────────────────────────────────────
# Canonical label set (keep amused as-is from GT)
# Maps each pipe's raw labels → canonical label

LABEL_MAP = {
    # pipe1 labels (already clean)
    "happy":     "happy",
    "sad":       "sad",
    "angry":     "anger",
    "fearful":   "fearful",
    "surprised": "surprised",
    "disgust":   "disgust",
    "neutral":   "neutral",
    "calm":      "calm",

    # pipe2 abbreviations
    "hap":       "happy",
    "ang":       "anger",
    "neu":       "neutral",

    # variants
    "anger":     "anger",
    # "sad" already covered above

    # GT labels that pipes don't have → map to closest
    # We keep "amused" in GT but map it to "happy" when looking at pipe predictions
    # (pipes can't predict amused, so we treat happy as the closest match)
    "amused":    "amused",   # kept as-is in GT
    "ps":        "surprised",  # pleasant surprise in GT → surprised
}

# When a GT label is "amused", what pipe label counts as a "hit"?
# This is used for top-k accuracy when GT has no direct match in pipe label set.
AMUSED_PIPE_EQUIVALENTS = {"happy"}  # treat "happy" as correct for "amused"

# All GT labels that should be treated as "happy" family for pipe scoring
GT_TO_PIPE_EQUIV = {
    "amused": AMUSED_PIPE_EQUIVALENTS,
}


def normalize_label(label: str) -> str:
    return LABEL_MAP.get(label.lower(), label.lower())


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────

def load_data(path: str) -> list[dict]:
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"Loaded {len(records)} records from {path}")
    return records


# ─────────────────────────────────────────────
# PREDICTION EXTRACTION
# ─────────────────────────────────────────────

def get_top_prediction(pipe_output: list[dict]) -> str:
    """Return the normalized top-1 label from a pipe output."""
    top = max(pipe_output, key=lambda x: x["score"])
    return normalize_label(top["label"])


def get_top_k_labels(pipe_output: list[dict], k: int = 3) -> list[str]:
    """Return top-k normalized labels sorted by score descending."""
    sorted_preds = sorted(pipe_output, key=lambda x: x["score"], reverse=True)
    return [normalize_label(p["label"]) for p in sorted_preds[:k]]


def get_top1_score(pipe_output: list[dict]) -> float:
    return max(p["score"] for p in pipe_output)


def reciprocal_rank(pipe_output: list[dict], gt: str) -> float:
    """
    MRR: 1/rank of correct label. Handles amused→happy equivalence.
    """
    sorted_preds = sorted(pipe_output, key=lambda x: x["score"], reverse=True)
    equivalents = GT_TO_PIPE_EQUIV.get(gt, {normalize_label(gt)})
    for rank, pred in enumerate(sorted_preds, start=1):
        if normalize_label(pred["label"]) in equivalents:
            return 1.0 / rank
    return 0.0


def is_top1_correct(pipe_output: list[dict], gt: str) -> bool:
    top = normalize_label(max(pipe_output, key=lambda x: x["score"])["label"])
    equivalents = GT_TO_PIPE_EQUIV.get(gt, {normalize_label(gt)})
    return top in equivalents


def is_topk_correct(pipe_output: list[dict], gt: str, k: int = 3) -> bool:
    top_k = get_top_k_labels(pipe_output, k)
    equivalents = GT_TO_PIPE_EQUIV.get(gt, {normalize_label(gt)})
    return bool(set(top_k) & equivalents)


# ─────────────────────────────────────────────
# BUILD RESULTS DATAFRAME
# ─────────────────────────────────────────────

def build_results(records: list[dict]) -> pd.DataFrame:
    rows = []
    for rec in records:
        gt = normalize_label(rec["emotion_gt"].lower())
        row = {
            "audio_path": rec["audio_path"],
            "gt": gt,
        }
        for pipe in ["pipe1", "pipe2", "pipe3"]:
            po = rec[pipe]
            row[f"{pipe}_top1"]          = get_top_prediction(po)
            row[f"{pipe}_top1_score"]    = get_top1_score(po)
            row[f"{pipe}_top1_correct"]  = is_top1_correct(po, gt)
            row[f"{pipe}_top3_correct"]  = is_topk_correct(po, gt, k=3)
            row[f"{pipe}_mrr"]           = reciprocal_rank(po, gt)
        rows.append(row)
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# METRICS SUMMARY
# ─────────────────────────────────────────────

def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    results = []
    for pipe in ["pipe1", "pipe2", "pipe3"]:
        acc    = df[f"{pipe}_top1_correct"].mean()
        top3   = df[f"{pipe}_top3_correct"].mean()
        mrr    = df[f"{pipe}_mrr"].mean()
        avg_conf = df[f"{pipe}_top1_score"].mean()

        # F1 — map GT and predictions together
        # For amused GT, we treat "happy" prediction as correct by remapping GT→happy for F1
        gt_for_f1   = df["gt"].apply(lambda x: "happy" if x == "amused" else x)
        pred_for_f1 = df[f"{pipe}_top1"]

        labels = sorted(set(gt_for_f1) | set(pred_for_f1))
        f1_weighted = f1_score(gt_for_f1, pred_for_f1, labels=labels,
                               average="weighted", zero_division=0)
        f1_macro    = f1_score(gt_for_f1, pred_for_f1, labels=labels,
                               average="macro",    zero_division=0)

        results.append({
            "pipe":        pipe,
            "top1_acc":    acc,
            "top3_acc":    top3,
            "mrr":         mrr,
            "f1_weighted": f1_weighted,
            "f1_macro":    f1_macro,
            "avg_confidence": avg_conf,
        })
    return pd.DataFrame(results).set_index("pipe")


# ─────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────

PIPE_COLORS = {
    "pipe1": "#4C9BE8",
    "pipe2": "#E8834C",
    "pipe3": "#5EC88A",
}

def set_style():
    plt.rcParams.update({
        "figure.facecolor": "#0F1117",
        "axes.facecolor":   "#1A1D27",
        "axes.edgecolor":   "#2E3250",
        "axes.labelcolor":  "#C8CDE0",
        "xtick.color":      "#8890AA",
        "ytick.color":      "#8890AA",
        "text.color":       "#C8CDE0",
        "grid.color":       "#2A2E40",
        "grid.linewidth":   0.6,
        "font.family":      "monospace",
        "axes.titlesize":   12,
        "axes.labelsize":   10,
    })


def plot_summary_metrics(metrics: pd.DataFrame, out: str):
    set_style()
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Classifier Performance Overview", fontsize=14, color="#E8EAF6", y=1.01)

    metric_cols  = ["top1_acc", "top3_acc", "f1_weighted", "f1_macro", "mrr"]
    metric_names = ["Top-1 Acc", "Top-3 Acc", "F1 Weighted", "F1 Macro", "MRR"]

    # --- Bar chart: main metrics
    ax = axes[0]
    x = np.arange(len(metric_cols))
    width = 0.25
    for i, pipe in enumerate(["pipe1", "pipe2", "pipe3"]):
        vals = [metrics.loc[pipe, c] for c in metric_cols]
        bars = ax.bar(x + i * width, vals, width, label=pipe,
                      color=PIPE_COLORS[pipe], alpha=0.85)
    ax.set_xticks(x + width)
    ax.set_xticklabels(metric_names, rotation=25, ha="right", fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.set_title("Key Metrics")
    ax.legend(fontsize=8)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)

    # --- Radar chart
    ax2 = axes[1]
    ax2.set_facecolor("#1A1D27")
    categories = metric_names
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    ax2 = fig.add_subplot(132, polar=True)
    ax2.set_facecolor("#1A1D27")
    ax2.spines["polar"].set_color("#2E3250")

    for pipe in ["pipe1", "pipe2", "pipe3"]:
        vals = [metrics.loc[pipe, c] for c in metric_cols]
        vals += vals[:1]
        ax2.plot(angles, vals, color=PIPE_COLORS[pipe], linewidth=2, label=pipe)
        ax2.fill(angles, vals, color=PIPE_COLORS[pipe], alpha=0.1)

    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(categories, size=8, color="#8890AA")
    ax2.set_ylim(0, 1)
    ax2.set_title("Radar Comparison", pad=15, color="#C8CDE0")
    ax2.tick_params(colors="#8890AA")
    ax2.yaxis.set_tick_params(labelcolor="#555")
    ax2.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8)

    # --- Avg confidence bar
    ax3 = axes[2]
    pipes = ["pipe1", "pipe2", "pipe3"]
    confs = [metrics.loc[p, "avg_confidence"] for p in pipes]
    bars = ax3.bar(pipes, confs, color=[PIPE_COLORS[p] for p in pipes], alpha=0.85)
    ax3.set_ylim(0, 1.1)
    ax3.set_title("Avg Top-1 Confidence")
    ax3.yaxis.grid(True)
    ax3.set_axisbelow(True)
    for bar, val in zip(bars, confs):
        ax3.text(bar.get_x() + bar.get_width()/2, val + 0.02,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=9, color="#C8CDE0")

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0F1117")
    plt.close()
    print(f"Saved: {out}")


def plot_confusion_matrices(df: pd.DataFrame, out: str):
    set_style()
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Confusion Matrices (GT rows, Predicted cols)", fontsize=13,
                 color="#E8EAF6")

    gt_mapped = df["gt"].apply(lambda x: "happy" if x == "amused" else x)

    for ax, pipe in zip(axes, ["pipe1", "pipe2", "pipe3"]):
        pred = df[f"{pipe}_top1"]
        labels = sorted(set(gt_mapped) | set(pred))
        cm = confusion_matrix(gt_mapped, pred, labels=labels)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

        sns.heatmap(cm_norm, annot=True, fmt=".2f", ax=ax,
                    xticklabels=labels, yticklabels=labels,
                    cmap="Blues", linewidths=0.4, linecolor="#2A2E40",
                    annot_kws={"size": 8})
        ax.set_title(pipe, color="#C8CDE0")
        ax.set_xlabel("Predicted", color="#8890AA")
        ax.set_ylabel("Ground Truth", color="#8890AA")
        ax.tick_params(axis="x", rotation=45)
        ax.tick_params(axis="y", rotation=0)

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0F1117")
    plt.close()
    print(f"Saved: {out}")


def plot_per_class_f1(df: pd.DataFrame, out: str):
    set_style()
    gt_mapped = df["gt"].apply(lambda x: "happy" if x == "amused" else x)
    all_labels = sorted(set(gt_mapped))

    per_class = {}
    for pipe in ["pipe1", "pipe2", "pipe3"]:
        pred = df[f"{pipe}_top1"]
        labels = sorted(set(gt_mapped) | set(pred))
        report = classification_report(gt_mapped, pred, labels=labels,
                                       output_dict=True, zero_division=0)
        per_class[pipe] = {lbl: report.get(lbl, {}).get("f1-score", 0)
                           for lbl in all_labels}

    x = np.arange(len(all_labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(14, 5))
    for i, pipe in enumerate(["pipe1", "pipe2", "pipe3"]):
        vals = [per_class[pipe][lbl] for lbl in all_labels]
        ax.bar(x + i * width, vals, width, label=pipe,
               color=PIPE_COLORS[pipe], alpha=0.85)

    ax.set_xticks(x + width)
    ax.set_xticklabels(all_labels, rotation=30, ha="right")
    ax.set_ylim(0, 1.1)
    ax.set_title("Per-Class F1 Score by Classifier")
    ax.set_ylabel("F1 Score")
    ax.legend()
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0F1117")
    plt.close()
    print(f"Saved: {out}")


def plot_confidence_distribution(df: pd.DataFrame, out: str):
    set_style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    fig.suptitle("Top-1 Confidence Distribution (correct vs incorrect)",
                 color="#E8EAF6", fontsize=13)

    for ax, pipe in zip(axes, ["pipe1", "pipe2", "pipe3"]):
        correct   = df[df[f"{pipe}_top1_correct"]][f"{pipe}_top1_score"]
        incorrect = df[~df[f"{pipe}_top1_correct"]][f"{pipe}_top1_score"]
        bins = np.linspace(0, 1, 25)
        ax.hist(correct,   bins=bins, alpha=0.7, color="#5EC88A", label="Correct")
        ax.hist(incorrect, bins=bins, alpha=0.7, color="#E85C5C", label="Incorrect")
        ax.set_title(pipe, color="#C8CDE0")
        ax.set_xlabel("Confidence Score")
        ax.legend(fontsize=8)
        ax.yaxis.grid(True)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0F1117")
    plt.close()
    print(f"Saved: {out}")


def plot_calibration(df: pd.DataFrame, out: str):
    """Reliability diagram: mean predicted confidence vs actual accuracy per bin."""
    set_style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    fig.suptitle("Calibration Curves (Reliability Diagrams)",
                 color="#E8EAF6", fontsize=13)

    bins = np.linspace(0, 1, 11)

    for ax, pipe in zip(axes, ["pipe1", "pipe2", "pipe3"]):
        conf    = df[f"{pipe}_top1_score"].values
        correct = df[f"{pipe}_top1_correct"].values.astype(float)

        bin_means, bin_accs, bin_counts = [], [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (conf >= lo) & (conf < hi)
            if mask.sum() > 0:
                bin_means.append(conf[mask].mean())
                bin_accs.append(correct[mask].mean())
                bin_counts.append(mask.sum())

        ax.plot([0, 1], [0, 1], "w--", linewidth=1, alpha=0.4, label="Perfect calibration")
        ax.plot(bin_means, bin_accs, "o-", color=PIPE_COLORS[pipe],
                linewidth=2, markersize=6, label=pipe)
        ax.fill_between(bin_means, bin_accs, bin_means,
                        alpha=0.15, color=PIPE_COLORS[pipe])
        ax.set_title(pipe, color="#C8CDE0")
        ax.set_xlabel("Mean Predicted Confidence")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
        ax.yaxis.grid(True)
        ax.xaxis.grid(True)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Fraction Correct")
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0F1117")
    plt.close()
    print(f"Saved: {out}")


def plot_accuracy_per_gt_class(df: pd.DataFrame, out: str):
    set_style()
    gt_classes = sorted(df["gt"].unique())
    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(gt_classes))
    width = 0.25

    for i, pipe in enumerate(["pipe1", "pipe2", "pipe3"]):
        accs = []
        for gt_cls in gt_classes:
            mask = df["gt"] == gt_cls
            accs.append(df[mask][f"{pipe}_top1_correct"].mean() if mask.sum() > 0 else 0)
        ax.bar(x + i * width, accs, width, label=pipe,
               color=PIPE_COLORS[pipe], alpha=0.85)

    ax.set_xticks(x + width)
    ax.set_xticklabels(gt_classes, rotation=30, ha="right")
    ax.set_ylim(0, 1.1)
    ax.set_title("Top-1 Accuracy per Ground Truth Emotion")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0F1117")
    plt.close()
    print(f"Saved: {out}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Emotion classifier comparison")
    parser.add_argument("--input",  required=True, help="Path to .jsonl file")
    parser.add_argument("--output", default="./results", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Load & build
    records = load_data(args.input)
    df      = build_results(records)
    metrics = compute_metrics(df)

    # Print summary
    print("\n" + "="*60)
    print("METRICS SUMMARY")
    print("="*60)
    print(metrics.round(4).to_string())
    print()

    # Save CSV
    csv_path = os.path.join(args.output, "metrics_summary.csv")
    metrics.to_csv(csv_path)
    print(f"Saved metrics CSV: {csv_path}")

    df_path = os.path.join(args.output, "full_results.csv")
    df.to_csv(df_path, index=False)
    print(f"Saved full results: {df_path}")

    # Plots
    plot_summary_metrics(metrics,
        os.path.join(args.output, "01_summary_metrics.png"))
    plot_confusion_matrices(df,
        os.path.join(args.output, "02_confusion_matrices.png"))
    plot_per_class_f1(df,
        os.path.join(args.output, "03_per_class_f1.png"))
    plot_confidence_distribution(df,
        os.path.join(args.output, "04_confidence_distribution.png"))
    plot_calibration(df,
        os.path.join(args.output, "05_calibration_curves.png"))
    plot_accuracy_per_gt_class(df,
        os.path.join(args.output, "06_accuracy_per_class.png"))

    print("\nDone! All outputs saved to:", args.output)


if __name__ == "__main__":
    main()