# src/analysis/generate_report_figures.py
# One-shot script to produce every figure and table the final report needs,
# using the .joblib results that train2d.py and train3d.py have already saved.
# No retraining required.
#
# Outputs (all under report_figures/):
#   * combined_results_table.csv  — all 18 datasets in one table
#   * combined_results_table.tex  — LaTeX booktabs version (drop into NeurIPS template)
#   * comparison_vs_medmnist.csv  — your AUC/ACC alongside the published baseline
#                                    (you fill in the baseline column from the paper)
#   * auc_comparison_2d.png
#   * auc_comparison_3d.png
#   * accuracy_comparison_2d.png
#   * accuracy_comparison_3d.png
#   * loss_curves_grid_2d.png     (already produced by overfit_diagnostics.py)
#   * loss_curves_grid_3d.png     (already produced by overfit_diagnostics.py)
#   * dataset_size_vs_auc.png     — scatter showing how training-set size
#                                    correlates with final AUC
#   * task_difficulty.png          — AUC grouped by task type (binary,
#                                    multi-class, multi-label, ordinal)

import os
import sys
import csv
import joblib
import numpy as np
import matplotlib.pyplot as plt

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, ".."))

from train2d import DATASETS_2D
from train3d import DATASETS_3D


MODELS_2D_DIR = "models_2d"
MODELS_3D_DIR = "models_3d"
OUTPUT_DIR = "report_figures"


def load_results(models_dir):
    rows = []
    if not os.path.exists(models_dir):
        print(f"[skip] missing {models_dir}")
        return rows
    for fname in sorted(os.listdir(models_dir)):
        if not fname.endswith(".joblib"):
            continue
        try:
            r = joblib.load(os.path.join(models_dir, fname))
            rows.append(r)
        except Exception as e:
            print(f"[skip] {fname}: {e}")
    return rows


def attach_meta(results, datasets_meta, dim_label):
    """Add task / n_classes / type info from DATASETS_2D / DATASETS_3D."""
    for r in results:
        meta = datasets_meta.get(r["dataset"], {})
        r["task"] = meta.get("task", "?")
        r["n_classes"] = meta.get("n_classes", 0)
        r["n_train"] = meta.get("n_train", 0)
        r["type"] = dim_label
    return results


def write_combined_csv(all_results, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Dataset", "Type", "Task", "Classes",
                    "Val AUC", "Val Accuracy", "Duration"])
        for r in sorted(all_results, key=lambda x: (x["type"], x["dataset"])):
            w.writerow([
                r["dataset"], r["type"], r["task"], r["n_classes"],
                f"{r['auc']:.4f}", f"{r['accuracy']:.4f}",
                r.get("duration", "—"),
            ])
    print(f"  wrote {path}")


def write_combined_latex(all_results, path):
    """LaTeX booktabs table — drop straight into the NeurIPS template."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rows = sorted(all_results, key=lambda x: (x["type"], x["dataset"]))

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Validation AUC and Accuracy across all 18 MedMNIST v2 datasets. "
        r"Our CNN architectures (Section~\ref{sec:method}) were trained with per-dataset "
        r"hyperparameter tuning. Bold indicates AUC $\geq 0.95$.}",
        r"\label{tab:full_results}",
        r"\small",
        r"\begin{tabular}{llcrrr}",
        r"\toprule",
        r"Dataset & Task & Classes & Val AUC & Val Acc & Time \\",
        r"\midrule",
    ]
    last_type = None
    for r in rows:
        if r["type"] != last_type:
            if last_type is not None:
                lines.append(r"\midrule")
            lines.append(rf"\multicolumn{{6}}{{l}}{{\textit{{{r['type']} datasets}}}} \\")
            last_type = r["type"]
        auc_str = f"\\textbf{{{r['auc']:.3f}}}" if r["auc"] >= 0.95 else f"{r['auc']:.3f}"
        ds_name = r["dataset"].replace("_", r"\_")
        lines.append(
            f"{ds_name} & {r['task']} & {r['n_classes']} & "
            f"{auc_str} & {r['accuracy']:.3f} & {r.get('duration', '--')} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  wrote {path}")


def write_comparison_template(all_results, path):
    """
    Comparison-with-MedMNIST-paper CSV. We fill in your numbers; you fill in
    the published ResNet-18 (28x28) numbers from Table 3 / Table 4 of:
      Yang et al., "MedMNIST v2 — A large-scale lightweight benchmark for 2D
      and 3D biomedical image classification," Scientific Data 10, 41 (2023).
    Available at https://www.nature.com/articles/s41597-022-01721-8

    Reading the paper's tables and pasting the numbers in is faster and more
    reliable than scraping the PDF.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "Dataset", "Type",
            "Ours: AUC", "Ours: Acc",
            "MedMNIST ResNet-18 (28): AUC", "MedMNIST ResNet-18 (28): Acc",
            "Notes",
        ])
        for r in sorted(all_results, key=lambda x: (x["type"], x["dataset"])):
            w.writerow([
                r["dataset"], r["type"],
                f"{r['auc']:.4f}", f"{r['accuracy']:.4f}",
                "", "",  # to be filled in from Table 3/4 of Yang et al. 2023
                "",
            ])
    print(f"  wrote {path}  (fill in baseline columns from MedMNIST v2 paper)")


def plot_auc_bar(results, title, output_path, baseline=None):
    """Bar chart of AUC by dataset, sorted descending."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    rows = sorted(results, key=lambda x: -x["auc"])
    names = [r["dataset"] for r in rows]
    aucs = [r["auc"] for r in rows]

    fig, ax = plt.subplots(figsize=(max(8, len(rows) * 0.7), 5))

    # Color bars by AUC: green ≥0.95, blue 0.85-0.95, orange 0.7-0.85, red <0.7
    def bar_color(auc):
        if auc >= 0.95: return "#2E7D32"
        if auc >= 0.85: return "#1976D2"
        if auc >= 0.70: return "#F57C00"
        return "#C62828"

    colors = [bar_color(a) for a in aucs]
    bars = ax.bar(names, aucs, color=colors, edgecolor="white", linewidth=0.5)

    if baseline is not None:
        ax.axhline(baseline, color="black", linestyle="--", linewidth=1, alpha=0.6,
                   label=f"Baseline avg = {baseline:.3f}")
        ax.legend(loc="lower right", fontsize=9)

    for bar, auc in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{auc:.3f}",
                ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("AUC", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=40, ha="right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {output_path}")


def plot_accuracy_bar(results, title, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    rows = sorted(results, key=lambda x: -x["accuracy"])
    names = [r["dataset"] for r in rows]
    accs = [r["accuracy"] for r in rows]

    fig, ax = plt.subplots(figsize=(max(8, len(rows) * 0.7), 5))
    bars = ax.bar(names, accs, color="#1976D2", edgecolor="white", linewidth=0.5)
    for bar, a in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{a:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=40, ha="right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {output_path}")


def plot_size_vs_auc(results, output_path):
    """Scatter: training set size vs AUC. Tells the story of data scale."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sizes = [r["n_train"] for r in results if r["n_train"] > 0]
    aucs = [r["auc"] for r in results if r["n_train"] > 0]
    names = [r["dataset"] for r in results if r["n_train"] > 0]
    types = [r["type"] for r in results if r["n_train"] > 0]

    fig, ax = plt.subplots(figsize=(9, 6))
    for t, c, m in [("2D", "#1976D2", "o"), ("3D", "#D32F2F", "s")]:
        xs = [sz for sz, ty in zip(sizes, types) if ty == t]
        ys = [a for a, ty in zip(aucs, types) if ty == t]
        ax.scatter(xs, ys, c=c, marker=m, s=80, alpha=0.7,
                   edgecolors="white", linewidth=1, label=f"{t} dataset")

    for n, sz, a in zip(names, sizes, aucs):
        ax.annotate(n, (sz, a), fontsize=7, alpha=0.75,
                    xytext=(5, 3), textcoords="offset points")

    ax.set_xscale("log")
    ax.set_xlabel("Training-set size (log scale)", fontsize=11)
    ax.set_ylabel("Validation AUC", fontsize=11)
    ax.set_title("Dataset Size vs. Final AUC", fontsize=12, fontweight="bold")
    ax.set_ylim(0.5, 1.02)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {output_path}")


def plot_task_difficulty(all_results, output_path):
    """Box plot of AUC grouped by task type — shows which tasks are hardest."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    by_task = {}
    for r in all_results:
        by_task.setdefault(r["task"], []).append(r["auc"])

    if not by_task:
        return

    tasks = sorted(by_task.keys())
    data = [by_task[t] for t in tasks]

    fig, ax = plt.subplots(figsize=(9, 5))
    bp = ax.boxplot(data, labels=tasks, patch_artist=True, widths=0.5)
    for patch in bp["boxes"]:
        patch.set_facecolor("#90CAF9")

    # Overlay individual points
    for i, vals in enumerate(data, start=1):
        x_jitter = np.random.normal(i, 0.04, size=len(vals))
        ax.scatter(x_jitter, vals, c="#1565C0", s=30, alpha=0.7,
                   edgecolors="white", linewidth=0.5, zorder=3)

    ax.set_ylabel("Validation AUC", fontsize=11)
    ax.set_title("AUC by Task Type", fontsize=12, fontweight="bold")
    ax.set_ylim(0.5, 1.02)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {output_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Generating report figures into {OUTPUT_DIR}/")

    results_2d = load_results(MODELS_2D_DIR)
    results_3d = load_results(MODELS_3D_DIR)
    attach_meta(results_2d, DATASETS_2D, "2D")
    attach_meta(results_3d, DATASETS_3D, "3D")

    if not results_2d and not results_3d:
        print("No saved results found. Run train2d.py / train3d.py first.")
        return

    all_results = results_2d + results_3d

    # Tables
    write_combined_csv(all_results,
                       os.path.join(OUTPUT_DIR, "combined_results_table.csv"))
    write_combined_latex(all_results,
                         os.path.join(OUTPUT_DIR, "combined_results_table.tex"))
    write_comparison_template(all_results,
                              os.path.join(OUTPUT_DIR, "comparison_vs_medmnist.csv"))

    # Bar charts
    if results_2d:
        plot_auc_bar(results_2d,
                     "Validation AUC — 2D MedMNIST datasets",
                     os.path.join(OUTPUT_DIR, "auc_comparison_2d.png"))
        plot_accuracy_bar(results_2d,
                          "Validation Accuracy — 2D MedMNIST datasets",
                          os.path.join(OUTPUT_DIR, "accuracy_comparison_2d.png"))

    if results_3d:
        plot_auc_bar(results_3d,
                     "Validation AUC — 3D MedMNIST datasets",
                     os.path.join(OUTPUT_DIR, "auc_comparison_3d.png"))
        plot_accuracy_bar(results_3d,
                          "Validation Accuracy — 3D MedMNIST datasets",
                          os.path.join(OUTPUT_DIR, "accuracy_comparison_3d.png"))

    # Cross-cutting analyses (these ARE the "more analysis from existing
    # results" you asked for)
    plot_size_vs_auc(all_results,
                     os.path.join(OUTPUT_DIR, "dataset_size_vs_auc.png"))
    plot_task_difficulty(all_results,
                         os.path.join(OUTPUT_DIR, "task_difficulty.png"))

    # Print headline numbers
    print("\n" + "=" * 60)
    print("  Headline numbers")
    print("=" * 60)
    if results_2d:
        avg_auc_2d = np.mean([r["auc"] for r in results_2d])
        avg_acc_2d = np.mean([r["accuracy"] for r in results_2d])
        print(f"  2D: avg AUC = {avg_auc_2d:.4f}  avg Acc = {avg_acc_2d:.4f}  "
              f"(n={len(results_2d)})")
    if results_3d:
        avg_auc_3d = np.mean([r["auc"] for r in results_3d])
        avg_acc_3d = np.mean([r["accuracy"] for r in results_3d])
        print(f"  3D: avg AUC = {avg_auc_3d:.4f}  avg Acc = {avg_acc_3d:.4f}  "
              f"(n={len(results_3d)})")
    print("=" * 60)
    print(f"\nAll outputs in: {os.path.abspath(OUTPUT_DIR)}/")


if __name__ == "__main__":
    main()
