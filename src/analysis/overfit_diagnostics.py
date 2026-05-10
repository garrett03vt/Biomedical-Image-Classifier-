# overfit_analysis.py
# Load saved model results and generate train/val loss curve plots.
# Detects overfitting (val loss diverges from train loss) and
# underfitting (both losses stay high).
#
# Usage:
#   python overfit_analysis.py          — analyses all saved models
#   python overfit_analysis.py 2d       — 2D models only
#   python overfit_analysis.py 3d       — 3D models only
#
# Plots are saved to overfit_plots/

import os
import sys
import joblib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


MODELS_2D_DIR  = "models_2d"
MODELS_3D_DIR  = "models_3d"
OUTPUT_DIR     = "overfit_plots"

# Thresholds for flagging
OVERFIT_GAP    = 0.15   # val_loss - train_loss > this at the final epoch → overfit

# Underfit floor depends on the loss structure of the task. With label
# smoothing of α on a K-class CE loss, even a perfect classifier has
# train loss ≈ -((1-α)·log(1-α + α/K) + (K-1)·(α/K)·log(α/K)). For
# common configs this floor can be substantial:
#   * 11-class with α=0.1 → floor ≈ 0.36
#   * 3-class  with α=0.1 → floor ≈ 0.43
#   * 2-class  with α=0.1 → floor ≈ 0.33
# Without per-model smoothing info we can't compute the exact floor, so
# we use a conservative absolute threshold that doesn't false-positive on
# smoothed multi-class losses. If a final-epoch val loss is well above
# this floor *and* far from the train loss, it's actually underfit.
UNDERFIT_FLOOR_TRAIN = 0.7   # train loss must exceed this to flag underfit
UNDERFIT_FLOOR_VAL   = 0.7   # val loss must also exceed this


def load_all_results(models_dir, suffix):
    results = []
    if not os.path.exists(models_dir):
        print(f"  [skip] Folder not found: {models_dir}")
        return results

    for fname in sorted(os.listdir(models_dir)):
        if not fname.endswith(".joblib"):
            continue
        path = os.path.join(models_dir, fname)
        try:
            r = joblib.load(path)
        except Exception as e:
            print(f"  [skip] Could not load {fname}: {e}")
            continue

        # Skip files that pre-date loss tracking
        if "train_losses" not in r or "val_losses" not in r:
            print(f"  [skip] {fname} — no loss curves (retrain to generate them)")
            continue

        r["_suffix"] = suffix
        results.append(r)

    return results


def diagnose(train_losses, val_losses):
    if not train_losses or not val_losses:
        return "no data"

    final_train = train_losses[-1]
    final_val   = val_losses[-1]
    gap         = final_val - final_train

    # Underfitting requires both losses to be high in absolute terms.
    # A model with label smoothing on a multi-class task can have train
    # loss of 0.6+ while being a near-perfect classifier (AUC > 0.99),
    # so we use a higher floor than v1 and require it on both sides.
    if final_train > UNDERFIT_FLOOR_TRAIN and final_val > UNDERFIT_FLOOR_VAL:
        return "UNDERFITTING"
    if gap > OVERFIT_GAP:
        return "OVERFITTING"
    return "OK"


def plot_loss_curve(ax, train_losses, val_losses, title, diagnosis, fontsize_scale=1.0):
    epochs = range(1, len(train_losses) + 1)

    ax.plot(epochs, train_losses, label="Train loss", linewidth=2.0, color="#2196F3")
    ax.plot(epochs, val_losses,   label="Val loss",   linewidth=2.0, color="#F44336", linestyle="--")

    # Shade the gap between curves
    ax.fill_between(epochs, train_losses, val_losses,
                    where=[v > t for t, v in zip(train_losses, val_losses)],
                    alpha=0.15, color="#F44336", label="Val > Train")
    ax.fill_between(epochs, train_losses, val_losses,
                    where=[v <= t for t, v in zip(train_losses, val_losses)],
                    alpha=0.15, color="#4CAF50", label="Train > Val")

    color_map = {"OK": "#4CAF50", "OVERFITTING": "#F44336", "UNDERFITTING": "#FF9800", "no data": "#9E9E9E"}
    color = color_map.get(diagnosis, "#9E9E9E")

    s = fontsize_scale
    ax.set_title(f"{title}\n[{diagnosis}]",
                 fontsize=int(11 * s), color=color, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=int(10 * s))
    ax.set_ylabel("Loss", fontsize=int(10 * s))
    ax.tick_params(labelsize=int(9 * s))
    ax.legend(fontsize=int(9 * s))
    ax.grid(True, alpha=0.3)


def make_grid_plot(results, title, output_path, ncols=4, fontsize_scale=1.0):
    n = len(results)
    if n == 0:
        print(f"  No results to plot for: {title}")
        return

    ncols = min(ncols, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5.5, nrows * 4.2))
    fig.suptitle(title, fontsize=int(14 * fontsize_scale), fontweight="bold", y=1.01)

    if nrows == 1 and ncols == 1:
        axes = [[axes]]
    elif nrows == 1:
        axes = [axes]
    elif ncols == 1:
        axes = [[ax] for ax in axes]

    flat_axes = [ax for row in axes for ax in row]

    for i, r in enumerate(results):
        train_losses = r.get("train_losses", [])
        val_losses   = r.get("val_losses",   [])
        dataset      = r.get("dataset", "unknown")
        auc          = r.get("auc", float("nan"))
        acc          = r.get("accuracy", float("nan"))
        diag         = diagnose(train_losses, val_losses)

        plot_label = f"{dataset}\nAUC={auc:.3f}  Acc={acc:.3f}"
        plot_loss_curve(flat_axes[i], train_losses, val_losses, plot_label, diag,
                        fontsize_scale=fontsize_scale)

    for j in range(len(results), len(flat_axes)):
        flat_axes[j].set_visible(False)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {output_path}")


def make_individual_plots(results, suffix, subdir):
    """One PNG per dataset — fully report-ready, large fonts, no grid."""
    if not results:
        return
    out_dir = os.path.join(OUTPUT_DIR, subdir)
    os.makedirs(out_dir, exist_ok=True)
    print(f"  Per-dataset plots → {out_dir}/")
    for r in results:
        train_losses = r.get("train_losses", [])
        val_losses   = r.get("val_losses",   [])
        if not train_losses or not val_losses:
            continue
        dataset = r.get("dataset", "unknown")
        auc = r.get("auc", float("nan"))
        acc = r.get("accuracy", float("nan"))
        diag = diagnose(train_losses, val_losses)

        fig, ax = plt.subplots(figsize=(7, 5))
        plot_loss_curve(
            ax, train_losses, val_losses,
            f"{dataset}  |  AUC={auc:.3f}  Acc={acc:.3f}",
            diag,
            fontsize_scale=1.4,
        )
        fig.tight_layout()
        out_path = os.path.join(out_dir, f"{dataset}_loss_curve.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    print(f"    Wrote {len(results)} files into {out_dir}/")


def make_split_grids(results, title_base, file_base, ncols=2, per_grid=4):
    """
    Split a long results list into multiple smaller grids.
    With per_grid=4 and ncols=2 you get 2x2 grids — each subplot is large
    enough to be readable when the figure is scaled to a report column.
    """
    if not results:
        return
    chunks = [results[i:i + per_grid] for i in range(0, len(results), per_grid)]
    print(f"  Split into {len(chunks)} smaller grid(s) of up to {per_grid} datasets each")
    for idx, chunk in enumerate(chunks, start=1):
        out_path = os.path.join(OUTPUT_DIR, f"{file_base}_part{idx}.png")
        make_grid_plot(
            chunk,
            f"{title_base} (part {idx} of {len(chunks)})",
            out_path,
            ncols=ncols,
            fontsize_scale=1.3,
        )


def make_summary_plot(all_results, output_path):
    if not all_results:
        return

    datasets    = [r["dataset"]   for r in all_results]
    aucs        = [r.get("auc",       float("nan")) for r in all_results]
    accs        = [r.get("accuracy",  float("nan")) for r in all_results]
    diagnoses   = [diagnose(r.get("train_losses", []), r.get("val_losses", [])) for r in all_results]

    color_map = {"OK": "#4CAF50", "OVERFITTING": "#F44336", "UNDERFITTING": "#FF9800", "no data": "#9E9E9E"}
    bar_colors = [color_map.get(d, "#9E9E9E") for d in diagnoses]

    fig, axes = plt.subplots(1, 2, figsize=(max(12, len(datasets) * 0.9), 5))

    for ax, values, ylabel, title in [
        (axes[0], aucs, "AUC",      "AUC by Dataset"),
        (axes[1], accs, "Accuracy", "Accuracy by Dataset"),
    ]:
        bars = ax.bar(datasets, values, color=bar_colors, edgecolor="white", linewidth=0.5)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.grid(axis="y", alpha=0.3)

        for bar, val in zip(bars, values):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=7)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#4CAF50", label="OK"),
        Patch(facecolor="#F44336", label="Overfitting"),
        Patch(facecolor="#FF9800", label="Underfitting"),
        Patch(facecolor="#9E9E9E", label="No data"),
    ]
    fig.legend(handles=legend_elements, loc="upper right", fontsize=9, title="Diagnosis")

    fig.suptitle("Model Performance Summary  (colour = overfit diagnosis)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {output_path}")


def print_text_summary(all_results):
    print("\n" + "=" * 72)
    print("  Overfitting / Underfitting Diagnostic Summary")
    print("=" * 72)
    print(f"  {'Dataset':<25} {'AUC':>7} {'Acc':>7} {'FinalTrain':>11} {'FinalVal':>9} {'Diagnosis'}")
    print("  " + "-" * 70)

    for r in all_results:
        train_losses = r.get("train_losses", [])
        val_losses   = r.get("val_losses",   [])
        ft = train_losses[-1] if train_losses else float("nan")
        fv = val_losses[-1]   if val_losses   else float("nan")
        diag = diagnose(train_losses, val_losses)
        print(
            f"  {r['dataset']:<25} "
            f"{r.get('auc', float('nan')):>7.4f} "
            f"{r.get('accuracy', float('nan')):>7.4f} "
            f"{ft:>11.4f} "
            f"{fv:>9.4f} "
            f"  {diag}"
        )

    print("=" * 72)
    print(f"\n  Thresholds used:")
    print(f"    OVERFITTING  : final val_loss - train_loss > {OVERFIT_GAP}")
    print(f"    UNDERFITTING : final train_loss > {UNDERFIT_FLOOR_TRAIN} AND final val_loss > {UNDERFIT_FLOOR_VAL}")
    print(f"  Note: label smoothing raises the CE loss floor, so a model with")
    print(f"        train loss 0.4-0.7 may still be a near-perfect classifier.")


def main():
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "all"

    results_2d, results_3d = [], []

    if mode in ("all", "2d"):
        results_2d = load_all_results(MODELS_2D_DIR, "2D")
        if results_2d:
            make_grid_plot(
                results_2d,
                "2D CNN — Train vs Val Loss Curves",
                os.path.join(OUTPUT_DIR, "loss_curves_2d.png"),
                ncols=4, fontsize_scale=1.0,
            )
            make_split_grids(
                results_2d,
                "2D CNN — Train vs Val Loss Curves",
                "loss_curves_2d_split",
                ncols=2, per_grid=4,
            )
            make_individual_plots(results_2d, "2d", "individual_2d")

    if mode in ("all", "3d"):
        results_3d = load_all_results(MODELS_3D_DIR, "3D")
        if results_3d:
            make_grid_plot(
                results_3d,
                "3D CNN — Train vs Val Loss Curves",
                os.path.join(OUTPUT_DIR, "loss_curves_3d.png"),
                ncols=3, fontsize_scale=1.0,
            )
            make_split_grids(
                results_3d,
                "3D CNN — Train vs Val Loss Curves",
                "loss_curves_3d_split",
                ncols=2, per_grid=4,
            )
            make_individual_plots(results_3d, "3d", "individual_3d")

    all_results = results_2d + results_3d

    if all_results:
        make_summary_plot(
            all_results,
            os.path.join(OUTPUT_DIR, "performance_summary.png"),
        )
        print_text_summary(all_results)
    else:
        print("\n  No results with loss curves found.")
        print("  Run train2d.py or train3d.py first, then re-run this script.")

    print(f"\n  All plots saved to: {os.path.abspath(OUTPUT_DIR)}/")


if __name__ == "__main__":
    main()