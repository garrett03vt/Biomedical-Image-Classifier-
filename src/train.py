# train.py
# Master training script — trains ALL MedMNIST datasets (2D and 3D).
# Optimized for: RTX 4090 (24 GB VRAM) | Ryzen 9 5900X (12C/24T) | 128 GB RAM
#
# 2D results → models_2d/
# 3D results → models_3d/

import sys
import time
import warnings

from tqdm import tqdm


try:
    import winsound
except ImportError:
    winsound = None

from train2d import train_single_2d, already_trained as already_trained_2d, DATASETS_2D
from train3d import train_single_3d, already_trained as already_trained_3d, DATASETS_3D
from utils import load_dataset

warnings.filterwarnings("ignore", category=FutureWarning)


def main():
    total_start = time.time()
    flags_2d    = list(DATASETS_2D.keys())
    flags_3d    = list(DATASETS_3D.keys())

    print("=" * 72)
    print("  MedMNIST Master Training Pipeline")
    print("  RTX 4090 | CNN only | 2D → models_2d/  3D → models_3d/")
    print("=" * 72)
    print(f"  2D datasets : {len(flags_2d)}")
    print(f"  3D datasets : {len(flags_3d)}")
    print(f"  Total       : {len(flags_2d) + len(flags_3d)}")
    sys.stdout.flush()

    # Pre-download all datasets
    all_flags = flags_2d + flags_3d
    print(f"\n[1/3] Pre-downloading all {len(all_flags)} datasets (skips if cached)...")
    sys.stdout.flush()
    for flag in tqdm(all_flags, desc="Downloading", unit="dataset"):
        load_dataset(flag)

    # Train 2D
    print(f"\n[2/3] Training {len(flags_2d)} 2D datasets  →  models_2d/")
    print("  " + "-" * 68)
    for flag in flags_2d:
        meta   = DATASETS_2D[flag]
        status = "✓ cached" if already_trained_2d(flag) else "needs training"
        print(f"  {flag:<20} {meta['task']:<14} batch={meta['batch']}  epochs={meta['epochs']}  [{status}]")
    sys.stdout.flush()

    results_2d = []
    for flag in tqdm(flags_2d, desc="2D datasets", unit="dataset"):
        try:
            result = train_single_2d(flag, lr=3e-4)
            results_2d.append((flag, result["auc"], result["accuracy"], "cnn2d", result["duration"], None))
        except Exception as e:
            tqdm.write(f"\n  [{flag}] ERROR: {e}")
            results_2d.append((flag, None, None, "cnn2d", "—", str(e)))

    # Train 3D
    print(f"\n[3/3] Training {len(flags_3d)} 3D datasets  →  models_3d/")
    print("  " + "-" * 68)
    for flag in flags_3d:
        meta   = DATASETS_3D[flag]
        status = "✓ cached" if already_trained_3d(flag) else "needs training"
        print(f"  {flag:<25} {meta['task']:<15} {meta['n_classes']} cls  [{status}]")
    sys.stdout.flush()

    results_3d = []
    for flag in tqdm(flags_3d, desc="3D datasets", unit="dataset"):
        try:
            result = train_single_3d(flag, lr=3e-4)
            results_3d.append((flag, result["auc"], result["accuracy"], "cnn3d", result["duration"], None))
        except Exception as e:
            tqdm.write(f"\n  [{flag}] ERROR: {e}")
            results_3d.append((flag, None, None, "cnn3d", "—", str(e)))

    # Summary
    total_elapsed  = time.time() - total_start
    total_duration = f"{int(total_elapsed // 60)}m {int(total_elapsed % 60)}s"

    def print_table(title, rows):
        sys.stdout.flush()
        print(f"\n{'=' * 72}")
        print(f"  {title}")
        print(f"{'=' * 72}")
        print(f"  {'Dataset':<25} {'AUC':>8} {'Accuracy':>10} {'Method':>7} {'Time':>10}")
        print("  " + "-" * 65)
        for flag, auc, acc, method, duration, err in sorted(rows):
            if err:
                print(f"  {flag:<25} ERROR: {err}")
            else:
                print(f"  {flag:<25} {auc:>8.4f} {acc:>10.4f} {method:>7} {duration:>10}")
        print(f"{'=' * 72}")
        sys.stdout.flush()

    print_table("2D Results  (models_2d/)", results_2d)
    print_table("3D Results  (models_3d/)", results_3d)

    good = [r for r in results_2d + results_3d if r[5] is None]
    bad  = [r for r in results_2d + results_3d if r[5] is not None]

    print(f"\n  Datasets trained successfully : {len(good)}")
    print(f"  Datasets with errors         : {len(bad)}")
    print(f"  Total wall-clock time        : {total_duration}")
    sys.stdout.flush()

    try:
        winsound.Beep(500, 400)
        winsound.Beep(700, 400)
        winsound.Beep(900, 600)
    except Exception:
        pass


if __name__ == "__main__":
    main()