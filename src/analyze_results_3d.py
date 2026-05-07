# src/analyze_results_3d.py
# Analyze saved 3D CNN result files and generate summary tables + graphs.

import os
import csv
import joblib
import matplotlib.pyplot as plt


RESULTS_DIR = "models_3d"
OUTPUT_CSV = os.path.join(RESULTS_DIR, "summary_results_3d.csv")


def duration_to_minutes(duration):
    if duration is None:
        return 0.0

    try:
        duration = duration.strip()
        minutes = 0
        seconds = 0

        if "m" in duration:
            minutes = int(duration.split("m")[0].strip())

        if "s" in duration:
            if "m" in duration:
                seconds_part = duration.split("m")[1].replace("s", "").strip()
            else:
                seconds_part = duration.replace("s", "").strip()

            if seconds_part:
                seconds = int(seconds_part)

        return minutes + seconds / 60.0

    except Exception:
        return 0.0


def load_results():
    results = []

    if not os.path.exists(RESULTS_DIR):
        print(f"Could not find folder: {RESULTS_DIR}")
        return results

    for filename in os.listdir(RESULTS_DIR):
        if filename.endswith(".joblib"):
            path = os.path.join(RESULTS_DIR, filename)

            try:
                result = joblib.load(path)

                dataset = result.get("dataset", filename.replace("_cnn3d_results.joblib", ""))
                auc = result.get("auc", None)
                accuracy = result.get("accuracy", result.get("acc", None))
                method = result.get("method", "cnn3d")
                duration = result.get("duration", "N/A")

                if auc is None or accuracy is None:
                    print(f"Skipping {filename}: missing AUC or accuracy.")
                    continue

                results.append({
                    "dataset": dataset,
                    "auc": auc,
                    "accuracy": accuracy,
                    "method": method,
                    "duration": duration,
                })

            except Exception as e:
                print(f"Could not read {filename}: {e}")

    results.sort(key=lambda r: r["dataset"])
    return results


def print_summary(results):
    print("\n3D CNN Performance Summary")
    print("=" * 75)
    print(f"{'Dataset':<25} {'AUC':>8} {'Accuracy':>10} {'Method':>12} {'Time':>12}")
    print("-" * 75)

    for r in results:
        print(
            f"{r['dataset']:<25} "
            f"{r['auc']:>8.4f} "
            f"{r['accuracy']:>10.4f} "
            f"{r['method']:>12} "
            f"{r['duration']:>12}"
        )

    print("=" * 75)


def save_csv(results):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Dataset", "AUC", "Accuracy", "Method", "Duration"])

        for r in results:
            writer.writerow([
                r["dataset"],
                f"{r['auc']:.4f}",
                f"{r['accuracy']:.4f}",
                r["method"],
                r["duration"],
            ])

    print(f"\nSaved CSV summary to: {OUTPUT_CSV}")


def plot_metric(results, metric, ylabel, filename):
    datasets = [r["dataset"] for r in results]
    values = [r[metric] for r in results]

    plt.figure(figsize=(12, 6))
    plt.bar(datasets, values)
    plt.xlabel("Dataset")
    plt.ylabel(ylabel)
    plt.title(f"3D CNN {ylabel} by Dataset")
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 1)
    plt.tight_layout()

    output_path = os.path.join(RESULTS_DIR, filename)
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved graph to: {output_path}")


def plot_training_time(results):
    datasets = [r["dataset"] for r in results]
    values = [duration_to_minutes(r["duration"]) for r in results]

    plt.figure(figsize=(12, 6))
    plt.bar(datasets, values)
    plt.xlabel("Dataset")
    plt.ylabel("Training Time (minutes)")
    plt.title("3D CNN Training Time by Dataset")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    output_path = os.path.join(RESULTS_DIR, "training_time_by_dataset.png")
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved graph to: {output_path}")


def main():
    results = load_results()

    if not results:
        print("No valid 3D result files found.")
        return

    print_summary(results)
    save_csv(results)

    plot_metric(results, "auc", "AUC", "auc_by_dataset.png")
    plot_metric(results, "accuracy", "Accuracy", "accuracy_by_dataset.png")
    plot_training_time(results)

    print("\n3D analysis complete.")


if __name__ == "__main__":
    main()