import argparse
import csv
from pathlib import Path


def load_history(path: Path) -> list[dict[str, float]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        return [
            {
                "epoch": int(row["epoch"]),
                "train_loss": float(row["train_loss"]),
                "train_accuracy": float(row["train_accuracy"]),
                "val_loss": float(row["val_loss"]),
                "val_accuracy": float(row["val_accuracy"]),
            }
            for row in reader
        ]


def load_per_class(path: Path) -> dict[str, dict[str, float]]:
    with path.open(newline="") as f:
        return {
            row["class_name"]: {
                "accuracy": float(row["accuracy"]),
                "correct": int(row["correct"]),
                "total": int(row["total"]),
            }
            for row in csv.DictReader(f)
        }


def parse_summary_accuracy(path: Path) -> tuple[int, int, float]:
    total = 0
    correct = 0
    accuracy = 0.0
    for line in path.read_text().splitlines():
        if line.startswith("Total images:"):
            total = int(line.split(":", 1)[1].strip())
        elif line.startswith("Correct:"):
            correct = int(line.split(":", 1)[1].strip())
        elif line.startswith("Accuracy:"):
            accuracy = float(line.split(":", 1)[1].strip())
    return correct, total, accuracy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare baseline and improved EuroSAT CNN reports.")
    parser.add_argument("--baseline-history", type=Path, default=Path("reports/training_history.csv"))
    parser.add_argument(
        "--improved-history",
        type=Path,
        default=Path("reports/improved_training_history.csv"),
    )
    parser.add_argument(
        "--baseline-per-class",
        type=Path,
        default=Path("reports/per_class_accuracy.csv"),
    )
    parser.add_argument(
        "--improved-per-class",
        type=Path,
        default=Path("reports/improved_per_class_accuracy.csv"),
    )
    parser.add_argument(
        "--baseline-prediction-summary",
        type=Path,
        default=Path("reports/prediction_samples_summary.txt"),
    )
    parser.add_argument(
        "--improved-prediction-summary",
        type=Path,
        default=Path("reports/improved_prediction_samples_summary.txt"),
    )
    parser.add_argument("--output-csv", type=Path, default=Path("reports/phase_2b_comparison.csv"))
    parser.add_argument("--output-md", type=Path, default=Path("reports/phase_2b_comparison.md"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline_history = load_history(args.baseline_history)
    improved_history = load_history(args.improved_history)
    baseline_best = max(baseline_history, key=lambda row: row["val_accuracy"])
    improved_best = max(improved_history, key=lambda row: row["val_accuracy"])
    baseline_final = baseline_history[-1]
    improved_final = improved_history[-1]

    baseline_pred = parse_summary_accuracy(args.baseline_prediction_summary)
    improved_pred = parse_summary_accuracy(args.improved_prediction_summary)

    baseline_classes = load_per_class(args.baseline_per_class)
    improved_classes = load_per_class(args.improved_per_class)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["metric", "baseline", "improved", "delta"],
        )
        writer.writeheader()
        rows = [
            ("best_val_accuracy", baseline_best["val_accuracy"], improved_best["val_accuracy"]),
            ("best_val_loss", baseline_best["val_loss"], improved_best["val_loss"]),
            ("final_train_accuracy", baseline_final["train_accuracy"], improved_final["train_accuracy"]),
            ("final_val_accuracy", baseline_final["val_accuracy"], improved_final["val_accuracy"]),
            ("sample_prediction_accuracy", baseline_pred[2], improved_pred[2]),
        ]
        for metric, baseline, improved in rows:
            writer.writerow(
                {
                    "metric": metric,
                    "baseline": f"{baseline:.6f}",
                    "improved": f"{improved:.6f}",
                    "delta": f"{improved - baseline:.6f}",
                }
            )
        for class_name in sorted(baseline_classes):
            baseline = baseline_classes[class_name]["accuracy"]
            improved = improved_classes[class_name]["accuracy"]
            writer.writerow(
                {
                    "metric": f"class_accuracy_{class_name}",
                    "baseline": f"{baseline:.6f}",
                    "improved": f"{improved:.6f}",
                    "delta": f"{improved - baseline:.6f}",
                }
            )

    lines = [
        "# Phase 2B Comparison",
        "",
        "| Metric | Baseline | Improved | Delta |",
        "| --- | ---: | ---: | ---: |",
        (
            f"| Best validation accuracy | {baseline_best['val_accuracy']:.4f} "
            f"| {improved_best['val_accuracy']:.4f} "
            f"| {improved_best['val_accuracy'] - baseline_best['val_accuracy']:+.4f} |"
        ),
        (
            f"| Best validation loss | {baseline_best['val_loss']:.4f} "
            f"| {improved_best['val_loss']:.4f} "
            f"| {improved_best['val_loss'] - baseline_best['val_loss']:+.4f} |"
        ),
        (
            f"| Sample prediction accuracy | {baseline_pred[2]:.4f} "
            f"| {improved_pred[2]:.4f} "
            f"| {improved_pred[2] - baseline_pred[2]:+.4f} |"
        ),
        "",
        "## Per-Class Validation Accuracy",
        "",
        "| Class | Baseline | Improved | Delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    for class_name in sorted(baseline_classes):
        baseline = baseline_classes[class_name]["accuracy"]
        improved = improved_classes[class_name]["accuracy"]
        lines.append(f"| {class_name} | {baseline:.4f} | {improved:.4f} | {improved - baseline:+.4f} |")
    args.output_md.write_text("\n".join(lines) + "\n")

    print(f"Comparison CSV: {args.output_csv}")
    print(f"Comparison Markdown: {args.output_md}")
    print(
        "Best validation accuracy delta: "
        f"{improved_best['val_accuracy'] - baseline_best['val_accuracy']:+.4f}"
    )


if __name__ == "__main__":
    main()
