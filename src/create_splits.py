import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

from torchvision import datasets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a deterministic stratified split manifest.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/EuroSAT_RGB"))
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--test-split", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--manifest", type=Path, default=Path("reports/split_manifest.csv"))
    parser.add_argument("--summary", type=Path, default=Path("reports/split_summary.csv"))
    return parser.parse_args()


def validate_splits(val_split: float, test_split: float) -> None:
    if val_split <= 0 or test_split <= 0:
        raise ValueError("Validation and test splits must both be greater than 0.")
    if val_split + test_split >= 1:
        raise ValueError("Validation split plus test split must be less than 1.")


def assign_splits(data_dir: Path, val_split: float, test_split: float, seed: int) -> list[dict[str, str]]:
    dataset = datasets.ImageFolder(data_dir)
    by_class: dict[int, list[str]] = defaultdict(list)
    for path, class_index in dataset.samples:
        by_class[class_index].append(path)

    rows: list[dict[str, str]] = []
    for class_index, paths in sorted(by_class.items()):
        rng = random.Random(seed + class_index)
        shuffled = sorted(paths)
        rng.shuffle(shuffled)

        total = len(shuffled)
        test_count = max(1, int(total * test_split))
        val_count = max(1, int(total * val_split))
        train_count = total - val_count - test_count
        if train_count < 1:
            raise ValueError(f"Class {dataset.classes[class_index]} does not have enough images.")

        split_paths = [
            ("test", shuffled[:test_count]),
            ("validation", shuffled[test_count : test_count + val_count]),
            ("training", shuffled[test_count + val_count :]),
        ]
        for split, selected_paths in split_paths:
            for path in selected_paths:
                rows.append(
                    {
                        "path": str(Path(path).relative_to(data_dir)),
                        "class_name": dataset.classes[class_index],
                        "split": split,
                    }
                )
    return sorted(rows, key=lambda row: (row["split"], row["class_name"], row["path"]))


def write_manifest(rows: list[dict[str, str]], manifest: Path) -> None:
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "class_name", "split"])
        writer.writeheader()
        writer.writerows(rows)


def write_summary(rows: list[dict[str, str]], summary: Path) -> None:
    counts = Counter((row["split"], row["class_name"]) for row in rows)
    split_totals = Counter(row["split"] for row in rows)
    class_names = sorted({row["class_name"] for row in rows})
    summary.parent.mkdir(parents=True, exist_ok=True)
    with summary.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["split", "class_name", "count", "split_total"],
        )
        writer.writeheader()
        for split in ["training", "validation", "test"]:
            for class_name in class_names:
                writer.writerow(
                    {
                        "split": split,
                        "class_name": class_name,
                        "count": counts[(split, class_name)],
                        "split_total": split_totals[split],
                    }
                )


def main() -> None:
    args = parse_args()
    validate_splits(args.val_split, args.test_split)
    rows = assign_splits(args.data_dir, args.val_split, args.test_split, args.seed)
    write_manifest(rows, args.manifest)
    write_summary(rows, args.summary)
    totals = Counter(row["split"] for row in rows)
    print(f"Split manifest: {args.manifest}")
    print(f"Split summary: {args.summary}")
    print(f"Training images: {totals['training']}")
    print(f"Validation images: {totals['validation']}")
    print(f"Test images: {totals['test']}")


if __name__ == "__main__":
    main()
