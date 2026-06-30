import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

import torch
from PIL import Image

from .config import CLASS_NAMES, DEFAULT_MODEL_NAME, default_checkpoint_for_model
from .dataset import build_transforms
from .model import MODEL_NAMES, build_model
from .utils import get_device, load_checkpoint


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run prediction on a deterministic sample from each class.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/EuroSAT_RGB"))
    parser.add_argument("--model", choices=MODEL_NAMES)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--samples-per-class", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-csv", type=Path, default=Path("reports/prediction_samples.csv"))
    parser.add_argument("--summary", type=Path, default=Path("reports/prediction_samples_summary.txt"))
    return parser.parse_args()


def sample_images(data_dir: Path, samples_per_class: int, seed: int) -> list[tuple[str, Path]]:
    if samples_per_class < 1:
        raise ValueError("--samples-per-class must be at least 1.")

    rows: list[tuple[str, Path]] = []
    rng = random.Random(seed)
    for class_dir in sorted(path for path in data_dir.iterdir() if path.is_dir()):
        images = sorted(path for path in class_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
        if not images:
            continue
        selected = images if samples_per_class >= len(images) else rng.sample(images, samples_per_class)
        for image_path in sorted(selected):
            rows.append((class_dir.name, image_path))
    return rows


def predict_image(model, transform, image_path: Path, class_names: list[str], device) -> tuple[str, float]:
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1).squeeze(0).cpu()
    confidence, class_index = probabilities.max(dim=0)
    return class_names[class_index.item()], confidence.item()


def write_summary(rows: list[dict[str, str]], summary_path: Path) -> None:
    total = len(rows)
    correct = sum(row["correct"] == "1" for row in rows)
    by_class_total = Counter(row["true_class"] for row in rows)
    by_class_correct: dict[str, int] = defaultdict(int)
    for row in rows:
        if row["correct"] == "1":
            by_class_correct[row["true_class"]] += 1

    lines = [
        f"Total images: {total}",
        f"Correct: {correct}",
        f"Accuracy: {correct / total:.4f}" if total else "Accuracy: n/a",
        "",
        "Per-class sample accuracy:",
    ]
    for class_name in sorted(by_class_total):
        class_total = by_class_total[class_name]
        class_correct = by_class_correct[class_name]
        lines.append(f"{class_name}: {class_correct}/{class_total} ({class_correct / class_total:.4f})")

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    device = get_device()
    checkpoint_path = args.checkpoint or default_checkpoint_for_model(args.model or DEFAULT_MODEL_NAME)
    checkpoint = load_checkpoint(checkpoint_path, device)
    model_name = args.model or checkpoint.get("model_name", DEFAULT_MODEL_NAME)
    class_names = checkpoint.get("class_names", CLASS_NAMES)

    model = build_model(num_classes=len(class_names), model_name=model_name).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    transform = build_transforms(train=False)

    samples = sample_images(args.data_dir, args.samples_per_class, args.seed)
    rows: list[dict[str, str]] = []
    for true_class, image_path in samples:
        predicted_class, confidence = predict_image(model, transform, image_path, class_names, device)
        rows.append(
            {
                "image": str(image_path),
                "true_class": true_class,
                "predicted_class": predicted_class,
                "confidence": f"{confidence:.6f}",
                "correct": "1" if true_class == predicted_class else "0",
            }
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image", "true_class", "predicted_class", "confidence", "correct"],
        )
        writer.writeheader()
        writer.writerows(rows)

    write_summary(rows, args.summary)
    correct = sum(row["correct"] == "1" for row in rows)
    print(f"Prediction CSV: {args.output_csv}")
    print(f"Prediction summary: {args.summary}")
    print(f"Model: {model_name}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Accuracy: {correct}/{len(rows)} ({correct / len(rows):.4f})")


if __name__ == "__main__":
    main()
