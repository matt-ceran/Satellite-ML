import argparse
from pathlib import Path

import torch
from torch import nn

from .config import CLASS_NAMES, DEFAULT_MODEL_NAME, NUM_CLASSES, default_checkpoint_for_model
from .dataset import make_dataloaders
from .model import MODEL_NAMES, build_model
from .reporting import save_confusion_matrix_image, save_per_class_accuracy_csv
from .utils import get_device, load_checkpoint, set_seed


@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes: int):
    model.eval()
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    total_loss = 0.0
    total_seen = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        predictions = logits.argmax(dim=1)

        total_loss += loss.item() * labels.numel()
        total_seen += labels.numel()
        for target, prediction in zip(labels.cpu(), predictions.cpu()):
            confusion[target, prediction] += 1

    correct = confusion.diag()
    per_class_total = confusion.sum(dim=1).clamp(min=1)
    per_class_accuracy = correct.float() / per_class_total.float()
    accuracy = correct.sum().item() / confusion.sum().item()
    return total_loss / total_seen, accuracy, per_class_accuracy, confusion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved EuroSAT CNN checkpoint.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/EuroSAT_RGB"))
    parser.add_argument("--model", choices=MODEL_NAMES)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--confusion-matrix-image",
        type=Path,
        default=Path("reports/confusion_matrix.png"),
    )
    parser.add_argument(
        "--per-class-csv",
        type=Path,
        default=Path("reports/per_class_accuracy.csv"),
    )
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    _, val_loader, data_class_names = make_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        val_split=args.val_split,
        seed=args.seed,
        num_workers=args.num_workers,
        smoke_test=args.smoke_test,
    )

    checkpoint_path = args.checkpoint or default_checkpoint_for_model(args.model or DEFAULT_MODEL_NAME)
    checkpoint = load_checkpoint(checkpoint_path, device)
    model_name = args.model or checkpoint.get("model_name", DEFAULT_MODEL_NAME)
    class_names = checkpoint.get("class_names", data_class_names or CLASS_NAMES)
    model = build_model(num_classes=len(class_names), model_name=model_name).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    criterion = nn.CrossEntropyLoss()
    val_loss, val_accuracy, per_class_accuracy, confusion = evaluate(
        model, val_loader, criterion, device, NUM_CLASSES
    )

    print(f"Validation loss: {val_loss:.4f}")
    print(f"Validation accuracy: {val_accuracy:.4f}")
    print(f"Model: {model_name}")
    print(f"Checkpoint: {checkpoint_path}")
    print("Per-class accuracy:")
    for index, name in enumerate(class_names):
        correct = confusion[index, index].item()
        total = confusion[index].sum().item()
        print(f"  {index:02d} {name}: {per_class_accuracy[index].item():.4f} ({correct}/{total})")

    print("Confusion matrix rows=true cols=pred:")
    print(confusion.tolist())

    confusion_rows = confusion.tolist()
    per_class_rows = per_class_accuracy.tolist()
    save_confusion_matrix_image(confusion_rows, class_names, args.confusion_matrix_image)
    save_per_class_accuracy_csv(class_names, per_class_rows, confusion_rows, args.per_class_csv)
    print(f"Confusion matrix image: {args.confusion_matrix_image}")
    print(f"Per-class CSV: {args.per_class_csv}")


if __name__ == "__main__":
    main()
