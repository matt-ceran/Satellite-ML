import argparse
from pathlib import Path

import torch
from torch import nn

from .config import DEFAULT_CHECKPOINT, DEFAULT_HISTORY_CSV
from .dataset import make_dataloaders
from .model import build_model
from .utils import accuracy_from_logits, get_device, save_history_csv, set_seed


def run_train_epoch(model, loader, criterion, optimizer, device) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        correct, seen = accuracy_from_logits(logits, labels)
        total_loss += loss.item() * seen
        total_correct += correct
        total_seen += seen

    return total_loss / total_seen, total_correct / total_seen


@torch.no_grad()
def run_eval_epoch(model, loader, criterion, device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)

        correct, seen = accuracy_from_logits(logits, labels)
        total_loss += loss.item() * seen
        total_correct += correct
        total_seen += seen

    return total_loss / total_seen, total_correct / total_seen


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a small CNN on EuroSAT RGB.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/EuroSAT_RGB"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--history-csv", type=Path, default=DEFAULT_HISTORY_CSV)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    train_loader, val_loader, class_names = make_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        val_split=args.val_split,
        seed=args.seed,
        num_workers=args.num_workers,
        smoke_test=args.smoke_test,
    )

    model = build_model(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = -1.0
    history: list[dict[str, float]] = []
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Classes: {class_names}")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = run_eval_epoch(model, val_loader, criterion, device)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
        }
        history.append(row)

        print(
            f"Epoch {epoch:03d} | "
            f"train loss {train_loss:.4f}, acc {train_acc:.3f} | "
            f"val loss {val_loss:.4f}, acc {val_acc:.3f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                    "epoch": epoch,
                    "val_accuracy": val_acc,
                },
                args.checkpoint,
            )

    save_history_csv(history, args.history_csv)
    print(f"Best checkpoint: {args.checkpoint} (val accuracy {best_val_acc:.3f})")
    print(f"Training history: {args.history_csv}")


if __name__ == "__main__":
    main()
