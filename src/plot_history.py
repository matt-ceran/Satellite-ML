import argparse
from pathlib import Path

from .config import DEFAULT_HISTORY_CSV
from .reporting import load_training_history, save_learning_curves


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Save learning-curve plots from training history.")
    parser.add_argument("--history-csv", type=Path, default=DEFAULT_HISTORY_CSV)
    parser.add_argument("--output", type=Path, default=Path("reports/learning_curves.png"))
    parser.add_argument("--title", default="EuroSAT CNN Learning Curves")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    history = load_training_history(args.history_csv)
    save_learning_curves(
        history,
        args.output,
        title=args.title,
        subtitle=f"Training and validation metrics from {args.history_csv}",
    )
    print(f"Learning curves: {args.output}")


if __name__ == "__main__":
    main()
