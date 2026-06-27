import argparse
from pathlib import Path

import torch
from PIL import Image

from .config import CLASS_NAMES, DEFAULT_CHECKPOINT
from .dataset import build_transforms
from .model import build_model
from .utils import get_device, load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict one EuroSAT RGB image.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()
    checkpoint = load_checkpoint(args.checkpoint, device)
    class_names = checkpoint.get("class_names", CLASS_NAMES)

    model = build_model(num_classes=len(class_names)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    image = Image.open(args.image).convert("RGB")
    tensor = build_transforms(train=False)(image).unsqueeze(0).to(device)

    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1).squeeze(0).cpu()

    confidence, class_index = probabilities.max(dim=0)
    print(f"Prediction: {class_names[class_index.item()]}")
    print(f"Confidence: {confidence.item():.4f}")


if __name__ == "__main__":
    main()
