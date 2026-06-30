import argparse
from pathlib import Path

import torch
from PIL import Image

from .config import CLASS_NAMES, DEFAULT_MODEL_NAME, default_checkpoint_for_model
from .dataset import build_transforms
from .model import MODEL_NAMES, build_model
from .utils import get_device, load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict one EuroSAT RGB image.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--model", choices=MODEL_NAMES)
    parser.add_argument("--checkpoint", type=Path)
    return parser.parse_args()


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

    image = Image.open(args.image).convert("RGB")
    tensor = build_transforms(train=False)(image).unsqueeze(0).to(device)

    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1).squeeze(0).cpu()

    confidence, class_index = probabilities.max(dim=0)
    print(f"Model: {model_name}")
    print(f"Prediction: {class_names[class_index.item()]}")
    print(f"Confidence: {confidence.item():.4f}")


if __name__ == "__main__":
    main()
