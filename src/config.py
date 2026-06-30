from pathlib import Path


IMAGE_SIZE = 64
NUM_CLASSES = 10
DEFAULT_MODEL_NAME = "small"
DEFAULT_CHECKPOINT = Path("models/eurosat_cnn.pt")
DEFAULT_IMPROVED_CHECKPOINT = Path("models/eurosat_improved_cnn.pt")
DEFAULT_HISTORY_CSV = Path("reports/training_history.csv")
DEFAULT_IMPROVED_HISTORY_CSV = Path("reports/improved_training_history.csv")

# EuroSAT RGB class folders are sorted alphabetically by ImageFolder.
CLASS_NAMES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]

# Simple normalization for a first from-scratch model.
RGB_MEAN = (0.5, 0.5, 0.5)
RGB_STD = (0.5, 0.5, 0.5)


def default_checkpoint_for_model(model_name: str) -> Path:
    if model_name == "improved":
        return DEFAULT_IMPROVED_CHECKPOINT
    return DEFAULT_CHECKPOINT


def default_history_for_model(model_name: str) -> Path:
    if model_name == "improved":
        return DEFAULT_IMPROVED_HISTORY_CSV
    return DEFAULT_HISTORY_CSV
