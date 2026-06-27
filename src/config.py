from pathlib import Path


IMAGE_SIZE = 64
NUM_CLASSES = 10
DEFAULT_CHECKPOINT = Path("models/eurosat_cnn.pt")
DEFAULT_HISTORY_CSV = Path("reports/training_history.csv")

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
