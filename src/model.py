import torch
from torch import nn

from .config import DEFAULT_MODEL_NAME, NUM_CLASSES


MODEL_NAMES = ("small", "improved")


class SmallEuroSATCNN(nn.Module):
    """Small CNN for 64x64 RGB image classification."""

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


class ConvBlock(nn.Module):
    """Two convolution layers followed by pooling for the improved CNN."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ImprovedEuroSATCNN(nn.Module):
    """Stronger from-scratch CNN with BatchNorm and global average pooling."""

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 32, dropout=0.05),
            ConvBlock(32, 64, dropout=0.05),
            ConvBlock(64, 128, dropout=0.10),
            ConvBlock(128, 256, dropout=0.10),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.30),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def build_model(
    num_classes: int = NUM_CLASSES,
    model_name: str = DEFAULT_MODEL_NAME,
) -> nn.Module:
    if model_name == "small":
        return SmallEuroSATCNN(num_classes=num_classes)
    if model_name == "improved":
        return ImprovedEuroSATCNN(num_classes=num_classes)
    raise ValueError(f"Unknown model_name {model_name!r}. Expected one of {MODEL_NAMES}.")
