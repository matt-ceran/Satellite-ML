from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from .config import CLASS_NAMES, IMAGE_SIZE, NUM_CLASSES, RGB_MEAN, RGB_STD


def build_transforms(train: bool) -> transforms.Compose:
    steps = [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))]
    if train:
        steps.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(20),
            ]
        )
    steps.extend([transforms.ToTensor(), transforms.Normalize(RGB_MEAN, RGB_STD)])
    return transforms.Compose(steps)


def make_dataloaders(
    data_dir: Path,
    batch_size: int,
    val_split: float,
    seed: int,
    num_workers: int,
    smoke_test: bool = False,
) -> tuple[DataLoader, DataLoader, list[str]]:
    if smoke_test:
        train_data = datasets.FakeData(
            size=128,
            image_size=(3, IMAGE_SIZE, IMAGE_SIZE),
            num_classes=NUM_CLASSES,
            transform=build_transforms(train=True),
        )
        val_data = datasets.FakeData(
            size=32,
            image_size=(3, IMAGE_SIZE, IMAGE_SIZE),
            num_classes=NUM_CLASSES,
            transform=build_transforms(train=False),
        )
        return (
            DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers),
            DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers),
            CLASS_NAMES,
        )

    if not data_dir.exists():
        raise FileNotFoundError(
            f"{data_dir} does not exist. Download EuroSAT RGB or run with --smoke-test."
        )

    base_data = datasets.ImageFolder(data_dir)
    class_names = base_data.classes
    if len(class_names) != NUM_CLASSES:
        raise ValueError(
            f"Expected {NUM_CLASSES} class folders in {data_dir}, found {len(class_names)}: "
            f"{class_names}"
        )
    if len(base_data) < 2:
        raise ValueError(f"Need at least 2 images in {data_dir} for a train/validation split.")

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(base_data), generator=generator).tolist()
    val_count = max(1, int(len(indices) * val_split))
    val_count = min(val_count, len(indices) - 1)
    val_indices = indices[:val_count]
    train_indices = indices[val_count:]

    train_data = datasets.ImageFolder(data_dir, transform=build_transforms(train=True))
    val_data = datasets.ImageFolder(data_dir, transform=build_transforms(train=False))

    return (
        DataLoader(
            Subset(train_data, train_indices),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        ),
        DataLoader(
            Subset(val_data, val_indices),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
        class_names,
    )
