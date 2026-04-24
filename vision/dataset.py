"""
Dataset and augmentation pipeline for traffic scene multi-label classification.
"""

import os
import json
import random
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageFilter
import numpy as np

LABELS = ["rain", "night", "congestion", "clear"]


# ---------------------------------------------------------------------------
# Augmentation transforms
# ---------------------------------------------------------------------------

def get_train_transforms(img_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.Lambda(lambda img: img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0, 1.5)))),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_val_transforms(img_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class TrafficSceneDataset(Dataset):
    """
    Expects a JSON annotation file with entries:
        [{"image": "path/to/img.jpg", "labels": ["rain", "congestion"]}, ...]
    """

    def __init__(
        self,
        annotations: List[Dict],
        transform: Optional[transforms.Compose] = None,
        label_list: List[str] = LABELS,
    ):
        self.annotations = annotations
        self.transform = transform
        self.label_list = label_list

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.annotations[idx]
        image = Image.open(item["image"]).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label_vec = torch.zeros(len(self.label_list), dtype=torch.float32)
        for lbl in item.get("labels", []):
            if lbl in self.label_list:
                label_vec[self.label_list.index(lbl)] = 1.0

        return image, label_vec


# ---------------------------------------------------------------------------
# Synthetic data generator (for demo / testing without real images)
# ---------------------------------------------------------------------------

def generate_dummy_annotations(
    data_dir: str,
    n_train: int = 200,
    n_val: int = 50,
    img_size: int = 224,
) -> Tuple[List[Dict], List[Dict]]:
    """Creates random RGB images and annotations for quick testing."""
    train_dir = Path(data_dir) / "train"
    val_dir = Path(data_dir) / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    def _make_split(directory: Path, n: int) -> List[Dict]:
        records = []
        for i in range(n):
            img = Image.fromarray(np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8))
            path = str(directory / f"img_{i:04d}.jpg")
            img.save(path)
            # Randomly assign 1-2 labels
            chosen = random.sample(LABELS, k=random.randint(1, 2))
            records.append({"image": path, "labels": chosen})
        return records

    train_ann = _make_split(train_dir, n_train)
    val_ann = _make_split(val_dir, n_val)

    # Save annotations
    for split, ann in [("train", train_ann), ("val", val_ann)]:
        with open(Path(data_dir) / f"{split}_annotations.json", "w") as f:
            json.dump(ann, f, indent=2)

    return train_ann, val_ann


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def load_annotations(json_path: str) -> List[Dict]:
    with open(json_path) as f:
        return json.load(f)


def build_dataloaders(
    train_ann: List[Dict],
    val_ann: List[Dict],
    batch_size: int = 32,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader]:
    train_ds = TrafficSceneDataset(train_ann, transform=get_train_transforms())
    val_ds = TrafficSceneDataset(val_ann, transform=get_val_transforms())

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    return train_loader, val_loader


if __name__ == "__main__":
    train_ann, val_ann = generate_dummy_annotations("../data")
    train_loader, val_loader = build_dataloaders(train_ann, val_ann, batch_size=8)
    imgs, labels = next(iter(train_loader))
    print(f"Batch shape: {imgs.shape}, Labels shape: {labels.shape}")
