"""
BDD100K Multi-Label Dataset
Parses official BDD100K JSON annotations and builds a PyTorch Dataset for
4-class binary classification: [rain, night, clear, daytime].
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# ── Label schema ──────────────────────────────────────────────────────────────
LABEL_NAMES = ["rain", "night", "clear", "daytime"]

# Rules: read directly from BDD100K "attributes" block
LABEL_RULES = {
    "rain":    lambda attr: int(attr.get("weather", "")    == "rainy"),
    "night":   lambda attr: int(attr.get("timeofday", "")  == "night"),
    "clear":   lambda attr: int(attr.get("weather", "")    == "clear"),
    "daytime": lambda attr: int(attr.get("timeofday", "")  == "daytime"),
}


def parse_bdd100k_json(json_path: str) -> Dict[str, np.ndarray]:
    """
    Parse a BDD100K label JSON file and return a mapping:
        filename → np.ndarray of shape (4,) with binary labels.

    Only images whose attributes block is fully present are kept.
    """
    with open(json_path, "r") as f:
        annotations = json.load(f)

    label_map: Dict[str, np.ndarray] = {}

    for entry in annotations:
        filename = entry.get("name", "")
        attributes = entry.get("attributes", {})

        # Skip entries that lack the attribute fields we need
        if not attributes or "weather" not in attributes or "timeofday" not in attributes:
            continue

        label_vec = np.array(
            [rule(attributes) for rule in LABEL_RULES.values()],
            dtype=np.float32,
        )
        label_map[filename] = label_vec

    return label_map


def build_dataset_entries(
    image_dir: str,
    json_path: str,
    max_samples: Optional[int] = None,
) -> Tuple[List[str], List[np.ndarray]]:
    """
    Cross-reference parsed labels with images that physically exist on disk.

    Args:
        image_dir:   Directory containing JPEG images.
        json_path:   Path to BDD100K label JSON.
        max_samples: Cap on number of samples (use for quick experiments).

    Returns:
        (image_paths, label_vectors) — parallel lists.
    """
    label_map = parse_bdd100k_json(json_path)
    image_dir = Path(image_dir)

    image_paths, label_vectors = [], []

    for filename, label_vec in label_map.items():
        img_path = image_dir / filename
        if img_path.exists():
            image_paths.append(str(img_path))
            label_vectors.append(label_vec)
            if max_samples and len(image_paths) >= max_samples:
                break

    print(
        f"[dataset] Loaded {len(image_paths)} samples from {image_dir} "
        f"(JSON had {len(label_map)} entries)"
    )
    return image_paths, label_vectors


# ── Transforms ────────────────────────────────────────────────────────────────

def get_train_transforms() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def get_val_transforms() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


# ── Dataset class ─────────────────────────────────────────────────────────────

class BDD100KDataset(Dataset):
    """
    PyTorch Dataset for BDD100K multi-label scene attribute classification.

    Each item returns:
        image  : FloatTensor of shape (3, 224, 224)
        labels : FloatTensor of shape (4,)  — [rain, night, clear, daytime]
    """

    def __init__(
        self,
        image_dir: str,
        json_path: str,
        transform: Optional[transforms.Compose] = None,
        max_samples: Optional[int] = None,
    ):
        self.image_paths, self.label_vectors = build_dataset_entries(
            image_dir, json_path, max_samples=max_samples
        )
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        labels = torch.tensor(self.label_vectors[idx], dtype=torch.float32)
        return img, labels

    def label_distribution(self) -> Dict[str, float]:
        """Return positive-rate per label — useful for class-weight tuning."""
        arr = np.stack(self.label_vectors)
        return {name: float(arr[:, i].mean()) for i, name in enumerate(LABEL_NAMES)}


# ── Quick smoke-test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python dataset.py <image_dir> <json_path> [max_samples]")
        sys.exit(1)

    image_dir = sys.argv[1]
    json_path = sys.argv[2]
    max_samples = int(sys.argv[3]) if len(sys.argv) > 3 else 100

    ds = BDD100KDataset(
        image_dir=image_dir,
        json_path=json_path,
        transform=get_val_transforms(),
        max_samples=max_samples,
    )
    print(f"Dataset size : {len(ds)}")
    print(f"Label names  : {LABEL_NAMES}")
    print(f"Distribution : {ds.label_distribution()}")

    img, lbl = ds[0]
    print(f"Image shape  : {img.shape}")
    print(f"Label vector : {lbl}")
