"""
Training script for traffic scene multi-label classifier.

Usage:
    python -m vision.train --data_dir data/ --epochs 20 --batch_size 32
"""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from vision.model import build_model
from vision.dataset import build_dataloaders, generate_dummy_annotations, load_annotations
from vision.metrics import compute_metrics, format_metrics


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0

    for images, labels in tqdm(loader, desc="  Train", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
):
    model.eval()
    total_loss = 0.0
    all_logits, all_targets = [], []

    for images, labels in tqdm(loader, desc="  Val  ", leave=False):
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item() * images.size(0)
        all_logits.append(logits.cpu())
        all_targets.append(labels.cpu())

    all_logits = torch.cat(all_logits)
    all_targets = torch.cat(all_targets)
    metrics = compute_metrics(all_logits, all_targets)
    val_loss = total_loss / len(loader.dataset)
    return val_loss, metrics


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ------------------------------------------------------------------ data
    ann_train = Path(args.data_dir) / "train_annotations.json"
    ann_val = Path(args.data_dir) / "val_annotations.json"

    if not ann_train.exists():
        print("Annotation files not found — generating dummy data...")
        train_ann, val_ann = generate_dummy_annotations(
            str(args.data_dir), n_train=200, n_val=50
        )
    else:
        train_ann = load_annotations(str(ann_train))
        val_ann = load_annotations(str(ann_val))

    train_loader, val_loader = build_dataloaders(
        train_ann, val_ann,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # ----------------------------------------------------------------- model
    model = build_model(pretrained=True, dropout=args.dropout).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=1e-4,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # -------------------------------------------------------------- training
    best_f1 = 0.0
    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    history = []

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs} — lr={scheduler.get_last_lr()[0]:.2e}")

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")
        print(format_metrics(metrics))

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, **metrics})

        if metrics["f1_macro"] > best_f1:
            best_f1 = metrics["f1_macro"]
            ckpt_path = models_dir / "best_vision_model.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "metrics": metrics,
            }, ckpt_path)
            print(f"  ✓ Best model saved (f1_macro={best_f1:.4f})")

    # Save training history
    with open(models_dir / "train_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print("\nTraining complete.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train traffic scene classifier")
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--models_dir", type=str, default="models")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--num_workers", type=int, default=2)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
