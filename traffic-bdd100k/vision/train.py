"""
Training script for BDD100K multi-label scene classifier.

Usage:
    python vision/train.py \
        --train_img   bdd100k/images/100k/train \
        --val_img     bdd100k/images/100k/val \
        --train_json  bdd100k/labels/bdd100k_labels_images_train.json \
        --val_json    bdd100k/labels/bdd100k_labels_images_val.json \
        --epochs 10 \
        --batch  32 \
        --max_train 2000 \
        --max_val   500 \
        --out_dir    models/
"""

import argparse
import os
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import BDD100KDataset, LABEL_NAMES, get_train_transforms, get_val_transforms
from metrics import MetricTracker, print_metrics
from model import BDD100KClassifier


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train BDD100K multi-label classifier")
    p.add_argument("--train_img",   required=True,  help="Train image directory")
    p.add_argument("--val_img",     required=True,  help="Val image directory")
    p.add_argument("--train_json",  required=True,  help="Train annotation JSON")
    p.add_argument("--val_json",    required=True,  help="Val annotation JSON")
    p.add_argument("--epochs",      type=int,   default=10)
    p.add_argument("--batch",       type=int,   default=32)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--max_train",   type=int,   default=2000,  help="Max training samples")
    p.add_argument("--max_val",     type=int,   default=500,   help="Max val samples")
    p.add_argument("--out_dir",     default="models/",         help="Checkpoint directory")
    p.add_argument("--workers",     type=int,   default=4,     help="DataLoader workers")
    p.add_argument("--freeze_epochs", type=int, default=2,
                   help="Epochs to freeze backbone (fine-tune head only)")
    return p.parse_args()


# ── Device selection ──────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():          # Apple Silicon
        return torch.device("mps")
    return torch.device("cpu")


# ── Freeze / unfreeze backbone ────────────────────────────────────────────────

def set_backbone_trainable(model: BDD100KClassifier, trainable: bool) -> None:
    for name, param in model.named_parameters():
        if "fc" not in name:
            param.requires_grad = trainable


# ── One epoch ─────────────────────────────────────────────────────────────────

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    tracker: MetricTracker,
    train: bool,
) -> float:
    model.train() if train else model.eval()
    total_loss = 0.0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            logits = model(images)
            loss   = criterion(logits, labels)

            if train and optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            tracker.update(logits, labels)

    return total_loss / len(loader.dataset)


# ── Main training loop ────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    device = get_device()
    print(f"[train] Device: {device}")

    # ── Datasets & loaders ───────────────────────────────────────────────────
    train_ds = BDD100KDataset(
        image_dir=args.train_img,
        json_path=args.train_json,
        transform=get_train_transforms(),
        max_samples=args.max_train,
    )
    val_ds = BDD100KDataset(
        image_dir=args.val_img,
        json_path=args.val_json,
        transform=get_val_transforms(),
        max_samples=args.max_val,
    )

    print(f"[train] Train size: {len(train_ds)}  Val size: {len(val_ds)}")
    print(f"[train] Label distribution (train): {train_ds.label_distribution()}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=args.workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )

    # ── Model ────────────────────────────────────────────────────────────────
    model = BDD100KClassifier(pretrained=True).to(device)

    # ── Loss & optimizer ─────────────────────────────────────────────────────
    # BCEWithLogitsLoss is numerically stable (no explicit sigmoid in forward)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    # ── Checkpoint dir ───────────────────────────────────────────────────────
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_macro_f1 = 0.0
    train_tracker = MetricTracker()
    val_tracker   = MetricTracker()

    # ── Epoch loop ───────────────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Phase 1: freeze backbone, train head only
        if epoch <= args.freeze_epochs:
            set_backbone_trainable(model, trainable=False)
            for g in optimizer.param_groups:
                g["lr"] = args.lr * 10   # higher LR for the fresh head
        else:
            set_backbone_trainable(model, trainable=True)
            for g in optimizer.param_groups:
                g["lr"] = args.lr

        # Training pass
        train_loss = run_epoch(
            model, train_loader, criterion, optimizer, device, train_tracker, train=True
        )
        train_metrics = train_tracker.compute()
        train_tracker.reset()

        # Validation pass
        val_loss = run_epoch(
            model, val_loader, criterion, None, device, val_tracker, train=False
        )
        val_metrics = val_tracker.compute()
        val_tracker.reset()

        scheduler.step()
        elapsed = time.time() - t0

        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}  ({elapsed:.1f}s)")
        print(f"  Train loss: {train_loss:.4f}   Val loss: {val_loss:.4f}")
        print_metrics(train_metrics, prefix="[Train]")
        print_metrics(val_metrics,   prefix="[Val]  ")

        # Save best checkpoint by macro F1
        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            ckpt_path = out_dir / "best_model.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metrics": val_metrics,
                },
                ckpt_path,
            )
            print(f"  ✓ New best — saved to {ckpt_path} (macro F1={best_macro_f1:.4f})")

    print(f"\n[train] Done. Best val macro F1: {best_macro_f1:.4f}")


if __name__ == "__main__":
    args = parse_args()
    train(args)
