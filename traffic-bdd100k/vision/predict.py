"""
Inference script — run the trained BDD100K classifier on a single image
or a directory of images.

Usage (single image):
    python vision/predict.py \
        --checkpoint models/best_model.pth \
        --image      path/to/image.jpg

Usage (directory):
    python vision/predict.py \
        --checkpoint models/best_model.pth \
        --image_dir  bdd100k/images/100k/val \
        --max        50 \
        --gradcam    True
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
from PIL import Image

from dataset import LABEL_NAMES, get_val_transforms
from gradcam import batch_visualize, visualize_gradcam
from model import BDD100KClassifier


# ── Device ────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Load checkpoint ───────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, device: torch.device) -> BDD100KClassifier:
    model = BDD100KClassifier(pretrained=False)
    ckpt  = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    model.to(device)

    epoch   = ckpt.get("epoch", "?")
    metrics = ckpt.get("val_metrics", {})
    print(f"[predict] Loaded checkpoint  epoch={epoch}")
    if metrics:
        print(f"[predict] Val macro F1 at save: {metrics.get('macro_f1', 0):.4f}")
    return model


# ── Single-image prediction ───────────────────────────────────────────────────

def predict_image(
    model: BDD100KClassifier,
    image_path: str,
    device: torch.device,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Returns a dict of {label: probability} and prints a human-readable result.
    """
    transform = get_val_transforms()
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.sigmoid(logits).squeeze().cpu().tolist()

    result: Dict[str, float] = {}
    print(f"\nImage: {Path(image_path).name}")
    print(f"{'Label':<12} {'Prob':>6}  {'Pred':>5}")
    print("-" * 28)
    for name, prob in zip(LABEL_NAMES, probs):
        pred = "YES" if prob >= threshold else "no"
        print(f"{name:<12} {prob:>6.3f}  {pred:>5}")
        result[name] = round(prob, 4)

    active = [n for n, p in result.items() if p >= threshold]
    print(f"\nActive labels: {active or ['(none)']}")
    return result


# ── Batch prediction ──────────────────────────────────────────────────────────

def predict_directory(
    model: BDD100KClassifier,
    image_dir: str,
    device: torch.device,
    max_images: int,
    threshold: float = 0.5,
    gradcam: bool = False,
    gradcam_out_dir: str = "gradcam_outputs/",
) -> List[Dict]:
    image_dir = Path(image_dir)
    image_files = sorted(image_dir.glob("*.jpg"))[:max_images]

    if not image_files:
        raise FileNotFoundError(f"No .jpg images found in {image_dir}")

    results = []
    for img_path in image_files:
        preds = predict_image(model, str(img_path), device, threshold)
        results.append({"image": str(img_path.name), "predictions": preds})

    # Optionally generate Grad-CAM heatmaps
    if gradcam:
        img_paths = [str(p) for p in image_files]
        batch_visualize(model, img_paths, device, out_dir=gradcam_out_dir, threshold=threshold)

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="BDD100K multi-label inference")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--image",      default=None, help="Single image path")
    p.add_argument("--image_dir",  default=None, help="Directory of images")
    p.add_argument("--max",        type=int, default=20, help="Max images (dir mode)")
    p.add_argument("--threshold",  type=float, default=0.5)
    p.add_argument("--gradcam",    action="store_true", help="Save Grad-CAM heatmaps")
    p.add_argument("--gradcam_dir", default="gradcam_outputs/")
    p.add_argument("--json_out",   default=None, help="Save predictions as JSON")
    return p.parse_args()


if __name__ == "__main__":
    args   = parse_args()
    device = get_device()
    model  = load_model(args.checkpoint, device)

    if args.image:
        predict_image(model, args.image, device, threshold=args.threshold)

    elif args.image_dir:
        results = predict_directory(
            model=model,
            image_dir=args.image_dir,
            device=device,
            max_images=args.max,
            threshold=args.threshold,
            gradcam=args.gradcam,
            gradcam_out_dir=args.gradcam_dir,
        )
        if args.json_out:
            with open(args.json_out, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\n[predict] Saved {len(results)} predictions → {args.json_out}")
    else:
        print("Provide --image or --image_dir")
