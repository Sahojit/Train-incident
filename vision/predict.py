import argparse
from pathlib import Path
from typing import Dict, List, Optional

import torch
from PIL import Image

from vision.model import build_model, LABELS
from vision.dataset import get_val_transforms
from vision.gradcam import visualize_all_labels


def load_model(checkpoint_path: Optional[str], device: torch.device) -> torch.nn.Module:
    model = build_model(pretrained=False)
    if checkpoint_path and Path(checkpoint_path).exists():
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print("No checkpoint found — using random weights (for demo only)")
    model.to(device).eval()
    return model


def predict_image(
    image_path: str,
    model: torch.nn.Module,
    device: torch.device,
    threshold: float = 0.5,
) -> Dict:
    transform = get_val_transforms()
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits).squeeze(0).cpu()

    results = {}
    for label, prob in zip(LABELS, probs.tolist()):
        results[label] = {
            "probability": round(prob, 4),
            "detected": prob >= threshold,
        }

    detected = [lbl for lbl, v in results.items() if v["detected"]]
    return {
        "image": image_path,
        "predictions": results,
        "detected_labels": detected,
        "threshold": threshold,
    }


def batch_predict(
    image_paths: List[str],
    model: torch.nn.Module,
    device: torch.device,
    threshold: float = 0.5,
) -> List[Dict]:
    return [predict_image(p, model, device, threshold) for p in image_paths]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict traffic scene labels")
    p.add_argument("--image", type=str, required=True)
    p.add_argument("--checkpoint", type=str, default="models/best_vision_model.pt")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--gradcam", action="store_true")
    p.add_argument("--gradcam_output", type=str, default="gradcam_output.png")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(args.checkpoint, device)
    result = predict_image(args.image, model, device, args.threshold)

    print("\n=== Traffic Scene Prediction ===")
    for label, info in result["predictions"].items():
        status = "✓" if info["detected"] else "✗"
        print(f"  [{status}] {label:<12}: {info['probability']:.4f}")
    print(f"\nDetected: {result['detected_labels']}")

    if args.gradcam:
        target_layer = model.backbone.layer4[-1].conv2
        visualize_all_labels(model, args.image, target_layer, args.gradcam_output)
