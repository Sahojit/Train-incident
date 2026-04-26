"""
Grad-CAM implementation for BDD100KClassifier.

Generates class-discriminative heatmaps by back-propagating gradients
through the final convolutional block (layer4) of ResNet18.

Usage:
    python vision/gradcam.py \
        --checkpoint models/best_model.pth \
        --image      path/to/image.jpg \
        --label_idx  0            # 0=rain 1=night 2=clear 3=daytime
        --out        gradcam_out.png
"""

import argparse
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from dataset import LABEL_NAMES, get_val_transforms
from model import BDD100KClassifier


# ── Grad-CAM core ─────────────────────────────────────────────────────────────

class GradCAM:
    """
    Computes Grad-CAM heatmaps for a given class index.

    The target layer is BDD100KClassifier.layer4 (the last ResNet conv block),
    which produces a 7×7 spatial feature map for 224×224 inputs.
    """

    def __init__(self, model: BDD100KClassifier, device: torch.device):
        self.model  = model.to(device)
        self.device = device
        self.model.eval()

        self._activations: Optional[torch.Tensor] = None
        self._gradients:   Optional[torch.Tensor] = None

        # Register forward & backward hooks on layer4
        self.model.layer4.register_forward_hook(self._save_activations)
        self.model.layer4.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self._activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    def generate(
        self,
        image_tensor: torch.Tensor,   # (1, 3, 224, 224)
        label_idx: int,
    ) -> np.ndarray:
        """
        Returns a (224, 224) float32 heatmap normalised to [0, 1].
        """
        image_tensor = image_tensor.to(self.device)
        image_tensor.requires_grad_(True)

        logits = self.model(image_tensor)          # (1, 4)

        self.model.zero_grad()
        # Back-prop the target class score
        logits[0, label_idx].backward()

        # Global average pool the gradients → channel weights  (C,)
        weights = self._gradients.squeeze(0).mean(dim=(1, 2))  # (C,)

        # Weighted sum of activation maps
        activations = self._activations.squeeze(0)             # (C, H, W)
        cam = torch.einsum("c,chw->hw", weights, activations)  # (H, W)
        cam = F.relu(cam)

        # Normalise and upsample to 224×224
        cam = cam.cpu().numpy()
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()

        cam_resized = cv2.resize(cam, (224, 224))
        return cam_resized.astype(np.float32)


# ── Visualisation helpers ─────────────────────────────────────────────────────

def overlay_heatmap(
    original_image: np.ndarray,    # (224, 224, 3)  uint8 RGB
    heatmap: np.ndarray,           # (224, 224)     float32 [0,1]
    alpha: float = 0.4,
) -> np.ndarray:
    """Overlay the Grad-CAM heatmap on the original image."""
    colormap   = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    colormap   = cv2.cvtColor(colormap, cv2.COLOR_BGR2RGB)
    overlaid   = cv2.addWeighted(original_image, 1 - alpha, colormap, alpha, 0)
    return overlaid


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Reverse ImageNet normalisation → uint8 (224, 224, 3)."""
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img  = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img  = img * std + mean
    img  = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img


def visualize_gradcam(
    model: BDD100KClassifier,
    image_path: str,
    label_idx: int,
    device: torch.device,
    out_path: Optional[str] = None,
    threshold: float = 0.5,
) -> None:
    import matplotlib.pyplot as plt  # lazy — not needed at import time
    """
    Full pipeline: load image → predict → generate heatmap → plot.
    """
    transform = get_val_transforms()
    img_pil   = Image.open(image_path).convert("RGB")
    tensor    = transform(img_pil).unsqueeze(0)

    grad_cam  = GradCAM(model, device)
    heatmap   = grad_cam.generate(tensor, label_idx=label_idx)

    # Predictions (for display)
    with torch.no_grad():
        logits = model(tensor.to(device))
        probs  = torch.sigmoid(logits).squeeze().cpu().numpy()

    orig_img = denormalize(tensor)
    overlay  = overlay_heatmap(orig_img, heatmap)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(orig_img);     axes[0].set_title("Original image");   axes[0].axis("off")
    axes[1].imshow(heatmap, cmap="jet"); axes[1].set_title(f"Grad-CAM heatmap\n[{LABEL_NAMES[label_idx]}]"); axes[1].axis("off")
    axes[2].imshow(overlay);      axes[2].set_title("Overlay");          axes[2].axis("off")

    # Prediction bar
    pred_str = "  ".join(
        f"{n}={p:.2f}{'✓' if p >= threshold else ''}"
        for n, p in zip(LABEL_NAMES, probs)
    )
    fig.suptitle(f"Predictions: {pred_str}", fontsize=10)
    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"[gradcam] Saved to {out_path}")
    else:
        plt.show()

    plt.close()


def batch_visualize(
    model: BDD100KClassifier,
    image_paths: List[str],
    device: torch.device,
    out_dir: str = "gradcam_outputs/",
    threshold: float = 0.5,
) -> None:
    """Generate Grad-CAM for every label on a list of images."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for img_path in image_paths:
        stem = Path(img_path).stem
        for idx, label in enumerate(LABEL_NAMES):
            out_path = out_dir / f"{stem}_{label}.png"
            visualize_gradcam(
                model=model,
                image_path=img_path,
                label_idx=idx,
                device=device,
                out_path=str(out_path),
                threshold=threshold,
            )


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="Path to best_model.pth")
    p.add_argument("--image",      required=True, help="Path to input image")
    p.add_argument("--label_idx",  type=int, default=0,
                   help="Label index: 0=rain 1=night 2=clear 3=daytime")
    p.add_argument("--out",        default=None,  help="Output PNG path")
    return p.parse_args()


if __name__ == "__main__":
    args   = parse_args()
    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )

    model  = BDD100KClassifier(pretrained=False)
    ckpt   = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    visualize_gradcam(
        model=model,
        image_path=args.image,
        label_idx=args.label_idx,
        device=device,
        out_path=args.out,
    )
