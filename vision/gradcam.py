from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

LABELS = ["rain", "night", "congestion", "clear"]


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self._gradients: Optional[torch.Tensor] = None
        self._activations: Optional[torch.Tensor] = None
        self._register_hooks()

    def _register_hooks(self) -> None:
        def forward_hook(module, input, output):
            self._activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self._gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(
        self,
        input_tensor: torch.Tensor,
        class_idx: Optional[int] = None,
    ) -> Tuple[np.ndarray, int]:
        self.model.eval()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = int(torch.sigmoid(output).argmax(dim=1).item())

        self.model.zero_grad()
        score = output[0, class_idx]
        score.backward()

        gradients = self._gradients
        activations = self._activations

        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1).squeeze(0)
        cam = F.relu(cam)

        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        return cam.cpu().numpy(), class_idx


def overlay_heatmap(
    original_image: np.ndarray,
    cam: np.ndarray,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    h, w = original_image.shape[:2]
    heatmap = cv2.resize(cam, (w, h))
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    if original_image.dtype != np.uint8:
        original_image = (original_image * 255).astype(np.uint8)

    return (alpha * heatmap_colored + (1 - alpha) * original_image).astype(np.uint8)


def visualize_all_labels(
    model: torch.nn.Module,
    image_path: str,
    target_layer: torch.nn.Module,
    output_path: str = "gradcam_output.png",
    label_names: List[str] = LABELS,
) -> Dict[str, np.ndarray]:
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    pil_img = Image.open(image_path).convert("RGB").resize((224, 224))
    original = np.array(pil_img)
    input_tensor = preprocess(pil_img).unsqueeze(0)

    gradcam = GradCAM(model, target_layer)
    results: Dict[str, np.ndarray] = {}
    overlays = []

    for idx, label in enumerate(label_names):
        cam, _ = gradcam.generate(input_tensor.clone().requires_grad_(True), class_idx=idx)
        overlay = overlay_heatmap(original.copy(), cam)
        results[label] = overlay
        overlays.append(overlay)

    grid = np.concatenate([original] + overlays, axis=1)
    Image.fromarray(grid).save(output_path)
    print(f"Grad-CAM grid saved to {output_path}")

    return results


if __name__ == "__main__":
    from vision.model import build_model

    model = build_model(pretrained=False)
    target_layer = model.backbone.layer4[-1].conv2

    dummy_img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    dummy_img.save("/tmp/test_traffic.jpg")
    visualize_all_labels(model, "/tmp/test_traffic.jpg", target_layer, "/tmp/gradcam_test.png")
