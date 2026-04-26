"""
ResNet18 multi-label classifier for BDD100K scene attributes.
Final FC layer replaced with a 4-output head; sigmoid is applied at
inference time (BCEWithLogitsLoss handles it internally during training).
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights


NUM_LABELS = 4  # rain, night, clear, daytime


def build_model(num_labels: int = NUM_LABELS, pretrained: bool = True) -> nn.Module:
    """
    Load ImageNet-pretrained ResNet18 and replace the classifier head.

    Args:
        num_labels: Number of binary output logits.
        pretrained: Use ImageNet weights (strongly recommended).

    Returns:
        nn.Module ready for training or inference.
    """
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)

    # Freeze backbone for the first few epochs (optional; controlled in train.py)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_labels)

    return model


class BDD100KClassifier(nn.Module):
    """
    Thin wrapper around ResNet18 that exposes intermediate feature maps
    needed by Grad-CAM (layer4 activations).
    """

    def __init__(self, num_labels: int = NUM_LABELS, pretrained: bool = True):
        super().__init__()
        base = build_model(num_labels=num_labels, pretrained=pretrained)

        # Split backbone so we can hook into the last conv block
        self.layer0  = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1  = base.layer1
        self.layer2  = base.layer2
        self.layer3  = base.layer3
        self.layer4  = base.layer4          # ← Grad-CAM target
        self.avgpool = base.avgpool
        self.fc      = base.fc

        self.gradients = None   # Optional[torch.Tensor], populated by hook

    # -- forward -----------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)   # raw logits; sigmoid applied externally

    # -- Grad-CAM hooks ----------------------------------------------------------
    def activations_hook(self, grad: torch.Tensor) -> None:
        """Backward hook — stores gradients flowing through layer4."""
        self.gradients = grad

    def get_activations(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass up to (and including) layer4, returning the feature map."""
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def register_grad_cam_hooks(self, x: torch.Tensor):
        """Run a partial forward, register the gradient hook, and return activations."""
        activations = self.get_activations(x)
        activations.requires_grad_(True)
        handle = activations.register_hook(self.activations_hook)
        return activations, handle


# ── Summary helper ────────────────────────────────────────────────────────────
def model_summary(model: nn.Module) -> None:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params    : {total:,}")
    print(f"Trainable params: {trainable:,}")


if __name__ == "__main__":
    model = BDD100KClassifier(pretrained=False)
    model_summary(model)
    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    print(f"Output shape: {out.shape}")   # (2, 4)
    print(f"Sample logits: {out[0].detach()}")
