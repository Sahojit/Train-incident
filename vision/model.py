"""
ResNet18-based multi-label classifier for traffic scene understanding.
Labels: rain, night, congestion, clear
"""

import torch
import torch.nn as nn
from torchvision import models


LABELS = ["rain", "night", "congestion", "clear"]
NUM_CLASSES = len(LABELS)


class TrafficSceneClassifier(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)

        # Freeze early layers, fine-tune later ones
        for name, param in backbone.named_parameters():
            if "layer4" not in name and "fc" not in name:
                param.requires_grad = False

        in_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Returns raw logits — apply sigmoid externally for inference
        return self.backbone(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        return (self.predict_proba(x) >= threshold).long()


def build_model(pretrained: bool = True, dropout: float = 0.3) -> TrafficSceneClassifier:
    return TrafficSceneClassifier(pretrained=pretrained, dropout=dropout)


if __name__ == "__main__":
    model = build_model()
    dummy = torch.randn(2, 3, 224, 224)
    logits = model(dummy)
    print(f"Output shape: {logits.shape}")  # (2, 4)
    print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
