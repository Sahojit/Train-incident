# Traffic Scene Understanding — BDD100K Multi-Label CNN

Multi-label scene attribute classification on the **Berkeley DeepDrive (BDD100K)** dataset using a fine-tuned ResNet18. The model simultaneously predicts four binary scene properties from a single image.

---

## Dataset — BDD100K

[BDD100K](https://bdd-data.berkeley.edu/) is a large-scale driving dataset with 100,000 images annotated with per-image scene attributes. We use **two label files**:

| File | Split | Images |
|------|-------|--------|
| `bdd100k_labels_images_train.json` | Training | 70,000 |
| `bdd100k_labels_images_val.json`   | Validation | 10,000 |

Each JSON entry carries an `attributes` block:

```json
{
  "name": "0000f77c-6257be58.jpg",
  "attributes": {
    "weather":   "clear",
    "timeofday": "daytime",
    "scene":     "city street"
  }
}
```

---

## Label Mapping Strategy

Four binary labels are derived **directly from the official annotation fields** — no synthetic augmentation, no guessing:

| Label | Source field | Positive condition |
|-------|-------------|-------------------|
| `rain` | `attributes.weather` | `== "rainy"` |
| `night` | `attributes.timeofday` | `== "night"` |
| `clear` | `attributes.weather` | `== "clear"` |
| `daytime` | `attributes.timeofday` | `== "daytime"` |

Each image maps to a `float32` vector of shape `(4,)`, e.g. `[0, 0, 1, 1]` (clear daytime).
Labels are not mutually exclusive — a small number of annotations have `weather=undefined` and will have zeros for both `rain` and `clear`.

---

## Model Architecture

```
Input (224×224×3)
    ↓
ResNet18 backbone (pretrained on ImageNet)
  └─ layer0  (conv1 + bn + relu + maxpool)
  └─ layer1–3 (residual blocks)
  └─ layer4   ← Grad-CAM target (7×7 feature maps)
    ↓
AdaptiveAvgPool2d → flatten → (512,)
    ↓
Linear(512, 4)   ← replaced head
    ↓
Raw logits (4,)  — sigmoid applied at inference / BCEWithLogitsLoss at training
```

**Why BCEWithLogitsLoss?** It fuses sigmoid + binary cross-entropy in a numerically stable way and naturally handles multi-label targets (no softmax competition between classes).

---

## Project Structure

```
traffic-bdd100k/
├── data/                        ← put BDD100K images & JSONs here
│   ├── bdd100k/images/100k/train/
│   ├── bdd100k/images/100k/val/
│   ├── bdd100k_labels_images_train.json
│   └── bdd100k_labels_images_val.json
│
├── vision/
│   ├── dataset.py   JSON parsing, label extraction, PyTorch Dataset, transforms
│   ├── model.py     ResNet18 wrapper with Grad-CAM hooks
│   ├── train.py     Training loop, cosine LR schedule, checkpoint saving
│   ├── metrics.py   Per-label F1, exact match ratio, Hamming loss
│   ├── gradcam.py   Grad-CAM heatmap generation and overlay visualisation
│   └── predict.py   Single-image and batch inference CLI
│
├── models/                      ← checkpoints saved here
├── notebooks/
│   └── explore_bdd100k.ipynb   Interactive EDA + metric review + Grad-CAM
│
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1 — Install dependencies

```bash
pip install -r requirements.txt
```

### 2 — Place dataset

```
traffic-bdd100k/data/
├── bdd100k_labels_images_train.json
├── bdd100k_labels_images_val.json
└── bdd100k/images/100k/train/*.jpg
                         val/*.jpg
```

### 3 — Smoke-test the dataset parser

```bash
cd traffic-bdd100k
python vision/dataset.py \
    data/bdd100k/images/100k/train \
    data/bdd100k_labels_images_train.json \
    100
```

### 4 — Train

```bash
python vision/train.py \
    --train_img   data/bdd100k/images/100k/train \
    --val_img     data/bdd100k/images/100k/val \
    --train_json  data/bdd100k_labels_images_train.json \
    --val_json    data/bdd100k_labels_images_val.json \
    --epochs      10 \
    --batch       32 \
    --max_train   2000 \
    --max_val     500 \
    --out_dir     models/
```

### 5 — Predict (single image)

```bash
python vision/predict.py \
    --checkpoint models/best_model.pth \
    --image      data/bdd100k/images/100k/val/0000f77c-6257be58.jpg
```

### 6 — Grad-CAM heatmaps

```bash
python vision/gradcam.py \
    --checkpoint models/best_model.pth \
    --image      data/bdd100k/images/100k/val/0000f77c-6257be58.jpg \
    --label_idx  2 \
    --out        outputs/gradcam_clear.png
```

---

## Training Process

| Phase | Epochs | Backbone | Head LR |
|-------|--------|----------|---------|
| Warm-up | 1–2 | Frozen | 10× base LR |
| Fine-tune | 3–10 | Trainable | base LR |

- **Optimizer:** Adam (`lr=1e-4`)
- **Scheduler:** CosineAnnealingLR (smooth LR decay over `T_max=epochs`)
- **Loss:** BCEWithLogitsLoss (one loss term per label, summed)
- **Checkpoint:** Saved whenever validation macro F1 improves

---

## Metrics

| Metric | Description |
|--------|-------------|
| Per-label F1 | Binary F1 computed independently for each of the 4 labels |
| Macro F1 | Mean of per-label F1 scores |
| Exact match ratio | Fraction of images where *all* 4 labels are correct simultaneously |
| Hamming loss | Fraction of individual label slots predicted incorrectly (lower = better) |

---

## Example Outputs

```
Epoch 10/10  (42.3s)
  Train loss: 0.2814   Val loss: 0.3102

[Val]  Metrics
  Macro F1      : 0.7841
  Exact match   : 0.6520
  Hamming loss  : 0.0812
  F1 [    rain]: 0.6914
  F1 [   night]: 0.8203
  F1 [   clear]: 0.8102
  F1 [daytime] : 0.8123
```

Grad-CAM heatmaps highlight *why* the model activates:
- **night** → activates on dark sky regions and headlight glare
- **rain** → activates on wet road reflections and blurred horizon
- **clear** → activates on bright sky and sharp horizon

---

## Notes

- Training on 2,000 images takes ~5–10 min on a modern GPU or ~45 min on CPU.
- Use `--max_train 500` for a quick sanity-check run (< 5 min on CPU).
- The dataset is class-imbalanced (daytime >> night). If F1 for minority classes is low, try weighted BCE by passing `pos_weight` to `BCEWithLogitsLoss`.
