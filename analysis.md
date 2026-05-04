# Traffic Scene Understanding — Project Analysis Reference

---

## 1. Problem Statement

Urban traffic management depends on two parallel data sources that are never connected:
- **Camera feeds** — visual information about road conditions
- **Incident text reports** — written descriptions of what happened, where, and how severe

No existing lightweight system fuses both sources and generates an actionable alert automatically. Manual monitoring is slow, inconsistent, and does not scale to city-wide traffic networks.

**What this project solves:**
- Automatically classify what is visually happening in a traffic scene
- Extract structured information from incident text
- Combine both signals into one human-readable alert

---

## 2. System Architecture — How Everything Connects

```
Traffic Image ──────────────────────────────────┐
                                                 ↓
                                        Vision Module (ResNet18)
                                                 ↓
                                    Scene Labels: rain, night,
                                    congestion, clear
                                                 ↓
                                                 │
Incident Text ──────────────────────────────────┤
                                                 ↓
                                       NLP Pipeline
                                    ┌──────────────────┐
                                    │ spaCy            │ → tokens, POS, deps
                                    │ BERT NER         │ → LOCATION, INCIDENT_TYPE, SEVERITY
                                    │ BERT Classifier  │ → accident/jam/closure/normal
                                    └──────────────────┘
                                                 ↓
                                                 │
                                    Both outputs combined
                                                 ↓
                                       Alert Generator
                                    (Claude few-shot / Template)
                                                 ↓
                               "🚨 Major accident at Sector 62.
                                Rain detected. Avoid this route."
                                                 ↓
                                    FastAPI REST Layer
                                    Streamlit Dashboard
```

---

## 3. Deep Learning — Vision Module

### 3.1 Why ResNet18

ResNet18 is an 18-layer deep CNN pretrained on ImageNet (1.2 million images, 1000 classes).

**Skip connections** are the key innovation in ResNet. In a plain deep network, gradients vanish as they travel back through many layers — the early layers stop learning. Skip connections create a shortcut path for gradients, bypassing one or more layers. This allows training of much deeper networks without degradation.

```
Input ──┬── Conv → BN → ReLU → Conv → BN ──┐
        │                                   + → ReLU → Output
        └───────────────────────────────────┘
                  (identity shortcut)
```

**Why not deeper ResNet (50/101):**
- More parameters need more data and compute
- ResNet18 is sufficient for 4 binary labels
- Faster training and inference
- Better for demo and CA submission

**Why not VGG:**
- No skip connections — harder to train deep
- Much larger parameter count
- No performance advantage here

**Why not train from scratch:**
- We have ~200 dummy images — far too little
- ImageNet features (edges, textures, shapes) transfer well to traffic scenes
- Training from scratch on 200 images would massively overfit

---

### 3.2 Transfer Learning Strategy

**Total parameters: 11,308,868**
**Trainable: 8,526,084**
**Frozen: 2,782,784**

```
ResNet18 Layer Structure:

Layer 1  → edges, gradients              ← FROZEN
Layer 2  → textures, patterns            ← FROZEN
Layer 3  → object parts, shapes          ← FROZEN
Layer 4  → high-level semantics          ← TRAINABLE
FC Head  → classification                ← REPLACED + TRAINABLE
```

**Why freeze layers 1–3:**
These layers learn universal visual features that are identical across all image domains. ImageNet training already made them excellent. Retraining them wastes compute and risks destroying good features with limited data.

**Why keep layer 4 trainable:**
Layer 4 learns domain-specific high-level patterns — what a congested road looks like, what rain on a lens looks like. These are different from ImageNet concepts and must be adapted.

---

### 3.3 Custom FC Head

The original ResNet18 FC layer maps 512 features to 1000 ImageNet classes. We replaced it entirely.

```
Original:  Linear(512 → 1000)

Ours:
  Dropout(0.3)
  Linear(512 → 256)
  ReLU
  Dropout(0.3)
  Linear(256 → 4)
```

**Dropout(0.3):** Randomly zeroes 30% of neurons during training. Forces the network to not rely on any single feature — acts as regularisation, reduces overfitting on small datasets.

**Two linear layers instead of one:** Adds a non-linear intermediate representation. The model can learn more complex decision boundaries before the final 4-label output.

**ReLU:** Introduces non-linearity between the two linear layers. Without it, two linear layers collapse into one linear transformation — no benefit.

---

### 3.4 Multi-Label Classification

The 4 labels — rain, night, congestion, clear — are **not mutually exclusive**.

One image can simultaneously be:
- Rainy + Night + Congested
- Clear + Daytime + Congested
- Rainy + Clear (clearing rain)

**Sigmoid vs Softmax:**

| | Sigmoid | Softmax |
|---|---|---|
| Applied to | Each output independently | All outputs together |
| Outputs sum to 1 | No | Yes |
| Multiple labels possible | Yes | No |
| Correct for our task | Yes | No |

Softmax forces competition — if one label gets high probability, others must drop. That is wrong for multi-label. Sigmoid treats each label as an independent binary question.

**Threshold:** If sigmoid output ≥ 0.5 → label detected. Adjustable — lower for higher recall, higher for higher precision.

---

### 3.5 Loss Function — BCEWithLogitsLoss

Binary Cross Entropy computed per label, per image, then averaged.

**Formula per label:**
```
loss = -[ y · log(σ(x)) + (1 - y) · log(1 - σ(x)) ]
```

Where:
- y = true label (0 or 1)
- x = raw logit output
- σ = sigmoid function

**Why BCEWithLogitsLoss over BCELoss:**

BCELoss requires sigmoid applied first, then CE computed. For large logit values, `exp(x)` in sigmoid overflows to infinity — numerical instability.

BCEWithLogitsLoss combines both operations using the log-sum-exp trick:
```
loss = max(x, 0) - x·y + log(1 + exp(-|x|))
```
Numerically stable for any logit value.

**Why not CrossEntropyLoss:**
CrossEntropyLoss applies softmax internally — makes all classes compete. Wrong for multi-label.

---

### 3.6 Data Augmentation

Applied **only during training**, not validation. Validation uses fixed transforms for consistent evaluation.

| Augmentation | Parameters | Reason |
|---|---|---|
| Resize | 256×256 | Standardise before crop |
| RandomCrop | 224×224 | Position invariance — vehicles appear anywhere in frame |
| RandomHorizontalFlip | p=0.5 | Roads are symmetric left-right — doubles effective data |
| ColorJitter | brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05 | Night, rain, glare all change exposure and colour balance |
| GaussianBlur | radius 0–1.5 | Rain causes lens blur and water droplets — simulates it |
| ToTensor | — | Converts PIL image to float tensor [0, 1] |
| Normalize | mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] | Must match ImageNet stats — pretrained backbone expects this exact scaling |

**Why Normalize with ImageNet stats specifically:**
The pretrained weights learned to expect inputs in this exact distribution. If you normalize differently, the feature maps produced by layers 1–3 are meaningless — the pretrained knowledge is wasted.

**Validation pipeline:**
Only Resize + ToTensor + Normalize. No random operations — we need reproducible, consistent evaluation scores.

---

### 3.7 Optimizer — Adam

Adam (Adaptive Moment Estimation) maintains:
- First moment: running mean of gradients
- Second moment: running mean of squared gradients

Uses these to compute an adaptive learning rate per parameter.

**Why Adam over SGD:**
SGD uses one global learning rate for all parameters. Some parameters need large updates, others need small — SGD handles this poorly without careful tuning. Adam adapts per parameter automatically.

**Why not SGD with momentum:**
Momentum SGD needs careful learning rate scheduling and warm-up. Adam converges faster with default hyperparameters — better for transfer learning where only part of the network is being fine-tuned.

**Key hyperparameters:**
- lr = 1e-3
- weight_decay = 1e-4 (L2 regularisation)
- Only parameters with `requires_grad=True` are passed — frozen layers receive no updates

---

### 3.8 Learning Rate Scheduler — CosineAnnealingLR

```
lr(t) = η_min + 0.5 × (η_max - η_min) × (1 + cos(π × t / T_max))
```

Starts at lr=1e-3, smoothly decays following a cosine curve to eta_min=1e-6 over all epochs.

```
lr
1e-3 |█
     | ██
     |   ██
     |     ███
     |        ████
     |            ██████████
1e-6 |_________________________ epochs
```

**Why cosine over StepLR:**
StepLR drops learning rate abruptly at fixed intervals — can cause loss to spike suddenly. Cosine decay is smooth and continuous — the model trains aggressively early, then fine-tunes precisely in later epochs without disruption.

**Why not ReduceLROnPlateau:**
Requires monitoring validation loss and has patience hyperparameter to tune. Cosine annealing is simpler — no extra decisions, deterministic schedule.

---

### 3.9 Evaluation Metrics

**Why not accuracy:**
Accuracy in multi-label is misleading. If `rain` appears in only 10% of images, predicting "not rain" always gives 90% accuracy. The model learns nothing but scores well.

**Per-label F1 Score:**
```
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2 × (Precision × Recall) / (Precision + Recall)
```
Computed separately for each of the 4 labels. Tells you: for this specific label, is the model balanced — not just predicting everything as positive (high recall, low precision) or being too conservative (high precision, low recall).

**Macro F1:**
Average of the 4 per-label F1 scores. Single number representing overall multi-label performance.

**Exact Match Ratio (Subset Accuracy):**
```
EMR = samples where all 4 predicted labels == all 4 true labels / N
```
Strictest metric. A sample with 3/4 correct labels counts as completely wrong. Best metric for production — you either got the full picture right or you didn't.

**Hamming Loss:**
```
HL = wrong label predictions / (N × num_labels)
```
Partial credit metric. Measures fraction of individual label slots that are wrong. Useful for understanding which specific labels are hard — complements EMR which gives no partial credit.

---

### 3.10 Grad-CAM — Explainability

**The problem it solves:**
Neural networks are black boxes. Even if the model predicts correctly, you don't know if it's looking at the right regions. A model could predict "rain" because of a watermark on the camera, not actual rain — Grad-CAM reveals this.

**How it works step by step:**

**Step 1 — Forward pass with hooks**
Two hooks are registered on `layer4[-1].conv2` (last conv layer):
- Forward hook captures the **feature maps** (activations) — what the layer produced
- Backward hook captures the **gradients** — how much each activation contributed to the final score

```python
forward_hook  → self._activations = output   # shape: (1, 512, 7, 7)
backward_hook → self._gradients   = grad     # shape: (1, 512, 7, 7)
```

**Step 2 — Backpropagate the label score**
Not the loss — the raw score of the target label. This gives gradients specifically for that label.

**Step 3 — Global average pool the gradients**
```
weights = gradients.mean(dim=(H, W))   # shape: (512,)
```
Each of the 512 channels gets one importance weight — average gradient across its 7×7 spatial grid.

**Step 4 — Weighted sum of feature maps**
```
CAM = Σ (weight_c × activation_c)     # shape: (7, 7)
```

**Step 5 — ReLU**
Keep only positive values. Negative values suppress the prediction — we only care about regions that activated it.

**Step 6 — Upscale and overlay**
7×7 → 224×224 via bilinear interpolation. Apply JET colormap (blue→green→yellow→red). Blend with original image at 45% opacity.

**Why layer4 and not earlier layers:**
Earlier layers detect low-level features (edges, gradients). They fire for everything — not useful for explaining a specific prediction. Layer4 is the deepest semantic layer — it understands traffic-specific concepts. Its activations directly relate to the final prediction.

**Output interpretation:**
- Red/Yellow = high gradient × high activation = model focused here for this label
- Blue = model ignored this region for this label

---

## 4. NLP Pipeline

### 4.1 spaCy Preprocessing

Runs first on every text input. `en_core_web_sm` runs as one optimised pipeline.

**Tokenization:**
Not a naive split on spaces. Handles contractions, hyphens, punctuation rules. "NH-8" → one token. "don't" → "do" + "n't".

**POS Tagging:**
Labels every token with Part-of-Speech:
- "severe" → ADJ
- "accident" → NOUN
- "occurred" → VERB
- "Sector 62" → PROPN + NUM

**Dependency Parsing:**
Builds a grammatical tree. Identifies which word is the ROOT verb, which nouns are subjects, which are objects. Allows extraction of Subject-Verb-Object triples from sentences.

**Noun Chunks:**
Extracts meaningful noun phrase units as single entities — "Major accident", "heavy congestion", "NH-8".

**Named Entities (spaCy built-in):**
Detects standard types — dates, cardinal numbers, organisations. Does NOT know INCIDENT_TYPE or SEVERITY — those require BERT fine-tuning.

**Why spaCy over NLTK:**
spaCy runs all operations in one optimised C++ pipeline. NLTK is modular, Python-based, slower, and its NER is rule-based and weaker.

---

### 4.2 BERT NER — Named Entity Recognition

**Entities extracted:**
- `LOCATION` — NH-8, Sector 62, MG Road, Ring Road
- `INCIDENT_TYPE` — accident, collision, pile-up, breakdown, road closure
- `SEVERITY` — minor, major, severe, heavy, light

**Model:** `BertForTokenClassification`
BERT encoder + linear classification head applied to every token's output vector.

**Why BERT over spaCy NER:**
spaCy's built-in NER knows: PERSON, ORG, GPE, DATE, CARDINAL, etc. It has never seen INCIDENT_TYPE or SEVERITY as entity types. Fine-tuning BERT teaches it these domain concepts.

**Why BERT over CRF (Conditional Random Field):**
Traditional NER uses CRF on top of hand-crafted features. BERT learns contextual representations — it understands that "major" before "accident" signals SEVERITY, but "major" in "Major Road" signals LOCATION context. CRF with hand features cannot capture this.

**BIO Tagging Scheme:**
```
Token    : "A    severe   accident  occurred  near  Sector   62"
BIO tag  :  O    B-SEV    B-INC     O         O     B-LOC    I-LOC
```
- B = Beginning of entity
- I = Inside / continuation of entity
- O = Outside — not an entity

7 total label types: O, B-LOCATION, I-LOCATION, B-INCIDENT_TYPE, I-INCIDENT_TYPE, B-SEVERITY, I-SEVERITY

**Subword alignment:**
BERT tokenizes by subwords, not words. "congestion" → ["con", "##gest", "##ion"]. Three subword tokens for one word.

Label alignment rule:
- First subword → gets the real BIO label
- Continuation subwords → get -100 (ignored in loss)
- Special tokens [CLS], [SEP] → get -100

This ensures loss is only computed on real word boundaries.

**Training:**
- AdamW (Adam + weight decay 0.01) — standard for BERT fine-tuning
- Linear warmup — starts lr near zero, ramps up over 10% of total steps. Prevents large gradient updates from destroying pretrained BERT weights early in training
- Gradient clipping at 1.0 — prevents exploding gradients which are common in transformer fine-tuning
- Trained on synthetically generated traffic sentences

---

### 4.3 BERT Text Classifier

**Task:** Given one incident text, predict one of 4 classes:
- accident
- jam
- road_closure
- normal

**Model:** `BertForSequenceClassification`
BERT encoder + linear head on the `[CLS]` token's final hidden state.

**Why [CLS] token:**
BERT prepends [CLS] to every input. After 12 transformer layers of bidirectional self-attention, this token attends to every other token in the sentence. Its final 768-dimensional vector is a compressed representation of the entire sentence — specifically designed for classification.

**Bidirectional self-attention — why it matters:**
BERT reads the entire sentence left-to-right AND right-to-left simultaneously. It understands that "not" before "clear" changes meaning completely. TF-IDF or left-to-right models miss this.

**Why not TF-IDF + Logistic Regression:**
TF-IDF is bag of words — order does not matter. "Road is clear" and "Road is not clear" produce nearly identical TF-IDF vectors. BERT's attention mechanism captures negation, context, and word relationships.

**Loss:** CrossEntropyLoss — applies softmax + negative log likelihood. Correct for single-label classification (one incident type per text).

**Why CrossEntropy here but BCE for vision:**
Text classification → one class per sample → CrossEntropy
Image classification → multiple labels per image → BCE per label

**Why two separate BERTs instead of one multi-task model:**
- NER output shape: `(sequence_length, 7)` — one label per token
- Classifier output shape: `(1, 4)` — one label per sentence
- They need different heads, different loss functions, different training dynamics
- Keeping them separate: each is independently trainable, debuggable, and replaceable without affecting the other

---

### 4.4 Alert Generator

**Inputs from both pipelines:**
- incident_type ← BERT Classifier
- location ← BERT NER (LOCATION entity)
- severity ← BERT NER (SEVERITY entity)
- vision_labels ← ResNet18 (rain, night, congestion, clear)

**Mode 1 — Claude API (Few-Shot Prompting):**

Sends a prompt containing 4 fully worked input→output examples to Claude. Claude generalises the pattern from those examples and generates the alert for the new input. No model training required.

```
Input: {accident, Sector 62, major, [rain, congestion]}
Output: 🚨 Major accident at Sector 62. Rain and congestion reported. Avoid route.
```

**Why few-shot over fine-tuning a generative model:**
- Fine-tuning needs hundreds of labelled alert examples — we have none
- Requires GPU compute and a full training pipeline
- 4 examples in a prompt achieves the same output quality
- New incident types only need a new example added to the prompt — no retraining

**Mode 2 — Template Fallback:**

Deterministic templates with placeholder filling. Vision labels map to fixed phrases:
```
rain       → "Rain conditions detected. "
night      → "Night-time visibility low. "
congestion → "Heavy congestion visible. "
clear      → ""
```

Four template banks (one per incident type), random template selected per call.

**Why templates at all:**
Guarantees the system always produces output — offline, API down, invalid key, rate limited. No single point of failure.

**Error handling:**
Any exception from Claude API (network error, rate limit, invalid key) is caught silently. Falls back to template automatically. The API endpoint never returns an error to the user.

---

## 5. FastAPI — Integration Layer

Four endpoints:

| Endpoint | Method | Input | Output |
|---|---|---|---|
| `/health` | GET | — | Device status |
| `/predict-image` | POST | Image file + threshold | Label probabilities + detected labels |
| `/analyze-text` | POST | JSON text | Entities + class + spaCy analysis |
| `/generate-alert` | POST | JSON structured data | Alert text + method used |
| `/full-pipeline` | POST | Image + text | All outputs combined |

**Lazy model loading with `@lru_cache`:**
Models are not loaded at server startup — they load on the first request and are cached for all subsequent requests. Startup is instant. Memory is only used when the endpoint is actually called.

**CORS middleware:**
Allows cross-origin requests — the dashboard (port 8501) can call the API (port 8000) from the browser without being blocked.

---

## 6. Dashboard — Streamlit

Five tabs:

| Tab | What it shows |
|---|---|
| Full Pipeline | Upload image + paste text → runs all modules → shows labels, entities, alert |
| Vision | Image upload → per-label confidence bars + Grad-CAM heatmap |
| NLP | Text input → entity chips + token table + incident class probabilities |
| Alert Generator | Manual form → instant alert with examples for all 4 types |
| About | Architecture summary + quick start commands |

Dark theme. No animations. Consistent card-based layout.

---

## 7. Key Design Decisions — Why Not Something Else

| Decision Made | Alternative Considered | Why We Chose Ours |
|---|---|---|
| ResNet18 | VGG, ResNet50 | Lighter, skip connections, sufficient for 4 labels |
| Partial fine-tuning | Full fine-tuning | Too little data to retrain early layers |
| BCEWithLogitsLoss | BCELoss, CrossEntropy | Numerically stable; multi-label requires independent sigmoid |
| Adam | SGD with momentum | Faster convergence, less tuning for partial fine-tuning |
| CosineAnnealingLR | StepLR | Smooth decay, no abrupt loss spikes |
| Grad-CAM on layer4 | Earlier layers, LIME, attention | Deepest semantic layer; no extra model needed |
| F1 + EMR + Hamming | Accuracy | Accuracy is misleading for multi-label |
| spaCy | NLTK | Single optimised pipeline, faster, better base NER |
| BERT NER | spaCy NER, regex, CRF | Domain entities not in standard NER; BERT learns context |
| BERT Classifier | TF-IDF + LogReg | Bidirectional context, handles negation and word order |
| Two separate BERTs | Multi-task BERT | Different output shapes, loss functions, training dynamics |
| Few-shot alert | Fine-tuned generative model | No training data; 4 examples sufficient; no compute needed |
| Template fallback | No fallback | System must never fail; offline reliability |

---

## 8. Project File Structure

```
traffic-system/
├── vision/
│   ├── model.py        ResNet18 classifier
│   ├── train.py        Training loop
│   ├── dataset.py      Dataset + augmentation + dummy data
│   ├── metrics.py      F1, Exact Match, Hamming Loss
│   ├── gradcam.py      Grad-CAM heatmap generation
│   └── predict.py      Inference + batch prediction
│
├── nlp/
│   ├── preprocess.py   spaCy pipeline
│   ├── ner_model.py    BERT NER fine-tuning + inference
│   ├── classifier.py   BERT text classifier
│   └── dataset.py      NER + CLS datasets + synthetic data
│
├── alerts/
│   └── generator.py    Few-shot Claude + template fallback
│
├── api/
│   └── main.py         FastAPI — 4 endpoints
│
├── dashboard/
│   └── app.py          Streamlit UI — 5 tabs
│
├── notebooks/
│   └── exploration.ipynb
│
├── requirements.txt
└── README.md
```

---

## 9. Quick Reference — Formulas

**Sigmoid:**
```
σ(x) = 1 / (1 + e^(-x))
```

**BCEWithLogitsLoss:**
```
L = -[ y·log(σ(x)) + (1-y)·log(1-σ(x)) ]
```

**F1 Score:**
```
F1 = 2·(P·R) / (P+R)    where P = TP/(TP+FP),  R = TP/(TP+FN)
```

**Exact Match Ratio:**
```
EMR = |{i : ŷᵢ = yᵢ}| / N
```

**Hamming Loss:**
```
HL = (1/N·L) · Σ Σ |ŷᵢⱼ - yᵢⱼ|
```

**CosineAnnealingLR:**
```
lr(t) = η_min + 0.5·(η_max - η_min)·(1 + cos(π·t/T))
```

**Grad-CAM:**
```
weights_c  = (1/Z) · Σ Σ (∂y^c / ∂A^k_ij)
CAM        = ReLU( Σ_k  weights_k · A^k )
```

---

## 10. Glossary

| Term | Meaning |
|---|---|
| Multi-label | One sample can have multiple correct labels simultaneously |
| Sigmoid | Squashes any value to (0,1) — used for independent binary decisions |
| Softmax | Squashes values to sum to 1 — used for single-class competition |
| BCEWithLogitsLoss | Binary cross entropy with built-in numerically stable sigmoid |
| Transfer learning | Using weights pretrained on one task as starting point for another |
| Fine-tuning | Continuing to train a pretrained model on a new dataset |
| Frozen layers | Layers whose weights are locked — not updated during training |
| BIO tagging | Labelling scheme for NER: Beginning, Inside, Outside |
| Subword tokenization | BERT splits words into frequent subword pieces |
| [CLS] token | Special BERT token whose final state represents the whole sentence |
| Grad-CAM | Gradient-weighted Class Activation Map — explains CNN predictions visually |
| Few-shot prompting | Giving a model a few examples in the prompt to guide its output |
| Hamming Loss | Fraction of incorrectly predicted label slots in multi-label classification |
| Exact Match Ratio | Fraction of samples where every label is predicted correctly |
| AdamW | Adam optimizer with decoupled weight decay regularisation |
| Warmup | Gradually increasing learning rate at the start of BERT fine-tuning |
| Gradient clipping | Capping gradient magnitude to prevent exploding gradients |
| lru_cache | Python decorator that caches function results — used for lazy model loading |
