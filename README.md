# Traffic Scene Understanding with Incident NLP Alert System

An end-to-end, production-quality system combining **computer vision** and **NLP** to
automatically detect traffic conditions, extract incident information from text, and
generate human-readable alerts.

---

## Problem Statement

Urban traffic management requires real-time awareness of road conditions. This system
addresses two complementary data sources:

1. **Camera feeds / images** — What are the current road conditions? (rain, night, congestion, clear)
2. **Incident reports / text feeds** — What happened, where, and how severe?

By fusing both signals, the system generates actionable alerts that drivers and traffic
management centres can act on immediately.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         INPUT LAYER                             │
│   Traffic Image                  Incident Text Report           │
└────────────┬────────────────────────────┬───────────────────────┘
             │                            │
             ▼                            ▼
┌────────────────────┐        ┌───────────────────────────────┐
│   VISION MODULE    │        │        NLP PIPELINE           │
│                    │        │                               │
│ ResNet18 backbone  │        │  spaCy: tokenise, POS, deps   │
│ + 4-label head     │        │  BERT NER: LOC/INC/SEV        │
│ BCEWithLogitsLoss  │        │  BERT Classifier: 4 classes   │
│ Grad-CAM explain.  │        │                               │
└────────────┬───────┘        └──────────────┬────────────────┘
             │                               │
             │  detected_labels              │  entities + class
             └──────────────┬────────────────┘
                            ▼
               ┌────────────────────────┐
               │    ALERT GENERATOR     │
               │                        │
               │ Claude (few-shot) or   │
               │ Template fallback      │
               └────────────┬───────────┘
                            │
                            ▼
               ┌────────────────────────┐
               │      FastAPI REST      │
               │  /predict-image        │
               │  /analyze-text         │
               │  /generate-alert       │
               │  /full-pipeline        │
               └────────────────────────┘
```

---

## Folder Structure

```
traffic-system/
├── vision/
│   ├── model.py        ResNet18 multi-label classifier
│   ├── train.py        Training loop with CosineAnnealingLR
│   ├── dataset.py      Dataset, augmentation, dummy data generator
│   ├── metrics.py      F1, exact match, Hamming loss
│   ├── gradcam.py      Grad-CAM explainability
│   └── predict.py      Inference + batch prediction
│
├── nlp/
│   ├── preprocess.py   spaCy tokenisation, POS, dependency parsing
│   ├── ner_model.py    BERT NER fine-tuning (LOC, INCIDENT, SEVERITY)
│   ├── classifier.py   BERT sequence classifier (4 traffic classes)
│   └── dataset.py      NER + classification datasets + synthetic data
│
├── alerts/
│   └── generator.py    Few-shot Claude alert + template fallback
│
├── api/
│   └── main.py         FastAPI app with 4 endpoints
│
├── data/               Raw images and text files
├── models/             Saved checkpoints
├── notebooks/          Exploration notebooks
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Install dependencies

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Train the vision model

```bash
# Generates dummy data automatically if no annotation files are found
python -m vision.train --epochs 20 --batch_size 32 --data_dir data
```

### 3. Train the NLP models

```bash
# NER model
python -m nlp.ner_model --train --epochs 5 --n_samples 500

# Text classifier
python -m nlp.classifier --train --epochs 5 --n_per_class 150
```

### 4. Start the API server

```bash
# Optional: set your API key for Claude-powered alerts
export ANTHROPIC_API_KEY=sk-ant-...

uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

API docs are automatically available at `http://localhost:8000/docs`

---

## Usage

### Predict from an image

```bash
curl -X POST http://localhost:8000/predict-image \
  -F "file=@/path/to/traffic.jpg" \
  -F "threshold=0.5"
```

Response:
```json
{
  "detected_labels": ["rain", "congestion"],
  "probabilities": {"rain": 0.82, "night": 0.12, "congestion": 0.91, "clear": 0.03},
  "threshold": 0.5
}
```

### Analyze incident text

```bash
curl -X POST http://localhost:8000/analyze-text \
  -H "Content-Type: application/json" \
  -d '{"text": "Major accident at Sector 62 causing severe congestion on NH-8"}'
```

Response:
```json
{
  "entities": [
    {"text": "Sector 62", "label": "LOCATION"},
    {"text": "accident", "label": "INCIDENT_TYPE"},
    {"text": "severe", "label": "SEVERITY"}
  ],
  "incident_class": "accident",
  "class_probabilities": {"accident": 0.91, "jam": 0.06, "road_closure": 0.02, "normal": 0.01},
  "spacy_analysis": { ... }
}
```

### Generate an alert

```bash
curl -X POST http://localhost:8000/generate-alert \
  -H "Content-Type: application/json" \
  -d '{
    "incident_type": "accident",
    "location": "Sector 62",
    "severity": "major",
    "vision_labels": ["rain", "congestion"]
  }'
```

Response:
```json
{
  "alert_text": "🚨 Major accident at Sector 62. Rain and heavy congestion reported. Avoid this route.",
  "method": "template",
  ...
}
```

### Full pipeline (image + text → alert)

```bash
curl -X POST http://localhost:8000/full-pipeline \
  -F "file=@traffic.jpg" \
  -F "text=Accident near Sector 62 with heavy vehicles involved" \
  -F "threshold=0.5"
```

---

## CLI Prediction

```bash
# Single image inference with optional Grad-CAM
python -m vision.predict --image data/images/test.jpg --checkpoint models/best_vision_model.pt --gradcam

# NER prediction
python -m nlp.ner_model --predict "Severe accident at NH-8 near Sector 62"

# Text classification
python -m nlp.classifier --predict "Road closure on MG Road due to construction"

# Alert generation (template mode)
python -c "
from alerts.generator import generate_alert
r = generate_alert('accident', 'Sector 62', 'major', ['rain', 'congestion'], force_template=True)
print(r['alert_text'])
"
```

---

## Example Outputs

| Scenario | Alert |
|---|---|
| Image: rain + congestion / Text: accident at Sector 62 | 🚨 Major accident at Sector 62. Rain and heavy congestion reported. Avoid this route. |
| Image: clear / Text: traffic jam on NH-8 | ⚠️ Heavy traffic jam on NH-8. Expect significant delays. Consider alternate routes. |
| Image: night / Text: road closure on MG Road | 🚧 Road closure on MG Road. Night-time visibility low. Please divert. |
| Image: clear / Text: normal traffic | ✅ Traffic flowing normally. No disruptions reported. |

---

## Key Design Decisions

- **Multi-label vision** — scenes can have multiple concurrent conditions (rain AND night AND congestion)
- **BCEWithLogitsLoss** — numerically stable binary cross-entropy; sigmoid applied at inference only
- **Partial fine-tuning** — ResNet18 layer4 + fc unfrozen; earlier layers frozen to prevent catastrophic forgetting
- **Grad-CAM** — per-label explanations show which image regions drive each prediction
- **Separate NER + classifier** — NER extracts structured entities; classifier determines intent class
- **Alert fallback** — Claude API used when key is available; deterministic templates ensure the system always produces output

---

## License

MIT
