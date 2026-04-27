from __future__ import annotations

import io
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

from vision.model import build_model, LABELS as VISION_LABELS
from vision.dataset import get_val_transforms
from vision.predict import load_model as load_vision_model
from nlp.preprocess import parse_document, document_to_dict
from nlp.ner_model import TrafficNERModel
from nlp.classifier import TrafficTextClassifier
from alerts.generator import generate_alert


app = FastAPI(
    title="Traffic Scene Understanding API",
    description="Multi-modal traffic analysis: vision + NLP + alert generation",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELS_DIR = Path(os.environ.get("MODELS_DIR", "models"))


@lru_cache(maxsize=1)
def get_vision_model():
    ckpt = str(MODELS_DIR / "best_vision_model.pt")
    return load_vision_model(ckpt if Path(ckpt).exists() else None, DEVICE)


@lru_cache(maxsize=1)
def get_ner_model():
    ckpt = str(MODELS_DIR / "ner_model.pt")
    return TrafficNERModel(
        checkpoint_path=ckpt if Path(ckpt).exists() else None,
        device=DEVICE,
    )


@lru_cache(maxsize=1)
def get_cls_model():
    ckpt = str(MODELS_DIR / "cls_model.pt")
    return TrafficTextClassifier(
        checkpoint_path=ckpt if Path(ckpt).exists() else None,
        device=DEVICE,
    )


class TextRequest(BaseModel):
    text: str


class AlertRequest(BaseModel):
    incident_type: str
    location: str
    severity: Optional[str] = None
    vision_labels: Optional[List[str]] = None
    api_key: Optional[str] = None


class VisionPrediction(BaseModel):
    detected_labels: List[str]
    probabilities: Dict[str, float]
    threshold: float


class NLPAnalysis(BaseModel):
    entities: List[Dict]
    incident_class: str
    class_probabilities: Dict[str, float]
    spacy_analysis: Dict


class AlertResponse(BaseModel):
    alert_text: str
    method: str
    incident_type: str
    location: str
    severity: Optional[str]
    vision_labels: List[str]


@app.get("/health")
def health_check():
    return {"status": "ok", "device": str(DEVICE)}


@app.post("/predict-image", response_model=VisionPrediction)
async def predict_image(
    file: UploadFile = File(...),
    threshold: float = Form(0.5),
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        contents = await file.read()
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not decode image: {e}")

    transform = get_val_transforms()
    tensor = transform(pil_img).unsqueeze(0).to(DEVICE)

    model = get_vision_model()
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits).squeeze(0).cpu().tolist()

    prob_dict = {label: round(p, 4) for label, p in zip(VISION_LABELS, probs)}
    detected = [lbl for lbl, p in prob_dict.items() if p >= threshold]

    return VisionPrediction(detected_labels=detected, probabilities=prob_dict, threshold=threshold)


@app.post("/analyze-text", response_model=NLPAnalysis)
async def analyze_text(request: TextRequest):
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    parsed = parse_document(text)
    spacy_data = document_to_dict(parsed)

    ner_model = get_ner_model()
    entities = ner_model.predict(text)

    cls_model = get_cls_model()
    cls_result = cls_model.predict(text)

    return NLPAnalysis(
        entities=entities,
        incident_class=cls_result["predicted_class"],
        class_probabilities=cls_result["probabilities"],
        spacy_analysis=spacy_data,
    )


@app.post("/generate-alert", response_model=AlertResponse)
async def generate_traffic_alert(request: AlertRequest):
    valid_types = {"accident", "jam", "road_closure", "normal"}
    if request.incident_type not in valid_types:
        raise HTTPException(status_code=400, detail=f"incident_type must be one of: {valid_types}")

    result = generate_alert(
        incident_type=request.incident_type,
        location=request.location,
        severity=request.severity,
        vision_labels=request.vision_labels or [],
        api_key=request.api_key,
    )

    return AlertResponse(**result)


@app.post("/full-pipeline")
async def full_pipeline(
    file: UploadFile = File(...),
    text: str = Form(...),
    threshold: float = Form(0.5),
    api_key: Optional[str] = Form(None),
):
    contents = await file.read()
    pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
    transform = get_val_transforms()
    tensor = transform(pil_img).unsqueeze(0).to(DEVICE)
    model = get_vision_model()

    with torch.no_grad():
        probs = torch.sigmoid(model(tensor)).squeeze(0).cpu().tolist()

    prob_dict = {label: round(p, 4) for label, p in zip(VISION_LABELS, probs)}
    detected_vision = [lbl for lbl, p in prob_dict.items() if p >= threshold]

    ner_model = get_ner_model()
    cls_model = get_cls_model()
    entities = ner_model.predict(text)
    cls_result = cls_model.predict(text)

    location = next((e["text"] for e in entities if e["label"] == "LOCATION"), "Unknown location")
    severity = next((e["text"] for e in entities if e["label"] == "SEVERITY"), None)

    alert = generate_alert(
        incident_type=cls_result["predicted_class"],
        location=location,
        severity=severity,
        vision_labels=detected_vision,
        api_key=api_key,
    )

    return {
        "vision": {"detected_labels": detected_vision, "probabilities": prob_dict},
        "nlp": {
            "entities": entities,
            "incident_class": cls_result["predicted_class"],
            "class_probabilities": cls_result["probabilities"],
        },
        "alert": alert,
    }
