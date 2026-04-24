"""
Smoke test — verifies every module can be imported and runs its core logic
without requiring trained checkpoints or a GPU.

Run from the traffic-system/ directory:
    python smoke_test.py
"""

import sys
import traceback

PASS = "  ✓"
FAIL = "  ✗"


def section(title: str):
    print(f"\n{'─'*50}\n{title}\n{'─'*50}")


def run(name: str, fn):
    try:
        fn()
        print(f"{PASS} {name}")
    except Exception:
        print(f"{FAIL} {name}")
        traceback.print_exc()


# ── Vision ──────────────────────────────────────────────────────────────────

section("VISION")

import torch

def test_model():
    from vision.model import build_model
    m = build_model(pretrained=False)
    out = m(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 4)

def test_dataset():
    from vision.dataset import generate_dummy_annotations, build_dataloaders
    import tempfile, os
    with tempfile.TemporaryDirectory() as tmp:
        train_ann, val_ann = generate_dummy_annotations(tmp, n_train=10, n_val=4, img_size=64)
        tl, vl = build_dataloaders(train_ann, val_ann, batch_size=4, num_workers=0)
        imgs, labels = next(iter(tl))
        assert imgs.shape == (4, 3, 224, 224)

def test_metrics():
    from vision.metrics import compute_metrics
    preds = torch.randn(16, 4)
    targets = torch.randint(0, 2, (16, 4)).float()
    m = compute_metrics(preds, targets)
    assert "f1_macro" in m and "hamming_loss" in m

def test_gradcam():
    from vision.model import build_model
    from vision.gradcam import GradCAM
    import numpy as np
    model = build_model(pretrained=False)
    layer = model.backbone.layer4[-1].conv2
    gc = GradCAM(model, layer)
    inp = torch.randn(1, 3, 224, 224)
    cam, idx = gc.generate(inp)
    assert cam.shape == (7, 7) or cam.ndim == 2

run("model.py — forward pass", test_model)
run("dataset.py — dummy data + dataloader", test_dataset)
run("metrics.py — compute all metrics", test_metrics)
run("gradcam.py — cam generation", test_gradcam)


# ── NLP ─────────────────────────────────────────────────────────────────────

section("NLP")

def test_preprocess():
    from nlp.preprocess import parse_document, extract_keywords
    doc = parse_document("Accident on NH-8 near Sector 62 causing heavy congestion.")
    assert doc.sentences
    kw = extract_keywords("Major collision on highway")
    assert isinstance(kw, list)

def test_nlp_dataset():
    from nlp.dataset import generate_ner_sample, generate_cls_samples
    s = generate_ner_sample()
    assert len(s.tokens) == len(s.ner_tags)
    texts, labels = generate_cls_samples(n_per_class=2)
    assert len(texts) == len(labels) == 8

run("preprocess.py — parse + keywords", test_preprocess)
run("dataset.py — NER + CLS samples", test_nlp_dataset)


# ── Alerts ──────────────────────────────────────────────────────────────────

section("ALERTS")

def test_alert_template():
    from alerts.generator import generate_alert
    r = generate_alert("accident", "Sector 62", "major", ["rain", "congestion"], force_template=True)
    assert r["method"] == "template"
    assert "Sector 62" in r["alert_text"]
    assert len(r["alert_text"]) > 10

def test_alert_all_types():
    from alerts.generator import generate_alert
    for inc in ("accident", "jam", "road_closure", "normal"):
        r = generate_alert(inc, "Test Road", force_template=True)
        assert r["alert_text"]

run("generator.py — template alert", test_alert_template)
run("generator.py — all incident types", test_alert_all_types)


# ── Summary ──────────────────────────────────────────────────────────────────

print("\n" + "═"*50)
print("Smoke test complete.")
print("NOTE: NER and Classifier models require 'transformers' and")
print("      a BERT download — run their --train scripts separately.")
print("═"*50)
