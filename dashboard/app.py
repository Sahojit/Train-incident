"""
Traffic Scene Understanding — Streamlit Dashboard
Run: streamlit run dashboard/app.py
"""

import io
import sys
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vision.model import build_model, LABELS as VISION_LABELS
from vision.dataset import get_val_transforms
from vision.gradcam import GradCAM, overlay_heatmap
from alerts.generator import generate_alert

try:
    from nlp.preprocess import parse_document, extract_keywords
    NLP_AVAILABLE = True
except Exception as e:
    NLP_AVAILABLE = False
    _NLP_MSG = str(e)


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Traffic Intelligence",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global styles ─────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* ── Reset Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem 2rem; max-width: 1200px; }

/* ── Body / background ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0e1117;
    color: #e0e4ef;
    font-family: 'Inter', 'Segoe UI', sans-serif;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #13151f;
    border-right: 1px solid #1f2233;
}
[data-testid="stSidebar"] .block-container { padding: 1.5rem 1rem; }

/* ── Tabs ── */
[data-testid="stTabs"] button {
    font-size: 0.85rem;
    font-weight: 500;
    color: #7a83a6;
    padding: 0.5rem 1rem;
    border-bottom: 2px solid transparent;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #e0e4ef;
    border-bottom: 2px solid #4f7dff;
}

/* ── Cards ── */
.card {
    background: #161b2e;
    border: 1px solid #1f2640;
    border-radius: 10px;
    padding: 1.25rem 1.4rem;
    margin-bottom: 1rem;
}

/* ── Section label ── */
.sec-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: #5a6285;
    margin-bottom: 0.6rem;
}

/* ── Label row (vision detection) ── */
.label-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.65rem 0.9rem;
    border-radius: 8px;
    margin-bottom: 0.45rem;
    border: 1px solid #1f2640;
    background: #12162a;
}
.label-name { font-size: 0.9rem; font-weight: 500; }
.label-badge-on  { font-size: 0.72rem; font-weight: 600; color: #4ade80; background: #0d2b1a; padding: 2px 10px; border-radius: 20px; border: 1px solid #1a4a30; }
.label-badge-off { font-size: 0.72rem; font-weight: 600; color: #4a5270; background: #12162a; padding: 2px 10px; border-radius: 20px; border: 1px solid #1f2640; }

/* ── Confidence bar ── */
.bar-track { background: #1a1e35; border-radius: 4px; height: 6px; margin-top: 5px; }
.bar-fill  { height: 6px; border-radius: 4px; }

/* ── Alert box ── */
.alert-accident { background:#1c1014; border:1px solid #e53e3e; border-left:4px solid #e53e3e; border-radius:8px; padding:1rem 1.2rem; color:#fed7d7; font-size:1rem; line-height:1.6; margin-bottom:0.8rem; }
.alert-jam      { background:#1c1a0e; border:1px solid #d69e2e; border-left:4px solid #d69e2e; border-radius:8px; padding:1rem 1.2rem; color:#fefcbf; font-size:1rem; line-height:1.6; margin-bottom:0.8rem; }
.alert-closure  { background:#0e1c1c; border:1px solid #38b2ac; border-left:4px solid #38b2ac; border-radius:8px; padding:1rem 1.2rem; color:#e6fffa; font-size:1rem; line-height:1.6; margin-bottom:0.8rem; }
.alert-normal   { background:#0e1c12; border:1px solid #48bb78; border-left:4px solid #48bb78; border-radius:8px; padding:1rem 1.2rem; color:#f0fff4; font-size:1rem; line-height:1.6; margin-bottom:0.8rem; }

/* ── Entity chip ── */
.chip { display:inline-block; padding:3px 10px; border-radius:5px; font-size:0.75rem; font-weight:600; margin:2px 3px; }
.chip-loc  { background:#1a2c48; color:#90cdf4; border:1px solid #2a4a72; }
.chip-inc  { background:#3b1515; color:#feb2b2; border:1px solid #6b2020; }
.chip-sev  { background:#3b2a0a; color:#fbd38d; border:1px solid #6b4a14; }

/* ── Metric tile ── */
.tile { background:#161b2e; border:1px solid #1f2640; border-radius:8px; padding:0.9rem 1rem; text-align:center; }
.tile-val { font-size:1.5rem; font-weight:700; color:#e0e4ef; }
.tile-lbl { font-size:0.7rem; color:#5a6285; text-transform:uppercase; letter-spacing:1px; margin-top:2px; }

/* ── Divider ── */
hr { border-color: #1f2640; margin: 1.2rem 0; }

/* ── Streamlit widget tweaks ── */
[data-testid="stTextArea"] textarea,
[data-testid="stTextInput"] input {
    background: #12162a;
    border: 1px solid #1f2640;
    color: #e0e4ef;
    border-radius: 7px;
}
[data-testid="stFileUploader"] {
    background: #12162a;
    border: 1px dashed #2a3050;
    border-radius: 8px;
}
button[kind="primary"] {
    background: #4f7dff;
    border: none;
    border-radius: 7px;
    font-weight: 600;
}
button[kind="primary"]:hover { background: #3d6be0; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

LABEL_ICONS  = {"rain": "🌧️", "night": "🌙", "congestion": "🚗", "clear": "☀️"}
CLASS_ICONS  = {"accident": "🚨", "jam": "⚠️", "road_closure": "🚧", "normal": "✅"}
COLOR_MAP    = {"rain": "#4299e1", "night": "#9f7aea", "congestion": "#f6ad55", "clear": "#68d391"}
ALERT_CLASS  = {"accident": "alert-accident", "jam": "alert-jam", "road_closure": "alert-closure", "normal": "alert-normal"}
CLS_COLORS   = {"accident": "#fc8181", "jam": "#f6ad55", "road_closure": "#76e4f7", "normal": "#68d391"}


# ── Cached loaders ────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_vision():
    m = build_model(pretrained=False)
    ckpt = MODELS_DIR / "best_vision_model.pt"
    if ckpt.exists():
        m.load_state_dict(torch.load(str(ckpt), map_location=DEVICE)["model_state_dict"])
    return m.to(DEVICE).eval()


@st.cache_resource(show_spinner=False)
def load_ner():
    if not NLP_AVAILABLE:
        return None
    from nlp.ner_model import TrafficNERModel
    ckpt = str(MODELS_DIR / "ner_model.pt")
    return TrafficNERModel(checkpoint_path=ckpt if Path(ckpt).exists() else None, device=DEVICE)


@st.cache_resource(show_spinner=False)
def load_cls():
    if not NLP_AVAILABLE:
        return None
    from nlp.classifier import TrafficTextClassifier
    ckpt = str(MODELS_DIR / "cls_model.pt")
    return TrafficTextClassifier(checkpoint_path=ckpt if Path(ckpt).exists() else None, device=DEVICE)


# ── Helpers ───────────────────────────────────────────────────────────────────

def run_vision(pil_img, threshold):
    model = load_vision()
    t = get_val_transforms()(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.sigmoid(model(t)).squeeze(0).cpu().tolist()
    prob_dict = {l: round(p, 4) for l, p in zip(VISION_LABELS, probs)}
    detected  = [l for l, p in prob_dict.items() if p >= threshold]
    return prob_dict, detected, model


def run_gradcam(model, pil_img, label_idx):
    gc  = GradCAM(model, model.backbone.layer4[-1].conv2)
    t   = get_val_transforms()(pil_img).unsqueeze(0)
    cam, _ = gc.generate(t, class_idx=label_idx)
    overlay = overlay_heatmap(np.array(pil_img.resize((224, 224))), cam, alpha=0.45)
    return Image.fromarray(overlay)


def run_nlp(text):
    if not NLP_AVAILABLE:
        return None, [], {"predicted_class": "normal", "probabilities": {}}
    parsed   = parse_document(text)
    entities = load_ner().predict(text) if load_ner() else []
    cls_res  = load_cls().predict(text) if load_cls() else {"predicted_class": "normal", "probabilities": {}}
    return parsed, entities, cls_res


def render_label_rows(prob_dict, threshold):
    for lbl in VISION_LABELS:
        p   = prob_dict[lbl]
        det = p >= threshold
        col = COLOR_MAP[lbl] if det else "#2a3050"
        pct = int(p * 100)
        badge = f"<span class='label-badge-on'>✓ detected</span>" if det else f"<span class='label-badge-off'>absent</span>"
        st.markdown(f"""
        <div class='label-row'>
          <span class='label-name'>{LABEL_ICONS[lbl]} {lbl}</span>
          <div style='flex:1;margin:0 1rem'>
            <div class='bar-track'>
              <div class='bar-fill' style='width:{pct}%;background:{col}'></div>
            </div>
          </div>
          <span style='font-size:0.8rem;color:#7a83a6;min-width:38px;text-align:right'>{p:.2f}</span>
          &nbsp;{badge}
        </div>""", unsafe_allow_html=True)


def render_alert(text, inc_type):
    css = ALERT_CLASS.get(inc_type, "alert-normal")
    st.markdown(f"<div class='{css}'>{text}</div>", unsafe_allow_html=True)


def render_entities(entities):
    if not entities:
        st.markdown("<span style='color:#4a5270;font-size:0.85rem'>No entities detected</span>", unsafe_allow_html=True)
        return
    chips = []
    for e in entities:
        css = {"LOCATION": "chip-loc", "INCIDENT_TYPE": "chip-inc", "SEVERITY": "chip-sev"}.get(e["label"], "chip-loc")
        chips.append(f"<span class='chip {css}'>{e['label']}: {e['text']}</span>")
    st.markdown(" ".join(chips), unsafe_allow_html=True)


def render_prob_bars(probs):
    for lbl, p in sorted(probs.items(), key=lambda x: -x[1]):
        col   = CLS_COLORS.get(lbl, "#4a5270")
        pct   = int(p * 100)
        st.markdown(f"""
        <div style='margin-bottom:0.5rem'>
          <div style='display:flex;justify-content:space-between;font-size:0.82rem;color:#a0aac8;margin-bottom:3px'>
            <span>{lbl}</span><span>{p:.3f}</span>
          </div>
          <div class='bar-track'>
            <div class='bar-fill' style='width:{pct}%;background:{col}'></div>
          </div>
        </div>""", unsafe_allow_html=True)


def sec(title):
    st.markdown(f"<div class='sec-label'>{title}</div>", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🚦 Traffic Intelligence")
    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown("**Vision settings**")
    threshold    = st.slider("Detection threshold", 0.1, 0.9, 0.5, 0.05)
    show_gradcam = st.checkbox("Grad-CAM heatmap", value=True)
    if show_gradcam:
        gradcam_label = st.selectbox("Heatmap for label", VISION_LABELS, index=2)
    else:
        gradcam_label = VISION_LABELS[2]

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("**Alert settings**")
    api_key        = st.text_input("Anthropic API key", type="password", placeholder="sk-ant-… (optional)")
    force_template = st.checkbox("Use template mode", value=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    nlp_ok = "🟢 Ready" if NLP_AVAILABLE else "🔴 Not found"
    st.markdown(f"""
    <div style='font-size:0.75rem;color:#5a6285;line-height:1.8'>
      Device &nbsp;<strong style='color:#a0aac8'>{DEVICE}</strong><br>
      NLP &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<strong style='color:#a0aac8'>{nlp_ok}</strong>
    </div>""", unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────────────────────────

st.markdown("""
<div style='padding:1.4rem 1.6rem 1.2rem;background:#161b2e;border:1px solid #1f2640;
            border-radius:10px;margin-bottom:1.5rem;display:flex;align-items:center;gap:1rem'>
  <div style='font-size:2.2rem'>🚦</div>
  <div>
    <div style='font-size:1.3rem;font-weight:700;color:#e0e4ef'>Traffic Scene Intelligence</div>
    <div style='font-size:0.83rem;color:#5a6285;margin-top:2px'>
      Multi-label vision · NER &amp; text classification · AI alert generation
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_full, tab_vision, tab_nlp, tab_alert, tab_about = st.tabs([
    "Full Pipeline", "Vision", "NLP", "Alert Generator", "About"
])


# ══════════════════════════════════════════════════════════════════════════════
# Full Pipeline
# ══════════════════════════════════════════════════════════════════════════════

with tab_full:
    left, right = st.columns([1, 1], gap="large")

    with left:
        sec("Input")
        uploaded = st.file_uploader("Upload traffic image", type=["jpg","jpeg","png"], key="fp_img",
                                    label_visibility="collapsed")
        incident_text = st.text_area("Incident report", height=90, key="fp_text",
            value="A major accident occurred on NH-8 near Sector 62 causing severe congestion.",
            label_visibility="collapsed")
        run_btn = st.button("Run pipeline", type="primary", use_container_width=True)

    with right:
        sec("Output")
        result_slot = st.empty()

    if run_btn:
        if not uploaded:
            st.warning("Upload an image to continue.")
        else:
            pil_img = Image.open(uploaded).convert("RGB")

            with st.spinner("Vision…"):
                prob_dict, detected, vis_model = run_vision(pil_img, threshold)
            with st.spinner("NLP…"):
                parsed, entities, cls_res = run_nlp(incident_text)
            inc_type = cls_res["predicted_class"] if cls_res else "normal"
            location = next((e["text"] for e in entities if e["label"] == "LOCATION"), "Unknown")
            severity = next((e["text"] for e in entities if e["label"] == "SEVERITY"), None)
            with st.spinner("Generating alert…"):
                alert = generate_alert(inc_type, location, severity, detected,
                                       api_key=api_key or None, force_template=force_template)

            with left:
                st.image(pil_img, use_container_width=True)

            with right:
                render_alert(alert["alert_text"], inc_type)

                st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
                sec("Vision labels")
                render_label_rows(prob_dict, threshold)

                if NLP_AVAILABLE:
                    st.markdown("<div style='height:0.2rem'></div>", unsafe_allow_html=True)
                    sec("Entities")
                    render_entities(entities)

                    icon = CLASS_ICONS.get(inc_type, "")
                    sec("Incident class")
                    st.markdown(
                        f"<span style='font-size:1rem;font-weight:600;color:#e0e4ef'>"
                        f"{icon} {inc_type}</span>"
                        f"<span style='font-size:0.72rem;color:#4a5270;margin-left:0.6rem'>"
                        f"via {alert['method']}</span>",
                        unsafe_allow_html=True,
                    )

            if show_gradcam:
                st.markdown("<hr>", unsafe_allow_html=True)
                sec(f"Grad-CAM — {gradcam_label}")
                c1, c2, _, _ = st.columns([1, 1, 1, 1])
                with st.spinner("Computing heatmap…"):
                    cam_img = run_gradcam(vis_model, pil_img, VISION_LABELS.index(gradcam_label))
                c1.image(pil_img.resize((224, 224)), caption="Original", use_container_width=True)
                c2.image(cam_img, caption=f"Grad-CAM: {gradcam_label}", use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# Vision
# ══════════════════════════════════════════════════════════════════════════════

with tab_vision:
    v_img = st.file_uploader("Upload traffic image", type=["jpg","jpeg","png"], key="v_img")

    if v_img:
        pil_img = Image.open(v_img).convert("RGB")
        with st.spinner("Running model…"):
            prob_dict, detected, vis_model = run_vision(pil_img, threshold)

        col_img, col_res = st.columns([1, 1], gap="large")

        with col_img:
            st.image(pil_img, use_container_width=True)
            if show_gradcam:
                with st.spinner("Grad-CAM…"):
                    cam_img = run_gradcam(vis_model, pil_img, VISION_LABELS.index(gradcam_label))
                st.image(cam_img, caption=f"Grad-CAM: {LABEL_ICONS[gradcam_label]} {gradcam_label}",
                         use_container_width=True)

        with col_res:
            sec("Detection results")
            render_label_rows(prob_dict, threshold)

            st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
            sec("Detected labels")
            if detected:
                chips = "".join(
                    f"<span style='display:inline-block;background:#0d2b1a;color:#4ade80;"
                    f"border:1px solid #1a4a30;border-radius:6px;padding:4px 12px;"
                    f"font-size:0.82rem;font-weight:600;margin:3px'>"
                    f"{LABEL_ICONS[l]} {l}</span>"
                    for l in detected
                )
                st.markdown(chips, unsafe_allow_html=True)
            else:
                st.markdown(
                    "<span style='color:#4a5270;font-size:0.85rem'>No labels above threshold</span>",
                    unsafe_allow_html=True,
                )

            st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
            sec("Summary")
            tcols = st.columns(4)
            for tc, lbl in zip(tcols, VISION_LABELS):
                p     = prob_dict[lbl]
                det   = p >= threshold
                color = "#4ade80" if det else "#4a5270"
                tc.markdown(
                    f"<div class='tile'>"
                    f"<div style='font-size:1.3rem'>{LABEL_ICONS[lbl]}</div>"
                    f"<div class='tile-val' style='color:{color}'>{p:.0%}</div>"
                    f"<div class='tile-lbl'>{lbl}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )


# ══════════════════════════════════════════════════════════════════════════════
# NLP
# ══════════════════════════════════════════════════════════════════════════════

with tab_nlp:
    if not NLP_AVAILABLE:
        st.error(f"spaCy / transformers not available — {_NLP_MSG}")
    else:
        nlp_text = st.text_area("Incident report", height=100,
            value="A severe pile-up occurred on NH-8 near Sector 62 causing major congestion during the night.")
        nlp_btn = st.button("Analyse", type="primary")

        if nlp_btn and nlp_text.strip():
            with st.spinner("Processing…"):
                parsed, entities, cls_res = run_nlp(nlp_text)

            col1, col2 = st.columns([1, 1], gap="large")

            with col1:
                sec("Named entities")
                render_entities(entities)

                if parsed:
                    st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
                    sec("Noun chunks")
                    st.markdown(
                        " ".join(
                            f"<span style='background:#12162a;border:1px solid #1f2640;"
                            f"border-radius:5px;padding:3px 9px;font-size:0.8rem;"
                            f"color:#a0aac8;margin:2px;display:inline-block'>{c}</span>"
                            for c in parsed.noun_chunks
                        ),
                        unsafe_allow_html=True,
                    )

                    st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
                    sec("Tokens (first 20)")
                    rows = [
                        {"Token": t.text, "Lemma": t.lemma, "POS": t.pos, "Dep": t.dep}
                        for t in parsed.tokens[:20] if t.is_alpha
                    ]
                    st.dataframe(rows, use_container_width=True, hide_index=True)

            with col2:
                sec("Incident classification")
                if cls_res:
                    inc  = cls_res["predicted_class"]
                    icon = CLASS_ICONS.get(inc, "")
                    st.markdown(
                        f"<div class='card' style='margin-bottom:1rem'>"
                        f"<div style='font-size:1.15rem;font-weight:700;color:#e0e4ef'>"
                        f"{icon} {inc}</div>"
                        f"<div style='font-size:0.75rem;color:#5a6285;margin-top:2px'>predicted class</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                    render_prob_bars(cls_res.get("probabilities", {}))


# ══════════════════════════════════════════════════════════════════════════════
# Alert Generator
# ══════════════════════════════════════════════════════════════════════════════

with tab_alert:
    col_form, col_out = st.columns([1, 1], gap="large")

    with col_form:
        sec("Configure alert")
        inc_sel  = st.selectbox("Incident type", ["accident", "jam", "road_closure", "normal"])
        loc_sel  = st.text_input("Location", value="Sector 62")
        sev_sel  = st.text_input("Severity", value="major", placeholder="optional")
        vis_sel  = st.multiselect("Vision labels", VISION_LABELS, default=["rain", "congestion"])
        gen_btn  = st.button("Generate", type="primary", use_container_width=True)

    with col_out:
        sec("Result")

        if gen_btn:
            with st.spinner("Generating…"):
                res = generate_alert(inc_sel, loc_sel or "Unknown", sev_sel or None,
                                     vis_sel, api_key=api_key or None, force_template=force_template)
            render_alert(res["alert_text"], inc_sel)
            st.markdown(
                f"<div style='font-size:0.72rem;color:#4a5270'>via {res['method']}</div>",
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
        sec("Examples")
        examples = [
            ("accident",     "Sector 62", "major", ["rain", "congestion"]),
            ("jam",          "NH-8",      "heavy", ["congestion"]),
            ("road_closure", "MG Road",   None,    ["clear"]),
            ("normal",       "Ring Road", None,    ["clear"]),
        ]
        for e_inc, e_loc, e_sev, e_vis in examples:
            r = generate_alert(e_inc, e_loc, e_sev, e_vis, force_template=True)
            render_alert(r["alert_text"], e_inc)


# ══════════════════════════════════════════════════════════════════════════════
# About
# ══════════════════════════════════════════════════════════════════════════════

with tab_about:
    c1, c2 = st.columns([1, 1], gap="large")

    with c1:
        sec("Stack")
        modules = [
            ("📷", "Vision",          "ResNet18 · BCEWithLogitsLoss · multi-label"),
            ("🔥", "Explainability",  "Grad-CAM — per-label activation maps"),
            ("🔤", "Preprocessing",   "spaCy — tokenise, POS, dependency parse"),
            ("🏷️", "NER",             "BERT fine-tuned — LOC / INCIDENT / SEVERITY"),
            ("🗂️", "Classification",  "BERT — accident / jam / closure / normal"),
            ("🚨", "Alert gen",       "Claude few-shot or template fallback"),
            ("⚡", "API",             "FastAPI · /predict-image /analyze-text /generate-alert"),
        ]
        for icon, name, desc in modules:
            st.markdown(
                f"<div class='card' style='padding:0.75rem 1rem;margin-bottom:0.4rem'>"
                f"<div style='display:flex;gap:0.7rem;align-items:center'>"
                f"<span style='font-size:1.1rem'>{icon}</span>"
                f"<div><div style='font-size:0.88rem;font-weight:600;color:#e0e4ef'>{name}</div>"
                f"<div style='font-size:0.75rem;color:#5a6285'>{desc}</div></div>"
                f"</div></div>",
                unsafe_allow_html=True,
            )

    with c2:
        sec("Vision labels")
        for lbl, desc in [("🌧️ rain","Wet road / rain visible"),("🌙 night","Low-light / night"),
                           ("🚗 congestion","Dense / slow traffic"),("☀️ clear","No adverse condition")]:
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;padding:0.5rem 0;"
                f"border-bottom:1px solid #1f2640;font-size:0.85rem'>"
                f"<span>{lbl}</span><span style='color:#5a6285'>{desc}</span></div>",
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        sec("Quick start")
        st.code("""\
# 1. Train vision model
python -m vision.train --epochs 20

# 2. Train NLP models
python -m nlp.ner_model --train --epochs 5
python -m nlp.classifier --train --epochs 5

# 3. Start API
uvicorn api.main:app --reload --port 8000

# 4. Run dashboard
streamlit run dashboard/app.py""", language="bash")
