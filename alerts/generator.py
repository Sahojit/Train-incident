"""
Alert generator for traffic incidents.

Strategy:
  1. If ANTHROPIC_API_KEY is set → use Claude API with few-shot prompting
  2. Otherwise → deterministic template-based fallback

Output example:
  "🚨 Accident at Sector 62. Heavy congestion expected. Avoid route."
"""

from __future__ import annotations

import os
import random
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Few-shot examples used in the Claude prompt
# ---------------------------------------------------------------------------

FEW_SHOT_EXAMPLES = [
    {
        "input": {
            "incident_type": "accident",
            "location": "Sector 62",
            "severity": "major",
            "vision_labels": ["congestion", "rain"],
        },
        "output": "🚨 Major accident at Sector 62. Heavy congestion and rain reported. Avoid this route.",
    },
    {
        "input": {
            "incident_type": "jam",
            "location": "NH-8",
            "severity": "heavy",
            "vision_labels": ["congestion"],
        },
        "output": "⚠️ Heavy traffic jam on NH-8. Expect significant delays. Consider alternate routes.",
    },
    {
        "input": {
            "incident_type": "road_closure",
            "location": "MG Road",
            "severity": "minor",
            "vision_labels": ["clear"],
        },
        "output": "🚧 Road closure on MG Road. Please use alternate routes.",
    },
    {
        "input": {
            "incident_type": "normal",
            "location": "Outer Ring Road",
            "severity": None,
            "vision_labels": ["clear"],
        },
        "output": "✅ Traffic flowing normally on Outer Ring Road. No incidents reported.",
    },
]


# ---------------------------------------------------------------------------
# Template-based fallback
# ---------------------------------------------------------------------------

TEMPLATES: Dict[str, List[str]] = {
    "accident": [
        "🚨 {severity_str}accident at {location}. {vision_str}Avoid this route.",
        "🚨 Traffic incident at {location}. {severity_str}accident reported. {vision_str}Use alternate roads.",
    ],
    "jam": [
        "⚠️ {severity_str}congestion on {location}. {vision_str}Expect delays.",
        "⚠️ Heavy traffic jam at {location}. {vision_str}Consider alternate routes.",
    ],
    "road_closure": [
        "🚧 Road closure at {location}. {vision_str}Please divert.",
        "🚧 {location} closed. {vision_str}Use alternate routes.",
    ],
    "normal": [
        "✅ Traffic normal at {location}. No disruptions reported.",
        "✅ Clear roads at {location}. Smooth flow.",
    ],
}

VISION_PHRASES = {
    "rain": "Rain conditions detected. ",
    "night": "Night-time visibility low. ",
    "congestion": "Heavy congestion visible. ",
    "clear": "",
}


def _template_alert(
    incident_type: str,
    location: str,
    severity: Optional[str],
    vision_labels: List[str],
) -> str:
    templates = TEMPLATES.get(incident_type, TEMPLATES["normal"])
    tmpl = random.choice(templates)

    severity_str = f"{severity.capitalize()} " if severity and severity.lower() not in ("none", "normal") else ""
    vision_str = "".join(VISION_PHRASES.get(lbl, "") for lbl in vision_labels)

    return tmpl.format(
        location=location,
        severity_str=severity_str,
        vision_str=vision_str,
    ).strip()


# ---------------------------------------------------------------------------
# Claude-based alert (few-shot)
# ---------------------------------------------------------------------------

def _build_prompt(
    incident_type: str,
    location: str,
    severity: Optional[str],
    vision_labels: List[str],
) -> str:
    examples_block = "\n\n".join(
        f"Input: {ex['input']}\nOutput: {ex['output']}"
        for ex in FEW_SHOT_EXAMPLES
    )
    current_input = {
        "incident_type": incident_type,
        "location": location,
        "severity": severity,
        "vision_labels": vision_labels,
    }
    return (
        "You are a traffic alert system. Generate concise, actionable traffic alerts.\n\n"
        "Examples:\n\n"
        f"{examples_block}\n\n"
        f"Input: {current_input}\n"
        "Output:"
    )


def _claude_alert(
    incident_type: str,
    location: str,
    severity: Optional[str],
    vision_labels: List[str],
    api_key: str,
) -> str:
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        prompt = _build_prompt(incident_type, location, severity, vision_labels)
        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=120,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text.strip()
    except Exception as e:
        # Fall back to template on any API error
        print(f"[AlertGenerator] Claude API error: {e}. Using template fallback.")
        return _template_alert(incident_type, location, severity, vision_labels)


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def generate_alert(
    incident_type: str,
    location: str,
    severity: Optional[str] = None,
    vision_labels: Optional[List[str]] = None,
    api_key: Optional[str] = None,
    force_template: bool = False,
) -> Dict[str, str]:
    """
    Generate a human-readable traffic alert.

    Args:
        incident_type:  One of accident | jam | road_closure | normal
        location:       Location string (e.g. "Sector 62")
        severity:       Optional severity level (e.g. "major")
        vision_labels:  List of vision model predictions (e.g. ["rain", "congestion"])
        api_key:        Anthropic API key; falls back to template if None/empty
        force_template: Skip Claude even if key is available (useful for testing)

    Returns:
        dict with keys: alert_text, method (claude | template)
    """
    vision_labels = vision_labels or []
    resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")

    if resolved_key and not force_template:
        alert_text = _claude_alert(incident_type, location, severity, vision_labels, resolved_key)
        method = "claude"
    else:
        alert_text = _template_alert(incident_type, location, severity, vision_labels)
        method = "template"

    return {
        "alert_text": alert_text,
        "method": method,
        "incident_type": incident_type,
        "location": location,
        "severity": severity,
        "vision_labels": vision_labels,
    }


if __name__ == "__main__":
    test_cases = [
        ("accident", "Sector 62", "major", ["rain", "congestion"]),
        ("jam", "NH-8", "heavy", ["congestion"]),
        ("road_closure", "MG Road", "minor", ["clear"]),
        ("normal", "Ring Road", None, ["clear"]),
    ]

    print("=== Alert Generator Demo (Template Mode) ===\n")
    for inc, loc, sev, vis in test_cases:
        result = generate_alert(inc, loc, sev, vis, force_template=True)
        print(f"[{inc.upper()}] {result['alert_text']}")
