# modules/llm/esa_router.py
import os
import re
from typing import Dict, Any, List
from openai import OpenAI

ROUTE_LABELS = [
    "TELECOM", "POWER", "DATA", "THERMAL", "PROPULSION", "AOCS",
    "PAYLOAD", "GROUND", "ORBIT", "SCIENCE_BACKGROUND", "OTHER"
]

SYSTEM_ROUTER_PROMPT = """You are a routing component for an ESA mission knowledge-graph system.

Task:
Given raw mission text, split it into sentences and assign each sentence exactly ONE label from this set:
TELECOM, POWER, DATA, THERMAL, PROPULSION, AOCS, PAYLOAD, GROUND, ORBIT, SCIENCE_BACKGROUND, OTHER

Rules:
- Do NOT invent facts or entities.
- Route based only on the sentence content.
- If a sentence is mainly climate/science motivation/background (not spacecraft engineering), label SCIENCE_BACKGROUND.
- If it mentions instruments/sensors/measurement payloads, label PAYLOAD.
- If it mentions orbit parameters (LEO, SSO, altitude, apogee/perigee, inclination), label ORBIT.
- If it mentions TT&C, RF, downlink/uplink, bands (X/Ka/S), antennas, transponders, label TELECOM.
- If unclear, use OTHER.

Also output:
- is_engineering (boolean): true if label is one of TELECOM/POWER/DATA/THERMAL/PROPULSION/AOCS/PAYLOAD/GROUND/ORBIT
- confidence (0..1)
- short rationale (max 15 words), quoting a key phrase from the sentence.
"""

ROUTER_SCHEMA: Dict[str, Any] = {
    "name": "esa_sentence_router",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "sentences": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "sentence": {"type": "string"},
                        "label": {"type": "string", "enum": ROUTE_LABELS},
                        "is_engineering": {"type": "boolean"},
                        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "rationale": {"type": "string"}
                    },
                    "required": ["sentence", "label", "is_engineering", "confidence", "rationale"]
                }
            }
        },
        "required": ["sentences"]
    }
}

def route_esa_text(text: str, model: str = "gpt-4o-mini", api_key: str | None = None) -> Dict[str, Any]:
    if not text or not text.strip():
        return {"sentences": []}

    client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    cleaned = re.sub(r"\s+", " ", text).strip()

    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM_ROUTER_PROMPT},
            {"role": "user", "content": f"Route the following text.\n\nTEXT:\n{cleaned}"}
        ],
        response_format={"type": "json_schema", "json_schema": ROUTER_SCHEMA},
    )

    if hasattr(resp, "output_parsed") and resp.output_parsed:
        return resp.output_parsed

    # Fallback parse
    out_text = getattr(resp, "output_text", None)
    if out_text:
        import json
        return json.loads(out_text)

    raise RuntimeError("No structured output returned.")

def engineering_only_text(routed: Dict[str, Any], min_conf: float = 0.55) -> str:
    keep: List[str] = []
    for s in routed.get("sentences", []):
        if s.get("is_engineering") and float(s.get("confidence", 0.0)) >= float(min_conf):
            keep.append(s["sentence"])
    return " ".join(keep)
