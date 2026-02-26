from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import re


@dataclass
class LabelResult:
    label: str
    confidence: float
    method: str  # "sentence" | "node" | "fallback"


class ESASubsystemLabeler:
    """
    Fast subsystem labelling for ESA mission documents.
    Uses:
      1) sentence context keywords (strong signal)
      2) node text keywords (direct match)
      3) fallback
    """

    def __init__(self):
        self.subsystem_keywords: Dict[str, List[str]] = {
            "TELECOM": [
                "rf", "telemetry", "telecommand", "ttc", "transmitter", "receiver",
                "antenna", "downlink", "uplink", "x-band", "ka-band", "s-band",
                "bit rate", "bandwidth", "modulation", "coding", "gain"
            ],
            "PROPULSION": [
                "propulsion", "thruster", "propellant", "tank", "valve", "nozzle",
                "hydrazine", "xenon", "monoprop", "biprop", "delta-v", "isp"
            ],
            "POWER": [
                "power", "electrical power", "solar array", "solar panel", "battery",
                "pdu", "power distribution", "voltage", "current", "watt"
            ],
            "THERMAL": [
                "thermal", "radiator", "heater", "heat pipe", "mli", "insulation",
                "temperature", "coating", "louver"
            ],
            "AOCS": [
                "aocs", "attitude", "orbit determination", "control subsystem",
                "reaction wheel", "momentum wheel", "star tracker", "gyroscope",
                "magnetorquer", "sun sensor", "earth sensor"
            ],
            "PAYLOAD": [
                "payload", "instrument", "spectrometer", "radiometer", "imager",
                "lidar", "camera", "sensor"
            ],
            "GROUND": [
                "ground segment", "ground station", "mission control", "mcc",
                "uplink", "downlink", "operations", "control center"
            ],
            "ORBIT": [
                "orbit", "leo", "low earth orbit", "sso", "sun-synchronous",
                "inclination", "altitude", "eccentricity", "raan", "semi-major axis",
                "orbital period"
            ],
            "DATA": [
                "level-1", "level 1", "level-2", "level 2", "l1", "l1b", "l2",
                "product", "retrieval", "radiance", "processing chain"
            ],
        }

        # Precompile regex for speed
        self._compiled: Dict[str, List[re.Pattern]] = {
            k: [re.compile(rf"\b{re.escape(term)}\b", flags=re.I) for term in v]
            for k, v in self.subsystem_keywords.items()
        }

    def label_node(self, node_text: str, sentence_text: str) -> LabelResult:
        node = (node_text or "").strip()
        sent = (sentence_text or "").strip()

        if not node:
            return LabelResult("OTHER", 0.10, "fallback")

        # 1) Sentence-level decision (strongest)
        sent_label, sent_score = self._best_label(sent)
        if sent_score >= 2:  # at least 2 keyword hits in sentence
            return LabelResult(sent_label, min(0.95, 0.70 + 0.10 * sent_score), "sentence")

        # 2) Node-level decision
        node_label, node_score = self._best_label(node)
        if node_score >= 1:
            return LabelResult(node_label, min(0.90, 0.55 + 0.15 * node_score), "node")

        # 3) Fallback
        return LabelResult("CONCEPT", 0.35, "fallback")

    def _best_label(self, text: str) -> Tuple[str, int]:
        best = ("OTHER", 0)
        if not text:
            return best

        for label, patterns in self._compiled.items():
            hits = 0
            for pat in patterns:
                if pat.search(text):
                    hits += 1
            if hits > best[1]:
                best = (label, hits)

        return best
