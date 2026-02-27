from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
import re

import networkx as nx
import spacy


# -----------------------------
# Data structures
# -----------------------------
@dataclass
class LabeledSpan:
    text: str
    label: str
    start: int
    end: int
    confidence: float = 0.60


# -----------------------------
# Builder
# -----------------------------
class LabelledSubsystemKGBuilder:
    """
    Labelled Subsystem KG Builder.

    Goal:
      - Extract candidate spans (noun phrases + key terms)
      - Assign each span a subsystem label (TELECOM/POWER/DATA/etc.)
      - Connect spans that co-occur in the same sentence

    This implementation is "demo-safe":
      - Works without a fine-tuned model (uses lexicon fallback).
      - Later, you can replace `label_span()` with SpaceBERT inference.
    """

    def __init__(
        self,
        spacy_model: str = "en_core_web_sm",
        enable_noun_chunks: bool = True,
    ):
        # We need sentence boundaries, so keep parser OR add sentencizer
        self.nlp = spacy.load(spacy_model)

        # Some spaCy pipelines don't split sentences well on pasted text:
        if "sentencizer" not in self.nlp.pipe_names:
            self.nlp.add_pipe("sentencizer")

        self.enable_noun_chunks = enable_noun_chunks

        # Simple subsystem lexicon (expand anytime)
        # Each subsystem has keywords that likely belong to it
        self.lexicon: Dict[str, List[str]] = {
            "TELECOM": [
                "telecom", "telecommunications", "ttc", "tt&c",
                "x-band", "ka-band", "s-band", "downlink", "uplink",
                "antenna", "transmitter", "receiver", "modulation", "link budget",
                "data rate", "bit rate",
            ],
            "POWER": [
                "solar array", "solar arrays", "battery", "batteries",
                "power distribution", "pdu", "pcdu", "bus voltage",
                "regulated", "power subsystem",
            ],
            "DATA": [
                "onboard data handling", "data handling", "mass memory",
                "storage", "secure storage", "processing", "onboard processing",
                "compression", "downlinked data", "data volume",
            ],
            "PAYLOAD": [
                "payload", "instrument", "imaging spectrometer", "spectrometer",
                "radiometer", "lidar", "sar", "altimeter", "camera",
                "calibrated radiance", "level-1", "level-1b", "level-2", "geophysical products",
            ],
            "ORBIT": [
                "orbit", "sun-synchronous", "sun synchronous", "leo",
                "low earth orbit", "altitude", "inclination", "ascending node",
            ],
            "GROUND": [
                "ground segment", "ground station", "mission operations",
                "mission control", "receiving station",
            ],
            "PROPULSION": [
                "propulsion", "thruster", "propellant", "tank", "delta-v",
                "orbit raising", "chemical propulsion", "electric propulsion",
            ],
            "THERMAL": [
                "thermal", "radiator", "heater", "insulation", "ml i", "mli",
                "temperature", "cooling",
            ],
            "AOCS": [
                "aocs", "adcs", "attitude", "orbit determination",
                "reaction wheel", "star tracker", "gyroscope", "magnetorquer",
            ],
        }

        self.stop_phrases = {
            "the mission", "the spacecraft", "the satellite", "the instrument",
            "the payload", "the system", "this mission", "this instrument"
        }

    # -----------------------------
    # Public API
    # -----------------------------
    def build(self, text: str, min_conf: float = 0.0) -> nx.Graph:
        doc = self.nlp(text)

        spans = self.extract_spans(doc, text)
        spans = [s for s in spans if s.confidence >= min_conf]

        g = nx.Graph()

        # nodes
        for s in spans:
            node = self.canon(s.text)
            if not node:
                continue
            if node not in g:
                g.add_node(
                    node,
                    entity_type=s.label,
                    confidence=float(s.confidence),
                    start=int(s.start),
                    end=int(s.end),
                )

        # edges: co-occurrence per sentence
        for sent in doc.sents:
            in_sent = []
            a0, a1 = sent.start_char, sent.end_char
            for s in spans:
                if s.start >= a0 and s.end <= a1:
                    node = self.canon(s.text)
                    if node:
                        in_sent.append(node)

            in_sent = list(dict.fromkeys(in_sent))
            for i in range(len(in_sent)):
                for j in range(i + 1, len(in_sent)):
                    u, v = in_sent[i], in_sent[j]
                    if u == v:
                        continue
                    if g.has_edge(u, v):
                        g[u][v]["weight"] += 1
                        g[u][v]["confidence"] = min(1.0, g[u][v]["confidence"] + 0.05)
                    else:
                        g.add_edge(u, v, predicate="co-occurs", label="co-occurs", weight=1, confidence=0.35)

        return g

    # -----------------------------
    # Span extraction
    # -----------------------------
    def extract_spans(self, doc, raw_text: str) -> List[LabeledSpan]:
        spans: List[LabeledSpan] = []

        # 1) noun chunks (good for mission terms)
        if self.enable_noun_chunks and doc.has_annotation("DEP"):
            for chunk in doc.noun_chunks:
                txt = chunk.text.strip()
                if not txt:
                    continue
                c = self.canon(txt)
                if not c or c.lower() in self.stop_phrases:
                    continue
                if len(c.split()) > 6:
                    continue

                label, conf = self.label_span(c)
                spans.append(LabeledSpan(text=c, label=label, start=chunk.start_char, end=chunk.end_char, confidence=conf))

        # 2) also extract important single tokens (X-band, CO2, CH4, Level-1B, etc.)
        token_patterns = [
            r"\b(?:x|s|ka)-band\b",
            r"\bco2\b",
            r"\bch4\b",
            r"\blevel-?\s?1b\b",
            r"\blevel-?\s?2\b",
            r"\bsun-?synchronous\b",
            r"\b\d+\s?km\b",
        ]
        for pat in token_patterns:
            for m in re.finditer(pat, raw_text, flags=re.I):
                txt = raw_text[m.start():m.end()]
                c = self.canon(txt)
                if not c:
                    continue
                label, conf = self.label_span(c)
                spans.append(LabeledSpan(text=c, label=label, start=m.start(), end=m.end(), confidence=conf))

        # de-dup by (start,end,label)
        uniq = {}
        for s in spans:
            key = (s.start, s.end, s.label)
            uniq[key] = s
        return list(uniq.values())

    # -----------------------------
    # Labelling (fallback)
    # -----------------------------
    def label_span(self, span_text: str) -> Tuple[str, float]:
        """
        Today: keyword/lexicon labeler (demo-ready).
        Later: replace this with SpaceBERT inference and confidence.
        """
        low = span_text.lower()

        best_label = "OTHER"
        best_score = 0

        for label, keys in self.lexicon.items():
            score = 0
            for k in keys:
                if k in low:
                    score += 1
            if score > best_score:
                best_score = score
                best_label = label

        # confidence heuristic
        if best_label == "OTHER":
            return "OTHER", 0.30
        if best_score >= 2:
            return best_label, 0.80
        return best_label, 0.60

    # -----------------------------
    # Normalization
    # -----------------------------
    def canon(self, s: str) -> str:
        x = (s or "").strip()
        x = re.sub(r"\s+", " ", x)
        x = x.strip(" ,.;:()[]{}\"'")
        # normalize unicode subscripts like CO₂ -> CO2
        sub_map = str.maketrans({"₀":"0","₁":"1","₂":"2","₃":"3","₄":"4","₅":"5","₆":"6","₇":"7","₈":"8","₉":"9"})
        x = x.translate(sub_map)
        # canonicalize short formulas like Co2 -> CO2
        if re.fullmatch(r"[A-Za-z]{1,3}\d{0,3}", x):
            x = x.upper()
        return x
