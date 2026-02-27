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

    Demo-safe:
      - Works without a fine-tuned model (uses lexicon fallback).
      - Later, replace `label_span()` with SpaceBERT inference.
    """

    def __init__(
        self,
        spacy_model: str = "en_core_web_sm",
        enable_noun_chunks: bool = True,
        enable_entities: bool = True,
        max_chunk_words: int = 6,
        min_node_chars: int = 3,
    ):
        # Keep parser for noun_chunks; add sentencizer to ensure sentence boundaries on pasted text.
        self.nlp = spacy.load(spacy_model)

        if "sentencizer" not in self.nlp.pipe_names:
            self.nlp.add_pipe("sentencizer")

        self.enable_noun_chunks = enable_noun_chunks
        self.enable_entities = enable_entities
        self.max_chunk_words = max_chunk_words
        self.min_node_chars = min_node_chars

        # --- Subsystem lexicon (expand anytime) ---
        self.lexicon: Dict[str, List[str]] = {
            "TELECOM": [
                "telecom", "telecommunications", "ttc", "tt&c",
                "x-band", "ka-band", "s-band", "downlink", "uplink",
                "antenna", "transmitter", "receiver", "modulation", "demodulation",
                "link budget", "rf", "frequency", "bandwidth",
                "data rate", "bit rate",
            ],
            "POWER": [
                "solar array", "solar arrays", "solar panel", "solar panels",
                "battery", "batteries", "power", "power subsystem",
                "power distribution", "pdu", "pcdu", "bus voltage",
                "regulated", "power distribution unit",
            ],
            "DATA": [
                "onboard data handling", "data handling", "command and data handling", "cdh",
                "mass memory", "memory", "storage", "secure storage",
                "processing", "onboard processing", "compression",
                "data volume", "data production",
            ],
            "PAYLOAD": [
                "payload", "instrument", "instruments",
                "imaging spectrometer", "spectrometer",
                "radiometer", "lidar", "laser", "sar", "altimeter", "camera",
                "calibrated radiance", "geophysical products",
                "level-1", "level-1b", "level-2",
            ],
            "ORBIT": [
                "orbit", "orbital", "sun-synchronous", "sun synchronous",
                "leo", "low earth orbit", "altitude", "inclination",
                "ascending node", "eccentricity", "semi-major axis", "period",
            ],
            "GROUND": [
                "ground segment", "ground station", "mission operations",
                "mission control", "control center", "mcc",
                "receiving station", "downlink station",
            ],
            "PROPULSION": [
                "propulsion", "thruster", "thrusters", "propellant", "tank", "tanks",
                "delta-v", "orbit raising", "chemical propulsion", "electric propulsion",
            ],
            "THERMAL": [
                "thermal", "radiator", "radiators", "heater", "heaters",
                "insulation", "mli", "multi-layer insulation",
                "temperature", "cooling", "heat pipe",
            ],
            "AOCS": [
                "aocs", "adcs", "attitude", "attitude control", "orbit determination",
                "reaction wheel", "momentum wheel", "star tracker", "gyroscope",
                "magnetorquer", "sun sensor", "earth sensor",
            ],
        }

        # Phrases we never want as nodes (common junk)
        self.stop_phrases = {
            "the mission", "the spacecraft", "the satellite", "the instrument",
            "the payload", "the system", "this mission", "this instrument",
            "the provision", "the response", "the range", "the connection",
        }

        # Leading words to strip (reduces duplicates like "the power distribution unit")
        self.leading_determiners_re = re.compile(r"^(the|a|an)\s+", flags=re.I)

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

            # keep "best" label if the node appears multiple times
            if node not in g:
                g.add_node(
                    node,
                    entity_type=s.label,
                    confidence=float(s.confidence),
                )
            else:
                # upgrade confidence/label if new span is stronger
                if float(s.confidence) > float(g.nodes[node].get("confidence", 0.0)):
                    g.nodes[node]["entity_type"] = s.label
                    g.nodes[node]["confidence"] = float(s.confidence)

        # edges: co-occurrence per sentence
        for sent in doc.sents:
            a0, a1 = sent.start_char, sent.end_char
            in_sent = []

            for s in spans:
                if s.start >= a0 and s.end <= a1:
                    node = self.canon(s.text)
                    if node:
                        in_sent.append(node)

            in_sent = list(dict.fromkeys(in_sent))
            if len(in_sent) < 2:
                continue

            for i in range(len(in_sent)):
                for j in range(i + 1, len(in_sent)):
                    u, v = in_sent[i], in_sent[j]
                    if u == v:
                        continue
                    if g.has_edge(u, v):
                        g[u][v]["weight"] += 1
                        g[u][v]["confidence"] = min(1.0, float(g[u][v]["confidence"]) + 0.05)
                    else:
                        g.add_edge(u, v, predicate="co-occurs", label="co-occurs", weight=1, confidence=0.35)

        return g

    # -----------------------------
    # Span extraction
    # -----------------------------
    def extract_spans(self, doc, raw_text: str) -> List[LabeledSpan]:
        spans: List[LabeledSpan] = []

        # 0) NER entities (optional) — helps catch mission names, orgs, places
        if self.enable_entities and hasattr(doc, "ents"):
            for ent in doc.ents:
                txt = ent.text.strip()
                c = self.canon(txt)
                if not c:
                    continue
                if c.lower() in self.stop_phrases:
                    continue
                if len(c.split()) > self.max_chunk_words:
                    continue
                label, conf = self.label_span(c)
                spans.append(LabeledSpan(text=c, label=label, start=ent.start_char, end=ent.end_char, confidence=conf))

        # 1) noun chunks (good for subsystem phrases)
        if self.enable_noun_chunks and doc.has_annotation("DEP"):
            for chunk in doc.noun_chunks:
                txt = chunk.text.strip()
                c = self.canon(txt)
                if not c:
                    continue
                if c.lower() in self.stop_phrases:
                    continue
                if len(c.split()) > self.max_chunk_words:
                    continue

                label, conf = self.label_span(c)
                spans.append(LabeledSpan(text=c, label=label, start=chunk.start_char, end=chunk.end_char, confidence=conf))

        # 2) key token patterns (X-band, CO2, Level-2, 100 km, etc.)
        token_patterns = [
            r"\b(?:x|s|ka)\s*-\s*band\b",
            r"\bco\s*2\b",
            r"\bch\s*4\b",
            r"\blevel\s*-\s*1b\b",
            r"\blevel\s*-\s*2\b",
            r"\blevel\s*1b\b",
            r"\blevel\s*2\b",
            r"\bsun\s*-\s*synchronous\b",
            r"\bsun\s*synchronous\b",
            r"\b\d+(?:\.\d+)?\s?km\b",
            r"\b\d+(?:\.\d+)?\s?w\b",
            r"\b\d+(?:\.\d+)?\s?hz\b",
            r"\b\d+(?:\.\d+)?\s?ghz\b",
        ]
        for pat in token_patterns:
            for m in re.finditer(pat, raw_text, flags=re.I):
                txt = raw_text[m.start():m.end()]
                c = self.canon(txt)
                if not c:
                    continue
                label, conf = self.label_span(c)
                spans.append(LabeledSpan(text=c, label=label, start=m.start(), end=m.end(), confidence=conf))

        # 3) de-dup by canonical text + sentence region (reduces duplicates heavily)
        # (start/end vary slightly between noun_chunks and ents; we mainly care about node text)
        best: Dict[str, LabeledSpan] = {}
        for s in spans:
            key = self.canon(s.text)
            if not key:
                continue
            if key not in best or s.confidence > best[key].confidence:
                best[key] = s

        return list(best.values())

    # -----------------------------
    # Labelling (fallback)
    # -----------------------------
    def label_span(self, span_text: str) -> Tuple[str, float]:
        """
        Keyword/lexicon labeler (demo-ready).
        Replace with SpaceBERT inference later.
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
        if best_score >= 3:
            return best_label, 0.85
        if best_score == 2:
            return best_label, 0.75
        return best_label, 0.60

    # -----------------------------
    # Normalization
    # -----------------------------
    def canon(self, s: str) -> str:
        x = re.sub(r"\s+", " ", x)
        x = x.strip(" ,.;:()[]{}\"'")
        x = re.sub(r"^(the|a|an)\s+", "", x, flags=re.I)

    # ignore very short junk
       if len(x) < 3:
           return ""

       return x

        # normalize unicode subscripts like CO₂ -> CO2
        sub_map = str.maketrans({
            "₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4",
            "₅": "5", "₆": "6", "₇": "7", "₈": "8", "₉": "9",
        })
        x = x.translate(sub_map)

        # strip quotes/punct and collapse whitespace
        x = re.sub(r"\s+", " ", x)
        x = x.strip(" ,.;:()[]{}\"'")

        # strip leading determiners ("the", "a", "an")
        x = self.leading_determiners_re.sub("", x).strip()

        # canonicalize short formulas like Co2 -> CO2, Ch4 -> CH4
        if re.fullmatch(r"[A-Za-z]{1,3}\d{0,3}", x):
            x = x.upper()

        # ignore tiny junk
        if len(x) < self.min_node_chars:
            return ""

        # avoid stop phrases after normalization
        if x.lower() in self.stop_phrases:
            return ""

        return x
