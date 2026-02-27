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


class LabelledSubsystemKGBuilder:
    """
    Mission-focused labelled KG builder (demo-safe).

    What it does:
      1) Extracts mission-relevant spans (mission name, objectives, measurements, region/orbit terms, etc.)
      2) Assigns each span a label using a mission-first taxonomy:
         - MISSION, PROGRAMME, SCIENCE_OBJECTIVE, MEASUREMENT, TARGET_REGION, MODELING, CONCEPT
         - SUBSYSTEM_* labels when strongly indicated (TELECOM, POWER, DATA, PAYLOAD, etc.)
      3) Connects spans that co-occur within the same sentence

    This works WITHOUT a fine-tuned SpaceBERT.
    Later, you can replace label_span() with SpaceBERT inference.
    """

    def __init__(
        self,
        spacy_model: str = "en_core_web_sm",
        enable_noun_chunks: bool = True,
        mission_mode: bool = True,
    ):
        self.nlp = spacy.load(spacy_model)

        # Ensure sentence splitting works for pasted text
        if "sentencizer" not in self.nlp.pipe_names:
            self.nlp.add_pipe("sentencizer")

        self.enable_noun_chunks = enable_noun_chunks
        self.mission_mode = mission_mode

        # -----------------------------------------
        # Mission-first taxonomy keywords
        # -----------------------------------------
        self.mission_lexicon: Dict[str, List[str]] = {
            "MISSION": [
                "mission", "earth explorer", "candidate mission", "phase 0", "phase a",
                "spacecraft", "satellite", "payload", "platform", "space segment",
                "ground segment", "user segment",
            ],
            "PROGRAMME": [
                "esa", "european space agency", "living planet programme", "earth explorer",
                "futureeo", "earth observation", "eo mission",
            ],
            "SCIENCE_OBJECTIVE": [
                "objective", "objectives", "aim", "aims", "goal", "goals", "quest",
                "questions", "pursuit", "provision of", "determine", "retrieving", "obtaining",
                "establishing", "characterisation", "characterize",
            ],
            "MEASUREMENT": [
                "measurement", "measurements", "in-situ", "simultaneous measurements",
                "retrieving", "retrieval", "estimates", "flux", "fluxes",
                "densities", "temperatures", "precipitation", "heating",
                "radiance", "spectral", "spectrometer", "radiometer", "lidar",
            ],
            "TARGET_REGION": [
                "thermosphere", "ionosphere", "upper atmosphere", "lower thermosphere",
                "lti", "altitudes", "100 km", "200 km", "region",
                "middle atmosphere",
            ],
            "MODELING": [
                "models", "model", "uncertainties", "scarcity of observations",
                "limited", "uncertain", "validation",
            ],
            "CONCEPT": [
                "process", "processes", "interaction", "interactions", "dynamics",
                "chemistry", "energetics", "system-level understanding",
                "earth system science", "connection to space",
                "gravity waves", "traveling atmospheric disturbances",
                "irregularities",
            ],
        }

        # -----------------------------------------
        # Subsystem labels (only when clearly present)
        # -----------------------------------------
        self.subsystem_lexicon: Dict[str, List[str]] = {
            "SUBSYSTEM_TELECOM": [
                "telecom", "telecommunications", "ttc", "tt&c",
                "downlink", "uplink", "x-band", "ka-band", "s-band",
                "antenna", "transmitter", "receiver", "modulation",
                "link budget", "data rate", "bit rate",
            ],
            "SUBSYSTEM_POWER": [
                "solar array", "solar arrays", "battery", "batteries",
                "power distribution", "pdu", "pcdu", "bus voltage",
                "power subsystem",
            ],
            "SUBSYSTEM_DATA": [
                "onboard data handling", "data handling", "mass memory",
                "secure storage", "storage", "processing", "onboard processing",
                "compression", "data volume",
            ],
            "SUBSYSTEM_PAYLOAD": [
                "payload", "instrument", "instruments",
                "imaging spectrometer", "spectrometer", "radiometer", "lidar", "sar",
                "altimeter", "camera",
                "level-1", "level-1b", "level-2", "geophysical products",
            ],
            "SUBSYSTEM_ORBIT": [
                "orbit", "sun-synchronous", "sun synchronous", "leo",
                "low earth orbit", "altitude", "inclination", "ascending node",
            ],
            "SUBSYSTEM_GROUND": [
                "ground segment", "ground station", "mission operations",
                "mission control", "receiving station",
            ],
            "SUBSYSTEM_PROPULSION": [
                "propulsion", "thruster", "propellant", "tank", "delta-v",
                "orbit raising", "chemical propulsion", "electric propulsion",
            ],
            "SUBSYSTEM_THERMAL": [
                "thermal", "radiator", "heater", "insulation", "mli",
                "temperature", "cooling",
            ],
            "SUBSYSTEM_AOCS": [
                "aocs", "adcs", "attitude", "orbit determination",
                "reaction wheel", "star tracker", "gyroscope", "magnetorquer",
            ],
        }

        # Junk filtering (mission-only demo)
        self.stop_phrases = {
            "the mission", "the spacecraft", "the satellite", "the instrument",
            "the payload", "the system", "this mission", "this instrument",
            "this region", "this situation", "this work", "this study",
            "the region", "the atmosphere", "the earth", "the questions",
            "a set", "the set", "the provision", "the pursuit",
        }

        self.determiner_prefix = ("the ", "a ", "an ", "this ", "that ", "these ", "those ")

        # Regex patterns for mission-style entities
        self.token_patterns = [
            # orbits / altitude
            r"\b(?:sun-?synchronous|low earth orbit|leo|near-?polar)\b",
            r"\b\d{2,4}\s?km\b",
            r"\baltitudes?\b",
            # ESA / programmes
            r"\bESA\b",
            r"\bLiving Planet Programme\b",
            r"\bEarth Explorer\b",
            # common science shorthand
            r"\bLTI\b",
            r"\bEPP\b",
        ]

    # -----------------------------
    # Public API
    # -----------------------------
    def build(self, text: str, min_conf: float = 0.0) -> nx.Graph:
        doc = self.nlp(text)

        spans = self.extract_spans(doc, text)
        spans = [s for s in spans if s.confidence >= min_conf]

        g = nx.Graph()

        # Nodes
        for s in spans:
            node = self.canon(s.text)
            if not node:
                continue

            # If already exists, keep the higher confidence label
            if node in g:
                old_c = float(g.nodes[node].get("confidence", 0.0))
                if s.confidence > old_c:
                    g.nodes[node]["entity_type"] = s.label
                    g.nodes[node]["confidence"] = float(s.confidence)
                continue

            g.add_node(
                node,
                entity_type=s.label,
                confidence=float(s.confidence),
                start=int(s.start),
                end=int(s.end),
            )

        # Edges: co-occurrence per sentence
        for sent in doc.sents:
            a0, a1 = sent.start_char, sent.end_char
            in_sent: List[str] = []

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

        # 1) Proper nouns / mission names: capture TitleCase sequences (Daedalus, Earth Explorer, Living Planet Programme)
        # Keep conservative: 1-5 tokens
        for m in re.finditer(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,4})\b", raw_text):
            txt = raw_text[m.start():m.end()]
            c = self.canon(txt)
            if not c:
                continue
            # Skip if it's sentence-start generic word
            if c.lower() in {"accordingly", "answers", "complex", "daedalus"} or len(c) < 3:
                pass
            label, conf = self.label_span(c)
            # boost if looks like a mission name (single capitalized token)
            if re.fullmatch(r"[A-Z][a-z]{2,}", c):
                conf = max(conf, 0.80)
                label = "MISSION"
            spans.append(LabeledSpan(text=c, label=label, start=m.start(), end=m.end(), confidence=conf))

        # 2) Noun chunks (mission terms, science objectives)
        if self.enable_noun_chunks and doc.has_annotation("DEP"):
            for chunk in doc.noun_chunks:
                txt = chunk.text.strip()
                c = self.canon(txt)
                if not c:
                    continue

                low = c.lower()

                # strip determiners and re-canon
                if low.startswith(self.determiner_prefix):
                    c2 = self.canon(re.sub(r"^(the|a|an|this|that|these|those)\s+", "", c, flags=re.I))
                    if c2:
                        c = c2
                        low = c.lower()

                # filters
                if low in self.stop_phrases:
                    continue
                if len(c.split()) > 7:
                    continue
                if len(c) < 3:
                    continue

                # Mission-mode: keep only "meaningful" chunks:
                # (a) chunk contains at least one mission/science keyword
                # OR (b) chunk is a named region/science term with capitals/acronyms
                if self.mission_mode:
                    keep = self._looks_mission_relevant(low, c)
                    if not keep:
                        continue

                label, conf = self.label_span(c)
                spans.append(
                    LabeledSpan(
                        text=c,
                        label=label,
                        start=chunk.start_char,
                        end=chunk.end_char,
                        confidence=conf,
                    )
                )

        # 3) Regex token patterns (EPP/LTI/100 km/etc.)
        for pat in self.token_patterns:
            for m in re.finditer(pat, raw_text, flags=re.I):
                txt = raw_text[m.start():m.end()]
                c = self.canon(txt)
                if not c:
                    continue
                label, conf = self.label_span(c)
                spans.append(LabeledSpan(text=c, label=label, start=m.start(), end=m.end(), confidence=conf))

        # De-dup by canonical text + label (keep max confidence)
        best: Dict[Tuple[str, str], LabeledSpan] = {}
        for s in spans:
            key = (self.canon(s.text), s.label)
            if key not in best or s.confidence > best[key].confidence:
                best[key] = s

        return list(best.values())

    def _looks_mission_relevant(self, low: str, original: str) -> bool:
        # Contains any mission/science lexicon keyword
        for keys in list(self.mission_lexicon.values()) + list(self.subsystem_lexicon.values()):
            for k in keys:
                if k in low:
                    return True

        # Contains acronyms or obvious science terms
        if re.search(r"\b[A-Z]{2,6}\b", original):  # LTI, EPP, ESA
            return True

        # Contains hyphenated technical term
        if "-" in original and len(original) <= 40:
            return True

        return False

    # -----------------------------
    # Labelling (lexicon fallback)
    # -----------------------------
    def label_span(self, span_text: str) -> Tuple[str, float]:
        """
        Lexicon labeler:
          - If a subsystem keyword hits strongly, return SUBSYSTEM_* label.
          - Else mission/science labels.
        """
        low = span_text.lower()

        # 1) Subsystem scoring (strong hits only)
        best_sub = None
        best_sub_score = 0
        for lab, keys in self.subsystem_lexicon.items():
            score = sum(1 for k in keys if k in low)
            if score > best_sub_score:
                best_sub_score = score
                best_sub = lab

        if best_sub and best_sub_score >= 2:
            return best_sub, 0.85
        if best_sub and best_sub_score == 1:
            # only accept 1-hit subsystem if the span is short/technical
            if len(span_text.split()) <= 3 or re.search(r"band|antenna|thruster|battery|orbit", low):
                return best_sub, 0.70

        # 2) Mission/science labels
        best_label = "OTHER"
        best_score = 0
        for lab, keys in self.mission_lexicon.items():
            score = sum(1 for k in keys if k in low)
            if score > best_score:
                best_score = score
                best_label = lab

        if best_label == "OTHER":
            return "OTHER", 0.30
        if best_score >= 2:
            return best_label, 0.80
        return best_label, 0.60

    # -----------------------------
    # Normalization (important)
    # -----------------------------
    def canon(self, s: str) -> str:
        x = (s or "").strip()
        if not x:
            return ""
        x = re.sub(r"\s+", " ", x)
        x = x.strip(" ,.;:()[]{}\"'")

        # Normalize unicode subscripts like CO₂ -> CO2
        sub_map = str.maketrans({
            "₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4",
            "₅": "5", "₆": "6", "₇": "7", "₈": "8", "₉": "9",
        })
        x = x.translate(sub_map)

        # Canonicalize short formulas like Co2 -> CO2
        if re.fullmatch(r"[A-Za-z]{1,3}\d{0,3}", x):
            x = x.upper()

        return x
