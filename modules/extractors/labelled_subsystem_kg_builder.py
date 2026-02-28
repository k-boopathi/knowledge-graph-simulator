from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
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
      1) Extracts mission-relevant spans (noun chunks + key patterns + TitleCase sequences)
      2) Labels spans using:
         - NER override (LOC / ORG / PERSON) when applicable
         - Subsystem labels (SUBSYSTEM_*) only when strongly indicated
         - Mission/science taxonomy otherwise
      3) Co-occurrence edges within sentence boundaries

    Works WITHOUT a fine-tuned SpaceBERT.
    Later, replace label_span() with SpaceBERT inference.
    """

    def __init__(
        self,
        spacy_model: str = "en_core_web_sm",
        enable_noun_chunks: bool = True,
        mission_mode: bool = True,
    ):
        # Full pipeline (includes NER)
        self.nlp = spacy.load(spacy_model)

        # Light pipeline for chunks/sents (disable ner only)
        self.nlp_light = spacy.load(spacy_model, disable=["ner"])

        # Ensure sentence splitting for pasted text
        if "sentencizer" not in self.nlp.pipe_names:
            self.nlp.add_pipe("sentencizer")
        if "sentencizer" not in self.nlp_light.pipe_names:
            self.nlp_light.add_pipe("sentencizer")

        self.enable_noun_chunks = enable_noun_chunks
        self.mission_mode = mission_mode

        # -----------------------------------------
        # Mission-first taxonomy keywords
        # -----------------------------------------
        self.mission_lexicon: Dict[str, List[str]] = {
            "MISSION": [
                "mission", "earth explorer", "candidate mission", "phase 0", "phase a",
                "spacecraft", "satellite", "space segment", "ground segment", "user segment",
                "selection", "phase", "payload",
            ],
            "PROGRAMME": [
                "esa", "european space agency", "living planet programme",
                "earth explorer", "earth observation", "eo mission",
            ],
            "SCIENCE_OBJECTIVE": [
                "objective", "objectives", "aim", "aims", "goal", "goals", "quest",
                "questions", "pursuit", "determine", "retrieving", "obtaining",
                "establishing", "characterisation", "characterize",
            ],
            "MEASUREMENT": [
                "measurement", "measurements", "in-situ", "simultaneous measurements",
                "retrieval", "estimates", "flux", "fluxes",
                "densities", "temperatures", "precipitation", "heating",
            ],
            "TARGET_REGION": [
                "thermosphere", "ionosphere", "upper atmosphere", "lower thermosphere",
                "lti", "altitude", "altitudes", "region", "middle atmosphere",
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
                "link budget", "data rate", "bit rate", "rf",
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

        # Strong junk filtering
        self.stop_phrases = {
            "the mission", "the spacecraft", "the satellite", "the instrument",
            "the payload", "the system", "this mission", "this instrument",
            "this region", "this situation", "this work", "this study",
            "the region", "the atmosphere", "the earth", "the questions",
            "a set", "the set", "the provision", "the pursuit",
            "the range", "the response", "the connection",
        }

        self.determiner_prefix = ("the ", "a ", "an ", "this ", "that ", "these ", "those ")

        # Regex patterns for mission-style entities
        self.token_patterns = [
            r"\b(?:sun-?synchronous|low earth orbit|leo|near-?polar)\b",
            r"\b\d{2,4}\s?km\b",
            r"\baltitudes?\b",
            r"\bESA\b",
            r"\bLiving Planet Programme\b",
            r"\bEarth Explorer\b",
            r"\bLTI\b",
            r"\bEPP\b",
        ]
        self.mission_whitelist = [
            "thermosphere", "ionosphere", "upper atmosphere", "lower thermosphere",
            "in-situ", "gravity waves", "energetics", "dynamics", "chemistry",
            "electro-magnetic", "electromagnetic", "particle precipitation", "epp",
            "radio wave", "irregularities", 
            "earth explorer", "living planet programme", "phase 0", "phase a",
            "spacecraft", "satellite", "payload", "instrument", "mission",
            "orbit", "altitude", "km",
    # subsystems (if they appear)
            "antenna", "downlink", "uplink", "battery", "solar", "propulsion", "thruster" ,
        ]

    

    # -----------------------------
    # Public API
    # -----------------------------
    def build(self, text: str, min_conf: float = 0.0) -> nx.Graph:
        doc_light = self.nlp_light(text)  # sents + noun_chunks
        doc_ner = self.nlp(text)          # ents

        spans = self.extract_spans(doc_light, doc_ner, text)
        spans = [s for s in spans if s.confidence >= min_conf]

        g = nx.Graph()

        # Nodes
        for s in spans:
            node = self.canon(s.text)
            if not node:
                continue

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

        # Edges: co-occurrence per sentence (use doc_light sentences)
        for sent in doc_light.sents:
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
    def extract_spans(self, doc_light, doc_ner, raw_text: str) -> List[LabeledSpan]:
        spans: List[LabeledSpan] = []

        # Build quick NER spans (for override)
        ner_spans = [(e.start_char, e.end_char, e.label_, e.text) for e in doc_ner.ents]

        # 1) TitleCase sequences (Daedalus, Living Planet Programme, Earth Explorer)
        for m in re.finditer(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,4})\b", raw_text):
            txt = raw_text[m.start():m.end()]
            c = self.canon(txt)
            if not c or len(c) < 3:
                continue

            # Avoid very generic TitleCase captures
            if c.lower() in {"accordingly", "answers", "complex", "region", "earth"}:
                continue

            label, conf = self.label_span(c, m.start(), m.end(), ner_spans)

            # Boost mission name if it's a single clean proper noun token
            if re.fullmatch(r"[A-Z][a-z]{2,}", c):
                conf = max(conf, 0.80)
                if label not in {"LOC", "ORG", "PERSON"}:
                    label = "MISSION"

            spans.append(LabeledSpan(text=c, label=label, start=m.start(), end=m.end(), confidence=conf))

        # 2) Noun chunks (mission/science terms)
        if self.enable_noun_chunks and doc_light.has_annotation("DEP"):
            for chunk in doc_light.noun_chunks:
                txt = chunk.text.strip()
                c = self.canon(txt)
                if not c:
                    continue

                low = c.lower()

                # Strip determiners
                if low.startswith(self.determiner_prefix):
                    c2 = self.canon(re.sub(r"^(the|a|an|this|that|these|those)\s+", "", c, flags=re.I))
                    if c2:
                        c = c2
                        low = c.lower()

                # Basic filters
                if low in self.stop_phrases:
                    continue
                if len(c.split()) > 7:
                    continue
                if len(c) < 3:
                    continue
                if self._is_mostly_stopwords(c):
                    continue

                # Mission-mode: keep only mission/science-relevant chunks
                if self.mission_mode and not self._looks_mission_relevant(low, c):
                    continue

                label, conf = self.label_span(c, chunk.start_char, chunk.end_char, ner_spans)
                spans.append(LabeledSpan(text=c, label=label, start=chunk.start_char, end=chunk.end_char, confidence=conf))

        # 3) Regex patterns (EPP/LTI/100 km/etc.)
        for pat in self.token_patterns:
            for m in re.finditer(pat, raw_text, flags=re.I):
                txt = raw_text[m.start():m.end()]
                c = self.canon(txt)
                if not c:
                    continue
                label, conf = self.label_span(c, m.start(), m.end(), ner_spans)
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
        # A) whitelist phrase/word hit = keep (high precision)
        for w in self.mission_whitelist:
            if w in low:
                return True

    # B) lexicon keyword hit = keep
        for keys in list(self.mission_lexicon.values()) + list(self.subsystem_lexicon.values()):
            for k in keys:
                if k in low:
                    return True

    # C) Acronyms (ESA, LTI, EPP)
        if re.search(r"\b[A-Z]{2,6}\b", original):
            return True

    # D) Hyphenated technical term
        if "-" in original and len(original) <= 40:
            return True
            
        return False

    def _is_mostly_stopwords(self, phrase: str) -> bool:
        toks = [t for t in re.split(r"\s+", phrase.strip()) if t]
        if not toks:
            return True
        # crude: if many tokens are short function-ish words
        short = sum(1 for t in toks if len(t) <= 2)
        return short >= max(2, len(toks) - 1)

    # -----------------------------
    # Labelling (NER override + lexicon fallback)
    # -----------------------------
    def label_span(self, span_text: str, start: int, end: int, ner_spans) -> Tuple[str, float]:
        low = span_text.lower()

        # 0) NER override (prevents Munich -> MISSION, etc.)
        ner_label = self._ner_override(start, end, ner_spans)
        if ner_label:
            # reasonable confidence for NER mapping
            return ner_label, 0.80

        # 1) Subsystem scoring (strong hits only)
        best_sub = None
        best_sub_score = 0
        for lab, keys in self.subsystem_lexicon.items():
            score = self._score_keys(low, keys)
            if score > best_sub_score:
                best_sub_score = score
                best_sub = lab

        # subsystem thresholds (conservative)
        if best_sub and best_sub_score >= 4:
            return best_sub, 0.90
        if best_sub and best_sub_score >= 2:
            return best_sub, 0.75

        # 2) Mission/science labels
        best_label = "OTHER"
        best_score = 0
        for lab, keys in self.mission_lexicon.items():
            score = self._score_keys(low, keys)
            if score > best_score:
                best_score = score
                best_label = lab

        if best_label == "OTHER" or best_score == 0:
            return "OTHER", 0.30
        if best_score >= 4:
            return best_label, 0.85
        if best_score >= 2:
            return best_label, 0.70
        return best_label, 0.55

    def _ner_override(self, start: int, end: int, ner_spans) -> Optional[str]:
        """
        If this span overlaps a strong NER entity, use that label.
        Helps prevent city/org names from being misclassified as MISSION/OTHER.
        """
        best = None
        best_overlap = 0

        for (a0, a1, lab, _) in ner_spans:
            overlap = max(0, min(end, a1) - max(start, a0))
            if overlap <= 0:
                continue
            if overlap > best_overlap:
                best_overlap = overlap
                best = lab

        if not best or best_overlap == 0:
            return None

        # map spaCy NER to our UI labels
        if best in {"GPE", "LOC", "FAC"}:
            return "LOC"
        if best == "ORG":
            return "ORG"
        if best == "PERSON":
            return "PERSON"

        return None

    def _score_keys(self, low: str, keys: List[str]) -> int:
        """
        Weighted keyword score:
          - +2 for whole-word match (or clean phrase boundary)
          - +1 for substring match
        """
        score = 0
        for k in keys:
            k_low = (k or "").lower().strip()
            if not k_low:
                continue
            if " " in k_low:
                if k_low in low:
                    score += 3
                continue
            if re.search(rf"\b{re.escape(k_low)}\b", low):
                score += 2
            elif k_low in low:
                score += 1
        return score
    # -----------------------------
    # Normalization
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
