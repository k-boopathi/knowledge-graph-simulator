from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import re

import networkx as nx
import spacy


@dataclass
class LabeledSpan:
    text: str
    subsystem: str
    structure: str
    start: int
    end: int
    confidence: float = 0.60


class LabelledSubsystemKGBuilder:
    """
    Mission KG builder with TWO labels per node:
      - subsystem_label: TELECOM/POWER/DATA/PAYLOAD/ORBIT/GROUND/...
      - structure_label: MISSION/ORBIT/PAYLOAD/INSTRUMENT/GROUND/TARGET_REGION/PARAMETER/ORG/LOC/...

    This massively improves accuracy without SpaceBERT:
      - NER protects ORG/LOC/PERSON
      - subsystem lexicon uses weighted boundary matching
      - structure taxonomy classifies mission description terms
    """

    def __init__(self, spacy_model: str = "en_core_web_sm", enable_noun_chunks: bool = True):
        self.nlp = spacy.load(spacy_model)
        if "sentencizer" not in self.nlp.pipe_names:
            self.nlp.add_pipe("sentencizer")

        self.enable_noun_chunks = enable_noun_chunks

        # -------------------------
        # Subsystem label space
        # -------------------------
        self.subsystem_lexicon: Dict[str, List[str]] = {
            "TELECOM": [
                "telecom", "telecommunication", "telecommunications", "ttc", "tt&c",
                "downlink", "uplink", "x-band", "ka-band", "s-band",
                "antenna", "transmitter", "receiver", "modulation",
                "rf", "link budget", "bit rate", "data rate", "telemetry",
            ],
            "POWER": [
                "solar array", "solar arrays", "battery", "batteries",
                "power distribution", "pdu", "pcdu", "bus voltage",
                "power conditioning", "power subsystem",
            ],
            "DATA": [
                "onboard data handling", "data handling", "cdh", "obdh",
                "mass memory", "storage", "secure storage",
                "processing", "onboard processing", "compression",
                "data volume", "downlinked data",
            ],
            "PAYLOAD": [
                "payload", "instrument", "instruments",
                "spectrometer", "imaging spectrometer", "radiometer",
                "lidar", "sar", "altimeter", "camera", "telescope",
                "level-1", "level-1b", "level-2", "geophysical products",
            ],
            "ORBIT": [
                "orbit", "sun-synchronous", "sun synchronous", "leo",
                "low earth orbit", "near-polar", "inclination", "ascending node",
                "altitude", "km", "apogee", "perigee",
            ],
            "GROUND": [
                "ground segment", "ground station", "mission operations",
                "mission control", "receiving station", "data centre", "data center",
            ],
            "PROPULSION": [
                "propulsion", "thruster", "propellant", "tank", "delta-v",
                "orbit raising", "chemical propulsion", "electric propulsion",
            ],
            "THERMAL": [
                "thermal", "radiator", "heater", "insulation", "mli",
                "temperature control", "cooling",
            ],
            "AOCS": [
                "aocs", "adcs", "attitude", "attitude control",
                "reaction wheel", "star tracker", "gyroscope", "magnetorquer",
            ],
        }

        # -------------------------
        # Mission structure label space
        # -------------------------
        self.structure_lexicon: Dict[str, List[str]] = {
            "MISSION": [
                "mission", "mission candidate", "earth explorer", "phase 0", "phase a",
                "selection", "candidate", "concept study",
            ],
            "ORBIT": [
                "orbit", "sun-synchronous", "sun synchronous", "leo",
                "low earth orbit", "inclination", "ascending node", "altitude",
            ],
            "PAYLOAD": [
                "payload", "payload suite", "instrument", "instruments",
            ],
            "INSTRUMENT": [
                "spectrometer", "radiometer", "lidar", "sar", "altimeter", "camera", "telescope",
            ],
            "GROUND": [
                "ground segment", "ground station", "mission operations", "mission control",
            ],
            "TARGET_REGION": [
                "thermosphere", "ionosphere", "upper atmosphere", "lower thermosphere",
                "stratosphere", "troposphere", "mesosphere", "lti", "region",
            ],
            "PARAMETER": [
                "altitude", "km", "temperature", "temperatures", "density", "densities",
                "flux", "fluxes", "heating", "radiance", "precipitation",
            ],
            "SUBSYSTEM": [
                "telecom subsystem", "power subsystem", "data handling subsystem",
                "propulsion subsystem", "thermal subsystem", "aocs",
            ],
        }

        # -------------------------
        # filtering junk chunks
        # -------------------------
        self.stop_phrases = {
            "the mission", "the spacecraft", "the satellite", "the instrument",
            "the payload", "the system", "this mission", "this instrument",
            "this region", "this situation", "this work", "this study",
            "the region", "the atmosphere", "the earth",
            "a set", "the set", "the provision", "the pursuit",
            "the range", "the response", "the connection",
        }

        self.determiner_prefix = ("the ", "a ", "an ", "this ", "that ", "these ", "those ")

        self.token_patterns = [
            r"\b(?:sun-?synchronous|low earth orbit|leo|near-?polar)\b",
            r"\b\d{2,4}\s?km\b",
            r"\bESA\b",
            r"\bEarth Explorer\b",
            r"\bLiving Planet Programme\b",
        ]

    # -----------------------------
    # Public
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

            if node in g:
                # keep higher-confidence labels
                if s.confidence > float(g.nodes[node].get("confidence", 0.0)):
                    g.nodes[node]["subsystem_label"] = s.subsystem
                    g.nodes[node]["structure_label"] = s.structure
                    g.nodes[node]["confidence"] = float(s.confidence)
                continue

            g.add_node(
                node,
                subsystem_label=s.subsystem,
                structure_label=s.structure,
                confidence=float(s.confidence),
                start=int(s.start),
                end=int(s.end),
            )

        # Edges
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
    # Spans
    # -----------------------------
    def extract_spans(self, doc, raw_text: str) -> List[LabeledSpan]:
        spans: List[LabeledSpan] = []

        ner_spans = [(e.start_char, e.end_char, e.label_, e.text) for e in doc.ents]

        # 1) TitleCase sequences (but do NOT force MISSION blindly)
        for m in re.finditer(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,4})\b", raw_text):
            txt = raw_text[m.start():m.end()]
            c = self.canon(txt)
            if not c or len(c) < 3:
                continue
            if c.lower() in {"accordingly", "answers", "complex", "region", "earth"}:
                continue

            subsystem, structure, conf = self.classify_span(c, m.start(), m.end(), ner_spans)
            spans.append(LabeledSpan(text=c, subsystem=subsystem, structure=structure, start=m.start(), end=m.end(), confidence=conf))

        # 2) noun chunks
        if self.enable_noun_chunks and doc.has_annotation("DEP"):
            for chunk in doc.noun_chunks:
                txt = chunk.text.strip()
                c = self.canon(txt)
                if not c:
                    continue
                low = c.lower()

                if low.startswith(self.determiner_prefix):
                    c2 = self.canon(re.sub(r"^(the|a|an|this|that|these|those)\s+", "", c, flags=re.I))
                    if c2:
                        c = c2
                        low = c.lower()

                if low in self.stop_phrases:
                    continue
                if len(c.split()) > 7:
                    continue
                if len(c) < 3:
                    continue

                subsystem, structure, conf = self.classify_span(c, chunk.start_char, chunk.end_char, ner_spans)
                if subsystem == "UNKNOWN" and structure == "UNKNOWN":
                    continue

                spans.append(LabeledSpan(text=c, subsystem=subsystem, structure=structure, start=chunk.start_char, end=chunk.end_char, confidence=conf))

        # 3) regex patterns
        for pat in self.token_patterns:
            for m in re.finditer(pat, raw_text, flags=re.I):
                txt = raw_text[m.start():m.end()]
                c = self.canon(txt)
                if not c:
                    continue
                subsystem, structure, conf = self.classify_span(c, m.start(), m.end(), ner_spans)
                spans.append(LabeledSpan(text=c, subsystem=subsystem, structure=structure, start=m.start(), end=m.end(), confidence=conf))

        # Dedup by node text; keep max confidence
        best: Dict[str, LabeledSpan] = {}
        for s in spans:
            key = self.canon(s.text)
            if key not in best or s.confidence > best[key].confidence:
                best[key] = s

        return list(best.values())

    # -----------------------------
    # Classification core
    # -----------------------------
    def classify_span(self, span_text: str, start: int, end: int, ner_spans) -> Tuple[str, str, float]:
        low = span_text.lower()

        # 0) NER override: protect ORG/LOC/PERSON
        ner = self._ner_override(start, end, ner_spans)
        if ner == "ORG":
            return "OTHER", "ORG", 0.85
        if ner == "LOC":
            return "OTHER", "LOC", 0.85
        if ner == "PERSON":
            return "OTHER", "OTHER", 0.75

        # 1) Subsystem score (weighted)
        best_sub, best_sub_score = self._best_label(low, self.subsystem_lexicon)
        subsystem = best_sub if best_sub_score >= 2 else "OTHER"

        # 2) Structure score (weighted)
        best_struct, best_struct_score = self._best_label(low, self.structure_lexicon)
        structure = best_struct if best_struct_score >= 2 else "OTHER"

        # 3) Special rules
        # numeric altitudes -> PARAMETER (structure) + ORBIT (subsystem)
        if re.search(r"\b\d{2,4}\s?km\b", low):
            structure = "PARAMETER"
            if subsystem in {"OTHER", "UNKNOWN"}:
                subsystem = "ORBIT"

        # instrument words -> INSTRUMENT (structure) + PAYLOAD (subsystem)
        if any(k in low for k in ["spectrometer", "radiometer", "lidar", "sar", "altimeter", "camera", "telescope"]):
            structure = "INSTRUMENT"
            subsystem = "PAYLOAD"

        # mission name heuristic: single proper noun token with no NER hit -> MISSION
        if re.fullmatch(r"[A-Z][a-z]{2,}", span_text) and structure == "OTHER":
            structure = "MISSION"

        # confidence heuristic
        score = max(best_sub_score, best_struct_score)
        conf = 0.55
        if score >= 6:
            conf = 0.90
        elif score >= 4:
            conf = 0.80
        elif score >= 2:
            conf = 0.65
        else:
            conf = 0.40

        # If both ended up OTHER, return UNKNOWN-ish
        if subsystem == "OTHER" and structure == "OTHER":
            return "UNKNOWN", "UNKNOWN", 0.30

        return subsystem, structure, conf

    def _best_label(self, low: str, lex: Dict[str, List[str]]) -> Tuple[str, int]:
        best_label = "OTHER"
        best_score = 0
        for lab, keys in lex.items():
            score = self._score_keys(low, keys)
            if score > best_score:
                best_score = score
                best_label = lab
        return best_label, best_score

    def _score_keys(self, low: str, keys: List[str]) -> int:
        score = 0
        for k in keys:
            k_low = k.lower().strip()
            if not k_low:
                continue
            # +2 for boundary match, +1 for substring match
            if re.search(rf"(?<!\w){re.escape(k_low)}(?!\w)", low):
                score += 2
            elif k_low in low:
                score += 1
        return score

    def _ner_override(self, start: int, end: int, ner_spans) -> Optional[str]:
        best = None
        best_overlap = 0
        for (a0, a1, lab, _) in ner_spans:
            overlap = max(0, min(end, a1) - max(start, a0))
            if overlap > best_overlap:
                best_overlap = overlap
                best = lab
        if not best or best_overlap == 0:
            return None
        if best in {"GPE", "LOC", "FAC"}:
            return "LOC"
        if best == "ORG":
            return "ORG"
        if best == "PERSON":
            return "PERSON"
        return None

    # -----------------------------
    # Normalization
    # -----------------------------
    def canon(self, s: str) -> str:
        x = (s or "").strip()
        if not x:
            return ""
        x = re.sub(r"\s+", " ", x)
        x = x.strip(" ,.;:()[]{}\"'")

        sub_map = str.maketrans({
            "₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4",
            "₅": "5", "₆": "6", "₇": "7", "₈": "8", "₉": "9",
        })
        x = x.translate(sub_map)

        if re.fullmatch(r"[A-Za-z]{1,3}\d{0,3}", x):
            x = x.upper()

        return x
