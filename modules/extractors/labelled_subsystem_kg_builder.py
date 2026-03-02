# modules/extractors/labelled_subsystem_kg_builder.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import os
import re
import csv
import json

import networkx as nx
import spacy


# -----------------------------
# Optional external ontology (safe fallback)
# -----------------------------
try:
    from modules.ontology import Ontology as ExternalOntology  # your existing ontology.py
except Exception:  # pragma: no cover
    ExternalOntology = None


@dataclass
class OntologyMapResult:
    ontology_class: str
    parent_class: str
    confidence: float
    method: str  # "subsystem" | "ner" | "value" | "fallback" | "vocab"


class SubsystemOntology:
    """
    Lightweight subsystem ontology + edge validation.
    """

    def __init__(self):
        # ontology classes (subsystems + generic types)
        self.classes = {
            "TelecomComponent",
            "PowerComponent",
            "DataHandlingComponent",
            "ThermalControlComponent",
            "PropulsionComponent",
            "AOCSComponent",
            "PayloadComponent",
            "GroundSegmentAsset",
            "OrbitParameter",
            "ScienceTerm",
            "Mission",          # ✅ added
            "Organization",
            "Location",
            "Person",
            "Value",
            "Concept",
            "Unknown",
        }

        # Allowed relation schema (domain, predicate, range)
        self.relations = {
            ("TelecomComponent", "has_component", "TelecomComponent"),
            ("PowerComponent", "has_component", "PowerComponent"),
            ("DataHandlingComponent", "has_component", "DataHandlingComponent"),
            ("ThermalControlComponent", "has_component", "ThermalControlComponent"),
            ("PropulsionComponent", "has_component", "PropulsionComponent"),
            ("AOCSComponent", "has_component", "AOCSComponent"),
            ("PayloadComponent", "has_component", "PayloadComponent"),
            ("GroundSegmentAsset", "has_component", "GroundSegmentAsset"),

            ("PayloadComponent", "measures", "ScienceTerm"),
            ("PayloadComponent", "measures", "Value"),
            ("TelecomComponent", "has_limit_max", "Value"),
            ("TelecomComponent", "has_limit_min", "Value"),
            ("TelecomComponent", "has_range", "Value"),

            ("OrbitParameter", "has_range", "Value"),
            ("OrbitParameter", "has_limit_max", "Value"),
            ("OrbitParameter", "has_limit_min", "Value"),

            ("Concept", "related_to", "Concept"),
            ("Concept", "co-occurs", "Concept"),
        }

    def map_node(self, node_text: str, subsystem_label: str, node_kind: str = "CONCEPT") -> OntologyMapResult:
        sub = (subsystem_label or "UNKNOWN").upper().strip()
        nk = (node_kind or "CONCEPT").upper().strip()

        # ✅ mission node kind
        if nk in {"MISSION"}:
            return OntologyMapResult("Mission", "Concept", 0.95, "vocab")

        if nk in {"ORG"}:
            return OntologyMapResult("Organization", "Concept", 0.85, "ner")
        if nk in {"LOC"}:
            return OntologyMapResult("Location", "Concept", 0.85, "ner")
        if nk in {"PERSON"}:
            return OntologyMapResult("Person", "Concept", 0.80, "ner")
        if nk in {"VALUE"}:
            return OntologyMapResult("Value", "Concept", 0.80, "value")
        if nk in {"SCI_TERM"}:
            return OntologyMapResult("ScienceTerm", "Concept", 0.75, "subsystem")

        m = {
            "TELECOM": "TelecomComponent",
            "POWER": "PowerComponent",
            "DATA": "DataHandlingComponent",
            "THERMAL": "ThermalControlComponent",
            "PROPULSION": "PropulsionComponent",
            "AOCS": "AOCSComponent",
            "PAYLOAD": "PayloadComponent",
            "GROUND": "GroundSegmentAsset",
            "ORBIT": "OrbitParameter",
        }.get(sub, None)

        if m:
            return OntologyMapResult(m, "Concept", 0.78, "subsystem")

        if sub in {"OTHER"}:
            return OntologyMapResult("Concept", "Concept", 0.55, "fallback")
        return OntologyMapResult("Unknown", "Concept", 0.35, "fallback")

    def is_valid(self, subj_class: str, predicate: str, obj_class: str) -> bool:
        # allow co-occurrence always
        if predicate == "co-occurs":
            return True
        return (subj_class, predicate, obj_class) in self.relations


# -----------------------------
# Data containers
# -----------------------------
@dataclass
class LabeledSpan:
    text: str
    subsystem: str
    start: int
    end: int
    confidence: float = 0.60
    node_kind: str = "CONCEPT"  # SUBSYSTEM_TERM | SCI_TERM | VALUE | ORG | LOC | PERSON | MISSION | CONCEPT
    source: str = "heuristic"   # acronym | science_vocab | mission_vocab | lexicon | ner | pattern | heuristic


class LabelledSubsystemKGBuilder:
    """
    ESA-friendly subsystem KG builder (Subsystem-only).
    """

    def __init__(
        self,
        spacy_model: str = "en_core_web_sm",
        enable_noun_chunks: bool = True,
        vocab_dir: Optional[str] = None,
        enable_ontology_validation: bool = True,
    ):
        # Keep parser + ner; disable lemmatizer for speed
        self.nlp = spacy.load(spacy_model, disable=["lemmatizer"])
        if "sentencizer" not in self.nlp.pipe_names:
            self.nlp.add_pipe("sentencizer", first=True)

        self.enable_noun_chunks = enable_noun_chunks
        self.enable_ontology_validation = enable_ontology_validation
        self._raw_text: str = ""

        # internal subsystem ontology (safe, always available)
        self.sub_ontology = SubsystemOntology()

        self.external_ontology = ExternalOntology() if ExternalOntology else None

        # -------------------------
        # Baseline ESA-tuned subsystem lexicon (expanded)
        # -------------------------
        self.subsystem_lexicon: Dict[str, List[str]] = {
            "TELECOM": [
                "telemetry", "telecommand", "ttc", "tt&c", "tt & c", "tm", "tc",
                "rf", "ranging", "doppler", "ccsds",
                "transmitter", "receiver", "transceiver", "transponder", "modem",
                "antenna", "lna", "pa", "amplifier", "filter", "diplexer", "duplexer",
                "downlink", "uplink", "link budget", "eirp", "g/t",
                "x-band", "x band", "ka-band", "ka band", "s-band", "s band",
                "bit rate", "data rate", "bandwidth", "frequency", "carrier",
                "modulation", "coding", "fec", "bpsk", "qpsk", "qam",
            ],
            "DATA": [
                "cdhs", "c&dh", "command and data handling", "data handling", "obdh",
                "on-board computer", "onboard computer", "obc", "avionics",
                "mass memory", "ssmm", "solid state mass memory", "memory", "storage",
                "packet", "packets", "telemetry packet", "compression", "data volume",
                "spacewire", "spw", "can bus",
                "mil-std-1553b", "mil-1553", "1553b",
                "remote interface unit", "riu", "rtu",
                "pdht", "payload data handling", "payload data handling and transmission",
                "processing chain", "level-0", "level-1", "level-1b", "level-2", "product generation",
                "fdir", "fault detection", "fault isolation", "fault recovery",
            ],
            "POWER": [
                "eps", "electrical power subsystem", "electrical power",
                "solar array", "solar arrays", "solar panel", "solar panels",
                "solar cell", "solar cells",
                "battery", "batteries", "battery assembly",
                "pcdu", "pdu", "power distribution", "power conditioning",
                "converter", "dc-dc", "dcdc", "regulator", "mppt",
                "bus voltage", "power bus", "unregulated bus",
                "28 v", "28v", "28-v", "28-vdc", "28 vdc",
                "lcl", "lcls", "fcl", "fcls", "latching current limiter", "current limiter",
                "power budget", "power consumption",
            ],
            "THERMAL": [
                "thermal", "thermal control", "tcs",
                "radiator", "radiators", "radiator panel", "radiator panels",
                "heater", "heaters", "heater line", "heater lines", "htr",
                "mli", "multi-layer insulation", "insulation",
                "heat pipe", "heat pipes", "loop heat pipe", "lhp",
                "thermal strap", "thermal doubler",
                "thermistor", "thermostat", "temperature sensor",
                "high emissivity", "coating", "louver", "osr",
                "temperature", "thermal stability", "thermal dissipation",
            ],
            "AOCS": [
                "aocs", "adcs", "gnc",
                "attitude", "attitude control", "attitude determination",
                "orbit determination", "pointing", "slew", "stability", "jitter",
                "reaction wheel", "reaction wheels", "momentum wheel", "rwa",
                "star tracker", "star trackers", "str",
                "gyro", "gyroscope", "gyros", "imu",
                "magnetorquer", "magnetorquers", "magnetometer", "mtq",
                "sun sensor", "sun sensors", "coarse sun sensor", "css",
                "earth sensor", "earth sensors",
                "gnss", "gps", "kalman filter",
            ],
            "PROPULSION": [
                "propulsion", "thruster", "thrusters",
                "propellant", "propellants", "propellant tank", "tank", "tanks",
                "feed system", "nozzle", "valve", "valves",
                "pressurant", "helium pressurant", "pressurization",
                "hydrazine", "xenon", "mmh", "nto",
                "blow-down", "blow down",
                "monopropellant", "bipropellant",
                "delta-v", "Δv", "orbit raising", "station keeping",
                "collision avoidance", "deorbit", "de-orbit",
                "hall thruster", "ion thruster", "het", "rit", "ppu",
            ],
            "PAYLOAD": [
                "payload", "payload suite", "instrument", "instruments",
                "spectrometer", "radiometer", "imager", "imaging",
                "lidar", "sar", "altimeter", "camera", "telescope",
                "detector", "detectors", "focal plane", "fpa",
                "calibration", "science data", "measurement",
            ],
            "GROUND": [
                "ground segment", "ground station", "tracking station", "estrack",
                "mission operations", "mission control",
                "operations centre", "operations center",
                "fos", "focc", "pdgs", "fds", "mcs", "mps", "mission planning",
                "kiruna", "svalbard", "ksat", "ssc",
                "payload data ground segment", "flight operations segment", "flight dynamics",
            ],
            "ORBIT": [
                "orbit", "leo", "low earth orbit", "sso", "sun-synchronous", "sun synchronous", "near-polar",
                "inclination", "altitude", "eccentricity", "raan",
                "semi-major axis", "orbital period", "apogee", "perigee",
                "ground track", "repeat cycle", "ltan", "local time of ascending node",
            ],
        }

        self.ground_strong_cues = [
            "pdgs", "focc", "fos", "estrack", "kiruna", "svalbard",
            "mission planning", "ground segment", "mcs", "mps", "fds",
        ]
        self.data_strong_cues = [
            "cdhs", "obc", "mass memory", "ssmm", "spacewire", "1553",
            "mil-std-1553", "pdht", "compression", "processing chain",
            "level-1", "level-2", "fdir",
        ]
        self.telecom_strong_cues = [
            "tt&c", "ranging", "x-band", "s-band", "ka-band",
            "antenna", "rf", "transceiver", "transponder",
            "downlink", "uplink",
        ]

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
            r"\bMIL-STD-1553B\b",
            r"\b(?:TT&C|CDHS|EPS|PCDU|PDHT|PDGS|FOCC|FOS|ESTRACK|SSMM|OBC|RIU|RTU|LCL|MPPT)\b",
        ]

        # -------------------------
        # Load curated vocab files
        # -------------------------
        if vocab_dir is None:
            here = os.path.dirname(os.path.abspath(__file__))       # .../modules/extractors
            project_root = os.path.abspath(os.path.join(here, "..", ".."))
            vocab_dir = os.path.join(project_root, "data", "vocab")

        self.vocab_dir = vocab_dir
        self.acronym_to_subsystem: Dict[str, str] = {}
        self.acronym_expansions: Dict[str, str] = {}
        self.science_terms: Dict[str, Tuple[str, float]] = {}          # term_low -> (subsystem, conf)
        self.missions: Dict[str, Tuple[str, str]] = {}                  # ✅ name_low -> (canonical, type)

        self._load_vocab_files(self.vocab_dir)

        self._acronym_override: Dict[str, str] = {}
        for acr, sub in self.acronym_to_subsystem.items():
            self._acronym_override[acr.upper()] = sub
            self._acronym_override[acr.upper().replace(" ", "")] = sub

        for term_low, (sub, _) in self.science_terms.items():
            if sub in self.subsystem_lexicon and len(term_low) >= 4 and term_low not in {"mission", "spacecraft", "system"}:
                self.subsystem_lexicon[sub].append(term_low)

        self._compiled: Dict[str, List[re.Pattern]] = {
            label: [self._compile_term(t) for t in sorted(set(terms), key=lambda x: (-len(x), x))]
            for label, terms in self.subsystem_lexicon.items()
        }

    # -----------------------------
    # Public
    # -----------------------------
    def build(self, text: str, min_conf: float = 0.0) -> nx.Graph:
        self._raw_text = text or ""
        doc = self.nlp(self._raw_text)

        spans = self.extract_spans(doc, self._raw_text)
        spans = [s for s in spans if s.confidence >= min_conf]

        g = nx.Graph()

        # Nodes
        for s in spans:
            node = self.canon(s.text)
            if not node:
                continue

            if node in g:
                if s.confidence > float(g.nodes[node].get("confidence", 0.0)):
                    g.nodes[node]["subsystem_label"] = s.subsystem
                    g.nodes[node]["confidence"] = float(s.confidence)
                    g.nodes[node]["node_kind"] = s.node_kind
                    g.nodes[node]["source"] = s.source
                continue

            g.add_node(
                node,
                subsystem_label=s.subsystem,
                confidence=float(s.confidence),
                node_kind=s.node_kind,
                source=s.source,
                start=int(s.start),
                end=int(s.end),
            )

        # Ontology mapping per node
        for n, attrs in g.nodes(data=True):
            res = self.sub_ontology.map_node(n, attrs.get("subsystem_label", "UNKNOWN"), attrs.get("node_kind", "CONCEPT"))
            attrs["ontology_class"] = res.ontology_class
            attrs["ontology_parent"] = res.parent_class
            attrs["ontology_conf"] = float(res.confidence)
            attrs["ontology_method"] = res.method

        # Edges
        for sent in doc.sents:
            a0, a1 = sent.start_char, sent.end_char
            in_sent: List[str] = []
            in_sent_conf: Dict[str, float] = {}

            for s in spans:
                if s.start >= a0 and s.end <= a1:
                    node = self.canon(s.text)
                    if node:
                        in_sent.append(node)
                        in_sent_conf[node] = max(in_sent_conf.get(node, 0.0), s.confidence)

            in_sent = list(dict.fromkeys(in_sent))

            # 1) Relation extraction
            rel_triples = self.extract_relations(sent, in_sent)
            for u, pred, v, conf in rel_triples:
                u2 = self.canon(u)
                v2 = self.canon(v)
                if not u2 or not v2 or u2 == v2:
                    continue

                if v2 not in g:
                    g.add_node(
                        v2,
                        subsystem_label="OTHER",
                        confidence=float(conf),
                        node_kind="VALUE" if self._looks_like_value(v2) else "CONCEPT",
                        source="pattern",
                        start=-1,
                        end=-1,
                    )
                    res = self.sub_ontology.map_node(v2, g.nodes[v2].get("subsystem_label", "UNKNOWN"), g.nodes[v2].get("node_kind", "CONCEPT"))
                    g.nodes[v2]["ontology_class"] = res.ontology_class
                    g.nodes[v2]["ontology_parent"] = res.parent_class
                    g.nodes[v2]["ontology_conf"] = float(res.confidence)
                    g.nodes[v2]["ontology_method"] = res.method

                self._add_edge(g, u2, v2, pred, conf)

            # 2) HUB-based co-occurrence
            if len(in_sent) >= 2:
                hub = max(in_sent, key=lambda n: in_sent_conf.get(n, 0.0))
                for n in in_sent:
                    if n == hub:
                        continue
                    self._add_edge(g, hub, n, "co-occurs", 0.35)

        if self.enable_ontology_validation:
            self._validate_edges(g)

        isolates = [n for n in list(g.nodes()) if g.degree(n) == 0]
        g.remove_nodes_from(isolates)

        return g

    # -----------------------------
    # Span extraction
    # -----------------------------
    def extract_spans(self, doc, raw_text: str) -> List[LabeledSpan]:
        spans: List[LabeledSpan] = []
        ner_spans = [(e.start_char, e.end_char, e.label_, e.text) for e in doc.ents]

        raw_low = raw_text.lower()

        # 0) Acronym tokens (high precision)
        for m in re.finditer(r"\b[A-Z][A-Z0-9&/\-]{1,20}\b", raw_text):
            txt = raw_text[m.start():m.end()]
            c = self.canon_display(txt)
            if not c:
                continue
            lab = self._acronym_override.get(c.upper()) or self._acronym_override.get(c.upper().replace(" ", ""))
            if lab:
                spans.append(LabeledSpan(text=c, subsystem=lab, start=m.start(), end=m.end(), confidence=0.92, node_kind="SUBSYSTEM_TERM", source="acronym"))

        # 1) TitleCase phrases
        for m in re.finditer(r"\b(?:[A-Z][a-zA-Z0-9\-]+(?:\s+[A-Z][a-zA-Z0-9\-]+){0,4})\b", raw_text):
            txt = raw_text[m.start():m.end()]
            c = self.canon_display(txt)
            if not c or len(c) < 3:
                continue

            low = c.lower()
            if low in {"accordingly", "answers", "complex", "region", "earth", "solar system"}:
                continue

            subsystem, conf, kind, source = self.classify_span(c, m.start(), m.end(), ner_spans)
            if subsystem == "UNKNOWN":
                continue
            spans.append(LabeledSpan(text=c, subsystem=subsystem, start=m.start(), end=m.end(), confidence=conf, node_kind=kind, source=source))

        # 2) noun chunks
        if self.enable_noun_chunks and doc.has_annotation("DEP"):
            for chunk in doc.noun_chunks:
                txt = chunk.text.strip()
                c = self.canon_display(txt)
                if not c:
                    continue
                low = c.lower()

                if low.startswith(self.determiner_prefix):
                    c2 = self.canon_display(re.sub(r"^(the|a|an|this|that|these|those)\s+", "", c, flags=re.I))
                    if c2:
                        c = c2
                        low = c.lower()

                if low in self.stop_phrases:
                    continue
                if len(c.split()) > 7:
                    continue
                if len(c) < 3:
                    continue

                subsystem, conf, kind, source = self.classify_span(c, chunk.start_char, chunk.end_char, ner_spans)
                if subsystem == "UNKNOWN":
                    continue
                spans.append(LabeledSpan(text=c, subsystem=subsystem, start=chunk.start_char, end=chunk.end_char, confidence=conf, node_kind=kind, source=source))

        # 3) regex anchors
        for pat in self.token_patterns:
            for m in re.finditer(pat, raw_text, flags=re.I):
                txt = raw_text[m.start():m.end()]
                c = self.canon_display(txt)
                if not c:
                    continue
                subsystem, conf, kind, source = self.classify_span(c, m.start(), m.end(), ner_spans)
                if subsystem == "UNKNOWN":
                    continue
                spans.append(LabeledSpan(text=c, subsystem=subsystem, start=m.start(), end=m.end(), confidence=conf, node_kind=kind, source=source))

        # 4) science terms (CSV)
        for term_low, (sub, conf0) in self.science_terms.items():
            if len(term_low) < 4:
                continue
            if re.search(rf"(?<!\w){re.escape(term_low)}(?!\w)", raw_low):
                idx = raw_low.find(term_low)
                if idx >= 0:
                    spans.append(
                        LabeledSpan(
                            text=raw_text[idx:idx + len(term_low)],
                            subsystem=sub,
                            start=idx,
                            end=idx + len(term_low),
                            confidence=min(0.88, max(0.60, float(conf0))),
                            node_kind="SCI_TERM",
                            source="science_vocab",
                        )
                    )

        best: Dict[str, LabeledSpan] = {}
        for s in spans:
            key = self.canon(s.text)
            if not key:
                continue
            if key not in best or s.confidence > best[key].confidence:
                best[key] = s

        return list(best.values())

    # -----------------------------
    # Classification (subsystem only)
    # -----------------------------
    def classify_span(self, span_text: str, start: int, end: int, ner_spans) -> Tuple[str, float, str, str]:
        low = span_text.lower().strip()

        # Acronym override
        up = span_text.strip().upper()
        sub = self._acronym_override.get(up) or self._acronym_override.get(up.replace(" ", ""))
        if sub:
            return sub, 0.92, "SUBSYSTEM_TERM", "acronym"

        # ✅ Mission vocab override (runs BEFORE NER override)
        if low in self.missions:
            # mission is not a subsystem, so keep subsystem_label="OTHER"
            return "OTHER", 0.95, "MISSION", "mission_vocab"

        # NER override: ORG/LOC/PERSON -> OTHER
        ner = self._ner_override(start, end, ner_spans)
        if ner in {"ORG", "LOC", "PERSON"}:
            return "OTHER", 0.70, ner, "ner"

        # Direct science-term hit
        if low in self.science_terms:
            s2, c0 = self.science_terms[low]
            return s2, float(min(0.88, max(0.60, c0))), "SCI_TERM", "science_vocab"

        # Lexicon scoring
        best_lab, best_hits = self._best_label(low)

        # Context tie-break
        window = self._context_window(start, end, self._raw_text, win=160)
        best_lab = self._tie_break(best_lab, window, span_text)

        # Confidence
        if best_hits >= 4:
            conf = 0.90
        elif best_hits >= 2:
            conf = 0.75
        elif best_hits >= 1:
            conf = 0.62
        else:
            if self._looks_like_value(low) or re.search(r"\b\d{2,4}\s?km\b", low):
                return "ORBIT", 0.60, "VALUE", "pattern"
            return "UNKNOWN", 0.30, "CONCEPT", "heuristic"

        kind = "SUBSYSTEM_TERM" if best_lab in {
            "TELECOM", "POWER", "DATA", "PROPULSION", "THERMAL", "AOCS", "GROUND", "ORBIT", "PAYLOAD"
        } else "CONCEPT"
        return best_lab, conf, kind, "lexicon"

    # -----------------------------
    # Relation extraction
    # -----------------------------
    def extract_relations(self, sent, span_nodes: List[str]) -> List[Tuple[str, str, str, float]]:
        rels: List[Tuple[str, str, str, float]] = []
        text = sent.text.strip()
        low = text.lower()

        if not span_nodes:
            return rels

        m = re.search(r"\bbetween\s+(\d+(?:\.\d+)?)\s*([a-zA-Z%°/]+)?\s+and\s+(\d+(?:\.\d+)?)\s*([a-zA-Z%°/]+)?\b", low)
        if m:
            unit = (m.group(2) or m.group(4) or "")
            vals = f"{m.group(1)}{unit}-{m.group(3)}{unit}"
            rels.append((span_nodes[0], "has_range", vals, 0.78))

        if any(k in low for k in ["no greater than", "at most", "maximum of", "not exceed", "≤", "<="]):
            mv = re.search(r"(?:no greater than|at most|maximum of|not exceed|≤|<=)\s+(\d+(?:\.\d+)?)\s*([a-zA-Z%°/]+)?", low)
            if mv:
                val = f"{mv.group(1)}{mv.group(2) or ''}"
                rels.append((span_nodes[0], "has_limit_max", val, 0.80))

        if any(k in low for k in ["no less than", "at least", "minimum of", "≥", ">="]):
            mv = re.search(r"(?:no less than|at least|minimum of|≥|>=)\s+(\d+(?:\.\d+)?)\s*([a-zA-Z%°/]+)?", low)
            if mv:
                val = f"{mv.group(1)}{mv.group(2) or ''}"
                rels.append((span_nodes[0], "has_limit_min", val, 0.80))

        for token in sent:
            if token.lemma_ in {"have", "use", "provide", "measure", "include"}:
                subj = [w for w in token.lefts if w.dep_ in {"nsubj", "nsubjpass"}]
                objs = [w for w in token.rights if w.dep_ in {"dobj", "attr", "pobj", "dative", "oprd"}]
                if not subj or not objs:
                    continue

                s = subj[0].text
                o = objs[0].text
                pred = {
                    "have": "has_component",
                    "include": "has_component",
                    "use": "uses",
                    "provide": "provides",
                    "measure": "measures",
                }.get(token.lemma_, "related_to")

                s_c = self.canon(s)
                o_c = self.canon(o)
                if (s_c in span_nodes) or (o_c in span_nodes):
                    rels.append((s, pred, o, 0.70))

        return rels

    # -----------------------------
    # Lexicon scoring + tie-break
    # -----------------------------
    def _best_label(self, text_low: str) -> Tuple[str, int]:
        best_label = "OTHER"
        best_hits = 0
        for label, patterns in self._compiled.items():
            hits = 0
            for pat in patterns:
                if pat.search(text_low):
                    hits += 1
            if hits > best_hits:
                best_hits = hits
                best_label = label
        return best_label, best_hits

    def _tie_break(self, label: str, context: str, node_text: str) -> str:
        s = (context or "").lower()
        n = (node_text or "").lower()

        if label in {"TELECOM", "DATA", "GROUND"}:
            if any(c in s for c in self.ground_strong_cues) or any(c in n for c in self.ground_strong_cues):
                return "GROUND"

        if label in {"TELECOM", "DATA"}:
            has_data = any(c in s for c in self.data_strong_cues) or any(c in n for c in self.data_strong_cues)
            has_tel = any(c in s for c in self.telecom_strong_cues) or any(c in n for c in self.telecom_strong_cues)
            if has_data and not has_tel:
                return "DATA"
            if has_tel and not has_data:
                return "TELECOM"
            if has_data and has_tel:
                if any(k in n for k in ["rf", "antenna", "transceiver", "transmitter", "receiver", "x-band", "s-band", "ka-band", "transponder"]):
                    return "TELECOM"
                return "DATA"

        return label

    def _compile_term(self, term: str) -> re.Pattern:
        t = term.strip()
        t_esc = re.escape(t)
        t_esc = t_esc.replace(r"\-", r"[\-\s]?")
        t_esc = t_esc.replace(r"\ ", r"\s+")
        return re.compile(rf"(?<!\w){t_esc}(?!\w)", flags=re.I)

    def _context_window(self, start: int, end: int, text: str, win: int = 160) -> str:
        a0 = max(0, start - win)
        a1 = min(len(text), end + win)
        return text[a0:a1]

    # -----------------------------
    # NER + value helpers
    # -----------------------------
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

    def _looks_like_value(self, txt: str) -> bool:
        return bool(re.search(
            r"\b\d+(?:\.\d+)?\s*(?:km|m|s|ms|w|mw|kw|hz|khz|mhz|ghz|v|mv|a|ma|db|dbi|kbps|mbps|gbps|kbit/s|mbit/s|gbit/s|°c|c|k|%)\b",
            txt.lower()
        ))

    # -----------------------------
    # Edge helpers + ontology validation
    # -----------------------------
    def _add_edge(self, g: nx.Graph, u: str, v: str, pred: str, conf: float) -> None:
        if not u or not v or u == v:
            return
        if g.has_edge(u, v):
            g[u][v]["weight"] = int(g[u][v].get("weight", 1)) + 1
            g[u][v]["confidence"] = float(min(1.0, float(g[u][v].get("confidence", 0.35)) + 0.05))
            if g[u][v].get("predicate") == "co-occurs" and pred != "co-occurs":
                g[u][v]["predicate"] = pred
                g[u][v]["label"] = pred
        else:
            g.add_edge(u, v, predicate=pred, label=pred, weight=1, confidence=float(conf))

    def _validate_edges(self, g: nx.Graph) -> None:
        for u, v, data in list(g.edges(data=True)):
            pred = str(data.get("predicate") or data.get("label") or "co-occurs")
            if pred == "co-occurs":
                continue

            su = g.nodes[u].get("ontology_class", "Concept")
            sv = g.nodes[v].get("ontology_class", "Concept")

            if not self.sub_ontology.is_valid(su, pred, sv):
                data["confidence"] = float(min(0.25, float(data.get("confidence", 0.35))))
                data["predicate"] = "co-occurs"
                data["label"] = "co-occurs"

    # -----------------------------
    # Normalization
    # -----------------------------
    def canon(self, s: str) -> str:
        x = (s or "").strip()
        if not x:
            return ""
        x = re.sub(r"\s+", " ", x).strip(" ,.;:()[]{}\"'")
        x = x.replace("–", "-").replace("—", "-")

        if re.fullmatch(r"[A-Z0-9&/\-]{2,}", x):
            pass
        else:
            x = x.lower()

        if x.endswith("ies") and len(x) > 4:
            x = x[:-3] + "y"
        elif x.endswith("s") and len(x) > 4 and not x.endswith("ss") and not x.endswith("bus"):
            x = x[:-1]
        return x

    def canon_display(self, s: str) -> str:
        x = (s or "").strip()
        if not x:
            return ""
        x = re.sub(r"\s+", " ", x).strip(" ,.;:()[]{}\"'")
        sub_map = str.maketrans({
            "₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4",
            "₅": "5", "₆": "6", "₇": "7", "₈": "8", "₉": "9",
        })
        return x.translate(sub_map)

    # -----------------------------
    # Vocab loading
    # -----------------------------
    def _load_vocab_files(self, vocab_dir: str) -> None:
        """
        Loads:
          - data/vocab/esa_acronym_map.json
          - data/vocab/esa_science_terms.csv
          - data/vocab/esa_missions.csv   ✅ added
        """
        if not vocab_dir or not os.path.isdir(vocab_dir):
            return

        acr_path = os.path.join(vocab_dir, "esa_acronym_map.json")
        if os.path.exists(acr_path):
            try:
                with open(acr_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    for subsystem, acrs in data.items():
                        sub = (subsystem or "").upper().strip()
                        if not isinstance(acrs, dict):
                            continue
                        for acr, expansion in acrs.items():
                            if not acr:
                                continue
                            a = str(acr).strip().upper()
                            self.acronym_to_subsystem[a] = sub
                            if expansion:
                                self.acronym_expansions[a] = str(expansion).strip()
            except Exception:
                pass

        sci_path = os.path.join(vocab_dir, "esa_science_terms.csv")
        if os.path.exists(sci_path):
            try:
                with open(sci_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        term = (row.get("term") or "").strip()
                        sub = (row.get("subsystem") or "").strip().upper()
                        conf_s = (row.get("confidence") or "").strip()
                        if not term or not sub:
                            continue
                        try:
                            conf = float(conf_s) if conf_s else 0.75
                        except Exception:
                            conf = 0.75
                        self.science_terms[term.lower()] = (sub, float(conf))
            except Exception:
                pass

        # ✅ Missions CSV
        missions_path = os.path.join(vocab_dir, "esa_missions.csv")
        if os.path.exists(missions_path):
            try:
                with open(missions_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        name = (row.get("name") or "").strip()
                        canonical = (row.get("canonical") or name).strip()
                        typ = (row.get("type") or "MISSION").strip().upper()
                        if not name:
                            continue
                        self.missions[name.lower()] = (canonical, typ)
            except Exception:
                pass
