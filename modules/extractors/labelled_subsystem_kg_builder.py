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
    node_kind: str = "CONCEPT"
    source: str = "heuristic"  # "ner" | "lexicon" | "pattern" | "heuristic"


class LabelledSubsystemKGBuilder:
    """
    Mission KG builder with TWO labels per node:
      - subsystem_label: TELECOM/POWER/DATA/PAYLOAD/ORBIT/GROUND/PROPULSION/THERMAL/AOCS/...
      - structure_label: MISSION/ORBIT/PAYLOAD/INSTRUMENT/GROUND/TARGET_REGION/PARAMETER/ORG/LOC/...

    Improvements vs previous version:
      - safer mission-name heuristic (no random TitleCase -> MISSION)
      - better canonicalization (lowercase, hyphen normalize, cheap singularization)
      - expanded subsystem lexicon (your listed terms)
      - relation extraction (has/uses/provides/measures + range/limit patterns)
      - co-occurrence edges are HUB-based (no sentence cliques)
      - node_kind + source stored for better visualization/debugging
    """

    def __init__(self, spacy_model: str = "en_core_web_sm", enable_noun_chunks: bool = True):
        # Keep parser + ner; disable lemmatizer for speed.
        self.nlp = spacy.load(spacy_model, disable=["lemmatizer"])
        if "sentencizer" not in self.nlp.pipe_names:
            self.nlp.add_pipe("sentencizer", first=True)

        self.enable_noun_chunks = enable_noun_chunks

        # Used for safer mission heuristic
        self._raw_text: str = ""

        # -------------------------
        # Subsystem label space (expanded)
        # -------------------------
        self.subsystem_lexicon: Dict[str, List[str]] = {
            "TELECOM": [
                "telecom", "telecommunication", "telecommunications", "ttc", "tt&c",
                "downlink", "uplink", "x-band", "ka-band", "s-band",
                "antenna", "transmitter", "receiver", "transceiver",
                "rf", "radio frequency",
                "link budget", "bit rate", "data rate", "telemetry",
                "modulation", "bandwidth", "frequency",
                "bpsk", "qpsk", "qam", "fec", "coding", "symbol rate", "carrier",
                "lna", "pa", "amplifier", "filter", "diplexer", "duplexer",
                "command", "commanding", "telecommand",
                "downlink rate", "uplink rate",
            ],
            "PROPULSION": [
                "propulsion", "thruster", "thrusters", "propellant", "propellants",
                "tank", "tanks", "nozzle", "pressurant", "feed system",
                "delta-v", "orbit raising", "station keeping",
                "chemical propulsion", "electric propulsion",
                "hydrazine", "xenon",
                "hall thruster", "ion thruster", "monopropellant", "bipropellant",
                "valve", "regulator", "blowdown",
            ],
            "POWER": [
                "power", "eps", "electrical power subsystem",
                "solar array", "solar arrays", "solar panel", "solar panels",
                "solar cell", "solar cells",
                "battery", "batteries",
                "power distribution", "pdu", "pcdu", "power conditioning",
                "converter", "dc-dc", "dcdc", "regulator",
                "bus", "power bus", "bus voltage", "pdu regulator",
                "pdu", "pdu board",
            ],
            "THERMAL": [
                "thermal", "thermal control", "thermal subsystem",
                "radiator", "radiators", "radiator panel", "radiator panels",
                "heater", "heaters", "insulation", "mli",
                "temperature control", "cooling", "warming",
                "heat pipe", "heat pipes", "thermal strap", "thermostat",
            ],
            "AOCS": [
                "aocs", "adcs", "attitude", "attitude control", "attitude determination",
                "orbit determination", "od", "kalman filter",
                "reaction wheel", "reaction wheels",
                "star tracker", "star trackers",
                "gyroscope", "gyroscopes", "gyro", "gyros",
                "magnetorquer", "magnetorquers",
                "magnetometer", "magnetometers",
                "sun sensor", "sun sensors", "earth sensor", "earth sensors",
            ],
            "DATA": [
                "data", "telemetry", "telecommand", "tm", "tc",
                "onboard data handling", "data handling", "cdh", "obdh",
                "onboard computer", "flight computer", "avionics",
                "mass memory", "memory", "storage", "secure storage",
                "processing", "onboard processing", "compression",
                "data volume", "downlinked data", "packet", "packets",
            ],
            "PAYLOAD": [
                "payload", "instrument", "instruments",
                "spectrometer", "imaging spectrometer", "radiometer",
                "lidar", "sar", "altimeter", "camera", "telescope",
                "detector", "detectors",
            ],
            "GROUND": [
                "ground", "ground segment", "ground station", "tracking station",
                "mission operations", "mission control",
                "receiving station", "data centre", "data center",
                "network", "antenna farm",
            ],
            "ORBIT": [
                "orbit", "sun-synchronous", "sun synchronous", "leo",
                "low earth orbit", "near-polar", "inclination",
                "ascending node", "raan", "altitude", "apogee", "perigee",
            ],
        }

        # -------------------------
        # Mission structure label space
        # -------------------------
        self.structure_lexicon: Dict[str, List[str]] = {
            "MISSION": [
                "mission", "mission candidate", "earth explorer", "phase 0", "phase a",
                "selection", "candidate", "concept study", "spacecraft", "satellite",
                "probe", "orbiter", "lander", "rover",
            ],
            "ORBIT": [
                "orbit", "sun-synchronous", "sun synchronous", "leo",
                "low earth orbit", "inclination", "ascending node", "altitude",
                "apogee", "perigee", "raan",
            ],
            "PAYLOAD": [
                "payload", "payload suite", "instrument", "instruments",
            ],
            "INSTRUMENT": [
                "spectrometer", "radiometer", "lidar", "sar", "altimeter", "camera", "telescope",
                "detector",
            ],
            "GROUND": [
                "ground segment", "ground station", "tracking station",
                "mission operations", "mission control",
            ],
            "TARGET_REGION": [
                "thermosphere", "ionosphere", "upper atmosphere", "lower thermosphere",
                "stratosphere", "troposphere", "mesosphere", "region",
            ],
            "PARAMETER": [
                "altitude", "temperature", "density",
                "flux", "heating", "radiance",
                "bandwidth", "frequency", "bit rate", "data rate",
                "voltage", "current", "power", "mass", "thrust",
            ],
            "SUBSYSTEM": [
                "telecom subsystem", "power subsystem", "data handling subsystem",
                "propulsion subsystem", "thermal subsystem", "aocs", "adcs",
            ],
            "ORG": [
                "esa", "nasa", "jaxa", "isro", "cnes", "dlr",
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

        # Regex patterns that are useful anchors (kept as nodes)
        self.token_patterns = [
            r"\b(?:sun-?synchronous|low earth orbit|leo|near-?polar)\b",
            r"\b\d{2,4}\s?km\b",
            r"\bESA\b",
            r"\bNASA\b",
            r"\bEarth Explorer\b",
            r"\bLiving Planet Programme\b",
        ]

        # Instrument keywords (used by special rules)
        self.instrument_keywords = ["spectrometer", "radiometer", "lidar", "sar", "altimeter", "camera", "telescope", "detector"]

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
                    g.nodes[node]["structure_label"] = s.structure
                    g.nodes[node]["confidence"] = float(s.confidence)
                    g.nodes[node]["node_kind"] = s.node_kind
                    g.nodes[node]["source"] = s.source
                continue

            g.add_node(
                node,
                subsystem_label=s.subsystem,
                structure_label=s.structure,
                confidence=float(s.confidence),
                node_kind=s.node_kind,
                source=s.source,
                start=int(s.start),
                end=int(s.end),
            )

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

            # unique preserving order
            in_sent = list(dict.fromkeys(in_sent))

            # 1) Relation extraction (adds meaningful edges)
            rel_triples = self.extract_relations(sent, in_sent)
            for u, pred, v, conf in rel_triples:
                u2 = self.canon(u)
                v2 = self.canon(v)
                if not u2 or not v2 or u2 == v2:
                    continue

                # Ensure relation object exists as node if it's not already
                if v2 not in g:
                    # relation objects may be numbers/values; treat as PARAMETER by default
                    g.add_node(
                        v2,
                        subsystem_label="OTHER",
                        structure_label="PARAMETER" if self._looks_like_value(v2) else "OTHER",
                        confidence=float(conf),
                        node_kind="VALUE" if self._looks_like_value(v2) else "CONCEPT",
                        source="pattern",
                        start=-1,
                        end=-1,
                    )

                self._add_edge(g, u2, v2, pred, conf)

            # 2) Co-occurrence edges (HUB-based, no cliques)
            if len(in_sent) >= 2:
                # pick hub with highest confidence in sentence
                hub = max(in_sent, key=lambda n: in_sent_conf.get(n, 0.0))
                for n in in_sent:
                    if n == hub:
                        continue
                    self._add_edge(g, hub, n, "co-occurs", 0.35)

        # Optional: remove isolated nodes (often junk)
        isolates = [n for n in g.nodes() if g.degree(n) == 0]
        g.remove_nodes_from(isolates)

        return g

    # -----------------------------
    # Spans
    # -----------------------------
    def extract_spans(self, doc, raw_text: str) -> List[LabeledSpan]:
        spans: List[LabeledSpan] = []

        ner_spans = [(e.start_char, e.end_char, e.label_, e.text) for e in doc.ents]

        # 1) TitleCase sequences (keep, but no longer blindly treated as mission)
        for m in re.finditer(r"\b(?:[A-Z][a-zA-Z0-9\-]+(?:\s+[A-Z][a-zA-Z0-9\-]+){0,4})\b", raw_text):
            txt = raw_text[m.start():m.end()]
            c = self.canon_display(txt)
            if not c or len(c) < 3:
                continue

            low = c.lower()
            if low in {"accordingly", "answers", "complex", "region", "earth", "solar system"}:
                continue

            subsystem, structure, conf, kind, source = self.classify_span(c, m.start(), m.end(), ner_spans)
            spans.append(LabeledSpan(text=c, subsystem=subsystem, structure=structure, start=m.start(), end=m.end(),
                                     confidence=conf, node_kind=kind, source=source))

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

                subsystem, structure, conf, kind, source = self.classify_span(c, chunk.start_char, chunk.end_char, ner_spans)
                if subsystem == "UNKNOWN" and structure == "UNKNOWN":
                    continue

                spans.append(LabeledSpan(text=c, subsystem=subsystem, structure=structure, start=chunk.start_char, end=chunk.end_char,
                                         confidence=conf, node_kind=kind, source=source))

        # 3) regex patterns
        for pat in self.token_patterns:
            for m in re.finditer(pat, raw_text, flags=re.I):
                txt = raw_text[m.start():m.end()]
                c = self.canon_display(txt)
                if not c:
                    continue
                subsystem, structure, conf, kind, source = self.classify_span(c, m.start(), m.end(), ner_spans)
                spans.append(LabeledSpan(text=c, subsystem=subsystem, structure=structure, start=m.start(), end=m.end(),
                                         confidence=conf, node_kind=kind, source=source))

        # Dedup by canonical node id; keep max confidence
        best: Dict[str, LabeledSpan] = {}
        for s in spans:
            key = self.canon(s.text)
            if not key:
                continue
            if key not in best or s.confidence > best[key].confidence:
                best[key] = s

        return list(best.values())

    # -----------------------------
    # Classification core
    # -----------------------------
    def classify_span(self, span_text: str, start: int, end: int, ner_spans) -> Tuple[str, str, float, str, str]:
        low = span_text.lower()

        # 0) NER override: protect ORG/LOC/PERSON
        ner = self._ner_override(start, end, ner_spans)
        if ner == "ORG":
            return "OTHER", "ORG", 0.88, "ORG", "ner"
        if ner == "LOC":
            return "OTHER", "LOC", 0.88, "LOC", "ner"
        if ner == "PERSON":
            return "OTHER", "OTHER", 0.75, "PERSON", "ner"

        # 1) Subsystem score (weighted)
        best_sub, best_sub_score = self._best_label(low, self.subsystem_lexicon)
        subsystem = best_sub if best_sub_score >= 2 else "OTHER"

        # 2) Structure score (weighted)
        best_struct, best_struct_score = self._best_label(low, self.structure_lexicon)
        structure = best_struct if best_struct_score >= 2 else "OTHER"

        # 3) Special rules

        # numeric altitudes / values -> PARAMETER (structure) + ORBIT (subsystem if unknown)
        if re.search(r"\b\d{2,4}\s?km\b", low) or self._looks_like_value(low):
            structure = "PARAMETER"
            if subsystem in {"OTHER", "UNKNOWN"}:
                subsystem = "ORBIT"

        # instrument words -> INSTRUMENT (structure) + PAYLOAD (subsystem)
        if any(k in low for k in self.instrument_keywords):
            structure = "INSTRUMENT"
            subsystem = "PAYLOAD"

        # safer mission heuristic: only if near mission cues
        if structure in {"OTHER", "UNKNOWN"} and self._mission_like(span_text, self._raw_text):
            structure = "MISSION"

        # Decide node_kind
        node_kind = "CONCEPT"
        if structure == "ORG":
            node_kind = "ORG"
        elif structure == "LOC":
            node_kind = "LOC"
        elif structure == "PARAMETER":
            node_kind = "PARAMETER"
        elif structure == "INSTRUMENT":
            node_kind = "INSTRUMENT"
        elif structure == "MISSION":
            node_kind = "MISSION"
        elif structure == "GROUND":
            node_kind = "GROUND_ASSET"
        elif subsystem in {"TELECOM", "POWER", "DATA", "PROPULSION", "THERMAL", "AOCS"}:
            node_kind = "SUBSYSTEM_TERM"

        # confidence heuristic
        score = max(best_sub_score, best_struct_score)
        conf = 0.55
        if score >= 7:
            conf = 0.92
        elif score >= 5:
            conf = 0.85
        elif score >= 3:
            conf = 0.72
        elif score >= 2:
            conf = 0.65
        else:
            conf = 0.40

        # If both ended up OTHER, return UNKNOWN-ish
        if subsystem == "OTHER" and structure == "OTHER":
            return "UNKNOWN", "UNKNOWN", 0.30, "CONCEPT", "heuristic"

        source = "lexicon" if (best_sub_score >= 2 or best_struct_score >= 2) else "heuristic"
        return subsystem, structure, conf, node_kind, source

    # -----------------------------
    # Relation Extraction
    # -----------------------------
    def extract_relations(self, sent, span_nodes: List[str]) -> List[Tuple[str, str, str, float]]:
        """
        Returns (subject, predicate, object, confidence).
        Objects may be values; we will create VALUE nodes for them.
        """
        rels: List[Tuple[str, str, str, float]] = []
        text = sent.text.strip()
        low = text.lower()

        if not span_nodes:
            return rels

        # Range pattern: between A and B (with optional unit)
        m = re.search(r"\bbetween\s+(\d+(?:\.\d+)?)\s*([a-zA-Z%°/]+)?\s+and\s+(\d+(?:\.\d+)?)\s*([a-zA-Z%°/]+)?\b", low)
        if m:
            unit1 = m.group(2) or ""
            unit2 = m.group(4) or ""
            unit = unit1 if unit1 else unit2
            vals = f"{m.group(1)}{unit}-{m.group(3)}{unit}"
            rels.append((span_nodes[0], "has_range", vals, 0.78))

        # Max/limit pattern
        if any(k in low for k in ["no greater than", "at most", "maximum of", "not exceed", "≤", "<="]):
            # Try to capture value after cue
            mv = re.search(r"(?:no greater than|at most|maximum of|not exceed|≤|<=)\s+(\d+(?:\.\d+)?)\s*([a-zA-Z%°/]+)?", low)
            if mv:
                val = f"{mv.group(1)}{mv.group(2) or ''}"
                rels.append((span_nodes[0], "has_limit_max", val, 0.80))
            else:
                rels.append((span_nodes[0], "has_limit_max", "max", 0.70))

        # Min pattern
        if any(k in low for k in ["no less than", "at least", "minimum of", "≥", ">="]):
            mv = re.search(r"(?:no less than|at least|minimum of|≥|>=)\s+(\d+(?:\.\d+)?)\s*([a-zA-Z%°/]+)?", low)
            if mv:
                val = f"{mv.group(1)}{mv.group(2) or ''}"
                rels.append((span_nodes[0], "has_limit_min", val, 0.80))

        # Verb-based relations using dependency parse
        # (cheap and imperfect, but adds meaning)
        for token in sent:
            if token.lemma_ in {"have", "use", "provide", "measure", "include"}:
                subj = [w for w in token.lefts if w.dep_ in {"nsubj", "nsubjpass"}]
                if not subj:
                    # sometimes subject is an ancestor
                    subj = [w for w in token.ancestors if w.dep_ in {"ROOT"}]
                # objects can be dobj/attr/pobj
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

                # Only add if at least one side is in known span nodes (prevents junk)
                s_c = self.canon(s)
                o_c = self.canon(o)
                if (s_c in span_nodes) or (o_c in span_nodes):
                    rels.append((s, pred, o, 0.70))

        return rels

    # -----------------------------
    # Scoring helpers
    # -----------------------------
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
    # Mission heuristic
    # -----------------------------
    def _mission_like(self, span_text: str, raw_text: str) -> bool:
        """
        Safer mission-name heuristic:
          - must look like a proper name token
          - must be near mission cues in the text
        """
        if not re.fullmatch(r"[A-Z][a-zA-Z0-9\-]{2,}", span_text):
            return False
        cues = ["mission", "spacecraft", "satellite", "probe", "orbiter", "lander", "rover"]
        idx = raw_text.find(span_text)
        if idx < 0:
            return False
        window = raw_text[max(0, idx - 80): idx + 80].lower()
        return any(c in window for c in cues)

    def _looks_like_value(self, txt: str) -> bool:
        # numbers + optional unit
        return bool(re.search(r"\b\d+(?:\.\d+)?\s*(?:km|m|s|w|mw|kw|hz|khz|mhz|ghz|v|mv|a|ma|db|dbi|°c|c|%)\b", txt.lower()))

    # -----------------------------
    # Graph edge helper
    # -----------------------------
    def _add_edge(self, g: nx.Graph, u: str, v: str, pred: str, conf: float) -> None:
        if not u or not v or u == v:
            return
        if g.has_edge(u, v):
            g[u][v]["weight"] = int(g[u][v].get("weight", 1)) + 1
            g[u][v]["confidence"] = float(min(1.0, float(g[u][v].get("confidence", 0.35)) + 0.05))
            # prefer non-cooccurs predicate if present
            if g[u][v].get("predicate") == "co-occurs" and pred != "co-occurs":
                g[u][v]["predicate"] = pred
                g[u][v]["label"] = pred
        else:
            g.add_edge(u, v, predicate=pred, label=pred, weight=1, confidence=float(conf))

    # -----------------------------
    # Normalization
    # -----------------------------
    def canon(self, s: str) -> str:
        """
        Canonical node id:
          - normalize whitespace
          - normalize hyphens
          - lowercase unless acronym-ish
          - cheap singularization
        """
        x = (s or "").strip()
        if not x:
            return ""
        x = re.sub(r"\s+", " ", x).strip(" ,.;:()[]{}\"'")
        x = x.replace("–", "-").replace("—", "-")

        # Keep acronyms uppercase (ESA, RF, TT&C, etc.)
        if re.fullmatch(r"[A-Z0-9&\-]{2,}", x):
            pass
        else:
            x = x.lower()

        # cheap singularization (avoid breaking "bus")
        if x.endswith("ies") and len(x) > 4:
            x = x[:-3] + "y"
        elif x.endswith("s") and len(x) > 4 and not x.endswith("ss") and not x.endswith("bus"):
            x = x[:-1]

        return x

    def canon_display(self, s: str) -> str:
        """
        Keeps a nice display string (trim), but does not force lowercase.
        Node ids still use canon().
        """
        x = (s or "").strip()
        if not x:
            return ""
        x = re.sub(r"\s+", " ", x).strip(" ,.;:()[]{}\"'")
        # normalize unicode subscripts just in case
        sub_map = str.maketrans({
            "₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4",
            "₅": "5", "₆": "6", "₇": "7", "₈": "8", "₉": "9",
        })
        x = x.translate(sub_map)
        return x
