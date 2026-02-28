from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import re

import networkx as nx
import spacy


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
    node_kind: str = "CONCEPT"
    source: str = "heuristic"  # "ner" | "lexicon" | "pattern" | "heuristic"


class LabelledSubsystemKGBuilder:
    """
    ESA-friendly subsystem KG builder.

    Outputs ONE label per node:
      - subsystem_label: TELECOM/POWER/DATA/PAYLOAD/ORBIT/GROUND/PROPULSION/THERMAL/AOCS/OTHER/UNKNOWN

    Key features:
      - ESA acronym + terminology tuned lexicon
      - noun chunks + TitleCase phrases + regex anchors
      - simple relation extraction (limits/ranges + have/use/provide/include)
      - HUB-based co-occurrence (no sentence cliques)
      - stable canonical node ids
    """

    def __init__(self, spacy_model: str = "en_core_web_sm", enable_noun_chunks: bool = True):
        # Keep parser + ner; disable lemmatizer for speed.
        self.nlp = spacy.load(spacy_model, disable=["lemmatizer"])
        if "sentencizer" not in self.nlp.pipe_names:
            self.nlp.add_pipe("sentencizer", first=True)

        self.enable_noun_chunks = enable_noun_chunks
        self._raw_text: str = ""

        # -------------------------
        # ESA-tuned subsystem lexicon
        # -------------------------
        self.subsystem_lexicon: Dict[str, List[str]] = {
            "TELECOM": [
                "telemetry", "telecommand", "ttc", "tt&c", "tt & c",
                "rf", "ranging", "doppler",
                "transmitter", "receiver", "transceiver",
                "antenna", "lna", "pa", "amplifier", "filter", "diplexer", "duplexer",
                "downlink", "uplink", "link budget", "eirp", "g/t",
                "x-band", "x band", "ka-band", "ka band", "s-band", "s band",
                "bit rate", "data rate", "bandwidth", "frequency", "modulation", "coding", "fec",
                "bpsk", "qpsk", "qam",
            ],
            "DATA": [
                # on-board data handling + avionics buses (ESA style)
                "cdhs", "c&dh", "command and data handling", "data handling", "obdh",
                "on-board computer", "onboard computer", "obc", "avionics",
                "mass memory", "ssmm", "solid state mass memory", "memory", "storage",
                "packet", "packets", "compression", "data volume",
                "spacewire", "spw", "can bus",
                "mil-std-1553b", "mil-1553", "1553b",
                "remote interface unit", "riu", "rtu",
                # payload data handling and transmission
                "pdht", "payload data handling", "payload data handling and transmission",
                # ground processing products often mentioned in ESA docs
                "processing chain", "level-0", "level-1", "level-1b", "level-2", "product generation",
            ],
            "POWER": [
                "eps", "electrical power subsystem", "electrical power",
                "solar array", "solar arrays", "solar panel", "solar panels", "solar cell", "solar cells",
                "battery", "batteries",
                "pcdu", "pdu", "power distribution", "power conditioning",
                "converter", "dc-dc", "dcdc", "regulator",
                "bus voltage", "power bus", "unregulated bus",
                "28 v", "28v", "28-v", "28-vdc", "28 vdc",
                "lcl", "lcls", "fcl", "fcls", "latching current limiter", "current limiter",
                "pdu board",
            ],
            "THERMAL": [
                "thermal", "thermal control", "tcs",
                "radiator", "radiators", "radiator panel",
                "heater", "heaters", "heater line",
                "mli", "insulation",
                "heat pipe", "heat pipes", "thermal strap", "thermal doubler",
                "thermistor", "thermostat",
                "high emissivity", "coating", "louver",
                "temperature",
            ],
            "AOCS": [
                "aocs", "adcs", "attitude", "attitude control", "attitude determination",
                "orbit determination", "pointing", "slew",
                "reaction wheel", "reaction wheels", "momentum wheel",
                "star tracker", "star trackers",
                "gyro", "gyroscope", "gyros",
                "magnetorquer", "magnetorquers", "magnetometer",
                "sun sensor", "sun sensors", "earth sensor", "earth sensors",
                "gnss", "gps", "kalman filter",
            ],
            "PROPULSION": [
                "propulsion", "thruster", "thrusters",
                "propellant", "propellants", "tank", "tanks",
                "feed system", "nozzle", "valve", "valves",
                "pressurant", "helium pressurant",
                "hydrazine", "xenon",
                "blow-down", "blow down",
                "monopropellant", "bipropellant",
                "delta-v", "Δv", "orbit raising", "station keeping",
                "collision avoidance", "deorbit",
                "hall thruster", "ion thruster", "ppu",
            ],
            "PAYLOAD": [
                "payload", "instrument", "instruments",
                "spectrometer", "radiometer", "imager", "imaging",
                "lidar", "sar", "altimeter", "camera", "telescope",
                "detector", "detectors",
            ],
            "GROUND": [
                "ground segment", "ground station", "tracking station", "estrack",
                "mission operations", "mission control",
                "operations centre", "operations center",
                "fos", "focc", "pdgs", "fds", "mcs", "mission planning",
                "kiruna", "svalbard", "ksat", "ssc",
                "payload data ground segment",
            ],
            "ORBIT": [
                "orbit", "leo", "low earth orbit", "sso", "sun-synchronous", "sun synchronous",
                "inclination", "altitude", "eccentricity", "raan",
                "semi-major axis", "orbital period", "apogee", "perigee",
            ],
        }

        # Tie-break helpers (ESA overlaps)
        self.ground_strong_cues = ["pdgs", "focc", "fos", "estrack", "kiruna", "svalbard", "mission planning", "ground segment"]
        self.data_strong_cues = ["cdhs", "obc", "mass memory", "ssmm", "spacewire", "1553", "mil-std-1553", "pdht", "compression", "processing chain", "level-1", "level-2"]
        self.telecom_strong_cues = ["tt&c", "ranging", "x-band", "s-band", "ka-band", "antenna", "rf", "transceiver"]

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
            r"\bMIL-STD-1553B\b",
            r"\b(?:TT&C|CDHS|EPS|PCDU|PDHT|PDGS|FOCC|FOS|ESTRACK|SSMM)\b",
        ]

        # Precompile lexicon patterns (fast boundary-ish matching)
        self._compiled: Dict[str, List[re.Pattern]] = {
            label: [self._compile_term(t) for t in terms]
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

            # 1) Relation extraction (adds meaningful edges)
            rel_triples = self.extract_relations(sent, in_sent)
            for u, pred, v, conf in rel_triples:
                u2 = self.canon(u)
                v2 = self.canon(v)
                if not u2 or not v2 or u2 == v2:
                    continue

                if v2 not in g:
                    # values become nodes
                    g.add_node(
                        v2,
                        subsystem_label="OTHER",
                        confidence=float(conf),
                        node_kind="VALUE" if self._looks_like_value(v2) else "CONCEPT",
                        source="pattern",
                        start=-1,
                        end=-1,
                    )

                self._add_edge(g, u2, v2, pred, conf)

            # 2) Co-occurrence edges (HUB-based, no cliques)
            if len(in_sent) >= 2:
                hub = max(in_sent, key=lambda n: in_sent_conf.get(n, 0.0))
                for n in in_sent:
                    if n == hub:
                        continue
                    self._add_edge(g, hub, n, "co-occurs", 0.35)

        # remove isolated nodes
        isolates = [n for n in g.nodes() if g.degree(n) == 0]
        g.remove_nodes_from(isolates)

        return g

    # -----------------------------
    # Span extraction
    # -----------------------------
    def extract_spans(self, doc, raw_text: str) -> List[LabeledSpan]:
        spans: List[LabeledSpan] = []
        ner_spans = [(e.start_char, e.end_char, e.label_, e.text) for e in doc.ents]

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
            if subsystem in {"UNKNOWN"}:
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
                if subsystem in {"UNKNOWN"}:
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
                spans.append(LabeledSpan(text=c, subsystem=subsystem, start=m.start(), end=m.end(), confidence=conf, node_kind=kind, source=source))

        # dedup by canonical id
        best: Dict[str, LabeledSpan] = {}
        for s in spans:
            key = self.canon(s.text)
            if not key:
                continue
            if key not in best or s.confidence > best[key].confidence:
                best[key] = s

        return list(best.values())

    # -----------------------------
    # Classification core (subsystem only)
    # -----------------------------
    def classify_span(self, span_text: str, start: int, end: int, ner_spans) -> Tuple[str, float, str, str]:
        low = span_text.lower()

        # NER override for ORG/LOC/PERSON: keep them as OTHER so they don't pollute subsystem graph
        ner = self._ner_override(start, end, ner_spans)
        if ner in {"ORG", "LOC", "PERSON"}:
            return "OTHER", 0.70, ner, "ner"

        # Score lexicon on span itself
        best_lab, best_hits = self._best_label(low)

        # Sentence-aware tie-break (use window around the span)
        window = self._context_window(start, end, self._raw_text, win=140)
        best_lab = self._tie_break(best_lab, window, span_text)

        # Confidence
        if best_hits >= 4:
            conf = 0.90
        elif best_hits >= 2:
            conf = 0.75
        elif best_hits >= 1:
            conf = 0.62
        else:
            # allow orbit/value anchors to survive
            if self._looks_like_value(low) or re.search(r"\b\d{2,4}\s?km\b", low):
                return "ORBIT", 0.60, "PARAMETER", "pattern"
            return "UNKNOWN", 0.30, "CONCEPT", "heuristic"

        # Node kind
        kind = "SUBSYSTEM_TERM" if best_lab in {"TELECOM", "POWER", "DATA", "PROPULSION", "THERMAL", "AOCS", "GROUND", "ORBIT", "PAYLOAD"} else "CONCEPT"
        source = "lexicon"
        return best_lab, conf, kind, source

    # -----------------------------
    # Relation extraction
    # -----------------------------
    def extract_relations(self, sent, span_nodes: List[str]) -> List[Tuple[str, str, str, float]]:
        rels: List[Tuple[str, str, str, float]] = []
        text = sent.text.strip()
        low = text.lower()

        if not span_nodes:
            return rels

        # Range: between A and B
        m = re.search(r"\bbetween\s+(\d+(?:\.\d+)?)\s*([a-zA-Z%°/]+)?\s+and\s+(\d+(?:\.\d+)?)\s*([a-zA-Z%°/]+)?\b", low)
        if m:
            unit = (m.group(2) or m.group(4) or "")
            vals = f"{m.group(1)}{unit}-{m.group(3)}{unit}"
            rels.append((span_nodes[0], "has_range", vals, 0.78))

        # Max
        if any(k in low for k in ["no greater than", "at most", "maximum of", "not exceed", "≤", "<="]):
            mv = re.search(r"(?:no greater than|at most|maximum of|not exceed|≤|<=)\s+(\d+(?:\.\d+)?)\s*([a-zA-Z%°/]+)?", low)
            if mv:
                val = f"{mv.group(1)}{mv.group(2) or ''}"
                rels.append((span_nodes[0], "has_limit_max", val, 0.80))

        # Min
        if any(k in low for k in ["no less than", "at least", "minimum of", "≥", ">="]):
            mv = re.search(r"(?:no less than|at least|minimum of|≥|>=)\s+(\d+(?:\.\d+)?)\s*([a-zA-Z%°/]+)?", low)
            if mv:
                val = f"{mv.group(1)}{mv.group(2) or ''}"
                rels.append((span_nodes[0], "has_limit_min", val, 0.80))

        # Verb relations (cheap)
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

        # Ground beats telecom/data if strong ground cues exist
        if label in {"TELECOM", "DATA", "GROUND"}:
            if any(c in s for c in self.ground_strong_cues) or any(c in n for c in self.ground_strong_cues):
                return "GROUND"

        # DATA vs TELECOM overlap in ESA docs
        if label in {"TELECOM", "DATA"}:
            has_data = any(c in s for c in self.data_strong_cues) or any(c in n for c in self.data_strong_cues)
            has_tel = any(c in s for c in self.telecom_strong_cues) or any(c in n for c in self.telecom_strong_cues)
            if has_data and not has_tel:
                return "DATA"
            if has_tel and not has_data:
                return "TELECOM"
            if has_data and has_tel:
                if any(k in n for k in ["rf", "antenna", "transceiver", "transmitter", "receiver", "x-band", "s-band", "ka-band"]):
                    return "TELECOM"
                return "DATA"

        return label

    def _compile_term(self, term: str) -> re.Pattern:
        t = term.strip()
        t_esc = re.escape(t)
        t_esc = t_esc.replace(r"\-", r"[\-\s]?")
        t_esc = t_esc.replace(r"\ ", r"\s+")
        return re.compile(rf"(?<!\w){t_esc}(?!\w)", flags=re.I)

    def _context_window(self, start: int, end: int, text: str, win: int = 140) -> str:
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
            if g[u][v].get("predicate") == "co-occurs" and pred != "co-occurs":
                g[u][v]["predicate"] = pred
                g[u][v]["label"] = pred
        else:
            g.add_edge(u, v, predicate=pred, label=pred, weight=1, confidence=float(conf))

    # -----------------------------
    # Normalization
    # -----------------------------
    def canon(self, s: str) -> str:
        x = (s or "").strip()
        if not x:
            return ""
        x = re.sub(r"\s+", " ", x).strip(" ,.;:()[]{}\"'")
        x = x.replace("–", "-").replace("—", "-")

        # Keep acronyms uppercase-ish
        if re.fullmatch(r"[A-Z0-9&\-]{2,}", x):
            pass
        else:
            x = x.lower()

        # cheap singularization (avoid "bus")
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
