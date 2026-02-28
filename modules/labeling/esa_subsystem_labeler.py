from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import re


@dataclass
class LabelResult:
    label: str
    confidence: float
    method: str  # "sentence" | "node" | "fallback"
    hits: int = 0


class ESASubsystemLabeler:
    """
    Fast subsystem labelling for ESA mission documents.

    Strategy:
      1) Sentence-level keyword evidence (strongest signal)
      2) Node-level keyword evidence
      3) Tie-breakers for ESA-style overlaps (GROUND vs TELECOM, DATA vs TELECOM)
      4) Fallback

    Notes:
      - ESA docs use lots of acronyms. We explicitly include those.
      - We treat 'DATA' as command/data handling + processing chain, not TT&C telemetry words.
    """

    def __init__(self):
        # ---------------------------------------------------------------------
        # Core ESA subsystem keywords (include canonical acronyms)
        # ---------------------------------------------------------------------
        self.subsystem_keywords: Dict[str, List[str]] = {
            # TELECOM / TT&C
            "TELECOM": [
                "telemetry", "telecommand", "ttc", "tt&c", "tt & c", "t t & c",
                "rf", "transmitter", "receiver", "transceiver", "antenna", "lna", "pa",
                "downlink", "uplink", "ranging", "doppler", "link budget",
                "x-band", "x band", "ka-band", "ka band", "s-band", "s band",
                "bit rate", "data rate", "bandwidth", "frequency", "modulation", "coding", "fec",
                "qpsk", "bpsk", "qam", "gain", "eirp", "g/t",
            ],

            # PROPULSION
            "PROPULSION": [
                "propulsion", "thruster", "thrusters", "propellant", "tank", "tanks",
                "feed system", "valve", "valves", "nozzle", "pressurant",
                "hydrazine", "xenon", "helium pressurant", "blow-down", "blow down",
                "monopropellant", "bipropellant", "delta-v", "Δv", "isp", "specific impulse",
                "hall thruster", "ion thruster", "ppu",
                "orbit raising", "station keeping", "collision avoidance", "deorbit",
            ],

            # POWER / EPS
            "POWER": [
                "eps", "electrical power subsystem", "electrical power",
                "solar array", "solar arrays", "solar panel", "solar panels", "solar cell",
                "battery", "batteries", "pcdu", "pcu", "pdu", "power distribution",
                "power conditioning", "converter", "dc-dc", "dcdc", "regulator",
                "bus voltage", "power bus", "28 v", "28v", "28-v", "unregulated bus",
                "lcl", "lcls", "fcl", "fcls", "current limiter", "latching current limiter",
                "voltage", "current", "watt", "power",
            ],

            # THERMAL / TCS
            "THERMAL": [
                "thermal", "thermal control", "tcs",
                "radiator", "radiators", "heater", "heaters", "heater line",
                "heat pipe", "heat pipes", "thermal strap", "thermal doubler",
                "mli", "insulation", "thermistor", "thermostat",
                "coating", "high emissivity", "louver", "temperature",
            ],

            # AOCS / ADCS
            "AOCS": [
                "aocs", "adcs", "attitude", "attitude control", "attitude determination",
                "orbit determination", "od", "pointing", "slew",
                "reaction wheel", "reaction wheels", "momentum wheel",
                "star tracker", "star trackers", "gyro", "gyroscope", "gyros",
                "magnetorquer", "magnetorquers", "magnetometer",
                "sun sensor", "earth sensor", "gnss", "gps",
                "kalman filter",
            ],

            # PAYLOAD
            "PAYLOAD": [
                "payload", "instrument", "instruments",
                "spectrometer", "radiometer", "imager", "imaging", "lidar", "sar",
                "altimeter", "camera", "telescope", "detector", "antenna reflector",
            ],

            # GROUND segment (Earth Explorer standard terms)
            "GROUND": [
                "ground segment", "ground station", "tracking station", "estrack",
                "mission operations", "mission control", "operations centre", "operations center",
                "fos", "focc", "pdgs", "fds", "mcs", "mission planning",
                "kiruna", "svalbard", "ksat", "ssc",
                "flight operations", "payload data ground segment",
            ],

            # ORBIT
            "ORBIT": [
                "orbit", "leo", "low earth orbit", "sso", "sun-synchronous", "sun synchronous",
                "inclination", "altitude", "eccentricity", "raan", "semi-major axis",
                "orbital period", "apogee", "perigee", "ground track",
            ],

            # DATA handling + processing chain (separate from TT&C telemetry words)
            "DATA": [
                "cdhs", "c&dh", "command and data handling", "data handling", "obdh",
                "on-board computer", "onboard computer", "obc", "avionics",
                "mass memory", "ssmm", "solid state mass memory", "memory", "storage",
                "packet", "packets", "telemetry packet", "telecommand packet",
                "mil-std-1553b", "mil-1553", "1553b", "spw", "spacewire", "can bus",
                "pdht", "payload data handling", "payload data handling and transmission",
                "processing chain", "level-0", "level-1", "level-1b", "level-2",
                "l0", "l1", "l1b", "l2", "product generation", "ground processing",
                "compression", "data volume", "downlinked data",
            ],
        }

        # ---------------------------------------------------------------------
        # Overlap-aware tie-break cues (ESA text patterns)
        # ---------------------------------------------------------------------
        self.ground_strong_cues = [
            "pdgs", "focc", "fos", "estrack", "kiruna", "svalbard", "mission planning",
            "flight operations", "payload data ground segment", "ground segment"
        ]
        self.data_strong_cues = [
            "cdhs", "obc", "mass memory", "ssmm", "spacewire", "1553", "mil-std-1553",
            "pdht", "compression", "processing chain", "level-1", "level-2"
        ]
        self.telecom_strong_cues = [
            "tt&c", "ranging", "x-band", "s-band", "ka-band", "antenna", "rf", "transceiver"
        ]

        # Precompile regex for speed (boundary-ish matching)
        self._compiled: Dict[str, List[re.Pattern]] = {
            label: [self._compile_term(t) for t in terms]
            for label, terms in self.subsystem_keywords.items()
        }

    # -----------------------------
    # Public
    # -----------------------------
    def label_node(self, node_text: str, sentence_text: str) -> LabelResult:
        node = (node_text or "").strip()
        sent = (sentence_text or "").strip()

        if not node:
            return LabelResult("OTHER", 0.10, "fallback", hits=0)

        # 1) Sentence-level decision (strongest)
        sent_label, sent_hits = self._best_label(sent)
        if sent_hits >= 2:
            label = self._tie_break(sent_label, sent, node)
            conf = min(0.97, 0.72 + 0.08 * sent_hits)
            return LabelResult(label, conf, "sentence", hits=sent_hits)

        # 2) Node-level decision
        node_label, node_hits = self._best_label(node)
        if node_hits >= 1:
            label = self._tie_break(node_label, sent, node)
            conf = min(0.92, 0.56 + 0.14 * node_hits)
            return LabelResult(label, conf, "node", hits=node_hits)

        # 3) Fallback: detect values / orbit numbers quickly (optional)
        if self._looks_like_orbit_param(node) or self._looks_like_orbit_param(sent):
            return LabelResult("ORBIT", 0.55, "fallback", hits=1)

        return LabelResult("OTHER", 0.35, "fallback", hits=0)

    # -----------------------------
    # Scoring
    # -----------------------------
    def _best_label(self, text: str) -> Tuple[str, int]:
        best_label = "OTHER"
        best_hits = 0
        if not text:
            return best_label, best_hits

        for label, patterns in self._compiled.items():
            hits = 0
            for pat in patterns:
                if pat.search(text):
                    hits += 1
            if hits > best_hits:
                best_hits = hits
                best_label = label

        return best_label, best_hits

    # -----------------------------
    # Tie breaking (ESA overlaps)
    # -----------------------------
    def _tie_break(self, label: str, sentence: str, node: str) -> str:
        s = (sentence or "").lower()
        n = (node or "").lower()

        # If ground cues exist, prefer GROUND
        if label in {"TELECOM", "DATA", "GROUND"}:
            if any(c in s for c in self.ground_strong_cues) or any(c in n for c in self.ground_strong_cues):
                return "GROUND"

        # Distinguish DATA vs TELECOM:
        # - telemetry/telecommand words appear in both contexts.
        # - if CDHS/OBC/1553/SpaceWire/SSMM present => DATA
        if label in {"TELECOM", "DATA"}:
            has_data = any(c in s for c in self.data_strong_cues) or any(c in n for c in self.data_strong_cues)
            has_tel = any(c in s for c in self.telecom_strong_cues) or any(c in n for c in self.telecom_strong_cues)
            if has_data and not has_tel:
                return "DATA"
            if has_tel and not has_data:
                return "TELECOM"
            if has_data and has_tel:
                # If the node itself is clearly RF/antenna/transceiver, keep TELECOM
                if any(k in n for k in ["rf", "antenna", "transceiver", "transmitter", "receiver", "x-band", "s-band", "ka-band"]):
                    return "TELECOM"
                return "DATA"

        return label

    # -----------------------------
    # Regex helpers
    # -----------------------------
    def _compile_term(self, term: str) -> re.Pattern:
        t = term.strip()
        # If term contains spaces or symbols, allow flexible whitespace/hyphenation
        # Example: "mil-std-1553b" should match "MIL STD 1553B" too.
        t_esc = re.escape(t)
        t_esc = t_esc.replace(r"\-", r"[\-\s]?")
        t_esc = t_esc.replace(r"\ ", r"\s+")
        # word boundaries around alphanumerics; keep permissive for acronyms like TT&C
        return re.compile(rf"(?<!\w){t_esc}(?!\w)", flags=re.I)

    def _looks_like_orbit_param(self, text: str) -> bool:
        if not text:
            return False
        low = text.lower()
        return bool(re.search(r"\b\d{2,4}\s?km\b", low)) or bool(re.search(r"\b\d{1,3}\.?\d*\s?deg\b", low)) or ("sun-synchronous" in low)
