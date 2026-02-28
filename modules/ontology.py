# modules/ontology.py
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Optional, Set, Tuple, List


@dataclass(frozen=True)
class OntologyMatch:
    ontology_class: str
    parent_class: str
    confidence: float
    method: str  # "acronym" | "pattern" | "subsystem" | "fallback"


class Ontology:
    """
    Backwards-compatible ontology:
      - preserves generic NER ontology (PERSON/ORG/LOC/PRODUCT/UNKNOWN)
      - extends with ESA spacecraft subsystem ontology
      - supports lightweight ontology mapping + relation validation
    """

    def __init__(self):
        # -------------------------
        # 1) Base NER classes (keep existing behavior)
        # -------------------------
        self.base_classes: Set[str] = {
            "PERSON",
            "ORG",
            "LOC",
            "PRODUCT",
            "UNKNOWN",
        }

        # -------------------------
        # 2) ESA Subsystem Classes (TBox-ish)
        # -------------------------
        self.subsystem_classes: Dict[str, str] = {
            "TELECOM": "esa:TelecommunicationsSubsystem",
            "POWER": "esa:ElectricalPowerSubsystem",
            "DATA": "esa:CommandAndDataHandlingSubsystem",
            "THERMAL": "esa:ThermalControlSubsystem",
            "AOCS": "esa:AttitudeAndOrbitControlSubsystem",
            "PROPULSION": "esa:PropulsionSubsystem",
            "PAYLOAD": "esa:PayloadSubsystem",
            "GROUND": "esa:GroundSegment",
            "ORBIT": "esa:OrbitDefinition",
            "OTHER": "esa:OtherConcept",
            "UNKNOWN": "esa:UnknownConcept",
        }

        # Component-level (optional, improves interpretability)
        # You can expand freely.
        self.component_patterns: List[Tuple[re.Pattern, str, str]] = [
            # (pattern, ontology_class, parent_subsystem_class)

            # POWER
            (re.compile(r"\bpcdu\b", re.I), "esa:PowerConditioningAndDistributionUnit", self.subsystem_classes["POWER"]),
            (re.compile(r"\bpdu\b", re.I), "esa:PowerDistributionUnit", self.subsystem_classes["POWER"]),
            (re.compile(r"\bmppt\b", re.I), "esa:MaximumPowerPointTracker", self.subsystem_classes["POWER"]),
            (re.compile(r"\blcl(s)?\b|\blatching current limiter(s)?\b", re.I), "esa:LatchingCurrentLimiter", self.subsystem_classes["POWER"]),
            (re.compile(r"\bbattery\b|\bbatteries\b", re.I), "esa:Battery", self.subsystem_classes["POWER"]),
            (re.compile(r"\bsolar array(s)?\b|\bsolar panel(s)?\b", re.I), "esa:SolarArray", self.subsystem_classes["POWER"]),

            # DATA / AVIONICS
            (re.compile(r"\bobc\b|\bon-?board computer\b", re.I), "esa:OnBoardComputer", self.subsystem_classes["DATA"]),
            (re.compile(r"\bcdhs\b|\bcommand and data handling\b|\bc&dh\b", re.I), "esa:CDHS", self.subsystem_classes["DATA"]),
            (re.compile(r"\briu\b|\bremote interface unit\b", re.I), "esa:RemoteInterfaceUnit", self.subsystem_classes["DATA"]),
            (re.compile(r"\brtu\b|\bremote terminal unit\b", re.I), "esa:RemoteTerminalUnit", self.subsystem_classes["DATA"]),
            (re.compile(r"\bssmm\b|\bsolid-?state mass memory\b|\bmass memory\b", re.I), "esa:MassMemory", self.subsystem_classes["DATA"]),
            (re.compile(r"\bspacewire\b|\bspw\b", re.I), "esa:SpaceWireBus", self.subsystem_classes["DATA"]),
            (re.compile(r"\bmil-?std-?1553b\b|\bmil-?1553\b|\b1553b\b", re.I), "esa:MIL1553Bus", self.subsystem_classes["DATA"]),
            (re.compile(r"\bfd ir\b|\bfdir\b", re.I), "esa:FDIR", self.subsystem_classes["DATA"]),
            (re.compile(r"\bpdht\b|\bpayload data handling\b", re.I), "esa:PayloadDataHandlingAndTransmission", self.subsystem_classes["DATA"]),

            # TELECOM
            (re.compile(r"\btt&c\b|\bttc\b", re.I), "esa:TTandCSubsystem", self.subsystem_classes["TELECOM"]),
            (re.compile(r"\bs-?band\b", re.I), "esa:SBandLink", self.subsystem_classes["TELECOM"]),
            (re.compile(r"\bx-?band\b", re.I), "esa:XBandLink", self.subsystem_classes["TELECOM"]),
            (re.compile(r"\bka-?band\b", re.I), "esa:KaBandLink", self.subsystem_classes["TELECOM"]),
            (re.compile(r"\bantenna(s)?\b", re.I), "esa:Antenna", self.subsystem_classes["TELECOM"]),
            (re.compile(r"\btransponder\b", re.I), "esa:Transponder", self.subsystem_classes["TELECOM"]),
            (re.compile(r"\btransceiver\b", re.I), "esa:Transceiver", self.subsystem_classes["TELECOM"]),

            # AOCS
            (re.compile(r"\breaction wheel(s)?\b|\brwa\b", re.I), "esa:ReactionWheel", self.subsystem_classes["AOCS"]),
            (re.compile(r"\bstar tracker(s)?\b|\bstr\b", re.I), "esa:StarTracker", self.subsystem_classes["AOCS"]),
            (re.compile(r"\bgyro(scope)?s?\b|\bimu\b", re.I), "esa:InertialMeasurementUnit", self.subsystem_classes["AOCS"]),
            (re.compile(r"\bmagnet(orquer|ometer)s?\b|\bmtq\b", re.I), "esa:MagneticActuatorOrSensor", self.subsystem_classes["AOCS"]),
            (re.compile(r"\bgnss\b|\bgps\b", re.I), "esa:GNSSReceiver", self.subsystem_classes["AOCS"]),

            # THERMAL
            (re.compile(r"\bmli\b", re.I), "esa:MultiLayerInsulation", self.subsystem_classes["THERMAL"]),
            (re.compile(r"\bheat pipe(s)?\b|\blhp\b", re.I), "esa:HeatPipe", self.subsystem_classes["THERMAL"]),
            (re.compile(r"\bradiator(s)?\b", re.I), "esa:Radiator", self.subsystem_classes["THERMAL"]),
            (re.compile(r"\bheater(s)?\b|\bhtr\b", re.I), "esa:Heater", self.subsystem_classes["THERMAL"]),

            # PROPULSION
            (re.compile(r"\bthruster(s)?\b|\bhet\b|\brit\b", re.I), "esa:Thruster", self.subsystem_classes["PROPULSION"]),
            (re.compile(r"\bpropellant\b|\bmmh\b|\bnto\b|\bhydrazine\b|\bxenon\b", re.I), "esa:Propellant", self.subsystem_classes["PROPULSION"]),
            (re.compile(r"\bdelta-?v\b|\bΔv\b", re.I), "esa:DeltaV", self.subsystem_classes["PROPULSION"]),
        ]

        # Acronym map for hard overrides (fast + accurate for ESA docs)
        self.acronym_map: Dict[str, str] = {
            # TELECOM
            "TTC": "TELECOM",
            "TT&C": "TELECOM",
            "TM": "TELECOM",
            "TC": "TELECOM",
            "RF": "TELECOM",
            "LGA": "TELECOM",
            "HGA": "TELECOM",
            "LNA": "TELECOM",
            "TWTA": "TELECOM",
            "SSPA": "TELECOM",

            # DATA
            "CDHS": "DATA",
            "OBDH": "DATA",
            "OBC": "DATA",
            "DPU": "DATA",
            "ICU": "DATA",
            "RIU": "DATA",
            "RTU": "DATA",
            "SSMM": "DATA",
            "SPACEWIRE": "DATA",
            "SPW": "DATA",
            "MIL-STD-1553B": "DATA",
            "MIL-1553B": "DATA",
            "1553B": "DATA",
            "FDIR": "DATA",
            "PDHT": "DATA",

            # POWER
            "EPS": "POWER",
            "PCDU": "POWER",
            "PDU": "POWER",
            "MPPT": "POWER",
            "LCL": "POWER",
            "LCLS": "POWER",
            "FCL": "POWER",
            "FCLS": "POWER",

            # THERMAL
            "TCS": "THERMAL",
            "MLI": "THERMAL",
            "LHP": "THERMAL",

            # AOCS
            "AOCS": "AOCS",
            "ADCS": "AOCS",
            "GNC": "AOCS",
            "IMU": "AOCS",
            "RWA": "AOCS",
            "MTQ": "AOCS",
            "STR": "AOCS",
            "GNSS": "AOCS",

            # PROPULSION
            "PPU": "PROPULSION",
            "HET": "PROPULSION",
            "RIT": "PROPULSION",
            "MMH": "PROPULSION",
            "NTO": "PROPULSION",

            # GROUND
            "FOS": "GROUND",
            "FOCC": "GROUND",
            "PDGS": "GROUND",
            "ESTRACK": "GROUND",
            "FDS": "GROUND",
            "MCS": "GROUND",
            "MPS": "GROUND",
        }

        # -------------------------
        # 3) Relations: keep your base + add ESA engineering relations
        # -------------------------
        self.base_relations: Set[Tuple[str, str, str]] = {
            ("PERSON", "founded", "ORG"),
            ("PERSON", "works_at", "ORG"),
            ("PERSON", "ceo_of", "ORG"),
            ("PERSON", "born_in", "LOC"),
            ("ORG", "located_in", "LOC"),
            ("ORG", "produces", "PRODUCT"),
            ("ORG", "acquired", "ORG"),
            ("PRODUCT", "created_by", "ORG"),
        }

        # ESA KG relations (subsystem graph relations)
        # types here are subsystem labels
        self.esa_relations: Set[Tuple[str, str, str]] = {
            ("POWER", "has_component", "POWER"),
            ("DATA", "has_component", "DATA"),
            ("TELECOM", "has_component", "TELECOM"),
            ("THERMAL", "has_component", "THERMAL"),
            ("AOCS", "has_component", "AOCS"),
            ("PROPULSION", "has_component", "PROPULSION"),
            ("PAYLOAD", "has_component", "PAYLOAD"),
            ("GROUND", "has_component", "GROUND"),

            # cross-subsystem relations
            ("PAYLOAD", "uses", "DATA"),
            ("PAYLOAD", "uses", "POWER"),
            ("PAYLOAD", "uses", "THERMAL"),
            ("PAYLOAD", "uses", "AOCS"),
            ("DATA", "uses", "POWER"),
            ("TELECOM", "uses", "POWER"),
            ("AOCS", "uses", "POWER"),
            ("PROPULSION", "uses", "POWER"),

            # parameter/value relations (we treat VALUE as a pseudo-type)
            ("TELECOM", "has_limit_max", "VALUE"),
            ("POWER", "has_limit_max", "VALUE"),
            ("PROPULSION", "has_limit_max", "VALUE"),
            ("THERMAL", "has_limit_max", "VALUE"),
            ("ORBIT", "has_range", "VALUE"),
            ("ORBIT", "has_limit_min", "VALUE"),
        }

        # Expose a single "classes" attribute for compatibility
        self.classes: Set[str] = set(self.base_classes) | set(self.subsystem_classes.keys()) | {"VALUE"}

        # Expose a single "relations" attribute for compatibility
        self.relations: Set[Tuple[str, str, str]] = set(self.base_relations) | set(self.esa_relations)

    # -------------------------
    # Backwards-compatible validation
    # -------------------------
    def is_valid(self, subj_type: str, relation: str, obj_type: str) -> bool:
        return (subj_type, relation, obj_type) in self.relations

    # -------------------------
    # Ontology mapping helpers
    # -------------------------
    def map_to_ontology(self, node_text: str, subsystem_label: str) -> OntologyMatch:
        """
        Map a node to an ontology class and parent class.

        Priority:
          1) Acronym map override (highest precision)
          2) Component patterns (PCDU, OBC, SSMM, etc.)
          3) Subsystem class fallback
        """
        txt = (node_text or "").strip()
        sub = (subsystem_label or "UNKNOWN").upper()

        # Acronym override
        u = txt.upper()
        if u in self.acronym_map:
            sub2 = self.acronym_map[u]
            parent = self.subsystem_classes.get(sub2, self.subsystem_classes["UNKNOWN"])
            # choose component if any matches too
            for pat, cls, parent_cls in self.component_patterns:
                if pat.search(txt):
                    return OntologyMatch(cls, parent_cls, 0.95, "acronym+pattern")
            return OntologyMatch(parent, parent, 0.92, "acronym")

        # Component patterns
        for pat, cls, parent_cls in self.component_patterns:
            if pat.search(txt):
                return OntologyMatch(cls, parent_cls, 0.85, "pattern")

        # Subsystem fallback
        parent = self.subsystem_classes.get(sub, self.subsystem_classes["UNKNOWN"])
        return OntologyMatch(parent, parent, 0.70, "subsystem")

    def subsystem_parent(self, subsystem_label: str) -> str:
        return self.subsystem_classes.get((subsystem_label or "UNKNOWN").upper(), self.subsystem_classes["UNKNOWN"])
