from __future__ import annotations
import csv
import re
from pathlib import Path
from typing import Dict, Optional

def _norm(s: str) -> str:
    # Uppercase + keep alnum and dash (Sentinel-1)
    s = s.strip().upper()
    s = re.sub(r"[^A-Z0-9\-]+", "", s)
    return s

def load_esa_missions_csv(path: str | Path) -> Dict[str, dict]:
    """
    Returns dict keyed by normalized name.
    Value contains canonical and type.
    """
    p = Path(path)
    out: Dict[str, dict] = {}
    with p.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            name = row.get("name", "") or ""
            canonical = row.get("canonical", "") or name
            etype = row.get("type", "") or "MISSION"
            if not name.strip():
                continue
            out[_norm(name)] = {"canonical": canonical, "type": etype}
    return out

class MissionTyper:
    def __init__(self, missions: Dict[str, dict]):
        self.missions = missions

    def override_entity_type(self, mention: str, predicted_type: str) -> str:
        k = _norm(mention)
        if k in self.missions:
            return self.missions[k]["type"]  # "MISSION"
        return predicted_type

    def canonicalize(self, mention: str) -> Optional[str]:
        k = _norm(mention)
        if k in self.missions:
            return self.missions[k]["canonical"]
        return None
