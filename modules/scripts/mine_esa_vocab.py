# scripts/mine_esa_vocab.py
from __future__ import annotations

import os
import re
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import List, Dict, Tuple

import fitz  # PyMuPDF

# Import cleaner from your project (adjust if needed)
from modules.extractors.esa_pdf_cleaner import ESAPDFCleaner


# -----------------------------
# Config
# -----------------------------
PDFS = [
    "/mnt/data/SP1330-1_CarbonSat.pdf",
    "/mnt/data/EE9-SKIM-RfMS-ESA-v1.0-FINAL.pdf",
    "/mnt/data/EE10_Hydroterra_Report-for-Assessment-v1.0_13Nov2020.pdf",
    "/mnt/data/EE10_Daedalus_Report-for-Assessment-v1.0_13Nov2020.pdf",
    "/mnt/data/EE10_Harmony_Report-for-Selection_21June2022.pdf",
    "/mnt/data/EE11_Seastar_Report_for_Assessment_v1.0_15Sept23.pdf",
    "/mnt/data/EE11_Nitrosat_Report_for_Assessment_v1.0_15Sept23.pdf",
    "/mnt/data/ESA_EE11_Report_for_Selection_CAIRT_v1.0.pdf",
]

OUT_DIR = os.path.join(os.getcwd(), "outputs")

# subsystem seed cues for suggestion (very lightweight)
SUBSYSTEM_SEEDS = {
    "TELECOM": ["tt&c", "ttc", "telemetry", "telecommand", "antenna", "x-band", "s-band", "ka-band", "rf", "downlink", "uplink", "transponder"],
    "DATA": ["command and data handling", "cdhs", "obc", "mass memory", "ssmm", "spacewire", "1553", "packet", "compression", "processing chain", "level-1", "level-2"],
    "POWER": ["pcdu", "eps", "pdu", "battery", "solar array", "lcl", "mppt", "bus voltage", "power conditioning"],
    "THERMAL": ["thermal", "tcs", "mli", "heat pipe", "radiator", "heater", "osr"],
    "AOCS": ["aocs", "adcs", "reaction wheel", "star tracker", "imu", "gyro", "magnetorquer", "gnss"],
    "PROPULSION": ["propulsion", "thruster", "propellant", "hydrazine", "xenon", "delta-v", "mmh", "nto", "ppu"],
    "GROUND": ["ground segment", "pdgs", "focc", "fos", "estrack", "ground station", "mission control"],
    "ORBIT": ["orbit", "leo", "sun-synchronous", "sso", "inclination", "altitude", "raan", "apogee", "perigee"],
    "PAYLOAD": ["payload", "instrument", "spectrometer", "radiometer", "lidar", "sar", "altimeter", "detector"],
}

STOPWORDS = set("""
a an and are as at be by for from has have if in into is it its of on or s such that the their this to was were will with without
""".split())

UNITS_PATTERN = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:km|m|cm|mm|s|ms|us|w|mw|kw|hz|khz|mhz|ghz|v|mv|a|ma|db|dbi|kbps|mbps|gbps|kbit/s|mbit/s|gbit/s|%|°c|k)\b",
    re.I
)

# Full Term (ACR)
ACR_DEF_PATTERN = re.compile(r"\b([A-Za-z][A-Za-z0-9\-/& ]{3,140}?)\s*\(([\w&/\-]{2,15})\)")

# Raw acronyms in text (all caps)
ACR_PATTERN = re.compile(r"\b[A-Z][A-Z0-9&/\-]{1,14}\b")


def ensure_out_dir():
    os.makedirs(OUT_DIR, exist_ok=True)


def guess_subsystem_from_context(text_window: str) -> Tuple[str, float]:
    """
    Guess subsystem based on presence of seed cues in a context window.
    Returns (label, score) where score is number of hits (proxy).
    """
    low = (text_window or "").lower()
    best_lab = "OTHER"
    best_hits = 0
    for lab, cues in SUBSYSTEM_SEEDS.items():
        hits = 0
        for c in cues:
            if c in low:
                hits += 1
        if hits > best_hits:
            best_hits = hits
            best_lab = lab
    return best_lab, float(best_hits)


def clean_pdf_text(path: str, cleaner: ESAPDFCleaner) -> str:
    res = cleaner.extract_clean_text(path)
    return res.text


def mine_acronym_definitions(text: str) -> Counter:
    pairs = Counter()
    for m in ACR_DEF_PATTERN.finditer(text):
        full = re.sub(r"\s+", " ", m.group(1)).strip()
        acr = m.group(2).strip()

        # keep uppercase-ish acronyms only
        if re.fullmatch(r"[A-Z0-9][A-Z0-9&/\-]{1,14}", acr) and not acr.isdigit():
            # avoid silly full terms
            if len(full) >= 4 and len(full) <= 140:
                pairs[(acr, full)] += 1
    return pairs


def mine_acronyms(text: str) -> Counter:
    c = Counter()
    for acr in ACR_PATTERN.findall(text):
        if len(acr) <= 1:
            continue
        # ignore common noise
        if acr in {"ESA", "ECSS", "EE", "UTC"}:
            continue
        c[acr] += 1
    return c


def mine_unit_params(text: str) -> Counter:
    c = Counter()
    for m in UNITS_PATTERN.finditer(text):
        c[m.group(0)] += 1
    return c


def mine_ngram_terms(text: str, n_min: int = 2, n_max: int = 4) -> Counter:
    """
    Very lightweight noun-phrase-ish n-grams by token regex.
    This avoids spaCy dependency to keep script simple.
    """
    # tokens: keep words + hyphen words
    toks = re.findall(r"[A-Za-z][A-Za-z\-]{1,}", text)
    toks = [t.lower() for t in toks if t.lower() not in STOPWORDS]
    c = Counter()
    L = len(toks)
    for n in range(n_min, n_max + 1):
        for i in range(0, L - n + 1):
            g = toks[i:i+n]
            # remove ngrams that are too generic
            if any(w in {"mission", "spacecraft", "satellite", "system", "study", "section"} for w in g):
                continue
            # require at least one "technical-ish" word (heuristic)
            if not any(len(w) >= 6 or "-" in w for w in g):
                continue
            phrase = " ".join(g)
            c[phrase] += 1
    return c


def main():
    ensure_out_dir()
    cleaner = ESAPDFCleaner()

    all_texts: Dict[str, str] = {}
    for p in PDFS:
        if not os.path.exists(p):
            print(f"Missing: {p}")
            continue
        print(f"Cleaning: {p}")
        all_texts[p] = clean_pdf_text(p, cleaner)

    # 1) Acronym definitions (Full (ACR))
    pair_counts = Counter()
    for t in all_texts.values():
        pair_counts.update(mine_acronym_definitions(t))

    # 2) Acronym raw frequency
    acr_counts = Counter()
    for t in all_texts.values():
        acr_counts.update(mine_acronyms(t))

    # 3) Unit-bearing params (science/engineering)
    unit_counts = Counter()
    for t in all_texts.values():
        unit_counts.update(mine_unit_params(t))

    # 4) N-gram term candidates (science+engineering phrases)
    term_counts = Counter()
    for t in all_texts.values():
        term_counts.update(mine_ngram_terms(t, 2, 4))

    # -----------------------------
    # Output: Acronym definitions CSV
    # -----------------------------
    acr_csv_path = os.path.join(OUT_DIR, "esa_acronyms.csv")
    with open(acr_csv_path, "w", encoding="utf-8") as f:
        f.write("acronym,count,full_term,suggested_subsystem,subsystem_hit_score\n")
        for (acr, full), cnt in pair_counts.most_common(800):
            # guess subsystem from full term text
            lab, score = guess_subsystem_from_context(full)
            f.write(f"\"{acr}\",{cnt},\"{full}\",\"{lab}\",{score}\n")

    # -----------------------------
    # Output: Acronym map suggestions JSON
    # -----------------------------
    suggestions = {}
    for (acr, full), cnt in pair_counts.most_common(800):
        lab, score = guess_subsystem_from_context(full)
        # require at least some signal or frequent definition
        if cnt >= 2 or score >= 2:
            # keep best by score then count
            cur = suggestions.get(acr)
            cand = {"suggested_subsystem": lab, "score": score, "count": cnt, "full_term": full}
            if cur is None:
                suggestions[acr] = cand
            else:
                if (cand["score"], cand["count"]) > (cur["score"], cur["count"]):
                    suggestions[acr] = cand

    sug_json_path = os.path.join(OUT_DIR, "esa_acronym_map_suggestions.json")
    with open(sug_json_path, "w", encoding="utf-8") as f:
        json.dump(suggestions, f, indent=2)

    # -----------------------------
    # Output: Science/engineering term candidates CSV
    # -----------------------------
    sci_csv_path = os.path.join(OUT_DIR, "esa_science_terms.csv")
    with open(sci_csv_path, "w", encoding="utf-8") as f:
        f.write("term,count,suggested_subsystem,subsystem_hit_score,term_type\n")

        # Unit params
        for val, cnt in unit_counts.most_common(400):
            lab, score = guess_subsystem_from_context(val)
            f.write(f"\"{val}\",{cnt},\"{lab}\",{score},\"unit_value\"\n")

        # N-gram phrases
        for term, cnt in term_counts.most_common(1200):
            lab, score = guess_subsystem_from_context(term)
            # keep more “technical” and frequent terms
            if cnt < 5:
                continue
            f.write(f"\"{term}\",{cnt},\"{lab}\",{score},\"ngram\"\n")

    # -----------------------------
    # Output: quick console summary
    # -----------------------------
    print("\nWrote:")
    print(" -", acr_csv_path)
    print(" -", sug_json_path)
    print(" -", sci_csv_path)

    print("\nTop raw acronyms (frequency):")
    for acr, cnt in acr_counts.most_common(40):
        print(f"{acr:12s} {cnt}")

    print("\nTop unit-bearing values:")
    for v, cnt in unit_counts.most_common(30):
        print(f"{v:12s} {cnt}")


if __name__ == "__main__":
    main()
