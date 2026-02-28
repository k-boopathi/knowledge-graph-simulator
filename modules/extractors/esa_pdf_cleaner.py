# modules/extractors/esa_pdf_cleaner.py
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import fitz  # PyMuPDF


@dataclass
class CleanResult:
    text: str
    removed_header_footer_lines: List[str]
    stats: Dict[str, int]


class ESAPDFCleaner:
    """
    ESA PDF -> text cleaner.

    Goals:
      - remove repeated headers/footers (report titles, page counters, disclaimers)
      - fix hyphenation across line breaks (e.g., "tele-\nmetry" -> "telemetry")
      - reflow lines into paragraphs (reduce broken sentences)
      - normalize bullets, spacing
      - keep headings (optional heuristic)

    Designed for ESA Earth Explorer reports where:
      - each page repeats the document title + version + date
      - page numbers and "Page X of Y" appear
      - hard line breaks appear mid-sentence
    """

    def __init__(
        self,
        header_footer_min_page_fraction: float = 0.45,
        min_line_len_to_keep: int = 3,
        keep_section_headings: bool = True,
        debug: bool = False,
    ):
        self.header_footer_min_page_fraction = header_footer_min_page_fraction
        self.min_line_len_to_keep = min_line_len_to_keep
        self.keep_section_headings = keep_section_headings
        self.debug = debug

        # common ESA-ish footer patterns
        self.footer_patterns = [
            re.compile(r"^\s*page\s*\d+\s*(of\s*\d+)?\s*$", re.I),
            re.compile(r"^\s*\d+\s*/\s*\d+\s*$"),
            re.compile(r"^\s*\d+\s*$"),
            re.compile(r"^\s*issue\s*:?.*$", re.I),
            re.compile(r"^\s*revision\s*:?.*$", re.I),
            re.compile(r"^\s*date\s*:?.*$", re.I),
            re.compile(r"^\s*copyright\s*:?.*$", re.I),
        ]

        # lines that often appear in headers
        self.header_hint_patterns = [
            re.compile(r"earth explorer", re.I),
            re.compile(r"living planet", re.I),
            re.compile(r"report for assessment", re.I),
            re.compile(r"report for selection", re.I),
            re.compile(r"european space agency", re.I),
            re.compile(r"\besa\b", re.I),
            re.compile(r"\bv\d+(\.\d+)?\b", re.I),  # v1.0, v2.1
        ]

        self.bullet_prefix = re.compile(r"^\s*([•●▪–—\-]|\(\w\)|\d+\.)\s+")

    # -----------------------------
    # Public API
    # -----------------------------
    def extract_clean_text(
        self,
        pdf_path: str,
        page_from: int = 0,
        page_to: Optional[int] = None,
        join_pages: bool = True,
    ) -> CleanResult:
        doc = fitz.open(pdf_path)
        try:
            pmax = doc.page_count
            a = max(0, page_from)
            b = pmax if page_to is None else min(pmax, page_to)

            # 1) read raw lines per page
            pages_lines: List[List[str]] = []
            for i in range(a, b):
                raw = doc.load_page(i).get_text("text") or ""
                lines = self._split_lines(raw)
                pages_lines.append(lines)

            # 2) detect repeating header/footer lines
            to_remove = self._detect_repeated_lines(pages_lines)

            # 3) remove header/footer + clean each page
            cleaned_pages: List[str] = []
            removed = set(to_remove)
            removed_list_sorted = sorted(list(removed))

            for i, lines in enumerate(pages_lines):
                page = self._clean_page_lines(lines, removed)
                page = self._reflow_page(page)
                cleaned_pages.append(page.strip())

            # 4) join pages with safe separator
            if join_pages:
                out = "\n\n".join([p for p in cleaned_pages if p])
            else:
                out = cleaned_pages[0] if cleaned_pages else ""

            stats = {
                "pages_read": (b - a),
                "removed_header_footer_lines": len(removed),
                "output_chars": len(out),
            }
            return CleanResult(text=out, removed_header_footer_lines=removed_list_sorted, stats=stats)
        finally:
            doc.close()

    # -----------------------------
    # Core steps
    # -----------------------------
    def _split_lines(self, raw: str) -> List[str]:
        raw = raw.replace("\r", "\n")
        lines = [ln.strip() for ln in raw.split("\n")]
        # remove empty duplicates early
        return [ln for ln in lines if ln and len(ln) >= self.min_line_len_to_keep]

    def _detect_repeated_lines(self, pages_lines: List[List[str]]) -> List[str]:
        """
        Find lines that occur on many pages -> treat as header/footer.
        We normalize lines before counting to handle small variations.
        """
        from collections import Counter

        norm_counts = Counter()
        norm_to_examples: Dict[str, str] = {}

        page_count = max(1, len(pages_lines))

        for lines in pages_lines:
            seen_on_page = set()
            for ln in lines:
                n = self._norm_header_footer(ln)
                if not n:
                    continue
                if n in seen_on_page:
                    continue
                seen_on_page.add(n)
                norm_counts[n] += 1
                norm_to_examples.setdefault(n, ln)

        threshold = max(2, int(self.header_footer_min_page_fraction * page_count))
        repeated = []

        for n, c in norm_counts.items():
            if c >= threshold:
                repeated.append(norm_to_examples.get(n, n))

        # Also remove obvious footer patterns even if not repeated enough
        for lines in pages_lines:
            for ln in lines:
                if self._looks_like_footer(ln) or self._looks_like_header(ln):
                    repeated.append(ln)

        # unique
        repeated = list(dict.fromkeys([r.strip() for r in repeated if r.strip()]))
        return repeated

    def _clean_page_lines(self, lines: List[str], removed_lines: set[str]) -> str:
        out_lines = []
        for ln in lines:
            if not ln:
                continue

            # remove if exact line matches detected header/footer lines
            if ln in removed_lines:
                continue

            # remove if it matches footer patterns
            if self._looks_like_footer(ln):
                continue

            # remove if it looks like a repeated header line (heuristic)
            if self._looks_like_header(ln) and self._norm_header_footer(ln) in {
                self._norm_header_footer(x) for x in removed_lines
            }:
                continue

            # clean weird spaces
            ln = re.sub(r"\s+", " ", ln).strip()
            out_lines.append(ln)

        # Preserve bullets as separate lines (we’ll reflow carefully)
        return "\n".join(out_lines)

    def _reflow_page(self, page_text: str) -> str:
        """
        Convert hard-broken lines into paragraphs while preserving bullets/headings.
        """
        if not page_text.strip():
            return ""

        lines = [ln.strip() for ln in page_text.split("\n") if ln.strip()]
        merged: List[str] = []

        def is_heading(ln: str) -> bool:
            if not self.keep_section_headings:
                return False
            # simple: "3.2 Payload" or "4 Thermal Control" or ALL CAPS short-ish
            if re.match(r"^\d+(\.\d+){0,3}\s+\S+", ln):
                return True
            if ln.isupper() and 3 <= len(ln) <= 80:
                return True
            return False

        def is_bullet(ln: str) -> bool:
            return bool(self.bullet_prefix.match(ln))

        buf = ""

        for ln in lines:
            # Fix hyphenation across line ends: done during join, but handle intra-line artifacts too
            ln = ln.replace("­", "")  # soft hyphen
            ln = re.sub(r"\s+", " ", ln).strip()

            if is_heading(ln):
                # flush buffer
                if buf.strip():
                    merged.append(buf.strip())
                    buf = ""
                merged.append(ln.strip())
                continue

            if is_bullet(ln):
                if buf.strip():
                    merged.append(buf.strip())
                    buf = ""
                # normalize bullet
                ln2 = self.bullet_prefix.sub("- ", ln)
                merged.append(ln2.strip())
                continue

            # Normal line: decide whether to append to buffer or start new paragraph
            if not buf:
                buf = ln
            else:
                # If previous ends with hyphen, join without space
                if buf.endswith("-"):
                    buf = buf[:-1] + ln
                # If previous ends with sentence punctuation, start new paragraph
                elif re.search(r"[.!?:;]\s*$", buf):
                    merged.append(buf.strip())
                    buf = ln
                else:
                    # If current line starts like a new section sentence (capital + long), still merge
                    buf = buf + " " + ln

        if buf.strip():
            merged.append(buf.strip())

        # Post pass: remove stray spacing, normalize multiple blank lines
        text = "\n\n".join(merged)
        text = re.sub(r"[ \t]+\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Fix broken hyphenation across newline boundaries:
        # "tele-\n\nmetry" -> "telemetry"
        text = re.sub(r"(\w)-\n\n(\w)", r"\1\2", text)

        return text.strip()

    # -----------------------------
    # Header/footer heuristics
    # -----------------------------
    def _norm_header_footer(self, ln: str) -> str:
        x = ln.strip().lower()
        if not x:
            return ""
        x = re.sub(r"\s+", " ", x)
        x = re.sub(r"\bpage\s*\d+(\s*of\s*\d+)?\b", "page", x)
        x = re.sub(r"\b\d+\s*/\s*\d+\b", "page", x)
        x = re.sub(r"\bv\d+(\.\d+)?\b", "v", x)
        x = re.sub(r"\d{1,2}\s+\w+\s+\d{4}", "date", x)  # "13 Nov 2020"
        x = re.sub(r"\d{4}-\d{2}-\d{2}", "date", x)
        x = re.sub(r"[^\w\s:.-]", "", x)
        return x.strip()

    def _looks_like_footer(self, ln: str) -> bool:
        s = ln.strip()
        for pat in self.footer_patterns:
            if pat.match(s):
                return True
        return False

    def _looks_like_header(self, ln: str) -> bool:
        s = ln.strip()
        if len(s) >= 120:
            return False
        hits = 0
        for pat in self.header_hint_patterns:
            if pat.search(s):
                hits += 1
        return hits >= 1
