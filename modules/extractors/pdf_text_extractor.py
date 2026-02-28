# modules/extractors/pdf_text_extractor.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from modules.extractors.esa_pdf_cleaner import ESAPDFCleaner


@dataclass
class PDFTextResult:
    raw_text: str
    cleaned_text: str
    removed_lines_count: int
    pages_read: int


class PDFTextExtractor:
    def __init__(self):
        self.cleaner = ESAPDFCleaner()

    def extract(self, pdf_path: str, page_from: int = 0, page_to: Optional[int] = None) -> PDFTextResult:
        # raw (no cleaning) – still useful for preview
        # we can use cleaner’s internal PyMuPDF access by just calling cleaner then reconstructing raw
        # simplest: call cleaner, and for raw use its cleaned output as baseline if you don’t want raw.
        # but we can read raw directly via PyMuPDF:
        import fitz

        doc = fitz.open(pdf_path)
        try:
            pmax = doc.page_count
            a = max(0, page_from)
            b = pmax if page_to is None else min(pmax, page_to)
            raw_pages = []
            for i in range(a, b):
                raw_pages.append(doc.load_page(i).get_text("text") or "")
            raw_text = "\n\n".join(raw_pages)
        finally:
            doc.close()

        cleaned = self.cleaner.extract_clean_text(pdf_path, page_from=page_from, page_to=page_to, join_pages=True)
        return PDFTextResult(
            raw_text=raw_text,
            cleaned_text=cleaned.text,
            removed_lines_count=len(cleaned.removed_header_footer_lines),
            pages_read=cleaned.stats.get("pages_read", 0),
        )
