# ESA Vocabulary Resources

This directory contains curated domain vocabulary extracted from ESA Earth Explorer mission reports.

## Files

### esa_acronym_map.json
Manually curated acronym-to-subsystem mapping.

Structure:
{
  "SUBSYSTEM": {
      "ACRONYM": "Full Expansion"
  }
}

Used at runtime by the Knowledge Graph builder to:
- Expand acronyms
- Improve subsystem classification
- Increase confidence of detected nodes

---

### esa_science_terms.csv

Domain-specific scientific and engineering terms extracted from ESA mission reports.

Columns:
- term: scientific phrase
- subsystem: mapped subsystem category
- confidence: heuristic reliability score (0–1)

Used to:
- Improve detection of scientific concepts
- Reduce UNKNOWN classification
- Improve subsystem recall

---

## How Generated

The initial candidates were mined using:

scripts/mine_esa_vocab.py

This script:
1. Cleans ESA PDFs (header/footer removal + line reflow)
2. Extracts acronym definitions (Full Term (ACR))
3. Mines frequent technical n-grams
4. Suggests subsystem mapping heuristically

Final vocabulary was manually curated to remove noise.

---

## Academic Methodology Note

The vocabulary creation process follows a corpus-driven domain adaptation approach:

1. Corpus: ESA Earth Explorer Assessment Reports
2. Automatic candidate extraction
3. Heuristic subsystem mapping
4. Manual validation
5. Integration into the KG builder

This improves:
- Precision of subsystem classification
- Recall of scientific concepts
- Stability across different ESA documents
