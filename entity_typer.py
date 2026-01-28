from typing import Dict, Optional
import spacy


LABEL_MAP = {
    "PERSON": "PERSON",
    "ORG": "ORG",
    "GPE": "LOC",
    "LOC": "LOC",
    "PRODUCT": "PRODUCT",
}

class EntityTyper:
    """
    Uses spaCy NER to assign node types. Works even if you use Ollama for extraction.
    """

    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def extract_types_from_text(self, text: str) -> Dict[str, str]:
        doc = self.nlp(text)
        types: Dict[str, str] = {}
        for ent in doc.ents:
            t = LABEL_MAP.get(ent.label_)
            if not t:
                continue
            key = ent.text.strip()
            # keep first-seen type
            types.setdefault(key, t)
        return types

    def type_for_node(self, node: str, type_map: Dict[str, str]) -> str:
        # direct match
        if node in type_map:
            return type_map[node]
        # case-insensitive fallback
        low = node.lower()
        for k, v in type_map.items():
            if k.lower() == low:
                return v
        return "UNKNOWN"
