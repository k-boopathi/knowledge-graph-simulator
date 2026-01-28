import re

STOP_ENTITIES = {
    "it", "they", "them", "this", "that", "these", "those",
    "company", "firm", "organization", "industry", "ceo",
    "electronics", "technology"
}

ROLE_WORDS = {"ceo", "founder", "president", "chairman", "chief"}

class EntityNormalizer:
    def __init__(self):
        self.alias = {
            "apple inc": "Apple",
            "apple inc.": "Apple",
            "apple": "Apple",
        }

    def clean(self, s: str) -> str:
        if not s:
            return ""
        s = s.strip()
        s = re.sub(r"\s+", " ", s)
        s = s.strip(" .,:;()[]{}\"'")
        return s

    def normalize(self, s: str) -> str:
        s = self.clean(s)
        if not s:
            return ""
        key = s.lower()
        return self.alias.get(key, s)

    def is_valid(self, s: str) -> bool:
        if not s:
            return False
        if s.isdigit():
            return False
        low = s.lower()
        if low in STOP_ENTITIES:
            return False
        if len(s) < 3:
            return False
        return True

    def is_role_like(self, s: str) -> bool:
        return s.lower() in ROLE_WORDS
