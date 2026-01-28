import json
import re
from google import genai


class LLMProcessor:
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("API key is required for LLMProcessor")

        self.client = genai.Client(api_key=api_key)
        self.model = "models/gemini-flash-latest"


    # -----------------------------
    # Sentence splitter
    # -----------------------------
    def split_sentences(self, text: str):
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s for s in sentences if len(s.strip()) > 3]

    # -----------------------------
    # Extract triples from ONE sentence
    # -----------------------------
    def extract_triples_from_sentence(self, sentence: str):
        prompt = f"""
You are a strict information extraction engine.

TASK:
Extract ALL factual Subject–Predicate–Object triples.

RULES:
- Output MUST be valid JSON
- Output MUST be a JSON array
- Do NOT explain anything
- Do NOT include markdown
- If no facts exist, output []

FORMAT:
[
  {{
    "subject": "...",
    "predicate": "...",
    "object": "..."
  }}
]

Sentence:
"{sentence}"
"""

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt
            )
        except Exception as e:
            print("LLM ERROR:", e)
            return []

        raw = response.text.strip()

        # 🔍 DEBUG
        print("\n--- SENTENCE ---")
        print(sentence)
        print("--- RAW OUTPUT ---")
        print(raw)

        if not raw or raw.lower().startswith("sorry"):
            return []

        try:
            data = json.loads(raw)
            if isinstance(data, list):
                return data
            return []
        except Exception as e:
            print("JSON PARSE ERROR:", e)
            return []

    # -----------------------------
    # Extract triples from MULTIPLE sentences
    # -----------------------------
    def extract_triples(self, text: str):
        if not self.client:
            return []

        all_triples = []
        sentences = self.split_sentences(text)

        for sentence in sentences:
            triples = self.extract_triples_from_sentence(sentence)
            print("PARSED TRIPLES:", triples)
            all_triples.extend(triples)

        return all_triples
