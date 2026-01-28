from google import genai
import json
from .base_extractor import BaseExtractor


class GeminiExtractor(BaseExtractor):
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        self.model = "models/gemini-flash-latest"

    def extract(self, text: str):
        prompt = f"""
Extract factual Subject–Predicate–Object triples from the text.

Return ONLY valid JSON in this format:
[
  {{
    "subject": "...",
    "predicate": "...",
    "object": "..."
  }}
]

Text:
{text}
"""

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt
            )
            return json.loads(response.text)

        except Exception as e:
            print("Extraction error:", e)
            return []
