import requests
import json
import re


class OllamaExtractor:
    """
    Local LLM extractor using Ollama.
    Requires Ollama running on http://localhost:11434
    """

    def __init__(self, model: str = "llama3.1:8b"):
        self.model = model
        self.endpoint = "http://localhost:11434/api/generate"

    def extract(self, text: str):
        """
        Returns list of triples as dicts:
        {
          "subject": "...",
          "predicate": "...",
          "object": "...",
          "confidence": 0.0-1.0
        }
        """

        prompt = f"""
You are an information extraction system.

TASK:
Extract factual Subject–Predicate–Object triples from the text.
Merge references across sentences (pronouns, repeated entities).
Normalize entity names (Apple Inc -> Apple).
Prefer meaningful relationships (founded by, located in, works at).

OUTPUT RULES:
- Return ONLY valid JSON
- Output MUST be a list
- Each item MUST contain:
  subject, predicate, object, confidence

FORMAT:
[
  {{
    "subject": "...",
    "predicate": "...",
    "object": "...",
    "confidence": 0.0
  }}
]

TEXT:
{text}
"""

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_p": 0.9
            }
        }

        try:
            response = requests.post(self.endpoint, json=payload, timeout=120)
            response.raise_for_status()
        except Exception as e:
            print("Ollama connection failed:", e)
            return []

        raw = response.json().get("response", "").strip()

        # ---- DEBUG (VERY IMPORTANT FOR YOU) ----
        print("\n--- OLLAMA RAW OUTPUT ---")
        print(raw)
        print("--- END RAW OUTPUT ---\n")

        # Remove markdown if any
        raw = re.sub(r"```json|```", "", raw).strip()

        try:
            data = json.loads(raw)
        except Exception as e:
            print("JSON parse failed:", e)
            return []

        cleaned = []
        for t in data:
            subj = t.get("subject")
            pred = t.get("predicate")
            obj = t.get("object")
            conf = float(t.get("confidence", 0.7))

            if subj and pred and obj:
                cleaned.append({
                    "subject": subj.strip(),
                    "predicate": pred.strip(),
                    "object": obj.strip(),
                    "confidence": round(conf, 2)
                })

        return cleaned
