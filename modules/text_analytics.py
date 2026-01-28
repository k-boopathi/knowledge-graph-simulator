import re
from dataclasses import dataclass
from typing import Dict, Any, Optional

import spacy


@dataclass
class TextAnalytics:
    text: str
    _nlp: Optional[object] = None

    @staticmethod
    def _get_nlp():
        # lightweight pipeline: tokenizer + sentence segmentation
        nlp = spacy.load("en_core_web_sm", disable=["ner", "tagger", "parser", "lemmatizer"])
        if "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")
        return nlp

    def _doc(self):
        if self._nlp is None:
            self._nlp = self._get_nlp()
        return self._nlp(self.text)

    def character_count(self) -> int:
        return len(self.text)

    def sentence_count(self) -> int:
        doc = self._doc()
        return sum(1 for _ in doc.sents)

    def token_count(self) -> int:
        doc = self._doc()
        return sum(1 for t in doc if not t.is_space)

    def word_count(self) -> int:
        # simple word-ish count
        return len(re.findall(r"\b[\w'-]+\b", self.text))

    def avg_sentence_length_words(self) -> float:
        sc = self.sentence_count()
        if sc == 0:
            return 0.0
        return round(self.word_count() / sc, 2)

    def summary(self) -> Dict[str, Any]:
        return {
            "characters": self.character_count(),
            "sentences": self.sentence_count(),
            "words": self.word_count(),
            "tokens": self.token_count(),
            "avg_sentence_length_words": self.avg_sentence_length_words(),
        }
