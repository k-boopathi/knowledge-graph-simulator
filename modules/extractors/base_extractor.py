# modules/extractors/base_extractor.py

from abc import ABC, abstractmethod
from typing import List, Dict


class BaseExtractor(ABC):
    @abstractmethod
    def extract(self, text: str) -> List[Dict[str, str]]:
        """
        Must return:
        [
          {"subject": "...", "predicate": "...", "object": "..."}
        ]
        """
        pass
