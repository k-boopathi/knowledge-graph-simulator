from __future__ import annotations

from typing import Dict, Tuple, List
from collections import defaultdict
import re
import spacy
import networkx as nx


class CooccurrenceGraphBuilder:
    """
    Web (co-occurrence) graph for general text + ESA/scientific content.

    Improvements:
    - Adds a named EntityRuler (esa_entity_ruler) for domain terms (CO2/CH4/IPCC, orbit terms, levels, etc.)
    - Clears patterns on init to avoid duplication in Streamlit reloads
    - Adds noun-chunk concepts (optional) to avoid "empty graph" on scientific paragraphs
    - Normalizes CO₂/CH₄ and splits simple "A and B" entities conservatively
    - Sentence-based co-occurrence + clique control
    """

    def __init__(self, model: str = "en_core_web_sm", enable_noun_chunks: bool = True):
        self.nlp = spacy.load(model)
        self.enable_noun_chunks = enable_noun_chunks

        # --- Add or reuse ESA EntityRuler (named) BEFORE ner ---
        if "esa_entity_ruler" in self.nlp.pipe_names:
            ruler = self.nlp.get_pipe("esa_entity_ruler")
        else:
            ruler = self.nlp.add_pipe("entity_ruler", name="esa_entity_ruler", before="ner")

        # Clear to avoid duplicate patterns (Streamlit reruns / multiple instances)
        ruler.clear()
        ruler.add_patterns(self._domain_patterns())

        # What entity labels we keep as nodes
        self.allowed_labels = {
            "PERSON", "ORG", "GPE", "LOC", "PRODUCT",
            "GAS", "CONCEPT"
        }

        # Map spaCy labels to our display labels
        self.label_map = {
            "GPE": "LOC",
            "LOC": "LOC",
        }

        # Stoplist to reduce junk nodes
        self.entity_stop = {
            "the company", "the government", "the world", "the state", "the university",
            "company", "government", "university",
            "this", "that", "these", "those",
            "it", "its", "their", "they",
        }

        # Extra stop words for noun chunks (to reduce noise)
        self.chunk_stop = {
            "one", "two", "three", "many", "most", "some",
            "important", "significant", "recent", "future",
        }

    def build(
        self,
        text: str,
        min_count: int = 1,
        max_edges_per_sentence: int = 12
    ) -> nx.Graph:
        """
        Build an undirected co-occurrence graph (more correct for co-occurrence).
        """
        doc = self.nlp(text)

        pair_counts: Dict[Tuple[str, str], int] = defaultdict(int)
        node_types: Dict[str, str] = {}

        for sent in doc.sents:
            sent_entities = self._entities_from_sentence(sent, node_types)

            # Reduce clique explosion if too many entities in one sentence
            if len(sent_entities) > max_edges_per_sentence:
                sent_entities = sorted(sent_entities, key=len, reverse=True)[:max_edges_per_sentence]

            uniq = list(dict.fromkeys(sent_entities))
            if len(uniq) < 2:
                continue

            for i in range(len(uniq)):
                for j in range(i + 1, len(uniq)):
                    a, b = uniq[i], uniq[j]
                    if a == b:
                        continue
                    key = tuple(sorted((a, b)))
                    pair_counts[key] += 1

        g = nx.Graph()

        for node, t in node_types.items():
            g.add_node(node, entity_type=t)

        for (a, b), w in pair_counts.items():
            if w < min_count:
                continue

            conf = self._confidence_from_count(w)

            g.add_edge(
                a, b,
                label="",
                predicate="co-occurs",
                confidence=conf,
                weight=w
            )

        return g

    def _entities_from_sentence(self, sent, node_types: Dict[str, str]) -> List[str]:
        names: List[str] = []

        # 1) spaCy entities (NER + EntityRuler-added ones)
        ents = [e for e in sent.ents if e.label_ in self.allowed_labels]
        for e in ents:
            extracted = self._normalize_and_split_entity(e.text)
            for name in extracted:
                if not name:
                    continue
                if name.lower() in self.entity_stop:
                    continue

                label = self.label_map.get(e.label_, e.label_)
                node_types.setdefault(name, label)
                names.append(name)

        # 2) Add noun chunks as CONCEPT nodes (optional)
        if self.enable_noun_chunks:
            existing_nodes = set(node_types.keys())

            for chunk in sent.noun_chunks:
                if any(tok.pos_ == "PRON" for tok in chunk):
                    continue

                phrase = self._normalize_entity(chunk.text)
                if not phrase:
                    continue

                # skip very long chunks
                if len(phrase.split()) > 5:
                    continue

                low = phrase.lower()
                if low in self.entity_stop:
                    continue
                if low in self.chunk_stop:
                    continue

                # Skip if already captured
                if phrase in existing_nodes:
                    continue

                # Skip if phrase is inside an existing entity span (avoid overlap noise)
                if any(phrase in e for e in existing_nodes):
                    continue

                node_types.setdefault(phrase, "CONCEPT")
                names.append(phrase)

        return names

    def _normalize_and_split_entity(self, s: str) -> List[str]:
        """
        Normalize entity and split simple conjunctions like:
        "CO2 and CH4" -> ["CO2", "CH4"]
        """
        s = self._normalize_entity(s)
        if not s:
            return []

        # Normalize CO₂/CH₄ to CO2/CH4
        s = s.replace("CO₂", "CO2").replace("CH₄", "CH4")

        # Conservative split on 'and'
        low = s.lower()
        if re.search(r"\band\b", low):
            parts = [p.strip() for p in re.split(r"\band\b", s, flags=re.I)]
            cleaned: List[str] = []

            for p in parts:
                p = self._normalize_entity(p)
                if p:
                    cleaned.append(p)

            # only split if it yields a small number of short parts
            if 1 < len(cleaned) <= 3 and all(len(x.split()) <= 4 for x in cleaned):
                return cleaned

        return [s]

    def _normalize_entity(self, s: str) -> str:
        s = (s or "").strip()
        if not s:
            return ""

        s = s.strip("\"'`")
        s = re.sub(r"\s+", " ", s)
        s = re.sub(r"[,\.;:\)\]]+$", "", s).strip()

        # Normalize curly apostrophes
        s = s.replace("’", "'")

        # Normalize common suffix punctuation
        s = s.replace("Inc.", "Inc").replace("Co.", "Co").replace("Ltd.", "Ltd")

        # Keep acronyms but title-case regular words lightly
        if len(s) > 1 and not s.isupper():
            s = " ".join([w if w.isupper() else w[:1].upper() + w[1:] for w in s.split()])

        # Normalize corporate words
        s = re.sub(
            r"\b(inc|ltd|co|corp|corporation|company)\b",
            lambda m: m.group(1).lower(),
            s,
            flags=re.I
        )
        s = re.sub(r"\s+", " ", s).strip()

        return s

    def _confidence_from_count(self, w: int) -> float:
        # Confidence grows with frequency but saturates
        return max(0.20, min(1.0, 0.20 + (1.0 - 0.20) * (1 - (0.70 ** w))))

    def _domain_patterns(self) -> List[dict]:
        """
        EntityRuler patterns to boost scientific/ESA entities.
        Add more patterns as you discover missing terms.
        """
        return [
            # ---------------------------
            # Gases
            # ---------------------------
            {"label": "GAS", "pattern": [{"LOWER": {"IN": ["co2", "ch4"]}}]},
            {"label": "GAS", "pattern": "CO₂"},
            {"label": "GAS", "pattern": "CH₄"},
            {"label": "GAS", "pattern": "carbon dioxide"},
            {"label": "GAS", "pattern": "methane"},

            # ---------------------------
            # Climate orgs / ESA orgs
            # ---------------------------
            {"label": "ORG", "pattern": "Intergovernmental Panel on Climate Change"},
            {"label": "ORG", "pattern": "IPCC"},
            {"label": "ORG", "pattern": "European Space Agency"},
            {"label": "ORG", "pattern": "ESA"},

            # ---------------------------
            # Orbit concepts
            # ---------------------------
            {"label": "CONCEPT", "pattern": "sun-synchronous orbit"},
            {"label": "CONCEPT", "pattern": "low Earth orbit"},
            {"label": "CONCEPT", "pattern": "near-polar orbit"},

            # ---------------------------
            # Mission concepts
            # ---------------------------
            {"label": "CONCEPT", "pattern": "Earth observation mission"},
            {"label": "CONCEPT", "pattern": "satellite mission"},

            # ---------------------------
            # Instruments / payload terms
            # ---------------------------
            {"label": "PRODUCT", "pattern": "imaging spectrometer"},
            {"label": "PRODUCT", "pattern": "spectrometer"},

            # ---------------------------
            # Data levels
            # ---------------------------
            {"label": "PRODUCT", "pattern": "Level-1"},
            {"label": "PRODUCT", "pattern": "Level-2"},
            {"label": "PRODUCT", "pattern": "Level 1"},
            {"label": "PRODUCT", "pattern": "Level 2"},

            # ---------------------------
            # Concepts often missed by NER
            # ---------------------------
            {"label": "CONCEPT", "pattern": "human activity"},
            {"label": "CONCEPT", "pattern": "sources and sinks"},
            {"label": "CONCEPT", "pattern": "carbon cycle"},
            {"label": "CONCEPT", "pattern": "climate change"},
            {"label": "CONCEPT", "pattern": "radiative forcing"},
            {"label": "CONCEPT", "pattern": "atmospheric concentrations"},
            {"label": "CONCEPT", "pattern": "greenhouse gases"},
        ]
