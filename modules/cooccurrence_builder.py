from __future__ import annotations
from typing import Dict, Tuple, List, Set
from collections import defaultdict
import re
import spacy
import networkx as nx


class CooccurrenceGraphBuilder:
    """
    Wordlit-style web graph (improved):
    - Nodes: normalized entities (spaCy NER)
    - Edges: co-occurrence in sentence (or within window)
    - Edge weight: frequency of co-occurrence
    - Reduced noise: no full clique explosion if configured
    """

    def __init__(self, model: str = "en_core_web_sm"):
        self.nlp = spacy.load(model)

        self.allowed_labels = {"PERSON", "ORG", "GPE", "LOC", "PRODUCT"}
        self.label_map = {"GPE": "LOC", "LOC": "LOC"}

        # Stoplist to reduce junk nodes
        self.entity_stop = {
            "the company", "the government", "the world", "the state", "the university",
            "company", "government", "university"
        }

    def build(
        self,
        text: str,
        min_count: int = 1,
        max_edges_per_sentence: int = 12
    ) -> nx.DiGraph:
        doc = self.nlp(text)

        pair_counts: Dict[Tuple[str, str], int] = defaultdict(int)
        node_types: Dict[str, str] = {}

        for sent in doc.sents:
            sent_entities = self._entities_from_sentence(sent, node_types)

            # Reduce clique explosion: if too many entities in one sentence,
            # only connect top ones (by length heuristic)
            if len(sent_entities) > max_edges_per_sentence:
                sent_entities = sorted(sent_entities, key=len, reverse=True)[:max_edges_per_sentence]

            uniq = list(dict.fromkeys(sent_entities))
            if len(uniq) < 2:
                continue

            # Undirected pairs (sorted)
            for i in range(len(uniq)):
                for j in range(i + 1, len(uniq)):
                    a, b = uniq[i], uniq[j]
                    if a == b:
                        continue
                    key = tuple(sorted((a, b)))
                    pair_counts[key] += 1

        g = nx.DiGraph()

        for node, t in node_types.items():
            g.add_node(node, entity_type=t)

        for (a, b), w in pair_counts.items():
            if w < min_count:
                continue

            conf = self._confidence_from_count(w)

            # Single directed edge to reduce visual clutter
            # Direction chosen by node degree later in visualization, so here just a -> b
            g.add_edge(
                a, b,
                label="",
                predicate="co-occurs",
                title=f"co-occurs (count {w})",
                confidence=conf,
                weight=float(w)
            )

        return g

    def _entities_from_sentence(self, sent, node_types: Dict[str, str]) -> List[str]:
        ents = [e for e in sent.ents if e.label_ in self.allowed_labels]
        names: List[str] = []

        for e in ents:
            name = self._normalize_entity(e.text)
            if not name:
                continue

            # Remove junk nodes
            if name.lower() in self.entity_stop:
                continue

            # Save entity type
            node_types.setdefault(name, self.label_map.get(e.label_, e.label_))

            names.append(name)

        return names

    def _normalize_entity(self, s: str) -> str:
        s = (s or "").strip()
        if not s:
            return ""

        s = s.strip("\"'`")
        s = re.sub(r"\s+", " ", s)
        s = re.sub(r"[,\.;:\)\]]+$", "", s).strip()

        # Normalize common suffix punctuation
        s = s.replace("Inc.", "Inc")
        s = s.replace("Co.", "Co")
        s = s.replace("Ltd.", "Ltd")

        # Normalize curly apostrophes
        s = s.replace("’", "'")

        # Lightweight title-case (keep acronyms)
        if len(s) > 1 and not s.isupper():
            s = " ".join([w if w.isupper() else w[:1].upper() + w[1:] for w in s.split()])

        # Normalize corporate words
        s = re.sub(r"\b(inc|ltd|co|corp|corporation|company)\b", lambda m: m.group(1).lower(), s, flags=re.I)
        s = re.sub(r"\s+", " ", s).strip()

        return s

    def _confidence_from_count(self, w: int) -> float:
        # Confidence grows with frequency but saturates
        # count 1 -> 0.35, count 2 -> 0.50, count 3 -> 0.62, etc.
        return max(0.20, min(1.0, 0.20 + (1.0 - 0.20) * (1 - (0.70 ** w))))
