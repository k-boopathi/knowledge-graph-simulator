# modules/cooccurrence_builder.py
from __future__ import annotations
from typing import Dict, Tuple, List
from collections import defaultdict
import spacy
import networkx as nx


class CooccurrenceGraphBuilder:
    """
    Wordlit-style web graph:
    - Nodes: entities (spaCy NER)
    - Edges: entities that co-occur in the same sentence (or within a token window)
    - Edge weight: frequency of co-occurrence
    """

    def __init__(self, model: str = "en_core_web_sm"):
        self.nlp = spacy.load(model)

        # Which entities count as nodes
        self.allowed_labels = {"PERSON", "ORG", "GPE", "LOC", "PRODUCT"}

        # Normalize some labels
        self.label_map = {"GPE": "LOC", "LOC": "LOC"}

    def build(self, text: str) -> nx.DiGraph:
        doc = self.nlp(text)

        # Count co-occurrences
        pair_counts: Dict[Tuple[str, str], int] = defaultdict(int)
        node_types: Dict[str, str] = {}

        for sent in doc.sents:
            ents = [e for e in sent.ents if e.label_ in self.allowed_labels]
            names = []
            for e in ents:
                name = e.text.strip()
                if not name:
                    continue
                names.append(name)
                node_types.setdefault(name, self.label_map.get(e.label_, e.label_))

            # unique entities in sentence
            uniq = list(dict.fromkeys(names))

            # build complete graph on sentence entities
            for i in range(len(uniq)):
                for j in range(i + 1, len(uniq)):
                    a, b = uniq[i], uniq[j]
                    pair_counts[(a, b)] += 1

        # Build directed graph (we'll add edges both ways for PyVis directed graph display)
        g = nx.DiGraph()

        for node, t in node_types.items():
            g.add_node(node, entity_type=t)

        for (a, b), w in pair_counts.items():
            # add bidirectional edges for directed display
            g.add_edge(a, b, label="co-occurs", predicate="co-occurs", confidence=min(1.0, 0.2 + w * 0.15), weight=w)
            g.add_edge(b, a, label="co-occurs", predicate="co-occurs", confidence=min(1.0, 0.2 + w * 0.15), weight=w)

        return g
