from __future__ import annotations
from typing import Dict, Tuple, List
from collections import defaultdict
import spacy
import networkx as nx

from modules.labeling.esa_subsystem_labeler import ESASubsystemLabeler


class LabelledSubsystemKGBuilder:
    """
    Builds a co-occurrence KG, but nodes are labelled by ESA subsystem.
    Nodes: entity strings (from EntityRuler/NER/noun chunks if you want)
    Node attribute: entity_type = subsystem label
    """

    def __init__(self, spacy_model: str = "en_core_web_sm"):
        self.nlp = spacy.load(spacy_model)
        self.labeler = ESASubsystemLabeler()

    def build(self, text: str, max_edges_per_sentence: int = 12) -> nx.Graph:
        doc = self.nlp(text)

        pair_counts: Dict[Tuple[str, str], int] = defaultdict(int)
        node_types: Dict[str, str] = {}
        node_conf: Dict[str, float] = {}

        for sent in doc.sents:
            # Use both named entities and noun chunks to get “enough” nodes
            candidates: List[str] = []
            candidates += [e.text.strip() for e in sent.ents]
            candidates += [c.text.strip() for c in sent.noun_chunks]

            # normalize + de-dupe
            uniq = []
            seen = set()
            for x in candidates:
                x = " ".join(x.split())
                if len(x) < 3:
                    continue
                k = x.lower()
                if k in seen:
                    continue
                seen.add(k)
                uniq.append(x)

            if len(uniq) > max_edges_per_sentence:
                uniq = sorted(uniq, key=len, reverse=True)[:max_edges_per_sentence]

            if len(uniq) < 2:
                continue

            # label nodes using sentence context
            for n in uniq:
                res = self.labeler.label_node(n, sent.text)
                node_types[n] = res.label
                node_conf[n] = max(node_conf.get(n, 0.0), res.confidence)

            # co-occurrence edges
            for i in range(len(uniq)):
                for j in range(i + 1, len(uniq)):
                    a, b = uniq[i], uniq[j]
                    if a == b:
                        continue
                    pair_counts[tuple(sorted((a, b)))] += 1

        g = nx.Graph()
        for n, t in node_types.items():
            g.add_node(n, entity_type=t, confidence=node_conf.get(n, 0.35))

        for (a, b), w in pair_counts.items():
            conf = min(1.0, 0.25 + 0.15 * w)
            g.add_edge(a, b, predicate="co-occurs", weight=w, confidence=conf)

        return g
