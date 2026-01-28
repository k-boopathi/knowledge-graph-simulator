# modules/inference_engine.py
from __future__ import annotations
from typing import List, Tuple
import networkx as nx


class InferenceEngine:
    """
    Adds inferred edges to connect components.

    Rule:
      If A succeeded B
      AND B --(ceo of)--> ORG
      THEN infer A --(ceo of)--> ORG  (lower confidence)
    """

    def __init__(self, inferred_confidence_multiplier: float = 0.7):
        self.mult = inferred_confidence_multiplier

    def apply(self, graph: nx.DiGraph) -> int:
        added = 0

        # Collect CEO-of edges: (person -> org)
        ceo_edges: List[Tuple[str, str, float]] = []
        for u, v, data in graph.edges(data=True):
            pred = (data.get("label") or data.get("predicate") or "").strip().lower()
            conf = float(data.get("confidence", data.get("weight", 0.8)))

            if pred == "ceo of":
                ceo_edges.append((u, v, conf))

        # Build lookup: person -> [(org, conf)]
        ceo_of = {}
        for person, org, conf in ceo_edges:
            ceo_of.setdefault(person, []).append((org, conf))

        # Find succession edges and infer CEO-of
        for a, b, data in graph.edges(data=True):
            pred = (data.get("label") or data.get("predicate") or "").strip().lower()
            succ_conf = float(data.get("confidence", data.get("weight", 0.8)))

            if pred not in {"succeed", "succeeded", "succeed by", "succeeded by"}:
                continue

            # We store edges as A -> B for "A succeed B" based on your extractor output.
            successor = a
            predecessor = b

            if predecessor not in ceo_of:
                continue

            for org, pred_ceo_conf in ceo_of[predecessor]:
                inferred_conf = round(min(succ_conf, pred_ceo_conf) * self.mult, 2)

                # Add only if missing
                if graph.has_edge(successor, org):
                    existing_label = (graph[successor][org].get("label") or "").lower()
                    if existing_label == "ceo of":
                        continue

                graph.add_edge(
                    successor,
                    org,
                    label="ceo of",
                    predicate="ceo of",
                    confidence=inferred_conf,
                    weight=inferred_conf,
                    inferred=True,
                    title=f"ceo of (inferred, conf {inferred_conf})"
                )
                added += 1

        return added
