import networkx as nx

from modules.ontology import Ontology
from modules.relation_mapper import RelationMapper
from modules.entity_typer import EntityTyper

class GraphManager:
    def __init__(self):
        self.graph = nx.DiGraph()

    def add_triples(self, triples):
        """
        Accepts:
          - (subj, pred, obj)
          - (subj, pred, obj, confidence)
          - {"subject":..., "predicate":..., "object":..., "confidence":...}
        """
        if not triples:
            return 0

        initial_edges = self.graph.number_of_edges()

        for t in triples:
            if isinstance(t, dict):
                subj = t.get("subject")
                pred = t.get("predicate")
                obj = t.get("object")
                conf = float(t.get("confidence", 0.75))
            else:
                if len(t) == 3:
                    subj, pred, obj = t
                    conf = 0.75
                elif len(t) == 4:
                    subj, pred, obj, conf = t
                    conf = float(conf)
                else:
                    continue

            if not subj or not pred or not obj:
                continue

            self.graph.add_edge(
                subj,
                obj,
                label=str(pred),
                predicate=str(pred),
                title=str(pred),
                confidence=conf,
                weight=conf
            )

        return self.graph.number_of_edges() - initial_edges

    def get_graph(self):
        return self.graph

    def reset_graph(self):
        self.graph.clear()

