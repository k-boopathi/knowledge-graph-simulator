import networkx as nx

from modules.ontology import Ontology
from modules.relation_mapper import RelationMapper
from modules.entity_typer import EntityTyper


class GraphManager:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.ontology = Ontology()
        self.mapper = RelationMapper()
        self.typer = EntityTyper()

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

            # -----------------------
            # Normalize input formats
            # -----------------------
            if isinstance(t, dict):
                subj = t.get("subject")
                pred = t.get("predicate")
                obj = t.get("object")
                conf = float(t.get("confidence", 0.6))
            else:
                if len(t) == 3:
                    subj, pred, obj = t
                    conf = 0.6
                elif len(t) == 4:
                    subj, pred, obj, conf = t
                    conf = float(conf)
                else:
                    continue

            if not subj or not pred or not obj:
                continue

            # -----------------------
            # Ontology mapping
            # -----------------------
            subj_type = self.typer.type_for_node(subj)
            obj_type = self.typer.type_for_node(obj)

            mapped_relation = self.mapper.map_relation(pred)

            # Semantic validation
            if mapped_relation and self.ontology.is_valid(subj_type, mapped_relation, obj_type):
                final_relation = mapped_relation
                conf = max(conf, 0.75)  # boost semantic confidence
            else:
                final_relation = "related_to"
                conf = min(conf, 0.35)  # weak fallback relation

            # -----------------------
            # Add edge
            # -----------------------
            self.graph.add_edge(
                subj,
                obj,
                label=final_relation,
                predicate=final_relation,
                title=f"{final_relation} ({subj_type} → {obj_type})",
                confidence=conf,
                weight=conf
            )

        return self.graph.number_of_edges() - initial_edges

    def get_graph(self):
        return self.graph

    def reset_graph(self):
        self.graph.clear()


