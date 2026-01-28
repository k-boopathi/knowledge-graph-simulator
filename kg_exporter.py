import json
import csv


class KGExporter:
    def __init__(self, graph):
        self.graph = graph

    def export_json(self, filepath):
        data = {
            "nodes": list(self.graph.nodes()),
            "edges": [
                {
                    "source": u,
                    "target": v,
                    "relation": d.get("label")
                }
                for u, v, d in self.graph.edges(data=True)
            ]
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def export_csv(self, filepath):
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["subject", "predicate", "object"])

            for u, v, d in self.graph.edges(data=True):
                writer.writerow([u, d.get("label"), v])
