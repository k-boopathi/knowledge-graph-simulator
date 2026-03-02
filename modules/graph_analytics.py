import networkx as nx

class GraphAnalytics:
    def __init__(self, graph):
        self.graph = graph

    def node_count(self):
        return self.graph.number_of_nodes()

    def edge_count(self):
        return self.graph.number_of_edges()

    def density(self):
        if self.node_count() <= 1:
            return 0
        return nx.density(self.graph)

    def degree_centrality(self):
        return nx.degree_centrality(self.graph)

    def betweenness_centrality(self):
        return nx.betweenness_centrality(self.graph)

    # ---------------------------
    # Robust attribute readers
    # ---------------------------
    def _get_subsystem(self, node):
        d = self.graph.nodes[node]
        return (d.get("subsystem_label") or d.get("subsystem") or "UNKNOWN")

    def _get_confidence(self, node):
        d = self.graph.nodes[node]
        c = d.get("confidence")
        if c is None:
            c = d.get("conf")
        try:
            return float(c)
        except Exception:
            return 0.0

    def unknown_rate(self):
        if self.node_count() == 0:
            return 0.0
        unk = 0
        for n in self.graph.nodes():
            if str(self._get_subsystem(n)).upper() == "UNKNOWN":
                unk += 1
        return unk / self.node_count()

    def other_rate(self):
        if self.node_count() == 0:
            return 0.0
        oth = 0
        for n in self.graph.nodes():
            if str(self._get_subsystem(n)).upper() == "OTHER":
                oth += 1
        return oth / self.node_count()

    def avg_node_confidence(self):
        if self.node_count() == 0:
            return 0.0
        vals = [self._get_confidence(n) for n in self.graph.nodes()]
        return sum(vals) / len(vals)

    def semantic_importance(self):
        degree = self.degree_centrality()
        betweenness = self.betweenness_centrality()

        scores = {}
        for node in self.graph.nodes():
            scores[node] = 0.6 * degree.get(node, 0.0) + 0.4 * betweenness.get(node, 0.0)
        return scores

    def top_entities(self, n=5):
        scores = self.semantic_importance()
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n]

