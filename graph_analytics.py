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

    def semantic_importance(self):
        degree = self.degree_centrality()
        betweenness = self.betweenness_centrality()

        scores = {}

        for node in self.graph.nodes():
            scores[node] = round(
                (0.6 * degree.get(node, 0)) +
                (0.4 * betweenness.get(node, 0)),
                4
            )

        return scores

    def top_entities(self, n=5):
        scores = self.semantic_importance()
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n]
