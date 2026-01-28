import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities


class CommunityDetector:
    def __init__(self, graph: nx.DiGraph):
        # Convert to undirected for community detection
        self.graph = graph.to_undirected()

    def detect_communities(self):
        """
        Returns a list of communities (sets of nodes)
        """
        if self.graph.number_of_nodes() == 0:
            return []

        communities = greedy_modularity_communities(self.graph)
        return [list(c) for c in communities]

    def labeled_communities(self):
        """
        Returns: {node: community_id}
        """
        mapping = {}
        communities = self.detect_communities()

        for idx, community in enumerate(communities):
            for node in community:
                mapping[node] = idx

        return mapping
