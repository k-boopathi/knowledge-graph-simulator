from modules.graph_manager import GraphManager
from modules.graph_analytics import GraphAnalytics

gm = GraphManager()
gm.add_triples([
    ("Apple", "founded_by", "Steve Jobs"),
    ("Apple", "headquartered_in", "Cupertino"),
    ("Steve Jobs", "born_in", "San Francisco"),
    ("Apple", "acquired", "Beats")
])

ga = GraphAnalytics(gm.get_graph())
print(ga.semantic_importance())
