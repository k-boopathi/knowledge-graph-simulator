class VisualStyler:
    def __init__(self, importance_scores: dict):
        self.importance_scores = importance_scores

    def node_size(self, node):
        """
        Scale node size based on importance
        """
        base = 15
        score = self.importance_scores.get(node, 1)
        return min(base + score * 5, 80)

    def node_color(self, community_id=None):
        palette = [
            "#ff6b6b", "#4ecdc4", "#ffe66d",
            "#a29bfe", "#fd79a8", "#74b9ff"
        ]

        if community_id is None:
            return "#ffffff"

        return palette[community_id % len(palette)]
