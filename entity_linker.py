from difflib import SequenceMatcher

class EntityLinker:
    """
    Dynamically links new entity strings to existing graph nodes
    using fuzzy string similarity (offline).
    """

    def __init__(self, threshold: float = 0.88):
        self.threshold = threshold

    def _sim(self, a: str, b: str) -> float:
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()

    def resolve(self, entity: str, existing_nodes) -> str:
        """
        If entity is similar to an existing node, return the existing node label.
        Else return entity as-is.
        """
        best_node = None
        best_score = 0.0

        for node in existing_nodes:
            score = self._sim(entity, node)
            if score > best_score:
                best_score = score
                best_node = node

        if best_node and best_score >= self.threshold:
            return best_node

        return entity