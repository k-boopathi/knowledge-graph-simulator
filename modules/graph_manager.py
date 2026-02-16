import re
import networkx as nx


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
            subj, pred, obj, conf = self._parse_triple(t)
            if not subj or not pred or not obj:
                continue

            subj = self._normalize_entity(subj)
            obj = self._normalize_entity(obj)
            pred = self._normalize_predicate(pred)

            if not subj or not obj or not pred:
                continue

            subj = self._resolve_alias(subj)
            obj = self._resolve_alias(obj)

            if subj == obj:
                continue

            self.graph.add_edge(
                subj,
                obj,
                label=pred,
                predicate=pred,
                title=pred,
                confidence=conf,
                weight=conf
            )

        return self.graph.number_of_edges() - initial_edges

    def _parse_triple(self, t):
        subj = pred = obj = None
        conf = 0.75

        if isinstance(t, dict):
            subj = t.get("subject")
            pred = t.get("predicate")
            obj = t.get("object")
            conf = float(t.get("confidence", 0.75))
        else:
            try:
                if len(t) == 3:
                    subj, pred, obj = t
                elif len(t) == 4:
                    subj, pred, obj, conf = t
                    conf = float(conf)
            except Exception:
                return None, None, None, 0.0

        return subj, pred, obj, conf

    def _normalize_entity(self, s: str) -> str:
        if s is None:
            return ""

        s = str(s).strip()

        # Remove surrounding quotes
        s = s.strip("\"'`")

        # Collapse whitespace
        s = re.sub(r"\s+", " ", s)

        # Remove trailing punctuation that causes duplicates
        s = re.sub(r"[,\.;:\)\]]+$", "", s).strip()

        # Normalize common org suffix punctuation
        s = s.replace("Inc.", "Inc")
        s = s.replace("Co.", "Co")
        s = s.replace("Ltd.", "Ltd")

        # Normalize apostrophes
        s = s.replace("’", "'")

        # Title-case simple names (avoid wrecking all-caps acronyms)
        if len(s) > 1 and not s.isupper():
            s = " ".join([w if w.isupper() else w[:1].upper() + w[1:] for w in s.split()])

        # Optional: collapse some corporate variants lightly
        # (does NOT remove "Inc" entirely, just normalizes spacing)
        s = re.sub(r"\b(inc|ltd|co|corp|corporation|company)\b", lambda m: m.group(1).lower(), s, flags=re.I)
        s = re.sub(r"\s+", " ", s).strip()

        return s

    def _normalize_predicate(self, p: str) -> str:
        if p is None:
            return ""
        p = str(p).strip()
        p = p.strip("\"'`")
        p = re.sub(r"\s+", " ", p)
        p = re.sub(r"[,\.;:]+$", "", p).strip()

        # Light normalization for web graph label
        if p.lower() in {"co occurs", "co-occur", "cooccurs"}:
            p = "co-occurs"

        return p

    def _resolve_alias(self, ent: str) -> str:
        """
        If graph already contains a more specific version of this entity,
        map to it. Example: "Jobs" -> "Steve Jobs" if present.
        """
        if not ent or self.graph.number_of_nodes() == 0:
            return ent

        # If exact node exists, keep it
        if ent in self.graph:
            return ent

        ent_l = ent.lower()

        # If single token like "Jobs", try to match to an existing multi-word node ending with that token
        if " " not in ent and len(ent) >= 3:
            candidates = []
            for n in self.graph.nodes():
                n_l = str(n).lower()
                if n_l.endswith(" " + ent_l):
                    candidates.append(n)

            # Choose the shortest candidate (usually "Steve Jobs" over longer titles)
            if candidates:
                candidates.sort(key=lambda x: len(str(x)))
                return candidates[0]

        # Merge common Apple variants into the most frequent Apple-like node (basic heuristic)
        if ent_l in {"apple", "apple inc", "apple computer", "apple computer company", "apple computer, inc"}:
            apple_candidates = [n for n in self.graph.nodes() if str(n).lower().startswith("apple")]
            if apple_candidates:
                apple_candidates.sort(key=lambda x: (-self.graph.degree(x), len(str(x))))
                return apple_candidates[0]

        return ent

    def get_graph(self):
        return self.graph

    def reset_graph(self):
        self.graph.clear()
