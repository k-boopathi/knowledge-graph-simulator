# modules/evaluation_metrics.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Any
import networkx as nx


@dataclass
class GraphMetrics:
    nodes: int
    edges: int
    density: float
    unknown_nodes: int
    other_nodes: int
    unknown_rate: float
    other_rate: float
    avg_node_confidence: float
    meaningful_edge_rate: float  # % edges not "co-occurs"


def compute_metrics(g: nx.Graph, label_key: str = "subsystem_label") -> GraphMetrics:
    n = g.number_of_nodes()
    e = g.number_of_edges()
    density = nx.density(g) if n > 1 else 0.0

    unknown = 0
    other = 0
    conf_sum = 0.0
    conf_count = 0

    for node, attrs in g.nodes(data=True):
        lab = str(attrs.get(label_key, "UNKNOWN")).upper()
        if lab == "UNKNOWN":
            unknown += 1
        if lab == "OTHER":
            other += 1
        if "confidence" in attrs:
            try:
                conf_sum += float(attrs["confidence"])
                conf_count += 1
            except Exception:
                pass

    avg_conf = (conf_sum / conf_count) if conf_count else 0.0

    meaningful = 0
    for _, _, data in g.edges(data=True):
        pred = str(data.get("predicate") or data.get("label") or "co-occurs").lower()
        if pred != "co-occurs":
            meaningful += 1

    meaningful_rate = (meaningful / e) if e else 0.0

    return GraphMetrics(
        nodes=n,
        edges=e,
        density=float(density),
        unknown_nodes=unknown,
        other_nodes=other,
        unknown_rate=(unknown / n) if n else 0.0,
        other_rate=(other / n) if n else 0.0,
        avg_node_confidence=float(avg_conf),
        meaningful_edge_rate=float(meaningful_rate),
    )


def metrics_to_dict(m: GraphMetrics) -> Dict[str, Any]:
    return asdict(m)
