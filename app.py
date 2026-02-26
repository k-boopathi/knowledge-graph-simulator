import os
import sys
import streamlit as st
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from modules.graph_manager import GraphManager
from modules.graph_analytics import GraphAnalytics
from modules.export_utils import graph_to_json_bytes
from modules.cooccurrence_builder import CooccurrenceGraphBuilder
from modules.extractors.labelled_subsystem_kg_builder import LabelledSubsystemKGBuilder


st.set_page_config(page_title="Knowledge Graph Simulator", layout="wide")
st.title("Knowledge Graph Simulator")
st.caption("Web Graph | Labelled Subsystem KG")

# -----------------------------
# Session State Init
# -----------------------------
if "graph_manager" not in st.session_state:
    st.session_state.graph_manager = GraphManager()

if "co_builder" not in st.session_state:
    st.session_state.co_builder = CooccurrenceGraphBuilder()

if "labelled_builder" not in st.session_state:
    st.session_state.labelled_builder = LabelledSubsystemKGBuilder()


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("Controls")

    graph_mode = st.radio(
        "Graph Mode",
        ["Web Graph", "Labelled Subsystem KG"]
    )

    min_conf = st.slider("Min confidence", 0.0, 1.0, 0.20, 0.05)
    show_edge_labels = st.checkbox("Show edge labels", value=True)

    if st.button("Clear Graph"):
        st.session_state.graph_manager.reset_graph()
        st.rerun()


# -----------------------------
# Layout
# -----------------------------
col_graph, col_side = st.columns([3, 2])

# -----------------------------
# GRAPH COLUMN
# -----------------------------
with col_graph:
    st.subheader("Graph Visualization")

    text_input = st.text_area(
        "Paste ESA text:",
        height=230,
    )

    if st.button("Build Graph"):
        if not text_input.strip():
            st.warning("Please paste text.")
        else:
            st.session_state.graph_manager.reset_graph()

            # -----------------------------
            # WEB GRAPH
            # -----------------------------
            if graph_mode == "Web Graph":
                g = st.session_state.co_builder.build(text_input)

            # -----------------------------
            # LABELLED SUBSYSTEM KG
            # -----------------------------
            else:
                g = st.session_state.labelled_builder.build(text_input)

            edges_for_manager = []
            for u, v, data in g.edges(data=True):
                conf = float(data.get("confidence", 0.4))
                edges_for_manager.append((u, "co-occurs", v, conf))

            st.session_state.graph_manager.add_triples(edges_for_manager)

    # -----------------------------
    # Visualization
    # -----------------------------
    g = st.session_state.graph_manager.get_graph()
    viz = nx.DiGraph()

    for n in g.nodes():
        t = g.nodes[n].get("entity_type", "OTHER")
        viz.add_node(n, entity_type=t)

    for u, v, data in g.edges(data=True):
        conf = float(data.get("confidence", 0.3))
        if conf >= min_conf:
            viz.add_edge(u, v, **data)

    if viz.number_of_nodes() > 0:
        net = Network(
            height="600px",
            width="100%",
            bgcolor="#111111",
            font_color="white",
            directed=True
        )

        color_map = {
            "TELECOM": "#ff6b6b",
            "PROPULSION": "#ffa94d",
            "POWER": "#ffd43b",
            "THERMAL": "#69db7c",
            "AOCS": "#4dabf7",
            "PAYLOAD": "#9775fa",
            "GROUND": "#20c997",
            "ORBIT": "#74c0fc",
            "DATA": "#adb5bd",
            "CONCEPT": "#b279a2",
            "OTHER": "#888888",
        }

        degrees = dict(viz.degree())

        for node, attrs in viz.nodes(data=True):
            t = attrs.get("entity_type", "OTHER")
            net.add_node(
                node,
                label=node,
                size=12 + degrees.get(node, 0) * 5,
                color=color_map.get(t, "#888888"),
                title=f"type: {t}",
            )

        for u, v, data in viz.edges(data=True):
            conf = float(data.get("confidence", 0.3))
            net.add_edge(
                u,
                v,
                label="co-occurs" if show_edge_labels else "",
                width=1 + min(6, conf * 6),
                arrows="to",
                title=f"conf {conf:.2f}",
            )

        output_file = os.path.join(BASE_DIR, "graph_output.html")
        net.save_graph(output_file)

        with open(output_file, "r", encoding="utf-8") as f:
            components.html(f.read(), height=650)

    else:
        st.info("Graph empty.")


# -----------------------------
# Analytics
# -----------------------------
with col_side:
    st.subheader("Analytics")

    base_g = st.session_state.graph_manager.get_graph()
    analytics = GraphAnalytics(base_g)

    st.metric("Nodes", analytics.node_count())
    st.metric("Edges", analytics.edge_count())
    st.metric("Density", round(analytics.density(), 4))

    st.divider()
    st.download_button(
        "Download JSON",
        graph_to_json_bytes(base_g),
        "graph.json"
    )

