import os
import re
import sys
import streamlit as st
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from modules.graph_manager import GraphManager
from modules.graph_analytics import GraphAnalytics
from modules.text_analytics import TextAnalytics
from modules.entity_typer import EntityTyper
from modules.export_utils import graph_to_json_bytes, graph_to_csv_bytes, graph_to_pdf_bytes
from modules.cooccurrence_builder import CooccurrenceGraphBuilder
from modules.extractors.spacy_relation_extractor import SpacyRelationExtractor
from modules.domains.carbonsat.lexicon import CARBONSAT_LEXICON


# -----------------------------
# ESA Type Mapper
# -----------------------------
def esa_type(entity: str) -> str:
    key = (entity or "").strip().lower()
    key = key.replace("(", " ").replace(")", " ").replace(",", " ")
    key = re.sub(r"\s+", " ", key).strip()

    t = CARBONSAT_LEXICON.get(key)
    if t:
        return t

    # substring fallback
    for k, v in CARBONSAT_LEXICON.items():
        if k in key:
            return v

    return "UNKNOWN"


st.set_page_config(page_title="Knowledge Graph Simulator", layout="wide")
st.title("Knowledge Graph Simulator")
st.caption("Web Graph and ESA Fact Graph. Analytics, coloring, export, filters.")

# -----------------------------
# Session state init
# -----------------------------
if "graph_manager" not in st.session_state:
    st.session_state.graph_manager = GraphManager()

if "entity_typer" not in st.session_state:
    st.session_state.entity_typer = EntityTyper()

if "co_builder" not in st.session_state:
    st.session_state.co_builder = CooccurrenceGraphBuilder()

if "fact_extractor" not in st.session_state:
    st.session_state.fact_extractor = SpacyRelationExtractor()

if "last_input_text" not in st.session_state:
    st.session_state.last_input_text = ""


# -----------------------------
# Sidebar controls
# -----------------------------
with st.sidebar:
    st.header("Controls")

    graph_mode = st.radio(
        "Graph Mode",
        ["Web Graph", "ESA Fact Graph"],
        index=0,
        help="Web Graph links co-occurring entities. ESA Fact Graph extracts subject–predicate–object triples.",
        key="graph_mode_radio"
    )

    st.divider()

    min_conf = st.slider("Min confidence", 0.0, 1.0, 0.20, 0.05)
    default_labels = (graph_mode == "ESA Fact Graph")
    show_edge_labels = st.checkbox("Show edge labels", value=default_labels)

    st.divider()

    # -----------------------------
    # Type Filters
    # -----------------------------
    if graph_mode == "Web Graph":
        show_types = {
            "PERSON": st.checkbox("Show PERSON", True),
            "ORG": st.checkbox("Show ORG", True),
            "LOC": st.checkbox("Show LOC", True),
            "PRODUCT": st.checkbox("Show PRODUCT", True),
            "GAS": st.checkbox("Show GAS", True),
            "CONCEPT": st.checkbox("Show CONCEPT", True),
            "UNKNOWN": st.checkbox("Show UNKNOWN", True),
        }
    else:
        show_types = {
            "MISSION": st.checkbox("Show MISSION", True),
            "GAS": st.checkbox("Show GAS", True),
            "PRODUCT": st.checkbox("Show PRODUCT", True),
            "ORBIT": st.checkbox("Show ORBIT", True),
            "PARAMETER": st.checkbox("Show PARAMETER", True),
            "ORG": st.checkbox("Show ORG", True),
            "LOC": st.checkbox("Show LOC", True),
            "UNKNOWN": st.checkbox("Show UNKNOWN", True),
        }

    st.divider()

    if st.button("Clear Graph"):
        st.session_state.graph_manager.reset_graph()
        st.session_state.last_input_text = ""
        st.success("Graph cleared")
        st.rerun()


# -----------------------------
# Layout
# -----------------------------
col_graph, col_side = st.columns([3, 2])

# -----------------------------
# Graph column
# -----------------------------
with col_graph:
    st.subheader("Graph Visualization")

    text_input = st.text_area(
        "Paste text:",
        height=230,
        placeholder=(
            "Example:\n"
            "CarbonSat is a proposed Earth observation mission developed by ESA.\n"
            "The mission aims to measure atmospheric carbon dioxide and methane.\n"
            "CarbonSat operates in low Earth orbit.\n"
        ),
    )

    if st.button("Build Graph"):
        if not text_input.strip():
            st.warning("Please paste some text first.")
        else:
            st.session_state.last_input_text = text_input
            st.session_state.graph_manager.reset_graph()

            # -----------------------------
            # WEB GRAPH
            # -----------------------------
            if graph_mode == "Web Graph":
                web_g = st.session_state.co_builder.build(text_input)

                edges_for_manager = []
                for u, v, data in web_g.edges(data=True):
                    label = str(data.get("label") or "co-occurs")
                    confidence = float(data.get("confidence", 0.30))
                    edges_for_manager.append((u, label, v, confidence))

                added = st.session_state.graph_manager.add_triples(edges_for_manager)

                st.success(
                    f"Web Graph built: {web_g.number_of_nodes()} entities, "
                    f"{web_g.number_of_edges()} links ({added} stored)."
                )

            # -----------------------------
            # ESA FACT GRAPH
            # -----------------------------
            else:
                triples = st.session_state.fact_extractor.extract(text_input)

                if triples:
                    added = st.session_state.graph_manager.add_triples(triples)
                    st.success(f"ESA Fact Graph built: {added} triples stored.")
                    with st.expander("Extracted triples"):
                        st.write(triples)
                else:
                    st.warning("No triples extracted. Try adding more extraction rules.")


    # -----------------------------
    # Build filtered graph
    # -----------------------------
    g = st.session_state.graph_manager.get_graph()
    viz = nx.DiGraph()

    if graph_mode == "Web Graph":
        type_map = st.session_state.entity_typer.extract_types_from_text(
            st.session_state.last_input_text
        )

        for n in g.nodes():
            t = st.session_state.entity_typer.type_for_node(n, type_map)

            if t in ("GPE", "FAC"):
                t = "LOC"

            if t not in show_types:
                t = "UNKNOWN"

            if show_types.get(t, False):
                viz.add_node(n, entity_type=t)

    else:
        for n in g.nodes():
            t = esa_type(n)

            if t not in show_types:
                t = "UNKNOWN"

            if show_types.get(t, False):
                viz.add_node(n, entity_type=t)

    for u, v, data in g.edges(data=True):
        conf = float(data.get("confidence", data.get("weight", 0.30)))
        if conf < min_conf:
            continue
        if u not in viz.nodes or v not in viz.nodes:
            continue
        viz.add_edge(u, v, **data)

    # -----------------------------
    # Render Graph
    # -----------------------------
    if viz.number_of_nodes() > 0 and viz.number_of_edges() > 0:

        net = Network(height="600px", width="100%", bgcolor="#111111",
                      font_color="white", directed=True)

        if graph_mode == "Web Graph":
            color_map = {
                "PERSON": "#4C78A8",
                "ORG": "#F58518",
                "LOC": "#54A24B",
                "PRODUCT": "#E45756",
                "GAS": "#E45756",
                "CONCEPT": "#B279A2",
                "UNKNOWN": "#B0B0B0",
            }
        else:
            color_map = {
                "MISSION": "#4C78A8",
                "GAS": "#E45756",
                "PRODUCT": "#F58518",
                "ORBIT": "#54A24B",
                "PARAMETER": "#B279A2",
                "ORG": "#72B7B2",
                "LOC": "#59A14F",
                "UNKNOWN": "#B0B0B0",
            }

        degrees = dict(viz.degree())

        for node, attrs in viz.nodes(data=True):
            t = attrs.get("entity_type", "UNKNOWN")
            net.add_node(
                node,
                label=node,
                size=12 + degrees.get(node, 0) * 5,
                color=color_map.get(t, "#B0B0B0"),
                title=f"type: {t}",
            )

        for u, v, data in viz.edges(data=True):
            raw_label = str(data.get("label") or data.get("predicate") or "related")
            conf = float(data.get("confidence", data.get("weight", 0.30)))
            width = 1 + min(6, conf * 6)

            net.add_edge(
                u,
                v,
                label=raw_label if show_edge_labels else "",
                width=width,
                arrows="to",
                title=f"{raw_label} (conf {conf:.2f})",
            )

        net.set_options("""
        {
          "physics": {
            "enabled": true,
            "solver": "forceAtlas2Based"
          }
        }
        """)

        output_file = os.path.join(BASE_DIR, "graph_output.html")
        net.save_graph(output_file)

        with open(output_file, "r", encoding="utf-8") as f:
            components.html(f.read(), height=650)

    else:
        st.info("Graph is empty or filtered out.")


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

    st.subheader("Exports")
    st.download_button("Download JSON",
                       graph_to_json_bytes(base_g),
                       "graph.json")
