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

# -----------------------------
# NEW: SpaceBERT subsystem-labelled KG builder
# -----------------------------
# Create this file:
#   modules/extractors/labelled_subsystem_kg_builder.py
# and ensure it defines:
#   class LabelledSubsystemKGBuilder:
#       def __init__(...): ...
#       def build(self, text: str, min_conf: float = 0.0) -> nx.Graph
try:
    from modules.extractors.labelled_subsystem_kg_builder import LabelledSubsystemKGBuilder
except Exception:
    LabelledSubsystemKGBuilder = None


# -----------------------------
# Helpers
# -----------------------------
def normalize_subsystem_label(x: str) -> str:
    """
    Map model labels to stable UI filter keys.
    Adjust this mapping to YOUR model's actual label strings.
    """
    s = (x or "").strip()
    if not s or s.upper() == "O":
        return "UNKNOWN"

    low = s.lower()

    # common normalizations (examples)
    if "tele" in low:
        return "TELECOM"
    if "prop" in low:
        return "PROPULSION"
    if "power" in low or "eps" in low:
        return "POWER"
    if "thermal" in low or "ttc" in low:
        return "THERMAL"
    if "aocs" in low or "adcs" in low or "attitude" in low:
        return "AOCS"
    if "payload" in low or "instrument" in low:
        return "PAYLOAD"
    if "ground" in low:
        return "GROUND"
    if "data" in low or "cdh" in low or "handling" in low:
        return "DATA"
    if "orbit" in low:
        return "ORBIT"
    if "concept" in low:
        return "CONCEPT"
    if "other" in low:
        return "OTHER"

    # fallback: keep the raw label uppercased if it is short
    if len(s) <= 24:
        return s.upper()

    return "UNKNOWN"


def get_labelled_show_types() -> dict:
    """
    Default subsystem filters to show in the sidebar.
    Keep these consistent with normalize_subsystem_label().
    """
    return {
        "TELECOM": True,
        "PROPULSION": True,
        "POWER": True,
        "THERMAL": True,
        "AOCS": True,
        "PAYLOAD": True,
        "GROUND": True,
        "DATA": True,
        "ORBIT": True,
        "CONCEPT": True,
        "OTHER": True,
        "UNKNOWN": True,
    }


def color_map_for_mode(mode: str) -> dict:
    # NOTE: PyVis accepts hex colors; keep your previous vibe.
    if mode == "Web Graph":
        return {
            "PERSON": "#4C78A8",
            "ORG": "#F58518",
            "LOC": "#54A24B",
            "PRODUCT": "#E45756",
            "GAS": "#E45756",
            "CONCEPT": "#B279A2",
            "UNKNOWN": "#B0B0B0",
        }
    return {
        "TELECOM": "#ff6b6b",
        "PROPULSION": "#ffa94d",
        "POWER": "#ffd43b",
        "THERMAL": "#69db7c",
        "AOCS": "#4dabf7",
        "PAYLOAD": "#9775fa",
        "GROUND": "#20c997",
        "DATA": "#adb5bd",
        "ORBIT": "#74c0fc",
        "CONCEPT": "#b279a2",
        "OTHER": "#888888",
        "UNKNOWN": "#B0B0B0",
    }


# -----------------------------
# App header
# -----------------------------
st.set_page_config(page_title="Knowledge Graph Simulator", layout="wide")
st.title("Knowledge Graph Simulator")
st.caption(
    "Web Graph (NER + co-occurrence) + Labelled Subsystem KG (SpaceBERT token classification). "
    "Analytics, coloring, export, filters."
)

# -----------------------------
# Session state init
# -----------------------------
if "graph_manager" not in st.session_state:
    st.session_state.graph_manager = GraphManager()

if "entity_typer" not in st.session_state:
    st.session_state.entity_typer = EntityTyper()

if "co_builder" not in st.session_state:
    st.session_state.co_builder = CooccurrenceGraphBuilder()

if "labelled_builder" not in st.session_state:
    # IMPORTANT: pass your actual fine-tuned model name/path if needed
    # Example: LabelledSubsystemKGBuilder("path/to/your-finetuned-model")
    st.session_state.labelled_builder = LabelledSubsystemKGBuilder() if LabelledSubsystemKGBuilder else None

if "last_input_text" not in st.session_state:
    st.session_state.last_input_text = ""

if "last_raw_graph" not in st.session_state:
    st.session_state.last_raw_graph = nx.Graph()

if "last_graph_mode" not in st.session_state:
    st.session_state.last_graph_mode = "Web Graph"

# -----------------------------
# Sidebar controls
# -----------------------------
with st.sidebar:
    st.header("Controls")

    graph_mode = st.radio(
        "Graph Mode",
        ["Web Graph", "Labelled Subsystem KG"],
        index=0,
        help="Web Graph: standard NER + co-occurrence. Labelled Subsystem KG: SpaceBERT labels nodes by subsystem.",
        key="graph_mode_radio",
    )

    st.session_state.last_graph_mode = graph_mode

    st.divider()

    min_conf = st.slider("Min confidence", 0.0, 1.0, 0.20, 0.05, key="min_conf_slider")
    show_edge_labels = st.checkbox("Show edge labels", value=True, key="show_edge_labels")

    st.divider()

    # -----------------------------
    # Type Filters
    # -----------------------------
    if graph_mode == "Web Graph":
        show_types = {
            "PERSON": st.checkbox("Show PERSON", True, key="web_show_person"),
            "ORG": st.checkbox("Show ORG", True, key="web_show_org"),
            "LOC": st.checkbox("Show LOC", True, key="web_show_loc"),
            "PRODUCT": st.checkbox("Show PRODUCT", True, key="web_show_product"),
            "GAS": st.checkbox("Show GAS", True, key="web_show_gas"),
            "CONCEPT": st.checkbox("Show CONCEPT", True, key="web_show_concept"),
            "UNKNOWN": st.checkbox("Show UNKNOWN", True, key="web_show_unknown"),
        }
    else:
        defaults = get_labelled_show_types()
        show_types = {
            k: st.checkbox(f"Show {k}", defaults[k], key=f"lab_show_{k.lower()}")
            for k in defaults.keys()
        }

        # optional: show what model is loaded
        if st.session_state.labelled_builder is None:
            st.warning("LabelledSubsystemKGBuilder not available (missing module).")
        else:
            st.caption("Subsystem labels come from your SpaceBERT token classifier.")

    st.divider()

    if st.button("Clear Graph", key="clear_graph_button"):
        st.session_state.graph_manager.reset_graph()
        st.session_state.last_input_text = ""
        st.session_state.last_raw_graph = nx.Graph()
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
            "The CubeSat Telecom subsystem shall include a transmitter antenna and RF power amplifier.\n"
            "The propulsion subsystem shall provide delta-v using cold gas thrusters.\n"
        ),
        key="text_input_area",
    )

    if st.button("Build Graph", key="build_graph_button"):
        if not text_input.strip():
            st.warning("Please paste some text first.")
        else:
            st.session_state.last_input_text = text_input
            st.session_state.graph_manager.reset_graph()

            # -----------------------------
            # Build raw graph (source-of-truth for node types)
            # -----------------------------
            if graph_mode == "Web Graph":
                raw_g = st.session_state.co_builder.build(text_input)

                # store into GraphManager for exports
                edges_for_manager = []
                for u, v, data in raw_g.edges(data=True):
                    label = str(data.get("label") or data.get("predicate") or "co-occurs")
                    confidence = float(data.get("confidence", 0.30))
                    edges_for_manager.append((u, label, v, confidence))
                st.session_state.graph_manager.add_triples(edges_for_manager)

                # Also store node types (best-effort) for exports/inspection
                base_g = st.session_state.graph_manager.get_graph()
                for n in base_g.nodes():
                    base_g.nodes[n]["entity_type"] = "UNKNOWN"

            else:
                if st.session_state.labelled_builder is None:
                    st.error(
                        "LabelledSubsystemKGBuilder not found. Create "
                        "`modules/extractors/labelled_subsystem_kg_builder.py` "
                        "and add the class."
                    )
                    raw_g = nx.Graph()
                else:
                    raw_g = st.session_state.labelled_builder.build(text_input, min_conf=0.0)

                    # store edges into GraphManager for exports
                    edges_for_manager = []
                    for u, v, data in raw_g.edges(data=True):
                        label = str(data.get("label") or data.get("predicate") or "co-occurs")
                        confidence = float(data.get("confidence", 0.40))
                        edges_for_manager.append((u, label, v, confidence))
                    st.session_state.graph_manager.add_triples(edges_for_manager)

                    # Copy node types from raw_g -> manager graph so exports include them
                    base_g = st.session_state.graph_manager.get_graph()
                    for n in base_g.nodes():
                        base_g.nodes[n]["entity_type"] = "UNKNOWN"

                    for n, attrs in raw_g.nodes(data=True):
                        if n in base_g.nodes:
                            t = attrs.get("entity_type") or "UNKNOWN"
                            base_g.nodes[n]["entity_type"] = normalize_subsystem_label(t)

            st.session_state.last_raw_graph = raw_g

            base_g = st.session_state.graph_manager.get_graph()
            st.success(f"{graph_mode} built: {base_g.number_of_nodes()} nodes, {base_g.number_of_edges()} edges.")

    # -----------------------------
    # Build filtered visualization graph
    # -----------------------------
    base_g = st.session_state.graph_manager.get_graph()
    viz = nx.DiGraph()

    if graph_mode == "Web Graph":
        type_map = st.session_state.entity_typer.extract_types_from_text(st.session_state.last_input_text)

        for n in base_g.nodes():
            t = st.session_state.entity_typer.type_for_node(n, type_map)
            if t in ("GPE", "FAC"):
                t = "LOC"
            if t not in show_types:
                t = "UNKNOWN"
            if show_types.get(t, False):
                viz.add_node(n, entity_type=t)

    else:
        for n in base_g.nodes():
            t = base_g.nodes[n].get("entity_type", "UNKNOWN")
            t = normalize_subsystem_label(t)
            if t not in show_types:
                t = "UNKNOWN"
            if show_types.get(t, False):
                viz.add_node(n, entity_type=t)

    for u, v, data in base_g.edges(data=True):
        conf = float(data.get("confidence", data.get("weight", 0.30)))
        if conf < min_conf:
            continue
        if u not in viz.nodes or v not in viz.nodes:
            continue
        viz.add_edge(u, v, **data)

    # -----------------------------
    # Render graph
    # -----------------------------
    if viz.number_of_nodes() > 0 and viz.number_of_edges() > 0:
        net = Network(height="600px", width="100%", bgcolor="#111111", font_color="white", directed=True)

        cmap = color_map_for_mode(graph_mode)
        degrees = dict(viz.degree())

        for node, attrs in viz.nodes(data=True):
            t = attrs.get("entity_type", "UNKNOWN")
            net.add_node(
                node,
                label=node,
                size=12 + degrees.get(node, 0) * 5,
                color=cmap.get(t, "#B0B0B0"),
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

        net.set_options(f"""
        {{
          "physics": {{
            "enabled": true,
            "solver": "forceAtlas2Based",
            "forceAtlas2Based": {{
              "gravitationalConstant": -90,
              "centralGravity": 0.03,
              "springLength": 160,
              "springConstant": 0.06,
              "avoidOverlap": 1
            }},
            "stabilization": {{ "enabled": true, "iterations": 1000 }}
          }},
          "edges": {{
            "smooth": {{ "enabled": true, "type": "dynamic" }},
            "font": {{ "size": {14 if show_edge_labels else 0}, "align": "middle" }}
          }},
          "nodes": {{
            "shape": "dot",
            "font": {{ "size": 18 }}
          }}
        }}
        """)

        output_file = os.path.join(BASE_DIR, "graph_output.html")
        net.save_graph(output_file)
        with open(output_file, "r", encoding="utf-8") as f:
            components.html(f.read(), height=650)
    else:
        if base_g.number_of_nodes() == 0:
            st.info("Graph is empty. Paste text and click Build Graph.")
        else:
            st.info("Graph built but filtered out. Lower min confidence or enable more types.")
            with st.expander("Debug info"):
                st.write("Stored nodes:", base_g.number_of_nodes())
                st.write("Stored edges:", base_g.number_of_edges())
                st.write("Shown nodes:", viz.number_of_nodes())
                st.write("Shown edges:", viz.number_of_edges())

# -----------------------------
# Analytics / exports column
# -----------------------------
with col_side:
    st.subheader("Text Analytics")
    if st.session_state.last_input_text.strip():
        ta = TextAnalytics(st.session_state.last_input_text)
        s = ta.summary()
        st.metric("Characters", s["characters"])
        st.metric("Sentences", s["sentences"])
        st.metric("Words", s["words"])
        st.metric("Tokens", s["tokens"])
        st.metric("Avg sentence length (words)", s["avg_sentence_length_words"])
    else:
        st.write("Paste text and build a graph.")

    st.divider()

    st.subheader("Graph Analytics")
    analytics = GraphAnalytics(base_g)
    st.metric("Nodes", analytics.node_count())
    st.metric("Edges", analytics.edge_count())
    st.metric("Density", round(analytics.density(), 4))
    cc = nx.number_weakly_connected_components(base_g) if base_g.number_of_nodes() else 0
    st.metric("Connected Components", cc)

    st.divider()

    st.subheader("Exports")
    json_bytes = graph_to_json_bytes(base_g)
    csv_bytes = graph_to_csv_bytes(base_g)
    top_entities = analytics.top_entities(10) if hasattr(analytics, "top_entities") else []
    pdf_bytes = graph_to_pdf_bytes(base_g, title="Knowledge Graph Report", top_entities=top_entities)

    st.download_button("Download Graph JSON", data=json_bytes, file_name="graph.json", mime="application/json")
    st.download_button("Download Edges CSV", data=csv_bytes, file_name="edges.csv", mime="text/csv")
    st.download_button("Download Report PDF", data=pdf_bytes, file_name="report.pdf", mime="application/pdf")

