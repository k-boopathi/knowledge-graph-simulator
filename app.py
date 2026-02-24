import streamlit as st
import sys
import os
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

# CarbonSat fact extractor (spaCy + domain gate)
from modules.extractors.spacy_relation_extractor import SpacyRelationExtractor
from modules.domains.carbonsat.lexicon import CARBONSAT_LEXICON


def carbonsat_type(entity: str) -> str:
    key = (entity or "").strip().lower()
    return CARBONSAT_LEXICON.get(key, "UNKNOWN")


st.set_page_config(page_title="Knowledge Graph Simulator", layout="wide")
st.title("Knowledge Graph Simulator")
st.caption("Web Graph and CarbonSat Fact Graph. Analytics, coloring, export, filters.")

if "graph_manager" not in st.session_state:
    st.session_state.graph_manager = GraphManager()

if "entity_typer" not in st.session_state:
    st.session_state.entity_typer = EntityTyper()

if "co_builder" not in st.session_state:
    st.session_state.co_builder = CooccurrenceGraphBuilder()

if "last_input_text" not in st.session_state:
    st.session_state.last_input_text = ""

if "fact_extractor" not in st.session_state:
    # Domain mode is the change you requested
    st.session_state.fact_extractor = SpacyRelationExtractor(domain_mode="carbonsat")


with st.sidebar:
    st.header("Controls")

    graph_mode = st.radio(
        "Graph Mode",
        ["Web Graph", "CarbonSat Fact Graph"],
        index=0,
        help="Web Graph links co-occurring entities. CarbonSat Fact Graph extracts validated triples for the ESA CarbonSat domain."
    )

    st.divider()

    min_conf = st.slider("Min confidence", 0.0, 1.0, 0.20, 0.05)

    show_edge_labels = st.checkbox("Show edge labels", value=False)
    if not show_edge_labels:
        st.caption("Edge labels are hidden for a cleaner graph.")

    st.divider()

    if graph_mode == "Web Graph":
        show_types = {
            "PERSON": st.checkbox("Show PERSON", True),
            "ORG": st.checkbox("Show ORG", True),
            "LOC": st.checkbox("Show LOC", True),
            "PRODUCT": st.checkbox("Show PRODUCT", True),
            "UNKNOWN": st.checkbox("Show UNKNOWN", True),
        }
    else:
        # CarbonSat domain type filters (lexicon-driven)
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


col_graph, col_side = st.columns([3, 2])

with col_graph:
    st.subheader("Graph Visualization")

    text_input = st.text_area(
        "Paste text:",
        height=230,
        placeholder=(
            "Example (CarbonSat domain):\n"
            "CarbonSat measures CO2 and CH4.\n"
            "CarbonSat produces Level-2 products.\n"
            "CarbonSat operates in a sun-synchronous orbit.\n"
        )
    )

    if st.button("Build Graph"):
        if not text_input.strip():
            st.warning("Please paste some text first.")
        else:
            st.session_state.last_input_text = text_input
            st.session_state.graph_manager.reset_graph()

            if graph_mode == "Web Graph":
                web_g = st.session_state.co_builder.build(text_input)

                edges_for_manager = []
                for u, v, data in web_g.edges(data=True):
                    label = data.get("label", "co-occurs")
                    confidence = float(data.get("confidence", 0.30))
                    edges_for_manager.append((u, label, v, confidence))

                st.session_state.graph_manager.add_triples(edges_for_manager)

                st.success(
                    f"Web Graph built: {web_g.number_of_nodes()} entities, {web_g.number_of_edges()} links."
                )

            else:
                triples = st.session_state.fact_extractor.extract(text_input)  # list of dicts

                if triples:
                    st.session_state.graph_manager.add_triples(triples)
                    st.success(f"CarbonSat Fact Graph built: {len(triples)} validated triples added.")
                    with st.expander("Extracted triples"):
                        st.write(triples)
                else:
                    st.warning("No validated CarbonSat triples extracted. Expand lexicon or relation mapping.")

    g = st.session_state.graph_manager.get_graph()

    # Type assignment for visualization
    viz = nx.DiGraph()

    if graph_mode == "Web Graph":
        type_map = st.session_state.entity_typer.extract_types_from_text(st.session_state.last_input_text)
        for n in g.nodes():
            t = st.session_state.entity_typer.type_for_node(n, type_map)
            if show_types.get(t, False):
                viz.add_node(n, entity_type=t)
    else:
        for n in g.nodes():
            t = carbonsat_type(n)
            if show_types.get(t, False):
                viz.add_node(n, entity_type=t)

    for u, v, data in g.edges(data=True):
        conf = float(data.get("confidence", data.get("weight", 0.0)))
        if conf < min_conf:
            continue
        if u not in viz.nodes or v not in viz.nodes:
            continue
        viz.add_edge(u, v, **data)

    if viz.number_of_nodes() > 0:
        net = Network(height="600px", width="100%", bgcolor="#111111", font_color="white", directed=True)

        if graph_mode == "Web Graph":
            color_map = {
                "PERSON": "#4C78A8",
                "ORG": "#F58518",
                "LOC": "#54A24B",
                "PRODUCT": "#E45756",
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
                title=f"type: {t}"
            )

        for u, v, data in viz.edges(data=True):
            raw_label = data.get("label") or data.get("predicate") or "related"
            conf = float(data.get("confidence", data.get("weight", 0.30)))
            width = 1 + min(6, conf * 6)

            edge_label = raw_label if show_edge_labels else ""
            edge_title = raw_label

            net.add_edge(
                u, v,
                label=edge_label,
                width=width,
                arrows="to",
                title=f"{edge_title} (conf {conf:.2f})"
            )

        edge_font_size = 14 if show_edge_labels else 0
        node_font_size = 18

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
    "stabilization": {{ "enabled": true, "iterations": 1400 }}
  }},
  "edges": {{
    "smooth": {{ "enabled": true, "type": "dynamic" }},
    "font": {{ "size": {edge_font_size}, "align": "middle" }}
  }},
  "nodes": {{
    "shape": "dot",
    "font": {{ "size": {node_font_size} }}
  }}
}}
""")

        output_file = os.path.join(BASE_DIR, "graph_output.html")
        net.save_graph(output_file)
        with open(output_file, "r", encoding="utf-8") as f:
            components.html(f.read(), height=650)
    else:
        st.info("Graph is empty (or filtered out). Lower min confidence or enable more types.")


with col_side:
    st.subheader("Analytics")

    if st.session_state.last_input_text.strip():
        ta = TextAnalytics(st.session_state.last_input_text)
        s = ta.summary()
        st.markdown("Text Analytics")
        st.metric("Characters", s["characters"])
        st.metric("Sentences", s["sentences"])
        st.metric("Words", s["words"])
        st.metric("Tokens", s["tokens"])
        st.metric("Avg sentence length (words)", s["avg_sentence_length_words"])
    else:
        st.markdown("Text Analytics")
        st.write("Paste text and build a graph.")

    st.divider()

    st.markdown("Graph Analytics")
    base_g = st.session_state.graph_manager.get_graph()
    analytics = GraphAnalytics(base_g)

    st.metric("Nodes", analytics.node_count())
    st.metric("Edges", analytics.edge_count())
    st.metric("Density", round(analytics.density(), 4))
    cc = nx.number_weakly_connected_components(base_g) if base_g.number_of_nodes() else 0
    st.metric("Connected Components", cc)

    st.divider()

    st.markdown("Exports")

    json_bytes = graph_to_json_bytes(base_g)
    csv_bytes = graph_to_csv_bytes(base_g)
    top_entities = analytics.top_entities(10) if hasattr(analytics, "top_entities") else []
    pdf_bytes = graph_to_pdf_bytes(base_g, title="Knowledge Graph Report", top_entities=top_entities)

    st.download_button("Download Graph JSON", data=json_bytes, file_name="graph.json", mime="application/json")
    st.download_button("Download Edges CSV", data=csv_bytes, file_name="edges.csv", mime="text/csv")
    st.download_button("Download Report PDF", data=pdf_bytes, file_name="report.pdf", mime="application/pdf")

