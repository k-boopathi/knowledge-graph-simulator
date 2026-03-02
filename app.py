from __future__ import annotations

import os
import sys
import json
import tempfile

import streamlit as st
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from modules.graph_analytics import GraphAnalytics
from modules.text_analytics import TextAnalytics
from modules.entity_typer import EntityTyper
from modules.export_utils import graph_to_json_bytes, graph_to_csv_bytes, graph_to_pdf_bytes
from modules.cooccurrence_builder import CooccurrenceGraphBuilder

# Labelled (subsystem-only) KG builder
try:
    from modules.extractors.labelled_subsystem_kg_builder import LabelledSubsystemKGBuilder
except Exception:
    LabelledSubsystemKGBuilder = None

# ESA PDF cleaner/extractor
try:
    from modules.extractors.pdf_text_extractor import PDFTextExtractor
except Exception:
    PDFTextExtractor = None

# Metrics
try:
    from modules.evaluation_metrics import compute_metrics, metrics_to_dict
except Exception:
    compute_metrics = None
    metrics_to_dict = None

# LLM Router (ESA sentence routing)
try:
    from modules.llm.esa_router import route_esa_text, engineering_only_text
except Exception:
    route_esa_text = None
    engineering_only_text = None


# -----------------------------
# Helpers
# -----------------------------
def safe_label(x: str) -> str:
    s = (x or "").strip()
    if not s or s.upper() == "O":
        return "UNKNOWN"
    return s.upper() if len(s) <= 32 else "UNKNOWN"


def color_map_web() -> dict:
    return {
        "PERSON": "#4C78A8",
        "ORG": "#F58518",
        "LOC": "#54A24B",
        "PRODUCT": "#E45756",
        "GAS": "#E45756",
        "CONCEPT": "#B279A2",
        "UNKNOWN": "#B0B0B0",
    }


def color_map_subsystem() -> dict:
    return {
        "TELECOM": "#ff6b6b",
        "POWER": "#ffd43b",
        "DATA": "#adb5bd",
        "PAYLOAD": "#9775fa",
        "ORBIT": "#74c0fc",
        "GROUND": "#20c997",
        "PROPULSION": "#ffa94d",
        "THERMAL": "#69db7c",
        "AOCS": "#4dabf7",
        "OTHER": "#888888",
        "UNKNOWN": "#B0B0B0",
    }


def get_show_types_web() -> dict:
    return {
        "PERSON": True,
        "ORG": True,
        "LOC": True,
        "PRODUCT": True,
        "GAS": True,
        "CONCEPT": True,
        "UNKNOWN": True,
    }


def get_show_types_subsystem() -> dict:
    return {
        "TELECOM": True,
        "POWER": True,
        "DATA": True,
        "PAYLOAD": True,
        "ORBIT": True,
        "GROUND": True,
        "PROPULSION": True,
        "THERMAL": True,
        "AOCS": True,
        "OTHER": True,
        "UNKNOWN": True,
    }


def write_temp_pdf_bytes(pdf_bytes: bytes) -> str:
    """Write bytes to a temp PDF and return path."""
    fd, path = tempfile.mkstemp(suffix=".pdf")
    os.close(fd)
    with open(path, "wb") as f:
        f.write(pdf_bytes)
    return path


def build_baseline_graph(text: str, co_builder: CooccurrenceGraphBuilder, entity_typer: EntityTyper) -> nx.DiGraph:
    raw_g = co_builder.build(text)

    g = nx.DiGraph()
    for n in raw_g.nodes():
        g.add_node(n)

    for u, v, data in raw_g.edges(data=True):
        g.add_edge(u, v, **data)
        if not g[u][v].get("label"):
            g[u][v]["label"] = g[u][v].get("predicate", "co-occurs")
        if "confidence" not in g[u][v]:
            g[u][v]["confidence"] = float(g[u][v].get("weight", 0.30))

    type_map = entity_typer.extract_types_from_text(text)
    for n in list(g.nodes()):
        t = entity_typer.type_for_node(n, type_map)
        if t in ("GPE", "FAC"):
            t = "LOC"
        if t not in get_show_types_web():
            t = "UNKNOWN"
        g.nodes[n]["entity_type"] = t

    return g


def build_improved_graph(text: str, builder: LabelledSubsystemKGBuilder) -> nx.DiGraph:
    raw_g = builder.build(text, min_conf=0.0)

    g = nx.DiGraph()
    for n, attrs in raw_g.nodes(data=True):
        g.add_node(n, **attrs)

    for u, v, data in raw_g.edges(data=True):
        g.add_edge(u, v, **data)
        if not g[u][v].get("label"):
            g[u][v]["label"] = g[u][v].get("predicate", "co-occurs")
        if "confidence" not in g[u][v]:
            g[u][v]["confidence"] = float(g[u][v].get("weight", 0.35))

    for n in list(g.nodes()):
        g.nodes[n]["subsystem_label"] = safe_label(g.nodes[n].get("subsystem_label", "UNKNOWN"))
        g.nodes[n]["confidence"] = float(g.nodes[n].get("confidence", 0.0))

    return g


def filter_graph(
    g: nx.DiGraph,
    label_key: str,
    show_types: dict,
    min_conf: float,
) -> nx.DiGraph:
    viz = nx.DiGraph()

    for n, attrs in g.nodes(data=True):
        t = safe_label(attrs.get(label_key, "UNKNOWN"))
        if t not in show_types:
            t = "UNKNOWN"

        if show_types.get(t, False):
            clean_attrs = dict(attrs)
            clean_attrs.pop("entity_type", None)
            viz.add_node(n, **clean_attrs)
            viz.nodes[n]["entity_type"] = t

    for u, v, data in g.edges(data=True):
        conf = float(data.get("confidence", data.get("weight", 0.30)))
        if conf < min_conf:
            continue
        if u not in viz.nodes or v not in viz.nodes:
            continue
        viz.add_edge(u, v, **data)

    return viz


def render_pyvis_graph(
    g_full: nx.DiGraph,
    g_viz: nx.DiGraph,
    label_key: str,
    show_edge_labels: bool,
    html_path: str,
    cmap: dict,
):
    if g_viz.number_of_nodes() == 0 or g_viz.number_of_edges() == 0:
        st.info("Graph built but filtered out. Lower min confidence or enable more types.")
        with st.expander("Debug info"):
            st.write("Stored nodes:", g_full.number_of_nodes())
            st.write("Stored edges:", g_full.number_of_edges())
            st.write("Shown nodes:", g_viz.number_of_nodes())
            st.write("Shown edges:", g_viz.number_of_edges())
        return

    net = Network(height="600px", width="100%", bgcolor="#111111", font_color="white", directed=True)
    degrees = dict(g_viz.degree())

    for node, attrs in g_viz.nodes(data=True):
        t = safe_label(attrs.get(label_key, attrs.get("entity_type", "UNKNOWN")))
        node_conf = float(g_full.nodes[node].get("confidence", 0.0)) if node in g_full.nodes else 0.0
        node_label = f"{node}\n[{t}]"

        tooltip_parts = [f"type: {t}", f"confidence: {node_conf:.2f}"]
        for k in ["node_kind", "source", "ontology_class", "ontology_parent", "ontology_conf", "ontology_method"]:
            if k in g_full.nodes.get(node, {}):
                tooltip_parts.append(f"{k}: {g_full.nodes[node].get(k)}")
        title = "<br>".join(tooltip_parts)

        net.add_node(
            node,
            label=node_label,
            size=12 + degrees.get(node, 0) * 5,
            color=cmap.get(t, "#B0B0B0"),
            title=title,
        )

    for u, v, data in g_viz.edges(data=True):
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

    net.save_graph(html_path)
    with open(html_path, "r", encoding="utf-8") as f:
        components.html(f.read(), height=650)


# -----------------------------
# App header
# -----------------------------
st.set_page_config(page_title="Knowledge Graph Simulator", layout="wide")
st.title("Knowledge Graph Simulator")
st.caption(
    "Baseline (NER + co-occurrence) vs Improved (ESA-cleaned text + subsystem labelling + ontology mapping). "
    "Includes PDF mode, comparison mode, analytics, and exports."
)

# Load OpenAI key from Streamlit secrets if available
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]


# -----------------------------
# Session state init
# -----------------------------
if "entity_typer" not in st.session_state:
    st.session_state.entity_typer = EntityTyper()

if "co_builder" not in st.session_state:
    st.session_state.co_builder = CooccurrenceGraphBuilder()

if "labelled_builder" not in st.session_state:
    st.session_state.labelled_builder = LabelledSubsystemKGBuilder() if LabelledSubsystemKGBuilder else None

if "pdf_extractor" not in st.session_state:
    st.session_state.pdf_extractor = PDFTextExtractor() if PDFTextExtractor else None

if "last_input_text" not in st.session_state:
    st.session_state.last_input_text = ""

if "baseline_graph" not in st.session_state:
    st.session_state.baseline_graph = nx.DiGraph()

if "improved_graph" not in st.session_state:
    st.session_state.improved_graph = nx.DiGraph()

# PDF caching for Streamlit reruns
if "pdf_tmp_path" not in st.session_state:
    st.session_state.pdf_tmp_path = None
if "pdf_raw_text" not in st.session_state:
    st.session_state.pdf_raw_text = ""
if "pdf_cleaned_text" not in st.session_state:
    st.session_state.pdf_cleaned_text = ""
if "pdf_pages_read" not in st.session_state:
    st.session_state.pdf_pages_read = 0
if "pdf_removed_lines" not in st.session_state:
    st.session_state.pdf_removed_lines = 0


# -----------------------------
# Sidebar controls
# -----------------------------
with st.sidebar:
    st.header("Controls")

    input_mode = st.radio(
        "Input Mode",
        ["Paste Text", "PDF (ESA)"],
        index=0,
        help="PDF mode uses ESA PDF cleaner (header/footer removal + line reflow).",
    )

    compare_mode = st.checkbox(
        "Comparison Mode (Baseline vs Improved)",
        value=True,
        help="Build and show two graphs side-by-side: Baseline (co-occurrence) vs Improved (subsystem labels + ontology).",
    )

    st.divider()

    use_llm_router = st.checkbox(
        "Use LLM routing (recommended for ESA PDF paragraphs)",
        value=True,
        help="Filters science background sentences before subsystem KG extraction.",
        disabled=(route_esa_text is None or engineering_only_text is None),
    )
    if route_esa_text is None or engineering_only_text is None:
        st.caption("LLM router not available. Add modules/llm/esa_router.py and install openai.")

    st.divider()
    min_conf = st.slider("Min confidence", 0.0, 1.0, 0.20, 0.05)
    show_edge_labels = st.checkbox("Show edge labels", value=True)

    st.divider()
    st.subheader("Filters")

    show_types_web_defaults = get_show_types_web()
    show_types_sub_defaults = get_show_types_subsystem()

    with st.expander("Baseline types (Web Graph)", expanded=False):
        show_types_web = {
            k: st.checkbox(f"Show {k}", show_types_web_defaults[k], key=f"web_show_{k.lower()}")
            for k in show_types_web_defaults.keys()
        }

    with st.expander("Improved types (Subsystem)", expanded=True):
        show_types_sub = {
            k: st.checkbox(f"Show {k}", show_types_sub_defaults[k], key=f"sub_show_{k.lower()}")
            for k in show_types_sub_defaults.keys()
        }

    st.divider()
    if st.button("Clear Graphs"):
        st.session_state.last_input_text = ""
        st.session_state.baseline_graph = nx.DiGraph()
        st.session_state.improved_graph = nx.DiGraph()
        st.success("Cleared.")
        st.rerun()

    if input_mode == "PDF (ESA)" and st.session_state.pdf_extractor is None:
        st.warning("PDFTextExtractor not available. Add modules/extractors/pdf_text_extractor.py and esa_pdf_cleaner.py.")
    if st.session_state.labelled_builder is None:
        st.warning("LabelledSubsystemKGBuilder not available. Add modules/extractors/labelled_subsystem_kg_builder.py.")


# -----------------------------
# Layout
# -----------------------------
col_graph, col_side = st.columns([3, 2])


# -----------------------------
# Input + build
# -----------------------------
with col_graph:
    st.subheader("Input")

    source_text = ""  # <-- IMPORTANT: this is what Build uses

    if input_mode == "PDF (ESA)":
        pdf_file = st.file_uploader("Upload ESA PDF", type=["pdf"])
        c1, c2 = st.columns(2)
        with c1:
            page_from = st.number_input("Page start (0-index)", min_value=0, value=10, step=1)
        with c2:
            page_to = st.number_input("Page end (exclusive, 0 = all)", min_value=0, value=30, step=1)

        # Extract immediately on upload/range change (cached in session_state)
        if pdf_file is not None and st.session_state.pdf_extractor is not None:
            # bytes once
            pdf_bytes = pdf_file.getvalue()

            # only rewrite temp file if changed
            pdf_sig = (pdf_file.name, len(pdf_bytes))
            if st.session_state.get("pdf_sig") != pdf_sig:
                st.session_state.pdf_sig = pdf_sig
                st.session_state.pdf_tmp_path = write_temp_pdf_bytes(pdf_bytes)

            page_to_val = None if int(page_to) == 0 else int(page_to)

            res = st.session_state.pdf_extractor.extract(
                st.session_state.pdf_tmp_path,
                page_from=int(page_from),
                page_to=page_to_val,
            )

            st.session_state.pdf_raw_text = res.raw_text
            st.session_state.pdf_cleaned_text = res.cleaned_text
            st.session_state.pdf_pages_read = res.pages_read
            st.session_state.pdf_removed_lines = res.removed_lines_count

            source_text = st.session_state.pdf_cleaned_text

            st.caption(
                f"Pages read: {st.session_state.pdf_pages_read} | "
                f"Removed header/footer lines: {st.session_state.pdf_removed_lines}"
            )

            with st.expander("Preview: RAW text (first 1500 chars)"):
                st.text(st.session_state.pdf_raw_text[:1500])

            with st.expander("Preview: CLEANED text (first 1500 chars)"):
                st.text(st.session_state.pdf_cleaned_text[:1500])

        else:
            source_text = ""

    else:
        pasted = st.text_area(
            "Paste text:",
            height=230,
            placeholder="Paste ESA mission text here…",
            key="text_input_area",
        )
        source_text = (pasted or "").strip()

    build_clicked = st.button("Build Graph(s)")

    if build_clicked:
        if not (source_text or "").strip():
            st.warning("Please provide some text (paste or PDF) first.")
        else:
            st.session_state.last_input_text = source_text

            if compare_mode:
                st.session_state.baseline_graph = build_baseline_graph(
                    source_text,
                    st.session_state.co_builder,
                    st.session_state.entity_typer,
                )

            if st.session_state.labelled_builder is not None:
                improved_input_text = source_text

                # --- LLM Routing Layer ---
                if use_llm_router and route_esa_text is not None and engineering_only_text is not None:
                    try:
                        routed = route_esa_text(source_text)
                        improved_input_text = engineering_only_text(routed, min_conf=0.55)

                        with st.expander("LLM Routing Debug"):
                            st.write(routed)
                            st.write("Engineering-only text:")
                            st.write(improved_input_text)

                        # If routing removes too much, fall back to original to avoid empty graphs
                        if len((improved_input_text or "").strip()) < 80:
                            st.warning("LLM routing kept very little engineering text; falling back to full text.")
                            improved_input_text = source_text

                    except Exception as e:
                        st.warning(f"LLM routing failed: {e}")
                        improved_input_text = source_text

                st.session_state.improved_graph = build_improved_graph(improved_input_text, st.session_state.labelled_builder)
            else:
                st.session_state.improved_graph = nx.DiGraph()

            if compare_mode:
                st.success(
                    f"Built baseline: {st.session_state.baseline_graph.number_of_nodes()} nodes, "
                    f"{st.session_state.baseline_graph.number_of_edges()} edges | "
                    f"improved: {st.session_state.improved_graph.number_of_nodes()} nodes, "
                    f"{st.session_state.improved_graph.number_of_edges()} edges."
                )
            else:
                st.success(
                    f"Built improved: {st.session_state.improved_graph.number_of_nodes()} nodes, "
                    f"{st.session_state.improved_graph.number_of_edges()} edges."
                )

    st.divider()
    st.subheader("Graph Visualization")

    if compare_mode:
        left, right = st.columns(2)

        with left:
            st.markdown("### Baseline (Web Graph)")
            g_base = st.session_state.baseline_graph
            if g_base.number_of_nodes() == 0:
                st.info("Baseline graph is empty. Build graphs first.")
            else:
                viz_base = filter_graph(
                    g_base,
                    label_key="entity_type",
                    show_types=show_types_web,
                    min_conf=min_conf,
                )
                render_pyvis_graph(
                    g_full=g_base,
                    g_viz=viz_base,
                    label_key="entity_type",
                    show_edge_labels=show_edge_labels,
                    html_path=os.path.join(BASE_DIR, "baseline_graph.html"),
                    cmap=color_map_web(),
                )

        with right:
            st.markdown("### Improved (Subsystem KG)")
            g_imp = st.session_state.improved_graph
            if g_imp.number_of_nodes() == 0:
                st.info("Improved graph is empty. Build graphs first.")
            else:
                viz_imp = filter_graph(
                    g_imp,
                    label_key="subsystem_label",
                    show_types=show_types_sub,
                    min_conf=min_conf,
                )
                render_pyvis_graph(
                    g_full=g_imp,
                    g_viz=viz_imp,
                    label_key="subsystem_label",
                    show_edge_labels=show_edge_labels,
                    html_path=os.path.join(BASE_DIR, "improved_graph.html"),
                    cmap=color_map_subsystem(),
                )
    else:
        g_imp = st.session_state.improved_graph
        if g_imp.number_of_nodes() == 0:
            st.info("Graph is empty. Paste text or upload a PDF and click Build.")
        else:
            viz_imp = filter_graph(
                g_imp,
                label_key="subsystem_label",
                show_types=show_types_sub,
                min_conf=min_conf,
            )
            render_pyvis_graph(
                g_full=g_imp,
                g_viz=viz_imp,
                label_key="subsystem_label",
                show_edge_labels=show_edge_labels,
                html_path=os.path.join(BASE_DIR, "graph_output.html"),
                cmap=color_map_subsystem(),
            )


# -----------------------------
# Analytics / exports
# -----------------------------
with col_side:
    st.subheader("Text Analytics")
    if (st.session_state.last_input_text or "").strip():
        ta = TextAnalytics(st.session_state.last_input_text)
        s = ta.summary()
        st.metric("Characters", s["characters"])
        st.metric("Sentences", s["sentences"])
        st.metric("Words", s["words"])
        st.metric("Tokens", s["tokens"])
        st.metric("Avg sentence length (words)", s["avg_sentence_length_words"])
    else:
        st.write("Paste text or upload a PDF and build graphs.")

    st.divider()
    st.subheader("Graph Analytics")

    def show_graph_metrics(title: str, g: nx.DiGraph, label_key: str):
        st.markdown(f"**{title}**")
        analytics = GraphAnalytics(g)
        st.metric("Nodes", analytics.node_count())
        st.metric("Edges", analytics.edge_count())
        st.metric("Density", round(analytics.density(), 4))
        cc = nx.number_weakly_connected_components(g) if g.number_of_nodes() else 0
        st.metric("Connected Components", cc)

        if compute_metrics is not None:
            m = compute_metrics(g, label_key=label_key)
            st.metric("UNKNOWN rate", round(m.unknown_rate, 3))
            st.metric("OTHER rate", round(m.other_rate, 3))
            st.metric("Avg node confidence", round(m.avg_node_confidence, 3))
            st.metric("Meaningful edge rate", round(m.meaningful_edge_rate, 3))

    if compare_mode:
        gb = st.session_state.baseline_graph
        gi = st.session_state.improved_graph

        if gb.number_of_nodes():
            show_graph_metrics("Baseline (Web Graph)", gb, label_key="entity_type")
        else:
            st.info("Baseline graph not built yet.")

        st.divider()

        if gi.number_of_nodes():
            show_graph_metrics("Improved (Subsystem KG)", gi, label_key="subsystem_label")
        else:
            st.info("Improved graph not built yet.")

        if compute_metrics is not None and gb.number_of_nodes() and gi.number_of_nodes():
            mb = compute_metrics(gb, label_key="entity_type")
            mi = compute_metrics(gi, label_key="subsystem_label")
            st.divider()
            st.subheader("Comparison Deltas")
            st.write("UNKNOWN rate drop:", round(mb.unknown_rate - mi.unknown_rate, 3))
            st.write("Meaningful edge rate increase:", round(mi.meaningful_edge_rate - mb.meaningful_edge_rate, 3))
    else:
        gi = st.session_state.improved_graph
        if gi.number_of_nodes():
            show_graph_metrics("Improved (Subsystem KG)", gi, label_key="subsystem_label")
        else:
            st.info("Graph not built yet.")

    st.divider()
    st.subheader("Exports")

    export_graph = st.session_state.improved_graph if st.session_state.improved_graph.number_of_nodes() else st.session_state.baseline_graph

    if export_graph.number_of_nodes() == 0:
        st.info("Build a graph to enable exports.")
    else:
        json_bytes = graph_to_json_bytes(export_graph)
        csv_bytes = graph_to_csv_bytes(export_graph)

        analytics = GraphAnalytics(export_graph)
        top_entities = analytics.top_entities(10) if hasattr(analytics, "top_entities") else []
        pdf_bytes = graph_to_pdf_bytes(export_graph, title="Knowledge Graph Report", top_entities=top_entities)

        st.download_button("Download Graph JSON", data=json_bytes, file_name="graph.json", mime="application/json")
        st.download_button("Download Edges CSV", data=csv_bytes, file_name="edges.csv", mime="text/csv")
        st.download_button("Download Report PDF", data=pdf_bytes, file_name="report.pdf", mime="application/pdf")

        if compute_metrics is not None and metrics_to_dict is not None:
            label_key = "subsystem_label" if "subsystem_label" in next(iter(export_graph.nodes(data=True)))[1] else "entity_type"
            m = compute_metrics(export_graph, label_key=label_key)
            metrics_json = json.dumps(metrics_to_dict(m), indent=2).encode("utf-8")
            st.download_button("Download Metrics JSON", data=metrics_json, file_name="metrics.json", mime="application/json")

        if compare_mode and st.session_state.baseline_graph.number_of_nodes() and st.session_state.improved_graph.number_of_nodes():
            st.divider()
            st.caption("Comparison exports (both graphs):")

            jb = graph_to_json_bytes(st.session_state.baseline_graph)
            cb = graph_to_csv_bytes(st.session_state.baseline_graph)
            st.download_button("Download Baseline JSON", data=jb, file_name="baseline_graph.json", mime="application/json")
            st.download_button("Download Baseline CSV", data=cb, file_name="baseline_edges.csv", mime="text/csv")

            ji = graph_to_json_bytes(st.session_state.improved_graph)
            ci = graph_to_csv_bytes(st.session_state.improved_graph)
            st.download_button("Download Improved JSON", data=ji, file_name="improved_graph.json", mime="application/json")
            st.download_button("Download Improved CSV", data=ci, file_name="improved_edges.csv", mime="text/csv")
