import io
import csv
import json
from typing import Tuple, List
import networkx as nx

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


def graph_to_json_bytes(g: nx.DiGraph) -> bytes:
    data = nx.node_link_data(g)
    return json.dumps(data, indent=2).encode("utf-8")


def graph_to_csv_bytes(g: nx.DiGraph) -> bytes:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["subject", "predicate", "object", "confidence", "inferred"])

    for u, v, data in g.edges(data=True):
        pred = data.get("label") or data.get("predicate") or "related"
        conf = data.get("confidence", data.get("weight", ""))
        inferred = data.get("inferred", False)
        writer.writerow([u, pred, v, conf, inferred])

    return buf.getvalue().encode("utf-8")


def graph_to_pdf_bytes(
    g: nx.DiGraph,
    title: str = "Knowledge Graph Report",
    top_entities: List[Tuple[str, float]] = None
) -> bytes:
    if top_entities is None:
        top_entities = []

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    width, height = letter

    y = height - 50
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, title)

    y -= 30
    c.setFont("Helvetica", 11)
    c.drawString(50, y, f"Nodes: {g.number_of_nodes()}   Edges: {g.number_of_edges()}")

    y -= 25
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Top Entities")
    y -= 18
    c.setFont("Helvetica", 10)
    if top_entities:
        for name, score in top_entities[:10]:
            c.drawString(60, y, f"- {name}  (score: {round(score, 3)})")
            y -= 14
            if y < 80:
                c.showPage()
                y = height - 50
                c.setFont("Helvetica", 10)
    else:
        c.drawString(60, y, "(none)")
        y -= 14

    y -= 10
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Edges (first 40)")
    y -= 18
    c.setFont("Helvetica", 9)

    count = 0
    for u, v, data in g.edges(data=True):
        pred = data.get("label") or data.get("predicate") or "related"
        conf = data.get("confidence", data.get("weight", ""))
        inferred = " (inferred)" if data.get("inferred") else ""
        line = f"{u} --[{pred} | {conf}]--> {v}{inferred}"
        c.drawString(55, y, line[:110])
        y -= 12
        count += 1
        if count >= 40:
            break
        if y < 60:
            c.showPage()
            y = height - 50
            c.setFont("Helvetica", 9)

    c.showPage()
    c.save()
    return buf.getvalue()
