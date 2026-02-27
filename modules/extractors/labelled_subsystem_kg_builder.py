from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import re

import torch
import spacy
import networkx as nx
from transformers import AutoTokenizer, AutoModelForTokenClassification


# -----------------------------
# Data structure
# -----------------------------
@dataclass
class LabeledSpan:
    text: str
    label: str
    start: int
    end: int
    confidence: float = 0.65


# -----------------------------
# Builder
# -----------------------------
class LabelledSubsystemKGBuilder:
    """
    Labelled Subsystem KG (SpaceBERT / token-classifier based):

    - Runs a HuggingFace token classification model (SpaceBERT/SpaceRoBERTa fine-tuned).
    - Converts token predictions -> word-level labels using tokenizer.word_ids()
    - Aggregates contiguous labeled words into spans (BIO supported, but also works without BIO)
    - Nodes = spans (normalized)
      node attributes:
        - entity_type: subsystem label (e.g., Telecom., Propulsion, Thermal, etc.)
        - confidence: average span confidence
        - start/end: char offsets in original text
    - Edges = co-occurrence within same sentence (undirected)
      edge attributes:
        - predicate/label: "co-occurs"
        - weight: count of co-occurrence
        - confidence: increases slightly with repeats

    Notes:
    - For sentence boundaries, we use a full spaCy pipeline in build() (sentencizer/parser).
      For tokenization inside predict_spans(), we use the same spaCy doc tokens.
    """

    def __init__(
        self,
        hf_model_name_or_path: str = "icelab/spaceroberta",
        spacy_model: str = "en_core_web_sm",
        device: Optional[str] = None,
        max_length: int = 512,
        use_amp: bool = False,
    ):
        self.spacy_model_name = spacy_model
        self.max_length = int(max_length)
        self.use_amp = bool(use_amp)

        # HF tokenizer + model
        self.tokenizer = AutoTokenizer.from_pretrained(hf_model_name_or_path)
        self.model = AutoModelForTokenClassification.from_pretrained(hf_model_name_or_path)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

        # label mapping
        self.id2label: Dict[int, str] = dict(self.model.config.id2label)

        # General unicode subscripts (₂ -> 2, etc.)
        self._sub_map = str.maketrans(
            {
                "₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4",
                "₅": "5", "₆": "6", "₇": "7", "₈": "8", "₉": "9",
                "₊": "+", "₋": "-", "₌": "=", "₍": "(", "₎": ")",
            }
        )

    # -----------------------------
    # Public API
    # -----------------------------
    def build(self, text: str, min_conf: float = 0.0) -> nx.Graph:
        """
        Build an undirected labelled KG:
        - nodes: predicted spans
        - edges: co-occurrence within a sentence
        """
        text = text or ""
        text = text.strip()
        g = nx.Graph()
        if not text:
            return g

        # Use a full spaCy pipeline so doc.sents works reliably
        nlp = spacy.load(self.spacy_model_name)
        doc = nlp(text)

        spans = self.predict_spans(text, doc)
        spans = [s for s in spans if float(s.confidence) >= float(min_conf)]

        # Add nodes (dedupe by canonical name)
        node_best: Dict[str, LabeledSpan] = {}
        for sp in spans:
            node = self._canon_node(sp.text)
            if not node:
                continue

            # Keep the best span if duplicates occur
            if node not in node_best or sp.confidence > node_best[node].confidence:
                node_best[node] = sp

        for node, sp in node_best.items():
            g.add_node(
                node,
                entity_type=self._clean_label(sp.label),
                confidence=float(sp.confidence),
                start=int(sp.start),
                end=int(sp.end),
            )

        # Sentence-based co-occurrence edges
        sent_bounds = [(s.start_char, s.end_char) for s in doc.sents] if doc.has_annotation("SENT_START") else []
        if sent_bounds:
            self._add_sentence_edges(g, spans, sent_bounds)
        else:
            # fallback: connect all nodes lightly
            nodes = list(g.nodes())
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    g.add_edge(nodes[i], nodes[j], predicate="co-occurs", label="co-occurs", weight=1, confidence=0.25)

        return g

    def predict_spans(self, text: str, doc) -> List[LabeledSpan]:
        """
        Token-classification inference -> word labels -> contiguous labeled spans.

        doc must be a spaCy Doc whose tokens align with 'text' (same text).
        """
        # spaCy token list
        word_list = [t.text for t in doc]
        if not word_list:
            return []

        encoded = self.tokenizer(
            word_list,
            return_tensors="pt",
            padding=True,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        with torch.no_grad():
            if self.use_amp and self.device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    out = self.model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                out = self.model(input_ids=input_ids, attention_mask=attention_mask)

            logits = out.logits  # [1, T, C]
            probs = torch.softmax(logits, dim=-1)
            pred_ids = torch.argmax(probs, dim=-1)  # [1, T]
            pred_conf = torch.gather(probs, 2, pred_ids.unsqueeze(-1)).squeeze(-1)  # [1, T]

        pred_ids_list = pred_ids[0].detach().cpu().tolist()
        pred_conf_list = pred_conf[0].detach().cpu().tolist()

        # Map subword tokens -> original word index
        word_ids = encoded.word_ids(batch_index=0)

        # word-level labels/conf (first subtoken strategy)
        word_labels: List[str] = ["O"] * len(word_list)
        word_confs: List[float] = [0.0] * len(word_list)

        for tok_i, w_i in enumerate(word_ids):
            if w_i is None:
                continue

            lab = self.id2label.get(int(pred_ids_list[tok_i]), "O")
            conf = float(pred_conf_list[tok_i])

            # keep label of first subtoken for each word
            if word_confs[w_i] == 0.0:
                word_labels[w_i] = lab
                word_confs[w_i] = conf

        # Aggregate to spans
        spans: List[LabeledSpan] = []
        i = 0
        while i < len(doc):
            lab = word_labels[i]
            if not lab or lab == "O":
                i += 1
                continue

            base = self._label_base(lab)
            start_i = i
            confs = [word_confs[i]]

            i += 1
            while i < len(doc):
                lab2 = word_labels[i]
                if not lab2 or lab2 == "O":
                    break
                if self._label_base(lab2) != base:
                    break
                confs.append(word_confs[i])
                i += 1

            start_char = doc[start_i].idx
            end_char = doc[i - 1].idx + len(doc[i - 1].text)
            span_text = text[start_char:end_char].strip()

            if span_text:
                spans.append(
                    LabeledSpan(
                        text=span_text,
                        label=base,
                        start=int(start_char),
                        end=int(end_char),
                        confidence=float(sum(confs) / max(1, len(confs))),
                    )
                )

        return spans

    # -----------------------------
    # Internals
    # -----------------------------
    def _add_sentence_edges(self, g: nx.Graph, spans: List[LabeledSpan], sent_bounds: List[Tuple[int, int]]) -> None:
        # Pre-index spans by sentence membership (simple scan; fast enough for typical text)
        for (a0, a1) in sent_bounds:
            sent_nodes: List[str] = []
            for sp in spans:
                if sp.start >= a0 and sp.end <= a1:
                    node = self._canon_node(sp.text)
                    if node and node in g.nodes:
                        sent_nodes.append(node)

            sent_nodes = list(dict.fromkeys(sent_nodes))
            if len(sent_nodes) < 2:
                continue

            for i in range(len(sent_nodes)):
                for j in range(i + 1, len(sent_nodes)):
                    u, v = sent_nodes[i], sent_nodes[j]
                    if u == v:
                        continue
                    if g.has_edge(u, v):
                        g[u][v]["weight"] = int(g[u][v].get("weight", 1)) + 1
                        g[u][v]["confidence"] = float(min(1.0, float(g[u][v].get("confidence", 0.35)) + 0.05))
                    else:
                        g.add_edge(u, v, predicate="co-occurs", label="co-occurs", weight=1, confidence=0.35)

    def _label_base(self, label: str) -> str:
        """
        Supports BIO (B-XXX / I-XXX) and also plain labels.
        """
        x = (label or "").strip()
        if not x:
            return "UNKNOWN"
        if x.startswith("B-") or x.startswith("I-"):
            return x.split("-", 1)[1].strip() if "-" in x else x
        return x

    def _clean_label(self, label: str) -> str:
        x = (label or "").strip()
        if not x or x == "O":
            return "UNKNOWN"
        return x

    def _canon_node(self, s: str) -> str:
        """
        Stronger normalization to reduce duplicate nodes:
        - unicode subscripts to normal digits
        - normalize whitespace
        - strip punctuation edges
        - collapse quotes/brackets
        """
        x = (s or "").strip()
        if not x:
            return ""

        x = x.translate(self._sub_map)
        x = re.sub(r"\s+", " ", x).strip()
        x = x.strip(" ,.;:()[]{}\"'")

        # If it's a short chemical-like token (Co2 -> CO2)
        if re.fullmatch(r"[A-Za-z]{1,3}\d{0,3}", x):
            x = x.upper()

        return x
