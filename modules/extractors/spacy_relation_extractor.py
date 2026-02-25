from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import re
import spacy

from modules.domains.carbonsat.lexicon import CARBONSAT_LEXICON
from modules.domains.carbonsat.relation_mapper import CarbonSatRelationMapper
from modules.domains.carbonsat.ontology import CarbonSatOntology


@dataclass
class Triple:
    subject: str
    predicate: str
    object: str
    confidence: float = 0.65


class SpacyRelationExtractor:
    """
    Offline relation extractor using spaCy dependency parses.

    CarbonSat domain mode:
    - Map raw predicate text into CarbonSat ontology predicates using CarbonSatRelationMapper
    - Type subject/object using CARBONSAT_LEXICON
    - Validate triples with CarbonSatOntology
    - Passive voice normalized to active where possible:
        "CO2 is measured by CarbonSat" -> (CarbonSat, measures, CO2)
    """

    def __init__(self, model: str = "en_core_web_sm", domain_mode: str = "carbonsat"):
        self.nlp = spacy.load(model)
        self.domain_mode = (domain_mode or "").lower().strip()
        self.mapper = CarbonSatRelationMapper() if self.domain_mode == "carbonsat" else None

    def extract(self, text: str) -> List[Dict[str, Any]]:
        doc = self.nlp(text)
        triples: List[Triple] = []

        for sent in doc.sents:
            triples.extend(self._extract_from_sentence(sent))

        # Global de-dupe
        uniq = {}
        for t in triples:
            key = (t.subject.lower(), t.predicate.lower(), t.object.lower())
            if key not in uniq:
                uniq[key] = t
        triples = list(uniq.values())

        return [
            {"subject": t.subject, "predicate": t.predicate, "object": t.object, "confidence": t.confidence}
            for t in triples
        ]

    def _extract_from_sentence(self, sent) -> List[Triple]:
        raw: List[Triple] = []

        # Case A: Active voice SVO
        for token in sent:
            if token.pos_ == "VERB":
                subj = self._get_subject(token)
                objs = self._get_objects(token)
                if subj and objs:
                    pred = self._predicate_phrase(token)
                    for obj in objs:
                        raw.append(Triple(subj, pred, obj, confidence=0.70))

        # Case B: Passive voice: "CO2 is measured by CarbonSat."
        for token in sent:
            if token.pos_ == "VERB":
                nsubjpass = self._get_passive_subject(token)
                agents = self._get_agent_objects(token)
                if nsubjpass and agents:
                    pred = self._predicate_phrase(token)
                    if "by" not in pred.lower():
                        pred = f"{pred} by"
                    for agent in agents:
                        raw.append(Triple(nsubjpass, pred, agent, confidence=0.75))

        # Case C: Copular / attribute
        raw.extend(self._extract_copular(sent))

        if self.domain_mode == "carbonsat":
            return self._carbonsat_filter_map(raw)

        return self._dedupe(raw)

    def _carbonsat_filter_map(self, triples: List[Triple]) -> List[Triple]:
        out: List[Triple] = []

        for t in triples:
            subj = self._norm_entity(t.subject)
            obj = self._norm_entity(t.object)
            pred_raw = (t.predicate or "").strip()

            if not subj or not obj or not pred_raw:
                continue

            # Passive normalization: "measured by"
            subj_active = subj
            obj_active = obj
            pred_for_map = pred_raw

            if pred_raw.lower().endswith(" by"):
                subj_active = obj
                obj_active = subj
                pred_for_map = pred_raw[:-3].strip()

            mapped = self.mapper.map_relation(pred_for_map) if self.mapper else None
            if not mapped:
                continue

            subj_type = self._type_from_lexicon(subj_active)
            obj_type = self._type_from_lexicon(obj_active)

            if CarbonSatOntology.is_valid(subj_type, mapped, obj_type):
                out.append(
                    Triple(
                        subj_active,
                        mapped,
                        obj_active,
                        confidence=max(t.confidence, 0.80),
                    )
                )

        return self._dedupe(out)

    def _type_from_lexicon(self, entity: str) -> str:
        key = (entity or "").strip().lower()
        return CARBONSAT_LEXICON.get(key, "UNKNOWN")

    def _dedupe(self, triples: List[Triple]) -> List[Triple]:
        uniq = {}
        for t in triples:
            key = (t.subject.lower(), t.predicate.lower(), t.object.lower())
            if key not in uniq:
                uniq[key] = t
        return list(uniq.values())

    def _get_subject(self, verb) -> Optional[str]:
        for child in verb.children:
            if child.dep_ == "nsubj":
                return self._span_text(child)
        return None

    def _get_passive_subject(self, verb) -> Optional[str]:
        for child in verb.children:
            if child.dep_ == "nsubjpass":
                return self._span_text(child)
        return None

    def _get_objects(self, verb) -> List[str]:
        """
        Returns multiple objects including conjunctions.
        Example: "measures CO2 and CH4" -> ["CO2", "CH4"]
        """
        objs: List[str] = []

        # Direct object
        for child in verb.children:
            if child.dep_ in ("dobj", "obj"):
                objs.extend(self._collect_conjuncts(child))
                break

        # Prepositional object: "operates in orbit" -> pobj of "in"
        if not objs:
            for prep in verb.children:
                if prep.dep_ == "prep":
                    for pobj in prep.children:
                        if pobj.dep_ == "pobj":
                            objs.extend(self._collect_conjuncts(pobj))
                            return self._dedupe_strings(objs)

        return self._dedupe_strings(objs)

    def _get_agent_objects(self, verb) -> List[str]:
        agents: List[str] = []
        for child in verb.children:
            if child.dep_ in ("agent", "prep") and child.text.lower() == "by":
                for pobj in child.children:
                    if pobj.dep_ == "pobj":
                        agents.extend(self._collect_conjuncts(pobj))
        return self._dedupe_strings(agents)

    def _collect_conjuncts(self, token) -> List[str]:
        """
        token + any conjuncts: "CO2 and CH4" -> ["CO2", "CH4"]
        """
        items = [self._span_text(token)]
        for c in token.conjuncts:
            items.append(self._span_text(c))
        return items

    def _dedupe_strings(self, items: List[str]) -> List[str]:
        seen = set()
        out = []
        for x in items:
            k = (x or "").strip().lower()
            if not k or k in seen:
                continue
            seen.add(k)
            out.append(x.strip())
        return out

    def _extract_copular(self, sent) -> List[Triple]:
        triples: List[Triple] = []

        root = None
        for tok in sent:
            if tok.dep_ == "ROOT":
                root = tok
                break
        if root is None:
            return triples

        subj = None
        for child in root.children:
            if child.dep_ in ("nsubj", "nsubjpass"):
                subj = self._span_text(child)
                break
        if not subj:
            return triples

        for child in root.children:
            if child.dep_ == "attr":
                triples.append(Triple(subj, "is", self._span_text(child), confidence=0.60))

        for child in root.children:
            if child.dep_ == "acomp":
                triples.append(Triple(subj, "is", self._span_text(child), confidence=0.55))

        for child in root.children:
            if child.dep_ in ("acomp", "xcomp", "advcl") and child.pos_ in ("VERB", "ADJ"):
                pred_head = child.lemma_ if child.lemma_ else child.text
                for prep in child.children:
                    if prep.dep_ == "prep":
                        for pobj in prep.children:
                            if pobj.dep_ == "pobj":
                                pred = f"{pred_head} {prep.text}"
                                for obj in self._collect_conjuncts(pobj):
                                    triples.append(Triple(subj, pred, obj, confidence=0.70))

        return triples

    def _predicate_phrase(self, verb) -> str:
        base = verb.lemma_ if verb.lemma_ else verb.text
        parts = [base]

        for child in verb.children:
            if child.dep_ == "prt":
                parts.append(child.text)

        for child in verb.children:
            if child.dep_ == "prep" and any(gc.dep_ == "pobj" for gc in child.children):
                parts.append(child.text)
                break

        return " ".join(parts)

    def _span_text(self, token) -> str:
        left = token.left_edge.i
        right = token.right_edge.i
        return token.doc[left:right + 1].text.strip()

    def _norm_entity(self, s: str) -> str:
        """
        Normalize entity strings slightly so lexicon matching works better.
        You can expand this if needed.
        """
        if not s:
            return ""
        x = s.strip()
        x = re.sub(r"\s+", " ", x)
        x = x.strip(" ,.;:()[]{}")
        return x
