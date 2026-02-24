from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any
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

    Updated for CarbonSat domain mode:
    - Map raw predicate text into CarbonSat ontology predicates using CarbonSatRelationMapper
    - Type subject/object using CARBONSAT_LEXICON
    - Validate triples with CarbonSatOntology
    - Passive voice is normalized to active when possible:
        "CO2 is measured by CarbonSat" -> (CarbonSat, measures, CO2)
    """

    def __init__(self, model: str = "en_core_web_sm", domain_mode: str = "carbonsat"):
        self.nlp = spacy.load(model)
        self.domain_mode = domain_mode.lower().strip()
        self.mapper = CarbonSatRelationMapper() if self.domain_mode == "carbonsat" else None

    def extract(self, text: str) -> List[Dict[str, Any]]:
        doc = self.nlp(text)
        triples: List[Triple] = []

        for sent in doc.sents:
            triples.extend(self._extract_from_sentence(sent))

        # De-duplicate globally
        uniq = {}
        for t in triples:
            key = (t.subject.lower(), t.predicate.lower(), t.object.lower())
            if key not in uniq:
                uniq[key] = t
        triples = list(uniq.values())

        # Return as list-of-dicts compatible with your GraphManager
        return [
            {
                "subject": t.subject,
                "predicate": t.predicate,
                "object": t.object,
                "confidence": t.confidence,
            }
            for t in triples
        ]

    def _extract_from_sentence(self, sent) -> List[Triple]:
        raw: List[Triple] = []

        # Case A: Active voice SVO
        # "CarbonSat measures CO2."
        for token in sent:
            if token.pos_ == "VERB":
                subj = self._get_subject(token)
                obj = self._get_object(token)
                if subj and obj:
                    pred = self._predicate_phrase(token)
                    raw.append(Triple(subj, pred, obj, confidence=0.70))

        # Case B: Passive voice
        # "CO2 is measured by CarbonSat."
        # nsubjpass = CO2, agent/by -> pobj = CarbonSat
        for token in sent:
            if token.pos_ == "VERB":
                nsubjpass = self._get_passive_subject(token)
                agent_obj = self._get_agent_object(token)
                if nsubjpass and agent_obj:
                    pred = self._predicate_phrase(token)
                    if "by" not in pred.lower():
                        pred = f"{pred} by"
                    raw.append(Triple(nsubjpass, pred, agent_obj, confidence=0.75))

        # Case C: Copular / attribute
        raw.extend(self._extract_copular(sent))

        # Domain filtering / mapping
        if self.domain_mode == "carbonsat":
            return self._carbonsat_filter_map(raw)

        # Otherwise return raw triples
        return self._dedupe(raw)

    def _carbonsat_filter_map(self, triples: List[Triple]) -> List[Triple]:
        """
        - Map predicates to CarbonSat predicates
        - Type subject/object using CARBONSAT_LEXICON
        - Validate with CarbonSatOntology
        - Normalize passive to active where possible (X is measured by Y -> Y measures X)
        """
        out: List[Triple] = []

        for t in triples:
            subj = (t.subject or "").strip()
            obj = (t.object or "").strip()
            pred_raw = (t.predicate or "").strip()

            if not subj or not obj or not pred_raw:
                continue

            # Passive normalization heuristic:
            # If predicate contains " by" and subject is the passive subject, object is agent.
            # Convert to active: agent -> predicate(without by) -> passive_subject
            subj_active = subj
            obj_active = obj
            pred_for_map = pred_raw

            if pred_raw.lower().endswith(" by"):
                # t.subject = passive subject, t.object = agent
                subj_active = obj
                obj_active = subj
                pred_for_map = pred_raw[:-3].strip()  # remove trailing "by"

            mapped = self.mapper.map_relation(pred_for_map)
            if not mapped:
                continue

            subj_type = self._type_from_lexicon(subj_active)
            obj_type = self._type_from_lexicon(obj_active)

            if CarbonSatOntology.is_valid(subj_type, mapped, obj_type):
                out.append(Triple(subj_active, mapped, obj_active, confidence=max(t.confidence, 0.80)))

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

    def _get_subject(self, verb) -> str | None:
        for child in verb.children:
            if child.dep_ in ("nsubj",):
                return self._span_text(child)
        return None

    def _get_object(self, verb) -> str | None:
        # direct object
        for child in verb.children:
            if child.dep_ in ("dobj", "obj"):
                return self._span_text(child)

        # prepositional object: "operates in orbit" -> pobj of "in"
        for prep in verb.children:
            if prep.dep_ == "prep":
                for pobj in prep.children:
                    if pobj.dep_ == "pobj":
                        return self._span_text(pobj)

        return None

    def _get_passive_subject(self, verb) -> str | None:
        for child in verb.children:
            if child.dep_ == "nsubjpass":
                return self._span_text(child)
        return None

    def _get_agent_object(self, verb) -> str | None:
        for child in verb.children:
            if child.dep_ in ("agent", "prep") and child.text.lower() == "by":
                for pobj in child.children:
                    if pobj.dep_ == "pobj":
                        return self._span_text(pobj)
        return None

    def _extract_copular(self, sent) -> List[Triple]:
        triples: List[Triple] = []

        root = None
        for tok in sent:
            if tok.dep_ == "ROOT":
                root = tok
                break
        if root is None:
            return triples

        # Find subject
        subj = None
        for child in root.children:
            if child.dep_ in ("nsubj", "nsubjpass"):
                subj = self._span_text(child)
                break
        if not subj:
            return triples

        # Attribute: "CarbonSat is a mission" -> attr
        for child in root.children:
            if child.dep_ == "attr":
                triples.append(Triple(subj, "is", self._span_text(child), confidence=0.60))

        # Adjectival complement: "Instrument is sensitive" -> acomp
        for child in root.children:
            if child.dep_ == "acomp":
                triples.append(Triple(subj, "is", self._span_text(child), confidence=0.55))

        # Prepositional complements: "headquartered in X" / "located in X"
        for child in root.children:
            if child.dep_ in ("acomp", "xcomp", "advcl") and child.pos_ in ("VERB", "ADJ"):
                pred_head = child.lemma_ if child.lemma_ else child.text
                for prep in child.children:
                    if prep.dep_ == "prep":
                        for pobj in prep.children:
                            if pobj.dep_ == "pobj":
                                pred = f"{pred_head} {prep.text}"
                                triples.append(Triple(subj, pred, self._span_text(pobj), confidence=0.70))

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
