from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import spacy


@dataclass
class Triple:
    subject: str
    predicate: str
    object: str
    confidence: float = 0.65


class SpacyRelationExtractor:
    """
    Offline relation extractor using spaCy dependency parses.

    Goal: extract (subject, predicate, object) when a clear grammatical link exists.
    Fallback strategies should be handled by your Web Graph builder, not here.
    """

    def __init__(self, model: str = "en_core_web_sm"):
        self.nlp = spacy.load(model)

    def extract(self, text: str) -> List[Dict[str, Any]]:
        doc = self.nlp(text)
        triples: List[Triple] = []

        for sent in doc.sents:
            triples.extend(self._extract_from_sentence(sent))

        # Return as list-of-dicts compatible with your GraphManager
        return [
            {"subject": t.subject, "predicate": t.predicate, "object": t.object, "confidence": t.confidence}
            for t in triples
        ]

    def _extract_from_sentence(self, sent) -> List[Triple]:
        triples: List[Triple] = []

        # Case A: Active voice SVO
        # "Steve Jobs founded Apple."
        for token in sent:
            if token.pos_ == "VERB":
                subj = self._get_subject(token)
                obj = self._get_object(token)
                if subj and obj:
                    pred = self._predicate_phrase(token)
                    triples.append(Triple(subj, pred, obj, confidence=0.70))

        # Case B: Passive voice
        # "Apple was founded by Steve Jobs."
        # nsubjpass = Apple, agent/by -> pobj = Steve Jobs
        for token in sent:
            if token.pos_ == "VERB":
                nsubjpass = self._get_passive_subject(token)
                agent_obj = self._get_agent_object(token)
                if nsubjpass and agent_obj:
                    pred = self._predicate_phrase(token)
                    # Make predicate explicitly passive-ish if "by" agent exists
                    if "by" not in pred.lower():
                        pred = f"{pred} by"
                    triples.append(Triple(nsubjpass, pred, agent_obj, confidence=0.75))

        # Case C: Copular / attribute
        # "Apple is a company." or "Apple is headquartered in Cupertino."
        triples.extend(self._extract_copular(sent))

        # De-duplicate
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

        # prepositional object: "works at Apple" -> pobj of "at"
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
        # agent phrase is often a prep "by" under the verb (or "agent" dep in some parses)
        for child in verb.children:
            if child.dep_ in ("agent", "prep") and child.text.lower() == "by":
                for pobj in child.children:
                    if pobj.dep_ == "pobj":
                        return self._span_text(pobj)
        return None

    def _extract_copular(self, sent) -> List[Triple]:
        triples: List[Triple] = []
        # Copular verb "is/was/are" with attribute or complement
        # We look for tokens with dep_="ROOT" and pos_="AUX"/"VERB" like "is"
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

        # Attribute: "Apple is a company" -> attr
        for child in root.children:
            if child.dep_ == "attr":
                triples.append(Triple(subj, "is", self._span_text(child), confidence=0.60))

        # Adjectival complement: "Apple is successful" -> acomp
        for child in root.children:
            if child.dep_ == "acomp":
                triples.append(Triple(subj, "is", self._span_text(child), confidence=0.55))

        # Prepositional complements: "headquartered in Cupertino"
        # Often "headquartered" is an acomp/advcl with prep "in"
        for child in root.children:
            if child.dep_ in ("acomp", "xcomp", "advcl") and child.pos_ in ("VERB", "ADJ"):
                # build predicate like "headquartered in"
                pred_head = child.lemma_ if child.lemma_ else child.text
                for prep in child.children:
                    if prep.dep_ == "prep":
                        for pobj in prep.children:
                            if pobj.dep_ == "pobj":
                                pred = f"{pred_head} {prep.text}"
                                triples.append(Triple(subj, pred, self._span_text(pobj), confidence=0.70))

        return triples

    def _predicate_phrase(self, verb) -> str:
        # build predicate from lemma + particles/prepositions
        base = verb.lemma_ if verb.lemma_ else verb.text

        parts = [base]

        # phrasal verb particle: "set up", "carry out"
        for child in verb.children:
            if child.dep_ == "prt":
                parts.append(child.text)

        # include immediate preposition if it directly links to an object (optional)
        # example: "headquartered in" usually not here (handled by copular), but "works at"
        for child in verb.children:
            if child.dep_ == "prep" and any(gc.dep_ == "pobj" for gc in child.children):
                parts.append(child.text)
                break

        return " ".join(parts)

    def _span_text(self, token) -> str:
        # expand to include compound names: "Steve Jobs", "New York"
        left = token.left_edge.i
        right = token.right_edge.i
        return token.doc[left:right + 1].text.strip()

