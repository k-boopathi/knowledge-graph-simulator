# modules/extractors/spacy_extractor.py
import spacy
from typing import List, Dict, Optional, Tuple


class SpacyOpenIEExtractor:
    """
    SpacyOpenIEExtractor v4
    Goal: produce a TRUE KG-like web (offline) by:
      1) Tightened Entity Gate: only real entities become nodes
         - NER: PERSON/ORG/GPE/LOC/PRODUCT
         - OR Proper nouns (PROPN) that look like named entities
      2) Coreference-lite:
         - "the company/it" -> last ORG
         - "he/she/they" -> last PERSON
      3) Org anchoring:
         - If sentence produces PERSON↔PERSON relation with no ORG, add a low-confidence anchor edge:
             PERSON --associated with--> last_org
      4) Role handling:
         - Extract "CEO of Apple" as PERSON --ceo of--> ORG (no CEO node)
      5) Predicate normalization:
         - map messy verbs to canonical labels
    """

    VALID_NER_LABELS = {"PERSON", "ORG", "GPE", "LOC", "PRODUCT"}

    ORG_COREF = {"it", "this", "that", "the company", "the firm", "the organization", "the business"}
    PERSON_COREF = {"he", "she", "him", "her", "they", "them"}

    CEO_TOKENS = {"ceo", "chief executive officer"}

    # If these appear as "entities", they are almost always junk nodes
    BAD_NODES = {
        "portfolio", "leadership", "products", "product", "industry", "sector",
        "consumer electronics", "electronics", "technology", "business", "company"
    }

    # Canonical predicate mapping (feel free to extend)
    PRED_MAP = {
        "expand": "expands",
        "expanded": "expands",
        "include": "has product",
        "includes": "has product",
        "design": "designs",
        "designs": "designs",
        "headquarter": "headquartered in",
        "headquartered": "headquartered in",
        "base": "based in",
        "based": "based in",
        "acquire": "acquired",
        "acquired": "acquired",
        "found": "founded",
        "founded": "founded",
        "succeed": "succeeded",
        "succeeded": "succeeded",
        "become": "became",
        "became": "became",
    }

    def __init__(self, model: str = "en_core_web_sm"):
        self.nlp = spacy.load(model)

    # -----------------------------
    # Public
    # -----------------------------
    def extract(self, text: str) -> List[Dict]:
        doc = self.nlp(text)
        triples: List[Dict] = []

        last_org: Optional[str] = None
        last_person: Optional[str] = None

        for sent in doc.sents:
            sent_triples, last_org, last_person = self._extract_sentence(sent, last_org, last_person)
            triples.extend(sent_triples)

        return triples

    # -----------------------------
    # Core logic
    # -----------------------------
    def _extract_sentence(
        self,
        sent,
        last_org: Optional[str],
        last_person: Optional[str]
    ) -> Tuple[List[Dict], Optional[str], Optional[str]]:

        triples: List[Dict] = []
        sdoc = sent.as_doc()

        # Collect entities in sentence
        ents = list(sdoc.ents)

        # Update last_org / last_person memory
        for e in ents:
            if e.label_ == "ORG":
                last_org = e.text
            elif e.label_ == "PERSON":
                last_person = e.text

        # -----------------------------
        # Helper: resolve coref
        # -----------------------------
        def resolve_coref(raw: str) -> str:
            low = raw.strip().lower()
            if low in self.ORG_COREF and last_org:
                return last_org
            if low in self.PERSON_COREF and last_person:
                return last_person
            return raw

        # -----------------------------
        # Helper: find entity span for token
        # -----------------------------
        def ent_for_token(tok) -> Optional[spacy.tokens.Span]:
            for e in ents:
                if e.start <= tok.i < e.end:
                    return e
            return None

        # -----------------------------
        # Helper: clean text node
        # -----------------------------
        def clean_node(name: str) -> str:
            return " ".join(name.strip().split()).strip(" .,:;()[]{}\"'")

        # -----------------------------
        # Entity Gate: Is this a "real entity"?
        # -----------------------------
        def is_real_entity(tok, ent: Optional[spacy.tokens.Span]) -> bool:
            if not tok:
                return False

            # Accept NER entities from allowed labels
            if ent and ent.label_ in self.VALID_NER_LABELS:
                val = clean_node(ent.text)
                if val.lower() in self.BAD_NODES:
                    return False
                return True

            # Accept Proper Noun that looks like a name
            # (Capitalized, and not a bad generic word)
            if tok.pos_ == "PROPN" and tok.text[:1].isupper():
                val = clean_node(tok.text)
                if val.lower() in self.BAD_NODES:
                    return False
                return True

            return False

        def node_text(tok, ent: Optional[spacy.tokens.Span]) -> str:
            if ent and ent.label_ in self.VALID_NER_LABELS:
                return clean_node(ent.text)
            return clean_node(tok.text)

        def pred_norm(p: str) -> str:
            p = p.strip()
            low = p.lower()
            return self.PRED_MAP.get(low, p)

        def conf_score(root_verb, strong: bool = False) -> float:
            base = 0.72
            if strong:
                base += 0.10
            if getattr(root_verb, "dep_", "") == "ROOT":
                base += 0.08
            return round(min(base, 0.95), 2)

        # Track whether sentence has ORG entity
        sentence_has_org = any(e.label_ == "ORG" for e in ents)

        # -----------------------------
        # CEO-of pattern: "X became CEO of ORG"
        # -----------------------------
        for v in sdoc:
            if v.pos_ != "VERB":
                continue

            # find subject token
            subj_tok = None
            for c in v.children:
                if c.dep_ in ("nsubj", "nsubjpass"):
                    subj_tok = c
                    break

            if subj_tok is None:
                continue

            subj_ent = ent_for_token(subj_tok)
            subj_raw = resolve_coref(subj_ent.text if subj_ent else subj_tok.text)
            subj_clean = clean_node(subj_raw)

            # CEO keyword present as attr/dobj/oprd?
            role_hit = None
            for c in v.children:
                if c.dep_ in ("attr", "dobj", "oprd"):
                    if clean_node(c.text).lower() in self.CEO_TOKENS:
                        role_hit = c
                        break

            if role_hit is not None:
                # look for "of ORG"
                for prep in v.children:
                    if prep.dep_ == "prep" and prep.text.lower() == "of":
                        for pobj in prep.children:
                            if pobj.dep_ == "pobj":
                                obj_ent = ent_for_token(pobj)
                                obj_raw = resolve_coref(obj_ent.text if obj_ent else pobj.text)
                                obj_clean = clean_node(obj_raw)

                                if is_real_entity(subj_tok, subj_ent) and is_real_entity(pobj, obj_ent):
                                    triples.append({
                                        "subject": subj_clean,
                                        "predicate": "ceo of",
                                        "object": obj_clean,
                                        "confidence": 0.84
                                    })

        # -----------------------------
        # Generic verb relations: subject-verb-object + prep objects
        # -----------------------------
        for v in sdoc:
            if v.pos_ != "VERB":
                continue

            subj_tok = None
            for c in v.children:
                if c.dep_ in ("nsubj", "nsubjpass"):
                    subj_tok = c
                    break
            if subj_tok is None:
                continue

            subj_ent = ent_for_token(subj_tok)
            subj_raw = resolve_coref(subj_ent.text if subj_ent else subj_tok.text)
            subj_clean = clean_node(subj_raw)

            pred = pred_norm(v.lemma_)

            # direct object
            obj_tok = None
            for c in v.children:
                if c.dep_ in ("dobj", "obj", "attr", "oprd"):
                    obj_tok = c
                    break

            if obj_tok is not None:
                obj_ent = ent_for_token(obj_tok)
                obj_raw = resolve_coref(obj_ent.text if obj_ent else obj_tok.text)
                obj_clean = clean_node(obj_raw)

                # Entity gate: both ends must be real entities
                if is_real_entity(subj_tok, subj_ent) and is_real_entity(obj_tok, obj_ent):
                    triples.append({
                        "subject": subj_clean,
                        "predicate": pred,
                        "object": obj_clean,
                        "confidence": conf_score(v, strong=(v.dep_ == "ROOT"))
                    })

                    # Org anchoring: if it's PERSON↔PERSON and no ORG mentioned, attach to last_org
                    if (
                        last_org
                        and not sentence_has_org
                        and subj_ent and subj_ent.label_ == "PERSON"
                        and obj_ent and obj_ent.label_ == "PERSON"
                    ):
                        triples.append({
                            "subject": subj_clean,
                            "predicate": "associated with",
                            "object": clean_node(last_org),
                            "confidence": 0.40
                        })
                        triples.append({
                            "subject": obj_clean,
                            "predicate": "associated with",
                            "object": clean_node(last_org),
                            "confidence": 0.40
                        })

            # prepositional objects: "headquartered in Cupertino"
            for prep in v.children:
                if prep.dep_ != "prep":
                    continue

                prep_word = prep.text.lower()
                # prefer prepositions that often represent facts
                if prep_word not in {"in", "at", "on", "of", "from", "to", "into", "within", "near", "by", "for"}:
                    continue

                for pobj in prep.children:
                    if pobj.dep_ != "pobj":
                        continue

                    obj_ent = ent_for_token(pobj)
                    obj_raw = resolve_coref(obj_ent.text if obj_ent else pobj.text)
                    obj_clean = clean_node(obj_raw)

                    # build predicate "verb + prep"
                    pred2 = pred_norm(f"{v.lemma_} {prep_word}")

                    if is_real_entity(subj_tok, subj_ent) and is_real_entity(pobj, obj_ent):
                        triples.append({
                            "subject": subj_clean,
                            "predicate": pred2,
                            "object": obj_clean,
                            "confidence": conf_score(v, strong=True)
                        })

        # Deduplicate exact triples
        seen = set()
        uniq = []
        for t in triples:
            key = (t["subject"], t["predicate"], t["object"])
            if key in seen:
                continue
            seen.add(key)
            uniq.append(t)

        return uniq, last_org, last_person
