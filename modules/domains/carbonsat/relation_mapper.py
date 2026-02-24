class CarbonSatRelationMapper:
    def map_relation(self, text_relation: str):
        if not text_relation:
            return None

        r = text_relation.lower().strip()

        if any(w in r for w in ["measure", "measures", "measuring", "observe", "observes", "detect", "detects"]):
            return "measures"

        if any(w in r for w in ["produce", "produces", "generate", "generates", "provide", "provides"]):
            return "produces"

        if any(w in r for w in ["carry", "carries", "equipped", "instrument", "payload"]):
            return "has_instrument"

        if "orbit" in r:
            return "has_orbit"

        if "swath" in r:
            return "has_swath"

        if "resolution" in r:
            return "has_resolution"

        if any(w in r for w in ["cover", "covers"]):
            return "covers"

        if any(w in r for w in ["operated", "operate", "managed"]):
            return "operated_by"

        return None
