class RelationMapper:

    def map_relation(self, text_relation: str):
        if not text_relation:
            return None

        r = text_relation.lower()

        if "found" in r:
            return "founded"

        if "work" in r or "employ" in r:
            return "works_at"

        if "ceo" in r:
            return "ceo_of"

        if "born" in r:
            return "born_in"

        if "headquarter" in r or "located" in r:
            return "located_in"

        if "produce" in r or "develop" in r or "make" in r:
            return "produces"

        if "acquire" in r or "buy" in r:
            return "acquired"

        return None
