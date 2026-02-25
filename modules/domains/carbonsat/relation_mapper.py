class CarbonSatRelationMapper:
    def map_relation(self, text_relation: str):
        if not text_relation:
            return None

        r = text_relation.lower().strip()

        # normalize common variants
        r = r.replace("-", " ")

        # key domain relations
        if "measure" in r or "monitor" in r:
            return "measures"
            
        if "use" in r or "include" in r or "carry" in r:
            return "uses"
        
        if "detect" in r or "retrieve" in r or "observe" in r:
            return "detects"

        if "produce" in r or "generate" in r or "deliver" in r or "provide" in r:
            return "produces"

        if "operate" in r or "fly" in r:
            return "operates_in"

        if "orbit" in r:
            return "operates_in"

        if "develop" in r or "propose" in r or "lead" in r:
            return "developed_by"

        if "design" in r or "build" in r:
            return "built_by"

        if "part of" in r or "programme" in r or "program" in r:
            return "part_of"

        # fallback: nothing mapped
        return None
