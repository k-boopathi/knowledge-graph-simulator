class CarbonSatOntology:
    ALLOWED = {
        ("MISSION", "measures", "GAS"),
        ("MISSION", "produces", "PRODUCT"),
        ("MISSION", "has_instrument", "INSTRUMENT"),
        ("MISSION", "has_orbit", "ORBIT"),
        ("MISSION", "has_swath", "PARAMETER"),
        ("MISSION", "has_resolution", "PARAMETER"),
        ("MISSION", "covers", "LOC"),
        ("MISSION", "operated_by", "ORG"),
    }

    @classmethod
    def is_valid(cls, subj_type: str, predicate: str, obj_type: str) -> bool:
        return (subj_type, predicate, obj_type) in cls.ALLOWED
