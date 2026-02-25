class CarbonSatOntology:
    """
    Very simple domain ontology validation:
    defines allowed (subject_type, predicate, object_type) combos.
    """

    VALID = {
        ("MISSION", "measures", "GAS"),
        ("MISSION", "detects", "GAS"),
        ("MISSION", "produces", "PRODUCT"),
        ("MISSION", "operates_in", "ORBIT"),
        ("MISSION", "developed_by", "ORG"),
        ("MISSION", "built_by", "ORG"),
        ("MISSION", "part_of", "ORG"),
        ("PRODUCT", "produced_by", "MISSION"),
    }

    @classmethod
    def is_valid(cls, subj_type: str, predicate: str, obj_type: str) -> bool:
        if not subj_type or not predicate or not obj_type:
            return False
        return (subj_type, predicate, obj_type) in cls.VALID
