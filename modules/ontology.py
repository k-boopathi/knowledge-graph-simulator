class Ontology:
    def __init__(self):

        # Entity classes
        self.classes = {
            "PERSON",
            "ORG",
            "LOC",
            "PRODUCT",
            "UNKNOWN"
        }

        # Allowed semantic relations (domain, relation, range)
        self.relations = {

            ("PERSON", "founded", "ORG"),
            ("PERSON", "works_at", "ORG"),
            ("PERSON", "ceo_of", "ORG"),
            ("PERSON", "born_in", "LOC"),

            ("ORG", "located_in", "LOC"),
            ("ORG", "produces", "PRODUCT"),
            ("ORG", "acquired", "ORG"),

            ("PRODUCT", "created_by", "ORG"),
        }

    def is_valid(self, subj_type, relation, obj_type):
        return (subj_type, relation, obj_type) in self.relations
