class Author:
    def __init__(self, name: str):
        self.name = name
        self.ndoc = 0
        self.production = []

    def add_doc(self, doc):
        self.production.append(doc)
        self.ndoc += 1

    def __str__(self):
        return f"{self.name} a publié {self.ndoc} documents."

