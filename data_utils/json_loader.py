import json
import os
from document_model import Document

# from configuration import Configuration


class JSONLoader:
    def __init__(
        self, hierarchical=True, restructure_doc=True, split_size_long_seqs=50
    ):
        super(JSONLoader).__init__()

        self.hierarchical = hierarchical
        self.restructure_doc = restructure_doc
        self.split_size_long_seqs = split_size_long_seqs

    def read_text(self, text: str):

        data = json.loads(text)
        return self._process_data(data)

    def read_file(self, filename: str):

        with open(filename, encoding="utf-8") as file:
            data = json.load(file)

        return self._process_data(data, filename)

    def _process_data(self, data, filename=None):
        tags = []
        sections = []
        text = ""
        sections.append(data["header"])
        sections.append(data["recitals"])
        sections.extend(data["main_body"])
        sections.append(data["attachments"])

        sentences = []
        for sec in sections:
            s = sec.replace(";", ".")
            sentences += s.split("\n")
            # replace ;\n
            # replace \n

        if not self.hierarchical:
            text = "\n".join(sections)
            sections = []
        for concept in data["concepts"]:
            tags.append(concept)

        if not filename:
            filename = data["celex_id"]

        return Document(
            text,
            tags,
            sentences=sentences,
            filename=os.path.basename(filename),
            restructure_doc=self.restructure_doc,
            split_size_long_seqs=self.split_size_long_seqs,
        )
