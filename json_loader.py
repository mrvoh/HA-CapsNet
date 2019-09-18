import json
import os
from document_model import Document
# from configuration import Configuration


class JSONLoader:

    def __init__(self, hierarchical=True):
        super(JSONLoader).__init__()

        self.hierarchical = hierarchical
    def read_file(self, filename: str):
        tags = []
        with open(filename, encoding='utf-8') as file:
            data = json.load(file)
        sections = []
        text = ''
        sections.append(data['header'])
        sections.append(data['recitals'])
        sections.extend(data['main_body'])
        sections.append(data['attachments'])

        sentences = []
        for sec in sections:
            s = sec.replace(';','.')
            sentences += s.split("\n")
            # replace ;\n
            # replace \n

        if not self.hierarchical:
            text = '\n'.join(sections)
            sections = []
        for concept in data['concepts']:
            tags.append(concept)


        # print(sentences)
        # print(sections)
        return Document(text, tags, sentences=sentences, filename=os.path.basename(filename))