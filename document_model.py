import spacy
import logging
import glob
import os
import tqdm
# from json_loader import JSONLoader
LOGGER = logging.getLogger(__name__)

def tokenize_doc_en(doc):

	tokens = []
	for token in doc:
		if '\n' in token.text:
			tokens.append(token)
		elif token.tag_ != '_SP' and token.text.strip(' '):
			tokens.append(token)

	return tokens


class Tagger(object):

	def __init__(self):
		self._spacy_tagger = spacy.load('en', disable=['parser', 'ner'])

	def tokenize_text(self, text: str):

		return [t for t in self._spacy_tagger(text)]


class Document:
	"""
	A document is a combination of text and the positions of the tags in that text.
	"""
	tagger = Tagger()

	def __init__(self, text, tags, sentences=None, filename=None):
		"""
		:param text: document text as a string
		:param tags: list of Tag objects
		"""

		print(text)
		self.tokens = [token.text for token in Document.tagger.tokenize_text(text)]

		if sentences:
			self.sentences = []
			for sentence in sentences:
				self.sentences.append([token.text for token in Document.tagger.tokenize_text(sentence)])
		self.tags = tags
		self.text = text
		self.filename = filename

