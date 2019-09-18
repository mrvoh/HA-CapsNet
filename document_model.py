import spacy
import logging
import glob
import os
import tqdm
# from json_loader import JSONLoader
LOGGER = logging.getLogger(__name__)
import unicodedata
# import sacremoses as sm
import re

# def tokenize_doc_en(doc):
#
# 	tokens = []
# 	for token in doc:
# 		if '\n' in token.text:
# 			tokens.append(token)
# 		elif token.tag_ != '_SP' and token.text.strip(' '):
# 			tokens.append(token)
#
# 	return tokens

def lowercase_and_remove_accent(text):
	"""
	Lowercase and strips accents from a piece of text based on
	https://github.com/facebookresearch/XLM/blob/master/tools/lowercase_and_remove_accent.py
	"""
	return text.lower()
	text = ' '.join(text)
	text = text.lower()
	text = unicodedata.normalize("NFD", text)
	output = []
	for char in text:
		cat = unicodedata.category(char)
		if cat == "Mn":
			continue
		output.append(char)
	return "".join(output).lower().split(' ')


def replace_unicode_punct(text):
	'''
	Port of https://github.com/moses-smt/mosesdecoder/blob/master/scripts/tokenizer/replace-unicode-punctuation.perl
	'''
	text = text.replace('，', ',')
	text = re.sub(r'。\s*', '. ', text)
	text = text.replace('、', ',')
	text = text.replace('”', '"')
	text = text.replace('“', '"')
	text = text.replace('∶', ':')
	text = text.replace('：', ':')
	text = text.replace('？', '?')
	text = text.replace('《', '"')
	text = text.replace('》', '"')
	text = text.replace('）', ')')
	text = text.replace('！', '!')
	text = text.replace('（', '(')
	text = text.replace('；', ';')
	text = text.replace('１', '"')
	text = text.replace('」', '"')
	text = text.replace('「', '"')
	text = text.replace('０', '0')
	text = text.replace('３', '3')
	text = text.replace('２', '2')
	text = text.replace('５', '5')
	text = text.replace('６', '6')
	text = text.replace('９', '9')
	text = text.replace('７', '7')
	text = text.replace('８', '8')
	text = text.replace('４', '4')
	text = re.sub(r'．\s*', '. ', text)
	text = text.replace('～', '~')
	text = text.replace('’', '\'')
	text = text.replace('…', '...')
	text = text.replace('━', '-')
	text = text.replace('〈', '<')
	text = text.replace('〉', '>')
	text = text.replace('【', '[')
	text = text.replace('】', ']')
	text = text.replace('％', '%')
	return text


def remove_non_printing_char(text):
	'''
	Port of https://github.com/moses-smt/mosesdecoder/blob/master/scripts/tokenizer/remove-non-printing-char.perl
	'''
	output = []
	for char in text:
		cat = unicodedata.category(char)
		if cat.startswith('C'):
			continue
		output.append(char)
	return "".join(output)

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


class Tagger(object):


	def __init__(self, preprocess = True):
		self._spacy_tagger = spacy.load('en', disable=['parser', 'ner'])
		self._spacy_tagger.add_pipe(self._spacy_tagger.create_pipe('sentencizer'))

		self.preprocess = preprocess

	def tokenize_text(self, text: str):

		if self.preprocess:
			text = lowercase_and_remove_accent(text)
			text = remove_non_printing_char(text)
			text = replace_unicode_punct(text)
		# x = [[t for t in sent] for sent in self._spacy_tagger(text).sents]
		return [[t for t in sent] for sent in self._spacy_tagger(text).sents]


class Document:
	"""
	A document is a combination of text and the positions of the tags in that text.
	"""
	tagger = Tagger()
	SHORT_SEN_TRESH = 3

	def __init__(self, text, tags, sentences=None, filename=None, discard_short_sents = True, split_size_long_seqs=50):
		"""
		:param text: document text as a string
		:param tags: list of Tag objects
		"""
		self.discard_short_sents = discard_short_sents
		self.split_size_long_seqs = split_size_long_seqs
		# print(text)
		# self.tokens = [[token.text for token in sent] for sent in Document.tagger.tokenize_text(text)]

		if sentences:
			self.sentences = []
			for sentence in sentences:
				self.sentences.extend([[token.text for token in sent] for sent in Document.tagger.tokenize_text(sentence)])#[token.text for token in Document.tagger.tokenize_text(sentence)])

		if self.discard_short_sents:
			self.sentences = [sent for sent in self.sentences if len(sent) > self.SHORT_SEN_TRESH]

		# Split seqs longer than split_size_long_seqs for computational efficiency
		self._split_long_seqs()

		self.tags = tags
		self.text = text
		self.filename = filename


	def _split_long_seqs(self):

		res = []
		split_punct = [',',';', ':','?','!', '/']

		# res = [re.split('[?,.;:\-!]'," ".join(sen)) if len(sen) > self.split_size_long_seqs else sen for sen in self.sentences]

		# first split on punctuation
		for sen in self.sentences:
			if len(sen) > self.split_size_long_seqs:
				s = re.split('[?,.;:\-!]'," ".join(sen))
				try:
					res.append(s.split(' '))
				except:
					s = [sub.split(' ') for sub in s]
					res.extend(s)
				# if s == sen:
				# 	res.append(s)
				# else:
				# 	res.extend(s)
			else:
				res.append(sen)

		assert len(res) >= len(self.sentences), "Splitting of long sentences went wrong"

		final = []
		# Else just split on index
		for sen in res:
			if len(sen) > self.split_size_long_seqs:

				# try:
				# 	res.append(s.split(' '))
				# except:
				# 	s = [sub.split(' ') for sub in s]
				# 	res.extend(s)
				final.extend(list(chunks(sen, self.split_size_long_seqs)))
			else:
				final.append(sen)

		assert len(final) >= len(res), "Splitting of long sentences went wrong"

		final = [[tok for tok in sen if tok != ''] for sen in final]
		self.sentences = final
