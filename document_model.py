import spacy
import logging
from pybacktrans import BackTranslator
# from json_loader import JSONLoader
LOGGER = logging.getLogger(__name__)
import unicodedata
# import sacremoses as sm
import re
from joblib import Parallel, delayed
from textblob import TextBlob
from textblob.translate import NotTranslated

LANG_MAP= {
    'af': 'afrikaans',
    'sq': 'albanian',
    'am': 'amharic',
    'ar': 'arabic',
    'hy': 'armenian',
    'az': 'azerbaijani',
    'eu': 'basque',
    'be': 'belarusian',
    'bn': 'bengali',
    'bs': 'bosnian',
    'bg': 'bulgarian',
    'ca': 'catalan',
    'ceb': 'cebuano',
    'ny': 'chichewa',
    'zh-cn': 'chinese (simplified)',
    'zh-tw': 'chinese (traditional)',
    'co': 'corsican',
    'hr': 'croatian',
    'cs': 'czech',
    'da': 'danish',
    'nl': 'dutch',
    'en': 'english',
    'eo': 'esperanto',
    'et': 'estonian',
    'tl': 'filipino',
    'fi': 'finnish',
    'fr': 'french',
    'fy': 'frisian',
    'gl': 'galician',
    'ka': 'georgian',
    'de': 'german',
    'el': 'greek',
    'gu': 'gujarati',
    'ht': 'haitian creole',
    'ha': 'hausa',
    'haw': 'hawaiian',
    'iw': 'hebrew',
    'hi': 'hindi',
    'hmn': 'hmong',
    'hu': 'hungarian',
    'is': 'icelandic',
    'ig': 'igbo',
    'id': 'indonesian',
    'ga': 'irish',
    'it': 'italian',
    'ja': 'japanese',
    'jw': 'javanese',
    'kn': 'kannada',
    'kk': 'kazakh',
    'km': 'khmer',
    'ko': 'korean',
    'ku': 'kurdish (kurmanji)',
    'ky': 'kyrgyz',
    'lo': 'lao',
    'la': 'latin',
    'lv': 'latvian',
    'lt': 'lithuanian',
    'lb': 'luxembourgish',
    'mk': 'macedonian',
    'mg': 'malagasy',
    'ms': 'malay',
    'ml': 'malayalam',
    'mt': 'maltese',
    'mi': 'maori',
    'mr': 'marathi',
    'mn': 'mongolian',
    'my': 'myanmar (burmese)',
    'ne': 'nepali',
    'no': 'norwegian',
    'ps': 'pashto',
    'fa': 'persian',
    'pl': 'polish',
    'pt': 'portuguese',
    'pa': 'punjabi',
    'ro': 'romanian',
    'ru': 'russian',
    'sm': 'samoan',
    'gd': 'scots gaelic',
    'sr': 'serbian',
    'st': 'sesotho',
    'sn': 'shona',
    'sd': 'sindhi',
    'si': 'sinhala',
    'sk': 'slovak',
    'sl': 'slovenian',
    'so': 'somali',
    'es': 'spanish',
    'su': 'sundanese',
    'sw': 'swahili',
    'sv': 'swedish',
    'tg': 'tajik',
    'ta': 'tamil',
    'te': 'telugu',
    'th': 'thai',
    'tr': 'turkish',
    'uk': 'ukrainian',
    'ur': 'urdu',
    'uz': 'uzbek',
    'vi': 'vietnamese',
    'cy': 'welsh',
    'xh': 'xhosa',
    'yi': 'yiddish',
    'yo': 'yoruba',
    'zu': 'zulu',
    'fil': 'Filipino',
    'he': 'Hebrew'
}

LANGUAGES = [k for k in LANG_MAP.keys()]

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


def translate(comment, language):
	if hasattr(comment, "decode"):
		comment = comment.decode("utf-8")

	text = TextBlob(comment)
	try:
		text = text.translate(to=language)
		text = text.translate(to="en")
	except NotTranslated:
		print('NOT TRANSLATED')
		pass

	return str(text)

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
	translator = BackTranslator()
	#
	# 	# SHORT_SEN_TRESH = 3

	def __init__(self, text, tags, sentences=None, filename=None, restructure_doc = True, split_size_long_seqs=50):
		"""
		:param text: document text as a string
		:param tags: list of Tag objects
		"""



		self.restructure_doc = restructure_doc # If set to True all sentences longer than split_size_long_seqs will be split and afterwards greedily merged to get length as close as possible to split_size_long_seqs
		self.split_size_long_seqs = split_size_long_seqs
		# print(text)
		# self.tokens = [[token.text for token in sent] for sent in Document.tagger.tokenize_text(text)]

		if sentences:
			self.sentences = []
			for sentence in sentences:
				self.sentences.extend([[token.text for token in sent] for sent in Document.tagger.tokenize_text(sentence)])#[token.text for token in Document.tagger.tokenize_text(sentence)])

		# if self.discard_short_sents:
		# 	self.sentences = [sent for sent in self.sentences if len(sent) > self.SHORT_SEN_TRESH]

		# Split seqs longer than split_size_long_seqs for computational efficiency
		if self.restructure_doc:
			self._split_long_seqs()

		self.tags = tags
		self.text = text
		self.filename = filename

	def back_translate(self, num_copies):

		sents = [" ".join(sen) for sen in self.sentences]
		copies = [Document(
			sentences=[translate(sen, LANGUAGES[i]) for sen in sents],
			text='',
			tags=self.tags)
			for i in range(num_copies)
		]

		return copies

	def back_translate1(self, num_copies): #TODO: look into https://github.com/PavelOstyakov/toxic/blob/master/tools/extend_dataset.py

		sents = [" ".join(sen) for sen in self.sentences]
		copies = [Document(
			sentences = [Document.translator.backtranslate(sen, src='en', mid=LANGUAGES[i]).text for sen in sents],
			text = '',
			tags=self.tags)
			for i in range(num_copies)
		]

		return copies

	def get_text_sentences(self):
		return [" ".join(sen) for sen in self.sentences]

	def _merge_short_seqs(self):

		sents = []
		curr_sen = []
		for sen in self.sentences:
			if len(curr_sen) + len(sen) <= self.split_size_long_seqs:
				curr_sen.extend(sen)
			else:
				sents.append(curr_sen)
				curr_sen = []

		if curr_sen != []:
			sents.append(curr_sen)

		self.sentences = sents

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
			else:
				res.append(sen)


		assert len(res) >= len(self.sentences), "Splitting of long sentences went wrong"

		final = []
		# Else just split on index
		for sen in res:
			if len(sen) > self.split_size_long_seqs:
				final.extend(list(chunks(sen, self.split_size_long_seqs)))
			else:
				final.append(sen)

		assert len(final) >= len(res), "Splitting of long sentences went wrong"

		final = [[tok for tok in sen if tok != ''] for sen in final]

		# Merge small sentences back together
		self.sentences = final
		self._merge_short_seqs()

	def __str__(self):

		res = "Doc name: {}\n Text: {}".format(self.filename, "\n".join([" ".join(sen) for sen in self.sentences]))
		return res


if __name__ == '__main__':

	sen = ['AFFILIATED PUBLICATIONS INC & lt ; AFP > SETS PAYOUT Qtrly div eight cts vs eight cts prior Pay June 1 Record May 15']
	d = Document('','',sentences=sen)
	print(d.sentences)
