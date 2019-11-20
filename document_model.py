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

from fastai.text.transform import Tokenizer as ULMFiTTokenizer
from fastai.text.transform import SpacyTokenizer

# language mapping for backtranslation
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

def translate(text, inter_lang, src_lang='en'):
	""""
	Translates text to inter_lang and then back to src_lang as means of data augmentation
	Translation performed using Google Translate API
	"""
	if hasattr(text, "decode"):
		text = text.decode("utf-8")

	text = TextBlob(text)
	try:
		text = text.translate(to=inter_lang)
		text = text.translate(to=src_lang)
	except NotTranslated:
		print('NOT TRANSLATED')
		pass

	return str(text)


class TextPreprocessor(object):
	""""
	Normalizes and tokenizes text
	"""
	def __init__(self, ulmfit_preprocessing = True, lang='en'):
		self._spacy_tagger = spacy.load(lang, disable=['parser', 'ner'])
		self._spacy_tagger.add_pipe(self._spacy_tagger.create_pipe('sentencizer'))

		self.ulmfit_preprocessing = ulmfit_preprocessing
		if ulmfit_preprocessing:
			self.ulmfit_tokenizer = ULMFiTTokenizer()
			self.s = SpacyTokenizer(lang) # used by ulmfit tokenizer

	def tokenize_text(self, text: str):

		if not self.ulmfit_preprocessing: # standard preprocessing
			text = lowercase_and_remove_accent(text)
			text = remove_non_printing_char(text)
			text = replace_unicode_punct(text)
		else: # ULMFiT-specific preprocessing
			text = ' '.join(self.ulmfit_tokenizer.process_text(text, self.s))


		return [[str(t) for t in sent] for sent in self._spacy_tagger(text).sents]


class Document:
	"""
	A document is a combination of text and the positions of the tags in that text.
	"""
	# text_preprocessor = TextPreprocessor()

	def __init__(self,tags, text_preprocessor, text=None,  sentences=None, filename=None, restructure_doc = True, split_size_long_seqs=50, discard_short_sents=False, short_seqs_thresh=3):
		"""
		:param text: document text as a string
		:param tags: list of Tag objects
		"""

		assert text or sentences, "To instantiate a Document, either a list of sentences or a full text must be given"

		self.discard_short_sents = discard_short_sents # whether or not to remove extremely short sentences from data
		self.short_seqs_thresh = short_seqs_thresh # Max length of sentences to be discarded (to e.g. remove headers from text)
		self.restructure_doc = restructure_doc # If set to True all sentences longer than split_size_long_seqs will be split and afterwards greedily merged to get length as close as possible to split_size_long_seqs
		self.split_size_long_seqs = split_size_long_seqs
		self.encoding = None

		if sentences: # Data already partially preprocessed
			self.sentences = []
			for sentence in sentences:
				self.sentences.extend([[token for token in sent] for sent in text_preprocessor.tokenize_text(sentence)])
		else: # Full preprocessing still needs to be done
			self.sentences = text_preprocessor.tokenize_text(str(text))

		if self.discard_short_sents:
			self.sentences = [sent for sent in self.sentences if len(sent) > self.short_seqs_thresh]

		# Split seqs longer than split_size_long_seqs for computational efficiency
		# After splitting, chunks are greedily merged back together to get the biggest chunks allowed
		if self.restructure_doc:
			self._split_long_seqs()

		self.tags = tags
		self.text = text
		self.filename = filename

	def set_encoding(self, enc):

		self.encoding = enc

	def back_translate(self, num_copies):
		""""
		Using this function might require (payed) credentials for the Google Translate API.
		"""
		sents = [" ".join(sen) for sen in self.sentences]
		copies = [Document(
			sentences=[translate(sen, LANGUAGES[i]) for sen in sents],
			text='',
			tags=self.tags)
			for i in range(num_copies)
		]

		return copies

	def get_text_sentences(self):
		# Return list of all sentences in doc
		return [" ".join(sen) for sen in self.sentences]


	##################################################################################
	# Helper functions
	##################################################################################

	def _merge_short_seqs(self):
		"""
		Greedily merge short sentences up until the split_size_long_seqs is reached
		This greatly improves computational efficiency
		"""
		sents = []
		curr_sen = []
		for sen in self.sentences:
			if len(curr_sen) + len(sen) <= self.split_size_long_seqs:
				curr_sen.extend(sen)
			else:
				sents.append(curr_sen)
				curr_sen = []

		if curr_sen != []: # When a doc contains only one sentence this statement is triggered
			sents.append(curr_sen)

		self.sentences = sents

	def _split_long_seqs(self):
		"""
		Split long sequences such that all sentences have a length <= split_size_long_seqs
		First long sequences are split on punctuation, then on index if still necessary
		"""
		res = []

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
