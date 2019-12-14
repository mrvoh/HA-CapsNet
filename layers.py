import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import copy
# from transformer import Encoder as TransformerEncoder
# from flair.embeddings import * #StackedEmbeddings, BertEmbeddings, ELMoEmbeddings, FlairEmbeddings
from torchnlp.encoders.text import stack_and_pad_tensors
from fastai.text import *
# from fastai

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

def _get_clones(module, N):
	return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class TransformerEncoder(nn.Module):
	r"""TransformerEncoder is a stack of N encoder layers

	Args:
		encoder_layer: an instance of the TransformerEncoderLayer() class (required).
		num_layers: the number of sub-encoder-layers in the encoder (required).
		norm: the layer normalization component (optional).

	Examples::
		>>> encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
		>>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
	"""

	def __init__(self, emb_dim, hidden_dim, encoder_layer, num_layers, norm=None, dropout = 0.1):
		super(TransformerEncoder, self).__init__()
		self.layers = _get_clones(encoder_layer, num_layers)
		self.num_layers = num_layers
		self.emb_to_hidden = nn.Linear(emb_dim, hidden_dim)
		self.drop = nn.Dropout(dropout)
		self.norm1 = nn.LayerNorm(hidden_dim)
		if norm:
			self.norm = norm
		else:
			self.norm = nn.LayerNorm(hidden_dim)

	def forward(self, src, mask=None, src_key_padding_mask=None):
		r"""Pass the input through the endocder layers in turn.

		Args:
			src: the sequnce to the encoder (required).
			mask: the mask for the src sequence (optional).
			src_key_padding_mask: the mask for the src keys per batch (optional).

		Shape:
			see the docs in Transformer class.
		"""


		output = F.relu(self.emb_to_hidden(src))
		output = self.norm1(self.drop(output))

		for i in range(self.num_layers):
			output = self.layers[i](output, src_mask=mask,
									src_key_padding_mask=src_key_padding_mask)

		if self.norm:
			output = self.norm(output)

		return output


###################################################################################################
# HIERARCHICAL DOC ENCODING LAYERS
###################################################################################################

class ULMFiTEncoder(nn.Module):
	def __init__(self, pretrained_path, num_tokens, dropout_factor):
		super(ULMFiTEncoder, self).__init__()
		# state_dict = torch.loa
		config = {'emb_sz': 400, 'n_hid': 1150, 'n_layers': 3, 'pad_token': 1, 'qrnn': False, 'bidir': False, 'hidden_p': 0.15, 'input_p': 0.25, 'embed_p': 0.02, 'weight_p': 0.2}
		config['vocab_sz'] = num_tokens

		lm = AWD_LSTM(**config)
		#TODO: save config in state_dict when finetuning
		# lm = get_language_model(AWD_LSTM, num_tokens, config=config, drop_mult=dropout_factor)
		#
		# lm.load_encoder(pretrained_path)
		sd = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
		lm.load_state_dict(sd)
		# self.bn = nn.BatchNorm1d(config['emb_sz'])
		self.ln = nn.LayerNorm(config['emb_sz'])
		# hacky way to extract only the AWD-LSTM from the language model (SequentialRNN) which also contains a linear decoder

		self.ulmfit = lm #next(lm.modules())[0]

	def freeze_to(self, l):
		# when l < 0 everything will be unfrozen
		stop_map = {
			0:10, # lstm layer 3
			1:8, # lstm layer 2
			2:6, # lstm layer 1
			3:-1 # all
		}

		for i, mod in enumerate(self.ulmfit.modules()):
			for param in mod.parameters():
				try: # hacky way to work around FastAI modules
					param.requires_grad = i > stop_map[l]
				except:
					continue

	def encode(self, x):
		# Encodes a str as a concatenation of the mean and max pooling of the final hidden state over the whole sequence

		self.ulmfit.reset() # since an internal state is kept
		h, c = self.ulmfit(x)

		mean_pool = h[-1].mean(dim=1)
		max_pool, _ = h[-1].max(dim=1)

		x = torch.cat([mean_pool, max_pool], dim=1).squeeze()
		return x

	def forward(self, x):
		# manually reset the hidden states
		self.ulmfit.reset()

		h, c = self.ulmfit(x)

		x = h[-1]#[:,-1,:] # final hidden state
		x = self.ln(x)
		# x = self.bn(x.permute(0,2,1)).permute(0,2,1)
		return x

class AttentionWordEncoder(nn.Module):
	def __init__(self, encoder_type, num_tokens, embed_size, word_hidden, bidirectional= True, dropout=0.1,
				 num_layers = 1, nhead = 4, use_bert = False, ulmfit_pretrained_path = None, dropout_factor_ulmfit = 1.0):

		super(AttentionWordEncoder, self).__init__()
		self.num_tokens = num_tokens
		self.embed_size = embed_size
		self.word_hidden = word_hidden
		self.dropout = dropout
		self.lookup = nn.Embedding(num_tokens, embed_size)
		self.encoder_type = encoder_type
		self.use_bert = use_bert
		self.ulmfit_pretrained_path = ulmfit_pretrained_path
		self.num_layers = num_layers
		self.nhead = nhead
		self.bidirectional = bidirectional
		self.dropout_factor_ulmfit = dropout_factor_ulmfit

		if encoder_type.lower() == 'gru':
			word_out = 2* word_hidden if bidirectional else word_hidden
			self.word_encoder = nn.GRU(embed_size, word_hidden, bidirectional=bidirectional)
		elif encoder_type.lower() == 'transformer':
			# TEST
			# encoder_layer = TransformerCapsEncoderLayer(word_hidden, 1, 2 * word_hidden, dropout)
			encoder_layer = nn.TransformerEncoderLayer(word_hidden, nhead, 2*word_hidden, dropout)
			self.word_encoder = TransformerEncoder(embed_size, word_hidden, encoder_layer, num_layers)

			word_out = word_hidden
		elif encoder_type.lower() == 'ulmfit':
			self.word_encoder = ULMFiTEncoder(ulmfit_pretrained_path, num_tokens, dropout_factor_ulmfit)
			word_out = 400

		self.weight_W_word = nn.Linear(word_out, word_out)
		self.weight_proj_word = nn.Parameter(torch.Tensor(word_out, 1))

		self.seq_dim = 0 if self.encoder_type.lower() == 'gru' else 1
		self.softmax_word = nn.Softmax(dim=self.seq_dim)
		self.drop1 = nn.Dropout(dropout)
		self.drop2 = nn.Dropout(dropout)
		self.weight_proj_word.data.normal_(0, 1 / np.sqrt(word_out))



	def forward(self, x):

		if self.encoder_type.lower() == 'ulmfit': # ULMFit flow
			x1 = self.word_encoder(x)
		else: # Regular word embeddings flow
			# embeddings
			x_emb = self.lookup(x)

			if self.encoder_type.lower() == 'gru':
				x1, _ = self.word_encoder(self.drop1(x_emb))
			elif self.encoder_type.lower() == 'transformer':
				x1 = self.word_encoder(self.drop1(x_emb))

		# compute attention
		Hw = torch.tanh(self.weight_W_word(x1))
		Hw = Hw + x1  # residual connection

		word_attn = self.drop2(Hw.matmul(self.weight_proj_word))
		word_attn_norm = self.softmax_word(word_attn)
		# get sentence representation as weighted avg
		sen_encoding = x1.mul(word_attn_norm).sum(self.seq_dim)

		return sen_encoding, word_attn_norm


	def get_init_params(self):

		params = {
		'embed_size':self.embed_size,
		'word_hidden':self.word_hidden,
		'dropout':self.dropout,
		'embed_size':self.embed_size,
		'word_encoder':self.encoder_type,
		'use_bert':self.use_bert,
		'ulmfit_pretrained_path':self.ulmfit_pretrained_path,
		'ulmfit_dropout_factor':self.dropout_factor_ulmfit,
		'num_layers_word':self.num_layers,
		'nhead_word':self.nhead,
		'bidirectional':self.bidirectional
		}

		return params



class AttentionSentEncoder(nn.Module):

	def __init__(self, encoder_type, sent_hidden, word_out, bidirectional=True,
				 num_layers=1, nhead=4, dropout=0.1):

		super(AttentionSentEncoder, self).__init__()

		self.encoder_type = encoder_type
		self.word_out = word_out
		self.sent_hidden = sent_hidden
		self.dropout = dropout
		self.num_layers = num_layers
		self.nhead = nhead
		self.dropout = dropout

		if encoder_type.lower() == 'gru':
			self.bidirectional = bidirectional
			# word_out = 2 * word_hidden if bidirectional else word_hidden
			sent_out = 2 * sent_hidden if bidirectional else sent_hidden
			self.sent_encoder = nn.GRU(word_out, sent_hidden, bidirectional=bidirectional)
		elif encoder_type.lower() == 'transformer':
			encoder_layer = nn.TransformerEncoderLayer(word_out, nhead, sent_hidden, dropout)
			# TEST
			# encoder_layer = TransformerCapsEncoderLayer(word_out, sent_hidden, nhead, sent_hidden, dropout)
			self.sent_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

			sent_out = sent_hidden

		self.drop1 = nn.Dropout(dropout)
		self.weight_W_sent = nn.Linear(sent_out, sent_out)
		self.weight_proj_sent = nn.Parameter(torch.Tensor(sent_out, 1))

		self.softmax_sent = nn.Softmax(dim=1)
		self.weight_proj_sent.data.normal_(0, 0.01)

	def forward(self, x):

		# word level gru
		if self.encoder_type.lower() == 'gru':
			x, _ = self.sent_encoder(self.drop1(x.permute(1,0,2)))
			x = x.permute(1,0,2)
		elif self.encoder_type.lower() == 'transformer':
			x = self.sent_encoder(x)

		# compute attention
		Hw = torch.tanh(self.weight_W_sent(x))
		sent_attn = Hw.matmul(self.weight_proj_sent)
		sent_attn_norm = self.softmax_sent(sent_attn)
		# get sentence representation as weighted avg
		doc_encoding = x.mul(sent_attn_norm).sum(1)

		return doc_encoding, sent_attn_norm

	def get_init_params(self):

		params = {
		'sent_hidden':self.sent_hidden,
		'dropout':self.dropout,
		'sent_encoder':self.encoder_type,
		'num_layers_sen':self.num_layers,
		'nhead_sen':self.nhead,
		'bidirectional':self.bidirectional
		}

		return params


class GRUMultiHeadAtt(nn.Module):
	""""
	GRU with LabelWise Attention Network
	"""

	def __init__(self, sent_hidden, word_out, nhead_doc, bidirectional=True, dropout=0.1, aggregate_output = True):
		super(GRUMultiHeadAtt, self).__init__()

		# self.batch_size = batch_size
		self.sent_hidden = sent_hidden
		self.nhead_doc = nhead_doc
		self.word_out = word_out
		self.bidirectional = bidirectional
		self.dropout = dropout
		self.aggregate_output = aggregate_output

		sent_gru_out = 2 * sent_hidden if bidirectional else sent_hidden

		self.score_normalizer = np.sqrt(sent_gru_out)

		self.sent_gru = nn.GRU(word_out, sent_hidden, bidirectional=bidirectional)
		self.U = nn.Linear(sent_gru_out, nhead_doc)

		self.softmax_sent = nn.Softmax(dim=1)

		# Regularization
		self.bn1 = nn.BatchNorm1d(word_out)
		self.drop1 = nn.Dropout(dropout)
		if self.aggregate_output:
			self.bn2 = nn.BatchNorm1d(sent_gru_out)
			self.drop2 = nn.Dropout(dropout)

			self.out = nn.Parameter(torch.Tensor(nhead_doc, sent_gru_out))
			self.out.data.normal_(0, 1 / np.sqrt(sent_gru_out))

	def get_init_params(self):

		params = {
		'sent_hidden':self.sent_hidden,
		'dropout':self.dropout,
		'sent_encoder':'gru',
		'nhead_doc':self.nhead_doc,
		'bidirectional':self.bidirectional
		}

		return params

	def forward(self, word_attention_vectors):


		word_attention_vectors = self.drop1(self.bn1(word_attention_vectors.permute(0, 2, 1))).permute(0, 2, 1)
		output_sent, _ = self.sent_gru(word_attention_vectors)
		B, N, d_c = output_sent.size()

		H = output_sent.permute(1, 0, 2)
		# Get labelwise attention scores per document
		# A: [B, N, L] -> softmax-normalized scores per sentence per label
		A = self.softmax_sent(self.U(
			H) / self.score_normalizer)  # TODO: check performance when scores are discounted --> inspired by transformer
		# Get labelwise representations of doc
		attention_expanded = torch.repeat_interleave(A, d_c, dim=2)
		H_expanded = H.repeat(1, 1, self.nhead_doc)

		V = (attention_expanded * H_expanded).view(N, B, self.nhead_doc, d_c).sum(dim=0)
		if not self.aggregate_output:
			return V, A
		else:
			# V = V.sum(dim=0)
			V = self.drop2(self.bn2(V.permute(0, 2, 1))).permute(0, 2, 1)

			y = V.mul(self.out)
			y = y.sum(dim=2)

			return y, A


###################################################################################################
# CAPSNET LAYERS
###################################################################################################

def squash_v1(x, axis):
	s_squared_norm = (x ** 2).sum(axis, keepdim=True)
	scale = torch.sqrt(s_squared_norm)/ (0.5 + s_squared_norm)
	return scale * x

def dynamic_routing(batch_size, b_ij, u_hat, input_capsule_num):
	num_iterations = 3

	for i in range(num_iterations):
		if True:
			leak = torch.zeros_like(b_ij).sum(dim=2, keepdim=True)
			leaky_logits = torch.cat((leak, b_ij),2)
			leaky_routing = F.softmax(leaky_logits, dim=2)
			c_ij = leaky_routing[:,:,1:,:].unsqueeze(4)
		else:
			c_ij = F.softmax(b_ij, dim=2).unsqueeze(4)
		v_j = squash_v1((c_ij * u_hat).sum(dim=1, keepdim=True), axis=3)
		if i < num_iterations - 1:
			b_ij = b_ij + (torch.cat([v_j] * input_capsule_num, dim=1) * u_hat).sum(3)

	poses = v_j.squeeze(1)
	activations = torch.sqrt((poses ** 2).sum(2))
	return poses, activations


def Adaptive_KDE_routing(batch_size, b_ij, u_hat):
	last_loss = 0.0
	while True:
		if False:
			leak = torch.zeros_like(b_ij).sum(dim=2, keepdim=True)
			leaky_logits = torch.cat((leak, b_ij),2)
			leaky_routing = F.softmax(leaky_logits, dim=2)
			c_ij = leaky_routing[:,:,1:,:].unsqueeze(4)
		else:
			c_ij = F.softmax(b_ij, dim=2).unsqueeze(4)
		c_ij = c_ij/c_ij.sum(dim=1, keepdim=True)
		v_j = squash_v1((c_ij * u_hat).sum(dim=1, keepdim=True), axis=3)
		dd = 1 - ((squash_v1(u_hat, axis=3)-v_j)** 2).sum(3)
		b_ij = b_ij + dd

		c_ij = c_ij.view(batch_size, c_ij.size(1), c_ij.size(2))
		dd = dd.view(batch_size, dd.size(1), dd.size(2))

		kde_loss = torch.mul(c_ij, dd).sum()/batch_size
		kde_loss = np.log(kde_loss.item())

		if abs(kde_loss - last_loss) < 0.05:
			break
		else:
			last_loss = kde_loss
	poses = v_j.squeeze(1)
	activations = torch.sqrt((poses ** 2).sum(2))
	return poses, activations


def KDE_routing(batch_size, b_ij, u_hat):
	num_iterations = 3
	for i in range(num_iterations):
		if False:
			leak = torch.zeros_like(b_ij).sum(dim=2, keepdim=True)
			leaky_logits = torch.cat((leak, b_ij),2)
			leaky_routing = F.softmax(leaky_logits, dim=2)
			c_ij = leaky_routing[:,:,1:,:].unsqueeze(4)
		else:
			c_ij = F.softmax(b_ij, dim=2).unsqueeze(4)

		c_ij = c_ij/c_ij.sum(dim=1, keepdim=True)
		v_j = squash_v1((c_ij * u_hat).sum(dim=1, keepdim=True), axis=3)

		if i < num_iterations - 1:
			dd = 1 - ((squash_v1(u_hat, axis=3)-v_j)** 2).sum(3)
			b_ij = b_ij + dd
	poses = v_j.squeeze(1)
	activations = torch.sqrt((poses ** 2).sum(2))
	return poses, activations

class FlattenCaps(nn.Module):
	def __init__(self):
		super(FlattenCaps, self).__init__()
	def forward(self, p, a):
		poses = p.view(p.size(0), p.size(2) * p.size(3) * p.size(4), -1)
		# p = p.permute(0,2,3,4,1)
		# poses = p.view(p.size(0), p.size(1) * p.size(2) * p.size(3), -1)
		activations = a.view(a.size(0), a.size(1) * a.size(2) * a.size(3), -1)
		return poses, activations

class PrimaryCaps(nn.Module):
	def __init__(self, num_capsules, in_channels, out_channels, kernel_size, stride):
		super(PrimaryCaps, self).__init__()

		self.capsules = nn.Conv1d(in_channels, out_channels * num_capsules, kernel_size, stride)

		torch.nn.init.xavier_uniform_(self.capsules.weight)

		self.out_channels = out_channels
		self.num_capsules = num_capsules

	def forward(self, x):
		batch_size = x.size(0)
		u = self.capsules(x)
		u = u.view(batch_size, self.num_capsules, self.out_channels, -1, 1)
		poses = squash_v1(u, axis=1)
		activations = torch.sqrt((poses ** 2).sum(1))
		return poses, activations

class FCCaps(nn.Module):
	def __init__(self, is_AKDE, output_capsule_num, input_capsule_num, in_channels, out_channels):
		super(FCCaps, self).__init__()

		self.in_channels = in_channels
		self.out_channels = out_channels
		self.input_capsule_num = input_capsule_num
		self.output_capsule_num = output_capsule_num

		self.W1 = nn.Parameter(torch.FloatTensor(1, input_capsule_num, output_capsule_num, out_channels, in_channels))
		torch.nn.init.xavier_uniform_(self.W1)

		self.is_AKDE = is_AKDE
		self.sigmoid = nn.Sigmoid()

	def forward(self, x ):
		batch_size = x.size(0)

		W1 = self.W1

		x = torch.stack([x] * self.output_capsule_num, dim=2).unsqueeze(4)

		W1 = W1.repeat(batch_size, 1, 1, 1, 1)
		u_hat = torch.matmul(W1, x)

		b_ij = Variable(torch.zeros(batch_size, self.input_capsule_num,  self.output_capsule_num, 1)).cuda()

		if self.is_AKDE:
			poses, activations = Adaptive_KDE_routing(batch_size, b_ij, u_hat)
		else:
			#poses, activations = dynamic_routing(batch_size, b_ij, u_hat, self.input_capsule_num)
			poses, activations = KDE_routing(batch_size, b_ij, u_hat)
		return poses, activations

	def forward_smart(self, x, y, labels):
		batch_size = x.size(0)
		variable_output_capsule_num = len(labels)
		W1 = self.W1[:,:,labels,:,:]

		x = torch.stack([x] * variable_output_capsule_num, dim=2).unsqueeze(4)

		W1 = W1.repeat(batch_size, 1, 1, 1, 1)
		u_hat = torch.matmul(W1, x)

		b_ij = Variable(torch.zeros(batch_size, self.input_capsule_num, variable_output_capsule_num, 1)).cuda()

		if self.is_AKDE == True:
			poses, activations = Adaptive_KDE_routing(batch_size, b_ij, u_hat)
		else:
			#poses, activations = dynamic_routing(batch_size, b_ij, u_hat, self.input_capsule_num)
			poses, activations = KDE_routing(batch_size, b_ij, u_hat)
		return poses, activations


class CapsNet_Text(nn.Module):
	def __init__(self, input_size, in_channels, num_classes, dim_caps, num_caps, num_compressed_caps, dropout_caps, lambda_reg_caps):
		super(CapsNet_Text, self).__init__()
		self.num_classes = num_classes
		self.dim_caps = dim_caps
		self.input_size = input_size
		self.in_channels = in_channels
		self.num_caps = num_caps
		self.num_compressed_caps = num_compressed_caps


		self.primary_capsules_doc = PrimaryCaps(num_capsules=num_caps, in_channels=in_channels, out_channels=dim_caps, kernel_size=1, stride=1)

		self.flatten_capsules = FlattenCaps()

		self.W_doc = nn.Parameter(torch.FloatTensor( input_size * dim_caps, num_compressed_caps)) # 14272 --> doc_enc_dim * num_caps * dim_caps
		torch.nn.init.xavier_uniform_(self.W_doc)

		self.fc_capsules_doc_child = FCCaps(True, output_capsule_num= num_classes, input_capsule_num=num_compressed_caps,
								  in_channels=num_caps, out_channels=num_caps)

		#TODO: test dropout
		self.drop = nn.Dropout2d(p=dropout_caps)

		# DOC RECONSTRUCTOR
		self.recon_error_lambda = lambda_reg_caps # factor to scale down reconstruction loss with
		self.rescale = nn.Parameter(torch.Tensor([7]))
		reconstruction_size = 800 #TODO: change
		self.reconstruct0 = nn.Linear(num_caps * num_classes, int((reconstruction_size * 2) / 3))
		self.reconstruct1 = nn.Linear(int((reconstruction_size * 2) / 3), int((reconstruction_size * 3) / 2))
		self.reconstruct2 = nn.Linear(int((reconstruction_size * 3) / 2), reconstruction_size)
		# self.bn = nn.BatchNorm2d(num_classes)

		self.relu = nn.ReLU(inplace=True)
		self.sigmoid = nn.Sigmoid()

	def compression(self, poses, W):
		poses = torch.matmul(poses.permute(0,2,1), W).permute(0,2,1)
		activations = torch.sqrt((poses ** 2).sum(2))
		return poses, activations

	def forward(self, doc):

		poses_doc, activations_doc = self.primary_capsules_doc(doc)

		poses, activations = self.flatten_capsules(poses_doc, activations_doc)
		poses = self.drop(poses)
		#TODO: test 1d vs 2d dropout as regularization
		poses, activations = self.compression(poses, self.W_doc)
		poses, activations = self.fc_capsules_doc_child(poses)
		return poses, activations

	def forward_old(self, data, labels): # Use when second model is used to limit label space to route for caps net
		# labels arg is preliminary prediction by other model
		data = self.embed(data)
		nets_doc_l = []
		for i in range(len(self.ngram_size)):
			nets = self.convs_doc[i](data)
			nets_doc_l.append(nets)
		nets_doc = torch.cat((nets_doc_l[0], nets_doc_l[1], nets_doc_l[2]), 2)
		poses_doc, activations_doc = self.primary_capsules_doc(nets_doc)
		poses, activations = self.flatten_capsules(poses_doc, activations_doc)
		poses, activations = self.compression(poses, self.W_doc)
		poses, activations = self.fc_capsules_doc_child(poses, activations, labels) #parallel model is used for restricting solution space
		# poses = poses.unsqueeze(2)
		return poses, activations

	def reconstruction_loss(self, doc_enc, input, size_average=True):
		# Get the lengths of capsule outputs.
		# input = self.bn(input)
		v_mag = torch.sqrt((input ** 2).sum(dim=2))

		# Get index of longest capsule output.
		_, v_max_index = v_mag.max(dim=1)
		v_max_index = v_max_index.data

		# Use just the winning capsule's representation (and zeros for other capsules) to reconstruct input image.
		batch_size = input.size(0)
		all_masked = [None] * batch_size
		for batch_idx in range(batch_size):
			# Get one sample from the batch.
			input_batch = input[batch_idx]

			# Copy only the maximum capsule index from this batch sample.
			# This masks out (leaves as zero) the other capsules in this sample.
			batch_masked = Variable(torch.zeros(input_batch.size())).cuda()
			batch_masked[v_max_index[batch_idx]] = input_batch[v_max_index[batch_idx]]
			all_masked[batch_idx] = batch_masked

		# Stack masked capsules over the batch dimension.
		masked = torch.stack(all_masked, dim=0)

		# Reconstruct doc encoding.
		masked = masked.view(input.size(0), -1)
		# masked = masked * torch.pow(10,self.rescale) #test
		output = self.relu(self.reconstruct0(masked))
		output = self.relu(self.reconstruct1(output))
		output = self.reconstruct2(output)
		# output = output.view(-1, self.image_channels, self.image_height, self.image_width)
		#test: normalize doc encoding and reconstruction
		# output = output.div(output.norm(p=2, dim=1, keepdim=True))
		# doc_enc = doc_enc.div(doc_enc.norm(p=2, dim=1, keepdim=True))
		# The reconstruction loss is the sum squared difference between the input image and reconstructed image.
		# Multiplied by a small number so it doesn't dominate the margin (class) loss.
		error = (output - doc_enc)
		# error = error.view(output.size(0), -1)
		error = error ** 2
		error = torch.sum(error, dim=1) * self.recon_error_lambda

		# Average over batch
		if size_average:
			error = error.mean()

		return error


###################################################################################################
# EXPERIMENT: Capsule Transformer
###################################################################################################

class TransformerCapsEncoderLayer(nn.Module):

	def __init__(self, d_model, out_size, nhead, dim_feedforward=50, dropout=0.1):
		super(TransformerCapsEncoderLayer, self).__init__()
		self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
		# # Implementation of Feedforward model
		#         # self.linear1 = Linear(d_model, dim_feedforward)
		#         # self.dropout = Dropout(dropout)
		#         # self.linear2 = Linear(dim_feedforward, d_model)
		self.d_model = d_model
		self.out_size = out_size
		self.feedforward_caps = CapsNet_Text(d_model, 1, out_size, dim_caps=12, num_caps=8, num_compressed_caps=dim_feedforward)
		self.scaling_param = nn.Parameter(torch.tensor([1e-8]))

		self.norm1 = nn.LayerNorm(d_model)
		self.norm2 = nn.LayerNorm(out_size)
		self.dropout1 = nn.Dropout(dropout)
		self.dropout2 = nn.Dropout(dropout)

	def forward(self, src, src_mask=None, src_key_padding_mask=None):
		r"""Pass the input through the endocder layer.

		Args:
			src: the sequnce to the encoder layer (required).
			src_mask: the mask for the src sequence (optional).
			src_key_padding_mask: the mask for the src keys per batch (optional).

		Shape:
			see the docs in Transformer class.
		"""
		src2 = self.self_attn(src, src, src, attn_mask=src_mask,
							  key_padding_mask=src_key_padding_mask)[0]
		src = src + self.dropout1(src2)
		src = self.norm1(src)
		B, N, d_c = src.size()
		src = src.view(-1,1,self.d_model)
		_, src2 = self.feedforward_caps(src)
		src = self.dropout2(src2.squeeze(2)).view(B,N,self.out_size)

		src = torch.log(src+self.scaling_param/(1-src+self.scaling_param))
		src = self.norm2(src)

		# src = src * torch.pow(10, self.scaling_param)

		return src