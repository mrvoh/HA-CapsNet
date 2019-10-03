import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import copy
from transformer import Encoder as TransformerEncoder

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

class AttentionWordEncoder(nn.Module):
	def __init__(self, encoder_type, num_tokens, embed_size, word_hidden, bidirectional= True, dropout=0.1,
				 num_layers = 1, nhead = 4):

		super(AttentionWordEncoder, self).__init__()
		self.num_tokens = num_tokens
		self.embed_size = embed_size
		self.word_hidden = word_hidden
		self.dropout = dropout
		self.lookup = nn.Embedding(num_tokens, embed_size)
		self.encoder_type = encoder_type

		if encoder_type.lower() == 'gru':
			self.bidirectional = bidirectional
			word_out = 2* word_hidden if bidirectional else word_hidden
			self.word_encoder = nn.GRU(embed_size, word_hidden, bidirectional=bidirectional)
		elif encoder_type.lower() == 'transformer':
			# TEST
			encoder_layer = TransformerCapsEncoderLayer(word_hidden, nhead, 2 * word_hidden, dropout)
			# encoder_layer = nn.TransformerEncoderLayer(word_hidden, nhead, 2*word_hidden, dropout)
			self.word_encoder = TransformerEncoder(embed_size, word_hidden, encoder_layer, num_layers)

			word_out = word_hidden

		self.weight_W_word = nn.Linear(word_out, word_out)
		self.weight_proj_word = nn.Parameter(torch.Tensor(word_out, 1))

		self.seq_dim = 0 if self.encoder_type.lower() == 'gru' else 1
		self.softmax_word = nn.Softmax(dim=self.seq_dim)
		self.drop1 = nn.Dropout(dropout)
		self.drop2 = nn.Dropout(dropout)
		self.weight_proj_word.data.normal_(0, 1 / np.sqrt(word_out))

	def forward(self, x):

		# embeddings
		x_emb = self.lookup(x)
		if self.encoder_type.lower() == 'gru':
			x1, _ = self.word_encoder(self.drop1(x_emb))
		elif self.encoder_type.lower() == 'transformer':
			x1 = self.word_encoder(self.drop1(x_emb))
			# x1 = x1.permute(1,0,2)
		# compute attention
		Hw = torch.tanh(self.weight_W_word(x1))
		Hw = Hw + x1  # residual connection

		word_attn = self.drop2(Hw.matmul(self.weight_proj_word))
		word_attn_norm = self.softmax_word(word_attn)
		# get sentence representation as weighted avg
		sen_encoding = x1.mul(word_attn_norm).sum(self.seq_dim)

		return sen_encoding, word_attn_norm


class AttentionSentEncoder(nn.Module):

	def __init__(self, encoder_type, sent_hidden, word_out, bidirectional=True,
				 num_layers=1, nhead=4, dropout=0.1):

		super(AttentionSentEncoder, self).__init__()

		self.encoder_type = encoder_type
		self.word_out = word_out
		self.sent_hidden = sent_hidden

		if encoder_type.lower() == 'gru':
			self.bidirectional = bidirectional
			# word_out = 2 * word_hidden if bidirectional else word_hidden
			sent_out = 2 * sent_hidden if bidirectional else sent_hidden
			self.sent_encoder = nn.GRU(word_out, sent_hidden, bidirectional=bidirectional)
		elif encoder_type.lower() == 'transformer':
			# encoder_layer = nn.TransformerEncoderLayer(word_out, nhead, sent_hidden, dropout)
			# TEST
			encoder_layer = TransformerCapsEncoderLayer(word_out, sent_hidden, nhead, sent_hidden, dropout)
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


class GRUMultiHeadAtt(nn.Module):
	""""
	GRU with LabelWise Attention Network
	"""

	def __init__(self, sent_hidden, word_out, num_att_heads, bidirectional=True, dropout=0.1, aggregate_output = True):
		super(GRUMultiHeadAtt, self).__init__()

		# self.batch_size = batch_size
		self.sent_gru_hidden = sent_hidden
		self.num_att_heads = num_att_heads
		self.word_out = word_out
		self.bidirectional = bidirectional
		self.dropout = dropout
		self.aggregate_output = aggregate_output

		sent_gru_out = 2 * sent_hidden if bidirectional else sent_hidden

		self.score_normalizer = np.sqrt(sent_gru_out)

		self.sent_gru = nn.GRU(word_out, sent_hidden, bidirectional=bidirectional)
		self.U = nn.Linear(sent_gru_out, num_att_heads)

		self.softmax_sent = nn.Softmax(dim=1)

		# Regularization
		self.bn1 = nn.BatchNorm1d(word_out)
		self.drop1 = nn.Dropout(dropout)
		if self.aggregate_output:
			self.bn2 = nn.BatchNorm1d(sent_gru_out)
			self.drop2 = nn.Dropout(dropout)

			self.out = nn.Parameter(torch.Tensor(num_att_heads, sent_gru_out))
			self.out.data.normal_(0, 1 / np.sqrt(sent_gru_out))

	def forward(self, word_attention_vectors):
		B, N, d_c = word_attention_vectors.size()

		word_attention_vectors = self.drop1(self.bn1(word_attention_vectors.permute(0, 2, 1))).permute(0, 2, 1)
		output_sent, _ = self.sent_gru(word_attention_vectors)

		H = output_sent.permute(1, 0, 2)
		# Get labelwise attention scores per document
		# A: [B, N, L] -> softmax-normalized scores per sentence per label
		A = self.softmax_sent(self.U(
			H) / self.score_normalizer)  # TODO: check performance when scores are discounted --> inspired by transformer
		# Get labelwise representations of doc
		attention_expanded = torch.repeat_interleave(A, d_c, dim=2)
		H_expanded = H.repeat(1, 1, self.num_att_heads)

		V = (attention_expanded * H_expanded).view(N, B, self.num_att_heads, d_c).sum(dim=0)
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
	def __init__(self, input_size, in_channels, num_classes, dim_caps, num_caps, num_compressed_caps):
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

	def compression(self, poses, W):
		poses = torch.matmul(poses.permute(0,2,1), W).permute(0,2,1)
		activations = torch.sqrt((poses ** 2).sum(2))
		return poses, activations

	def forward(self, doc):

		poses_doc, activations_doc = self.primary_capsules_doc(doc)
		poses, activations = self.flatten_capsules(poses_doc, activations_doc)
		poses, activations = self.compression(poses, self.W_doc)
		poses, activations = self.fc_capsules_doc_child(poses)
		return poses, activations

	def forward_smart(self, data, labels): # Use when second model is used to limit label space to route for caps net
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