import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from transformer import Encoder as TransformerEncoder

###################################################################################################
# HIERARCHICAL DOC ENCODING LAYERS
###################################################################################################

class AttentionWordEncoder(nn.Module):
	def __init__(self, encoder_type, num_tokens, embed_size, word_hidden, bidirectional= True, dropout=0.1,
				 max_seq_len = 50, num_layers = 1, nhead = 4):

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
			# encoder_layer = nn.TransformerEncoderLayer(max_seq_len, nhead, word_hidden, dropout)
			self.word_encoder = TransformerEncoder(
														embedding_size= embed_size,
														hidden_size = word_hidden, #todo: change setup might help
														num_layers= num_layers,
														num_heads = nhead,
														total_key_depth = word_hidden, #todo: check
														total_value_depth = word_hidden, #check as well
														filter_size = word_hidden,
														max_length = max_seq_len,
														attention_dropout=dropout,
														input_dropout=dropout,
														relu_dropout=dropout,
														layer_dropout=dropout
            )
			word_out = word_hidden

		self.weight_W_word = nn.Linear(word_out, word_out)
		self.weight_proj_word = nn.Parameter(torch.Tensor(word_out, 1))

		self.softmax_word = nn.Softmax(dim=0)
		self.drop1 = nn.Dropout(dropout)
		self.drop2 = nn.Dropout(dropout)
		self.weight_proj_word.data.normal_(0, 1 / np.sqrt(word_out))

	def forward(self, x):

		# embeddings
		x = self.lookup(x)
		if self.encoder_type.lower() == 'gru':
			x, _ = self.word_encoder(self.drop1(x))
		elif self.encoder_type.lower() == 'transformer':
			x = self.word_encoder(self.drop1(x))
		# compute attention
		Hw = torch.tanh(self.weight_W_word(x))
		Hw = Hw + x  # residual connection

		word_attn = self.drop2(Hw.matmul(self.weight_proj_word))
		word_attn_norm = self.softmax_word(word_attn)
		# get sentence representation as weighted avg
		sen_encoding = x.mul(word_attn_norm).sum(0)

		return sen_encoding, word_attn_norm


class AttentionSentEncoder(nn.Module):

	def __init__(self, encoder_type, sent_hidden, word_out, bidirectional=True,
				 max_seq_len=100, num_layers=1, nhead=4, dropout=0.1):

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
			# encoder_layer = nn.TransformerEncoderLayer(max_seq_len, nhead, sent_hidden, dropout)
			self.sent_encoder =  TransformerEncoder(
														embedding_size= word_out,
														hidden_size = sent_hidden, #todo: change setup might help
														num_layers= num_layers,
														num_heads = nhead,
														total_key_depth = sent_hidden, #todo: check
														total_value_depth = sent_hidden, #check as well
														filter_size = sent_hidden,
														max_length = max_seq_len,
														attention_dropout=dropout,
														input_dropout=dropout,
														relu_dropout=dropout,
														layer_dropout=dropout)
			sent_out = sent_hidden

		self.drop1 = nn.Dropout(dropout)
		self.weight_W_sent = nn.Linear(sent_out, sent_out)
		self.weight_proj_sent = nn.Parameter(torch.Tensor(sent_out, 1))

		self.softmax_sent = nn.Softmax(dim=1)
		self.weight_proj_sent.data.normal_(0, 0.1)

	def forward(self, x):

		# word level gru
		if self.encoder_type.lower() == 'gru':
			x, _ = self.sent_encoder(self.drop1(x))
		elif self.encoder_type.lower() == 'transformer':
			x = self.sent_encoder(self.drop1(x))
		# compute attention
		Hw = torch.tanh(self.weight_W_sent(x))
		sent_attn = Hw.matmul(self.weight_proj_sent)
		sent_attn_norm = self.softmax_sent(sent_attn)
		# get sentence representation as weighted avg
		doc_encoding = x.mul(sent_attn_norm).sum(1)

		return doc_encoding, sent_attn_norm


class GRULWAN(nn.Module):
	""""
	GRU with LabelWise Attention Network
	"""

	def __init__(self, sent_gru_hidden, word_gru_hidden, n_classes, bidirectional=True, dropout=0.1):
		super(GRULWAN, self).__init__()

		# self.batch_size = batch_size
		self.sent_gru_hidden = sent_gru_hidden
		self.n_classes = n_classes
		self.word_gru_hidden = word_gru_hidden
		self.bidirectional = bidirectional
		self.dropout = dropout

		word_gru_out = 2 * word_gru_hidden if bidirectional else word_gru_hidden
		sent_gru_out = 2 * sent_gru_hidden if bidirectional else sent_gru_hidden

		self.score_normalizer = np.sqrt(sent_gru_out)

		self.sent_gru = nn.GRU(word_gru_out, sent_gru_hidden, bidirectional=bidirectional)
		self.U = nn.Linear(sent_gru_out, n_classes)
		self.out = nn.Parameter(
			torch.Tensor(n_classes, sent_gru_out))  # nn.Parameter(torch.Tensor(2*sent_gru_hidden, n_classes))

		self.out.data.normal_(0, 1 / np.sqrt(sent_gru_out))
		self.softmax_sent = nn.Softmax(dim=1)

		# Regularization
		self.bn1 = nn.BatchNorm1d(word_gru_out)
		self.drop1 = nn.Dropout(dropout)

		self.bn2 = nn.BatchNorm1d(sent_gru_out)
		self.drop2 = nn.Dropout(dropout)

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
		H_expanded = H.repeat(1, 1, self.n_classes)

		V = (attention_expanded * H_expanded).view(N, B, self.n_classes, d_c).sum(dim=0)
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
        u = self.capsules(x).view(batch_size, self.num_capsules, self.out_channels, -1, 1)
        poses = squash_v1(u, axis=1)
        activations = torch.sqrt((poses ** 2).sum(1))
        return poses, activations

class FCCaps(nn.Module):
    def __init__(self, args, output_capsule_num, input_capsule_num, in_channels, out_channels):
        super(FCCaps, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_capsule_num = input_capsule_num
        self.output_capsule_num = output_capsule_num

        self.W1 = nn.Parameter(torch.FloatTensor(1, input_capsule_num, output_capsule_num, out_channels, in_channels))
        torch.nn.init.xavier_uniform_(self.W1)

        self.is_AKDE = args.is_AKDE
        self.sigmoid = nn.Sigmoid()


    def forward(self, x, y, labels):
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