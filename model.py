import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
# from torch.nn.modules.activation import MultiheadAttention

from torchnlp.encoders.text import stack_and_pad_tensors, pad_tensor
# ## Word attention model with bias

def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)
    return output


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)

        # perform linear operation and split into h heads

        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous() \
            .view(bs, -1, self.d_model)

        output = self.out(concat)

        return output




class AttentionWordRNN(nn.Module):
    def __init__(self, num_tokens, embed_size, word_gru_hidden, bidirectional= True, dropout=0.1):
        
        super(AttentionWordRNN, self).__init__()
        self.num_tokens = num_tokens
        self.embed_size = embed_size
        self.word_gru_hidden = word_gru_hidden
        self.bidirectional = bidirectional
        self.dropout = dropout

        word_gru_out = 2*word_gru_hidden if bidirectional else word_gru_hidden
        
        self.lookup = nn.Embedding(num_tokens, embed_size)
        self.word_gru = nn.GRU(embed_size, word_gru_hidden, bidirectional= bidirectional)
        self.weight_W_word = nn.Linear(word_gru_out,word_gru_out)
        self.weight_proj_word = nn.Parameter(torch.Tensor(word_gru_out, 1))

            
        self.softmax_word = nn.Softmax(dim=0)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.weight_proj_word.data.normal_(0,1/np.sqrt(word_gru_out))

    def forward(self, x):

        # embeddings
        y = self.lookup(x)
        # word level gru
        y, _ = self.word_gru(self.drop1(y))
        # compute attention
        Hw = torch.tanh(self.weight_W_word(y))
        word_attn = self.drop2(Hw.matmul(self.weight_proj_word))
        word_attn_norm = self.softmax_word(word_attn)
        # get sentence representation as weighted avg
        sen_encoding = y.mul(word_attn_norm).sum(0)

        return sen_encoding, word_attn_norm


class AttentionSentRNN(nn.Module):

    def __init__(self, sent_gru_hidden, word_gru_hidden,  bidirectional=True):

        super(AttentionSentRNN, self).__init__()

        # self.batch_size = batch_size
        self.word_gru_hidden = word_gru_hidden
        self.sent_gru_hidden = sent_gru_hidden
        self.bidirectional = bidirectional

        word_gru_out = 2 * word_gru_hidden if bidirectional else word_gru_hidden
        sent_gru_out = 2 * sent_gru_hidden if bidirectional else sent_gru_hidden

        self.sent_gru = nn.GRU(word_gru_out, sent_gru_hidden, bidirectional=bidirectional)
        self.weight_W_sent = nn.Linear(sent_gru_out, 2 * sent_gru_hidden)
        self.weight_proj_sent = nn.Parameter(torch.Tensor(sent_gru_out, 1))

        self.softmax_sent = nn.Softmax(dim=1)
        self.weight_proj_sent.data.normal_(0, 0.1)

    def forward(self, x):

        # word level gru
        x, _ = self.sent_gru(x)
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
    
    def __init__(self, sent_gru_hidden, word_gru_hidden, n_classes, bidirectional= True, dropout = 0.1):
        
        super(GRULWAN, self).__init__()
        
        # self.batch_size = batch_size
        self.sent_gru_hidden = sent_gru_hidden
        self.n_classes = n_classes
        self.word_gru_hidden = word_gru_hidden
        self.bidirectional = bidirectional
        self.dropout = dropout

        word_gru_out = 2 * word_gru_hidden if bidirectional else word_gru_hidden
        sent_gru_out = 2 * sent_gru_hidden if bidirectional else sent_gru_hidden


        self.sent_gru = nn.GRU(word_gru_out, sent_gru_hidden, bidirectional= bidirectional)
        self.U = nn.Linear(sent_gru_out, n_classes)
        self.out = nn.Parameter(torch.Tensor(n_classes, sent_gru_out)) # nn.Parameter(torch.Tensor(2*sent_gru_hidden, n_classes))

        self.out.data.normal_(0, 1/np.sqrt(sent_gru_out))
        self.softmax_sent = nn.Softmax(dim=1)

        # Regularization
        self.bn1 = nn.BatchNorm1d(word_gru_out)
        self.drop1 = nn.Dropout(dropout)

        self.bn2 = nn.BatchNorm1d(sent_gru_out)
        self.drop2 = nn.Dropout(dropout)
        
        
    def forward(self, word_attention_vectors):

        B, N, d_c = word_attention_vectors.size()

        word_attention_vectors = self.drop1(self.bn1(word_attention_vectors.permute(0,2,1))).permute(0,2,1)
        output_sent, _ = self.sent_gru(word_attention_vectors)

        H = output_sent.permute(1,0,2)
        # Get labelwise attention scores per document
        # A: [B, N, L] -> softmax-normalized scores per sentence per label
        A = self.softmax_sent(self.U(H))
        # Get labelwise representations of doc
        attention_expanded = torch.repeat_interleave(A, d_c, dim=2)
        H_expanded = H.repeat(1,1,self.n_classes)

        V = (attention_expanded * H_expanded).view(N, B, self.n_classes, d_c).sum(dim=0) #TODO: check here
        # V = (H.contiguous().view(-1) @ test.contiguous().view(-1, self.n_classes))
        V = self.drop2(self.bn2(V.permute(0,2,1))).permute(0,2,1)

        y = V.mul(self.out)
        y = y.sum(dim=2)

        return y, A


class GRULWAN1(nn.Module):
    """"
    GRU with LabelWise Attention Network
    """

    def __init__(self, sent_gru_hidden, word_gru_hidden, n_classes, bidirectional=True):

        super(GRULWAN1, self).__init__()

        # self.batch_size = batch_size
        self.sent_gru_hidden = sent_gru_hidden
        self.n_classes = n_classes
        self.word_gru_hidden = word_gru_hidden
        self.bidirectional = bidirectional

        self.multi_att = MultiHeadAttention(n_classes, 2*sent_gru_hidden)

        if bidirectional == True:
            self.sent_gru = nn.GRU(2 * word_gru_hidden, sent_gru_hidden, bidirectional=True)
            self.U = nn.Linear(2 * sent_gru_hidden, n_classes)
            self.out = nn.Linear(2 * sent_gru_hidden,
                                 n_classes)  # nn.Parameter(torch.Tensor(2*sent_gru_hidden, n_classes))
        else:
            self.sent_gru = nn.GRU(word_gru_hidden, sent_gru_hidden, bidirectional=False, dropout=0.2)
            self.U = nn.Linear(sent_gru_hidden, n_classes)
            self.out = nn.Linear(sent_gru_hidden, n_classes)

        self.softmax_sent = nn.Softmax(dim=1)

    def forward(self, word_attention_vectors):

        # B, N, d_c = word_attention_vectors.size()
        output_sent, _ = self.sent_gru(word_attention_vectors)
        # output_sent = output_sent.permute(1,0,2)


        attn_out, attn_weight = self.multi_att(output_sent, output_sent, output_sent)

         # = output_sent.permute(1, 0, 2)
        doc_enc = attn_out.sum(dim=1)

        y = self.out(doc_enc)
        return y, attn_weight





class HAN(nn.Module):

    def __init__(self, num_tokens, embed_size, sent_gru_hidden, word_gru_hidden, num_classes, bidirectional=True, dropout=0.1, **kwargs):
        super(HAN, self).__init__()

        # self.batch_size = batch_size
        self.sent_gru_hidden = sent_gru_hidden
        self.n_classes = num_classes
        self.word_gru_hidden = word_gru_hidden
        self.bidirectional = bidirectional

        sent_gru_out = 2*sent_gru_hidden if bidirectional else sent_gru_hidden

        self.sent_encoder = AttentionWordRNN(num_tokens, embed_size, word_gru_hidden, bidirectional)
        self.doc_encoder = AttentionSentRNN(sent_gru_hidden, word_gru_hidden, bidirectional)

        self.out = nn.Linear(sent_gru_out, num_classes)
        self.bn = nn.BatchNorm1d(sent_gru_out)
        self.drop = nn.Dropout(dropout)

    def set_embedding(self, embed_table):
        self.sent_encoder.lookup.load_state_dict({'weight': torch.tensor(embed_table)})

    def forward(self, sents, sents_len, doc_lens):
        sen_encodings, word_attn_weight = self.sent_encoder(sents)

        sen_encodings = sen_encodings.split(split_size=doc_lens)
        # stack and pad
        sen_encodings, _ = stack_and_pad_tensors(sen_encodings)  #
        # get predictions
        doc_encoding, sent_attn_weight = self.doc_encoder(sen_encodings)

        doc_encoding = self.drop(self.bn(doc_encoding))

        y_pred = self.out(doc_encoding)

        return y_pred, word_attn_weight, sent_attn_weight #TODO: split out forward and predict fn for attn_weights?

class HGRULWAN(nn.Module):

        def __init__(self, num_tokens, embed_size, sent_gru_hidden, word_gru_hidden, num_classes, bidirectional= True, dropout=0.1, **kwargs):
            super(HGRULWAN, self).__init__()

            # self.batch_size = batch_size
            self.sent_gru_hidden = sent_gru_hidden
            self.num_classes = num_classes
            self.word_gru_hidden = word_gru_hidden
            self.bidirectional = bidirectional

            self.sent_encoder = AttentionWordRNN(num_tokens, embed_size, word_gru_hidden, bidirectional, dropout=dropout)
            self.doc_encoder = GRULWAN(sent_gru_hidden, word_gru_hidden, num_classes, bidirectional, dropout=dropout)

        def set_embedding(self, embed_table):
            self.sent_encoder.lookup.load_state_dict({'weight': torch.tensor(embed_table)})

        def forward(self, sents, sents_len, doc_lens):
            sen_encodings, word_attn_weight = self.sent_encoder(sents)

            sen_encodings = sen_encodings.split(split_size=doc_lens)
            # stack and pad
            sen_encodings, _ = stack_and_pad_tensors(sen_encodings) #
            # get predictions
            y_pred, sent_attn_weight = self.doc_encoder(sen_encodings)
            return y_pred, word_attn_weight, sent_attn_weight