import torch
import torch.nn as nn
from torchnlp.encoders.text import stack_and_pad_tensors, pad_tensor
# ## Word attention model with bias
from layers import *
import fasttext
transpose = (lambda b: b.t_().squeeze(0).contiguous())

ULMFIT_OUT_SIZE = 400

class FastTextLearner:

    def __init__(self):
        self.model = None

    def train(self, train_path, dev_path=None, test_path=None, save_path = None, optimize_time=None, binary_classification=True,
              lr=0.1, epoch=1000,embed_size=300, K=5):
        # optimize_time given in seconds

        # train model
        if not optimize_time:
            self.model = fasttext.train_supervised(train_path, loss='ova' if binary_classification else 'hs', lr=lr, epoch=epoch, dim=embed_size)
        else: # optimize params..
            assert dev_path, "When FastText is optimized, a development set must also be given"
            self.model = fasttext.train_supervised(train_path, autotune_validation_file=dev_path, autotune_duration=optimize_time, loss='hs' if binary_classification else 'ova')
            fasttext.train_supervised()

        if test_path:
            N, p_k, r_k = self.model.test(test_path, K)
            print("FastText model attained P@{0}: {1:0.3f}, R@{0}: {2:0.3f}, F1@{0}: {3:0.3f}".format(K, p_k, r_k, (2*p_k*r_k)/(r_k+p_k)))

        # save if necessary
        if save_path:
            self.model.save_model(save_path) #TODO: quantization?



class HAN(nn.Module):

    def __init__(self, num_tokens, embed_size, word_hidden, sent_hidden, num_classes, bidirectional=True, dropout=0.1, word_encoder='GRU', sent_encoder='GRU',
                 num_layers_word = 1, num_layers_sen = 1, nhead_word = 4, nhead_sen = 4, ulmfit_pretrained_path = None, dropout_factor_ulmfit = 1.0,**kwargs):
        super(HAN, self).__init__()

        # self.batch_size = batch_size
        self.sent_gru_hidden = sent_hidden
        self.n_classes = num_classes
        self.word_hidden = word_hidden
        self.bidirectional = bidirectional
        self.word_contextualizer = word_encoder
        self.sent_contextualizer = sent_encoder

        if word_encoder.lower() == 'ulmfit':
            word_out = ULMFIT_OUT_SIZE # static ULMFiT value
        else:
            word_out = 2 * word_hidden if (bidirectional and word_encoder.lower() =='gru')  else word_hidden
        sent_out = 2 * sent_hidden if (bidirectional and sent_encoder.lower() =='gru')  else sent_hidden

        # self.sent_encoder = ULMFiTEncoder(kwargs['ulmfit_pretrained_path'], num_tokens)
        self.sent_encoder = AttentionWordEncoder(word_encoder, num_tokens, embed_size, word_hidden,
                                                 bidirectional=bidirectional,
                                                 num_layers=num_layers_word, nhead=nhead_word,
                                                 ulmfit_pretrained_path=ulmfit_pretrained_path,
                                                 dropout_factor_ulmfit=dropout_factor_ulmfit
                                                 )
        self.doc_encoder = AttentionSentEncoder(sent_encoder, sent_hidden, word_out,
                                                bidirectional=bidirectional,
                                                num_layers=num_layers_sen, nhead=nhead_sen)

        self.out = nn.Linear(sent_out, num_classes)
        self.bn = nn.BatchNorm1d(sent_out)
        self.drop = nn.Dropout(dropout)

    def set_embedding(self, embed_table):
        self.sent_encoder.lookup.load_state_dict({'weight': torch.tensor(embed_table)})

    def get_init_params(self):
        # returns the parameters needed to initialize the model as is
        params = self.sent_encoder.get_init_params()
        params.update(self.doc_encoder.get_init_params())

        params.update(
            {
                "model_name": "HAN",
            }
        )

        return params


    def forward(self, sents, sents_len, doc_lens):

        sen_encodings, word_attn_weight = self.sent_encoder(sents)

        sen_encodings = sen_encodings.split(split_size=doc_lens)
        # stack and pad
        sen_encodings, _ = stack_and_pad_tensors(sen_encodings)  #
        # get predictions
        doc_encoding, sent_attn_weight = self.doc_encoder(sen_encodings)

        doc_encoding = self.drop(self.bn(doc_encoding))

        y_pred = self.out(doc_encoding)

        return y_pred, word_attn_weight, sent_attn_weight, 0 # return 0 as reconstruction loss for caps nets


class HGRULWAN(nn.Module):

        def __init__(self, num_tokens, embed_size, sent_hidden, word_hidden, num_classes, word_encoder = 'gru', bidirectional= True, dropout=0.1,
                     ulmfit_pretrained_path=None, dropout_factor_ulmfit = 1.0, num_layers_word=1, nhead_word = 4, **kwargs):
            super(HGRULWAN, self).__init__()

            # self.batch_size = batch_size
            self.sent_gru_hidden = sent_hidden
            self.num_classes = num_classes
            self.word_gru_hidden = word_hidden
            self.bidirectional = bidirectional

            if word_encoder.lower() == 'ulmfit':
                word_out = ULMFIT_OUT_SIZE # static ULMFiT value
            else:
                word_out = 2 * word_hidden if (bidirectional and word_encoder.lower() == 'gru') else word_hidden

            # sent_out = 2 * sent_hidden if (bidirectional and sent_encoder.lower() == 'gru') else sent_hidden

            # self.sent_encoder = ULMFiTEncoder(kwargs['ulmfit_pretrained_path'], num_tokens)
            self.sent_encoder = AttentionWordEncoder(word_encoder, num_tokens, embed_size, word_hidden,
                                                     bidirectional=bidirectional,
                                                     num_layers=num_layers_word, nhead=nhead_word,
                                                     ulmfit_pretrained_path=ulmfit_pretrained_path,
                                                    dropout_factor_ulmfit=dropout_factor_ulmfit
                                                     )


            self.doc_encoder = GRUMultiHeadAtt(sent_hidden, word_out, nhead_doc=num_classes,
                                               bidirectional=bidirectional, dropout=dropout, aggregate_output=True)

        def set_embedding(self, embed_table):
            self.sent_encoder.lookup.load_state_dict({'weight': torch.tensor(embed_table)})

        def get_init_params(self):
            # returns the parameters needed to initialize the model as is
            params = self.sent_encoder.get_init_params()
            params.update(self.doc_encoder.get_init_params())

            params.update(
                {
                    "model_name": "HGRULWAN",
                }
            )

            return params

        def forward(self, sents, sents_len, doc_lens):

            sen_encodings, word_attn_weight = self.sent_encoder(sents)

            sen_encodings = sen_encodings.split(split_size=doc_lens)
            # stack and pad
            sen_encodings, _ = stack_and_pad_tensors(sen_encodings) #
            # get predictions
            y_pred, sent_attn_weight = self.doc_encoder(sen_encodings)
            return y_pred, word_attn_weight, sent_attn_weight, 0 # return 0 as reconstruction loss for caps nets


class HCapsNet(nn.Module):
    def __init__(self, num_tokens, embed_size, word_hidden, sent_hidden, num_classes, bidirectional=True, dropout=0.1, word_encoder='GRU', sent_encoder='GRU',
                 num_layers_word = 1, num_layers_sen = 1, nhead_word = 4, nhead_sen = 4, dim_caps=16, num_caps = 25, num_compressed_caps = 100,
                 dropout_caps = 0.2, lambda_reg_caps = 0.0005, ulmfit_pretrained_path=None, dropout_factor_ulmfit = 1.0,**kwargs):
        super(HCapsNet, self).__init__()

        # self.batch_size = batch_size
        self.num_tokens = num_tokens
        self.embed_size = embed_size
        self.sent_gru_hidden = sent_hidden
        self.n_classes = num_classes
        self.word_hidden = word_hidden
        self.bidirectional = bidirectional
        self.word_encoder = word_encoder
        self.sent_encoder = sent_encoder
        self.dropout = dropout
        self.lambda_reg_caps = lambda_reg_caps

        if word_encoder.lower() == 'ulmfit':
            word_out = ULMFIT_OUT_SIZE  # static ULMFiT value
        else:
            word_out = 2 * word_hidden if (bidirectional and word_encoder.lower() == 'gru') else word_hidden
        sent_out = 2 * sent_hidden if (bidirectional and sent_encoder.lower() =='gru')  else sent_hidden

        self.sent_encoder = AttentionWordEncoder(word_encoder, num_tokens, embed_size, word_hidden,
                                                 bidirectional=bidirectional,
                                                 num_layers=num_layers_word, nhead=nhead_word,
                                                 ulmfit_pretrained_path=ulmfit_pretrained_path,
                                                 dropout_factor_ulmfit=dropout_factor_ulmfit
                                                 )
        self.doc_encoder = AttentionSentEncoder(sent_encoder, sent_hidden, word_out,
                                                bidirectional=bidirectional,
                                                num_layers=num_layers_sen, nhead=nhead_sen)

        self.caps_classifier = CapsNet_Text(sent_out, 1, num_classes, dim_caps=dim_caps, num_caps=num_caps,
                                            num_compressed_caps=num_compressed_caps, dropout_caps = dropout_caps,
                                            lambda_reg_caps = lambda_reg_caps)
        # self.out = nn.Linear(sent_out, num_classes)
        # self.bn = nn.BatchNorm1d(sent_out)
        self.drop = nn.Dropout(dropout)


    def get_init_params(self):
        # returns the parameters needed to initialize the model as is
        params = self.sent_encoder.get_init_params()
        params.update(self.doc_encoder.get_init_params())

        params.update(
            {
                "model_name":"HCapsNet",
                "num_caps":self.caps_classifier.num_caps,
                "dim_caps":self.caps_classifier.dim_caps,
                "lambda_reg_caps":self.lambda_reg_caps,
                "num_compressed_caps":self.caps_classifier.num_compressed_caps,
            }
        )

        return params

    def set_embedding(self, embed_table):
        self.sent_encoder.lookup.load_state_dict({'weight': torch.tensor(embed_table)})


    def forward(self, sents, sents_len, doc_lens, encoding):

        sen_encodings, word_attn_weight = self.sent_encoder(sents)

        sen_encodings = sen_encodings.split(split_size=doc_lens)
        # stack and pad
        sen_encodings, _ = stack_and_pad_tensors(sen_encodings)  #
        # get predictions
        doc_encoding, sent_attn_weight = self.doc_encoder(sen_encodings)

        # doc_encoding = self.bn(doc_encoding).unsqueeze(1) #self.drop()
        doc_encoding = doc_encoding.unsqueeze(1)
        poses, activations = self.caps_classifier(doc_encoding)
        activations = activations.squeeze(2)

        #TODO TEMP: reconstuction
        rec_los = self.caps_classifier.reconstruction_loss(encoding, poses)

        return activations, word_attn_weight, sent_attn_weight, rec_los


class HCapsNetMultiHeadAtt(nn.Module):
    def __init__(self, num_tokens, embed_size, word_hidden, sent_hidden, num_classes, bidirectional=True, dropout=0.1, word_encoder='GRU', sent_encoder='GRU',
                 num_layers_word = 1, num_layers_sen = 1, nhead_word = 4, nhead_sen = 4, dim_caps=16, num_caps = 25, num_compressed_caps = 100, nhead_doc = 25,
                 dropout_caps=0.2, lambda_reg_caps = 0.0005, ulmfit_pretrained_path=None, dropout_factor_ulmfit=1.0, **kwargs):
        super(HCapsNetMultiHeadAtt, self).__init__()

        # self.batch_size = batch_size
        self.sent_gru_hidden = sent_hidden
        self.n_classes = num_classes
        self.word_hidden = word_hidden
        self.bidirectional = bidirectional
        self.word_contextualizer = word_encoder
        self.sent_contextualizer = sent_encoder
        self.lambda_reg_caps = lambda_reg_caps

        if word_encoder.lower() == 'ulmfit':
            word_out = ULMFIT_OUT_SIZE  # static ULMFiT value
        else:
            word_out = 2 * word_hidden if (bidirectional and word_encoder.lower() == 'gru') else word_hidden
        sent_out = 2 * sent_hidden if (bidirectional and sent_encoder.lower() =='gru')  else sent_hidden

        self.sent_encoder = AttentionWordEncoder(word_encoder, num_tokens, embed_size, word_hidden,
                                                 bidirectional=bidirectional,
                                                 num_layers=num_layers_word, nhead=nhead_word,
                                                 ulmfit_pretrained_path=ulmfit_pretrained_path,
                                                 dropout_factor_ulmfit=dropout_factor_ulmfit,
                                                 )
        self.doc_encoder = GRUMultiHeadAtt(sent_hidden, word_out, nhead_doc=nhead_doc,
                                           bidirectional=bidirectional, dropout=dropout, aggregate_output=False)

        self.caps_classifier = CapsNet_Text(sent_out, nhead_doc, num_classes,dim_caps=dim_caps, num_caps=num_caps
                                            , num_compressed_caps=num_compressed_caps, dropout_caps = dropout_caps,
                                            lambda_reg_caps = lambda_reg_caps)
        # self.out = nn.Linear(sent_out, num_classes)
        self.bn = nn.BatchNorm1d(sent_out)
        self.drop = nn.Dropout(dropout)

    def set_embedding(self, embed_table):
        self.sent_encoder.lookup.load_state_dict({'weight': torch.tensor(embed_table)})

    def get_init_params(self):
        # returns the parameters needed to initialize the model as is
        params = self.sent_encoder.get_init_params()
        params.update(self.doc_encoder.get_init_params())

        params.update(
            {
                "model_name":"HCapsNetMultiHeadAtt",
                "nhead_doc":self.caps_classifier.in_channels,
                "num_caps":self.caps_classifier.num_caps,
                "lambda_reg_caps":self.lambda_reg_caps,
                "dim_caps":self.caps_classifier.dim_caps,
                "num_compressed_caps":self.caps_classifier.num_compressed_caps,
            }
        )

        return params


    def forward(self, sents, sents_len, doc_lens, encoding):

        sen_encodings, word_attn_weight = self.sent_encoder(sents)

        # if self.word_contextualizer != self.sent_contextualizer:
        #     sen_encodings = sen_encodings.permute(1, 0, 2)

        sen_encodings = sen_encodings.split(split_size=doc_lens)
        # stack and pad
        sen_encodings, _ = stack_and_pad_tensors(sen_encodings)  #
        # get predictions
        doc_encoding, sent_attn_weight = self.doc_encoder(sen_encodings)

        doc_encoding = self.drop(self.bn(doc_encoding.permute(0,2,1))).permute(0,2,1)
        # doc_encoding = self.drop(self.bn(doc_encoding))

        poses, activations = self.caps_classifier(doc_encoding)
        activations = activations.squeeze(2)

        rec_loss = self.caps_classifier.reconstruction_loss(encoding, poses)

        return activations, word_attn_weight, sent_attn_weight, rec_loss