#!/usr/bin/env python
#coding:utf8
from .BasicModule import BasicModule
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import attention.transformer.Constants as Constants
from attention.transformer.Modules import BottleLinear as Linear
from attention.transformer.Layers import EncoderLayer, DecoderLayer

def position_encoding_init(n_position, d_pos_vec):
    ''' Init the sinusoid position encoding table '''

    # keep dim 0 for padding token position encoding zero vector
    position_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / d_pos_vec) for j in range(d_pos_vec)]
        if pos != 0 else np.zeros(d_pos_vec) for pos in range(n_position)])

    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2]) # dim 2i
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2]) # dim 2i+1
    return torch.from_numpy(position_enc).type(torch.FloatTensor)

def get_attn_padding_mask(seq_q, seq_k):
    ''' Indicate the padding-related part to mask '''
    assert seq_q.dim() == 2 and seq_k.dim() == 2
    mb_size, len_q = seq_q.size()
    mb_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(Constants.PAD).unsqueeze(1)   # bx1xsk
    pad_attn_mask = pad_attn_mask.expand(mb_size, len_q, len_k) # bxsqxsk
    return pad_attn_mask

def get_attn_subsequent_mask(seq):
    ''' Get an attention mask to avoid using the subsequent info.'''
    assert seq.dim() == 2
    attn_shape = (seq.size(0), seq.size(1), seq.size(1))
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    subsequent_mask = torch.from_numpy(subsequent_mask)
    if seq.is_cuda:
        subsequent_mask = subsequent_mask.cuda()
    return subsequent_mask

class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self,args, n_layers=6, n_head=4, d_k=25, d_v=25,
            d_word_vec=100, d_model=100, d_inner_hid=200, dropout=0.1, ):

        super(Encoder, self).__init__()

        self.d_model = d_model
        self.src_word_emb = nn.Embedding(args.embed_num, args.embed_dim)
        #print(args.embed_num)
        self.position_enc = nn.Embedding(100, d_word_vec, padding_idx=Constants.PAD)
        self.position_enc.weight.data = position_encoding_init(100, d_word_vec)
        self.position_enc.weight.requires_grad = False
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner_hid, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, src_seq, return_attns=False):
        # Word embedding look up
        #print(src_seq)
        enc_input = self.src_word_emb(src_seq)
        #print(enc_input.shape)
        src_pos = (torch.arange(0,enc_input.shape[1]).view(1,-1).expand(enc_input.shape[0],enc_input.shape[1])).type(torch.LongTensor).cuda()
        #print(src_pos.shape)
        # Position Encoding addition
        enc_input += self.position_enc(src_pos)
        if return_attns:
            enc_slf_attns = []

        enc_output = enc_input
        enc_slf_attn_mask = get_attn_padding_mask(src_seq, src_seq)
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output, slf_attn_mask=enc_slf_attn_mask)
            if return_attns:
                enc_slf_attns += [enc_slf_attn]
        #print("enc_output",enc_output.shape)
        if return_attns:
            return enc_output, enc_slf_attns
        else:
            return enc_output
        
class RNN(BasicModule):
    def __init__(self, args, embed=None):
        super(RNN,self).__init__()
        self.model_name = 'RNN'
        self.args = args

        V = args.embed_num
        D = args.embed_dim
        H = args.hidden_size
        S = args.seg_num
        P_V = args.pos_num
        P_D = args.pos_dim
        self.abs_pos_embed = nn.Embedding(P_V,P_D)
        self.rel_pos_embed = nn.Embedding(S,P_D)
        self.embed = nn.Embedding(V,D,padding_idx=0)
        self.encoder = Encoder(self.args)
        self.H = H
        if embed is not None:
            self.embed.weight.data.copy_(embed)

        self.word_RNN = nn.GRU(
                        input_size = D,
                        hidden_size = H,
                        batch_first = True,
                        bidirectional = True
                        )
        self.sent_RNN = nn.GRU(
                        input_size = H,
                        hidden_size = H,
                        batch_first = True,
                        bidirectional = True
                        )
        self.fc = nn.Linear(2*H,2*H)

        # Parameters of Classification Layer
        self.content = nn.Linear(2*H,1,bias=False)
        self.salience = nn.Bilinear(2*H,2*H,1,bias=False)
        self.novelty = nn.Bilinear(2*H,2*H,1,bias=False)
        self.abs_pos = nn.Linear(P_D,1,bias=False)
        self.rel_pos = nn.Linear(P_D,1,bias=False)
        self.bias = nn.Parameter(torch.FloatTensor(1).uniform_(-0.1,0.1))

    def max_pool1d(self,x,seq_lens):
        # x:[N,L,O_in]
        out = []
        #print(x.shape)
        for index,t in enumerate(x):
            #print(t.shape)
            t = t[:seq_lens[index],:]
            t = torch.t(t).unsqueeze(0)
            out.append(F.max_pool1d(t,t.size(2)))

        out = torch.cat(out).squeeze(2)
        return out
    def avg_pool1d(self,x,seq_lens):
        # x:[N,L,O_in]
        out = []
        for index,t in enumerate(x):
            t = t[:seq_lens[index],:]
            t = torch.t(t).unsqueeze(0)
            out.append(F.avg_pool1d(t,t.size(2)))

        out = torch.cat(out).squeeze(2)
        return out

    def forward(self,x,doc_lens):
        
        #print("inside forward")
        sent_lens = torch.sum(torch.sign(x),dim=1).data
        #x = self.embed(x)                                                      # (N,L,D)
        # word level GRU
        #H = self.args.hidden_size
        #x = self.word_RNN(x)[0]                                                 # (N,2*H,L)
        #print(x.shape)
        enc_output = self.encoder(x)
        #print("encoder final output",((enc_output.shape)))
        #word_out = self.avg_pool1d(x,sent_lens)
        #print(x)
        #print(len(sent_lens)
        word_out = self.max_pool1d(enc_output,sent_lens)
        #print(word_out.shape)
        # make sent features(pad with zeros)
        x = self.pad_doc(word_out,doc_lens)
        #print(x.shape)
        # sent level GRU
        sent_out = self.sent_RNN(x)[0]                                           # (B,max_doc_len,2*H)
        #docs = self.avg_pool1d(sent_out,doc_lens)                               # (B,2*H)
        docs = self.max_pool1d(sent_out,doc_lens)                                # (B,2*H)
        probs = []
        for index,doc_len in enumerate(doc_lens):
            #print('Index',index)
            valid_hidden = sent_out[index,:doc_len,:]                            # (doc_len,2*H)
            doc = F.tanh(self.fc(docs[index])).unsqueeze(0)
            s = Variable(torch.zeros(1,2*self.H)).cuda()
            for position, h in enumerate(valid_hidden):
                h = h.view(1, -1)                                                # (1,2*H)
                # get position embeddings
                abs_index = Variable(torch.LongTensor([[position]])).cuda()
                abs_features = self.abs_pos_embed(abs_index).squeeze(0)

                rel_index = int(round((position + 1) * 9.0 / doc_len))
                rel_index = Variable(torch.LongTensor([[rel_index]])).cuda()
                rel_features = self.rel_pos_embed(rel_index).squeeze(0)

                # classification layer
                content = self.content(h)
                salience = self.salience(h,doc)
                novelty = -1 * self.novelty(h,F.tanh(s))
                abs_p = self.abs_pos(abs_features)
                rel_p = self.rel_pos(rel_features)
                prob = F.sigmoid(content + salience + novelty + abs_p + rel_p + self.bias)
                s = s + torch.mm(prob,h)
                probs.append(prob)
        return torch.cat(probs).squeeze()
