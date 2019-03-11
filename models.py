"""Top-level model classes.

Author:
    Chris Chute (chute@stanford.edu)
"""

import layers
import torch
import torch.nn as nn


class BiDAF(nn.Module):
    """Baseline BiDAF model for SQuAD.

    Based on the paper:
    "Bidirectional Attention Flow for Machine Comprehension"
    by Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, Hannaneh Hajishirzi
    (https://arxiv.org/abs/1611.01603).

    Follows a high-level structure commonly found in SQuAD models:
        - Embedding layer: Embed word indices to get word vectors.
        - Encoder layer: Encode the embedded sequence.
        - Attention layer: Apply an attention mechanism to the encoded sequence.
        - Model encoder layer: Encode the sequence again.
        - Output layer: Simple layer (e.g., fc + softmax) to get final outputs.

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Number of features in the hidden state at each layer.
        drop_prob (float): Dropout probability.
    """
    def __init__(self, word_vectors, char_vectors, hidden_size, drop_prob=0.):
        super(BiDAF, self).__init__()
        self.embd_size = hidden_size
        self.d = self.embd_size * 2+1 # word_embedding + char_embedding + word_feature
        self.emb = layers.Embedding(word_vectors=word_vectors, char_vectors=char_vectors,
                                    hidden_size=self.embd_size,
                                    drop_prob=drop_prob)

        # layer size 需要改
        self.enc = layers.RNNEncoder(input_size=self.d,
                                     hidden_size=self.d,
                                     num_layers=1,
                                     drop_prob=drop_prob)

        self.att = layers.BiDAFAttention(hidden_size=2 * self.d,
                                         drop_prob=drop_prob)

        # self.selfMatch = layers.SelfMatcher(in_size = 8 * self.d,
        #                                  drop_prob=drop_prob)
        self.selfMatch = layers.StaticDotAttention(memory_size = 8 * self.d, 
                        input_size = 8 * self.d, attention_size = 8 * self.d,
                        batch_first=False, drop_prob=drop_prob)

        self.mod = layers.RNNEncoder(input_size=16 * self.d,
                                     hidden_size=self.d,
                                     num_layers=2,
                                     drop_prob=drop_prob)

        self.out = layers.BiDAFOutput(hidden_size=self.d,
                                      drop_prob=drop_prob)

    def forward(self, cc_idxs, qc_idxs, cw_idxs, qw_idxs,cwf):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        # char emb
        c_emb = self.emb(cc_idxs, cw_idxs)         # (batch_size, c_len, d)
        q_emb = self.emb(qc_idxs, qw_idxs)         # (batch_size, q_len, d)

        # word feature
        cwf = torch.unsqueeze(cwf, dim = 2)
        cwf = cwf.type(torch.cuda.FloatTensor)
        c_emb = torch.cat((c_emb, cwf), dim = 2)

        # qwf = torch.unsqueeze(qwf, dim = 2)
        # qwf = qwf.type(torch.cuda.FloatTensor)
        # q_emb = torch.cat((q_emb, qwf), dim = 2)
        s = q_emb.shape
        qf_emb = torch.zeros(s[0],s[1],1, device='cuda')
        q_emb = torch.cat((q_emb, qf_emb), dim = 2)
        assert c_emb.size(2) == self.d and q_emb.size(2) == self.d
        
        c_enc = self.enc(c_emb, c_len)    # (batch_size, c_len, 2 * d)
        q_enc = self.enc(q_emb, q_len)    # (batch_size, q_len, 2 * d)
        print("q_enc", q_enc.shape)
        assert c_enc.size(2) == 2 * self.d and q_enc.size(2) == 2 * self.d

        att = self.att(c_enc, q_enc,
                       c_mask, q_mask)    # (batch_size, c_len, 8 * d)
        print("att", att.shape)
        assert att.size(2) == 8 * self.d

        # selfMatch = self.selfMatch(att)
        selfMatch = self.selfMatch(att, att, c_mask)
        assert selfMatch.size(2) == 16 * self.d

        mod = self.mod(selfMatch, c_len)        # (batch_size, c_len, 2 * d)
        assert mod.size(2) == 2 * self.d

        out = self.out(att, mod, c_mask)  # 2 tensors, each (batch_size, c_len)

        return out
