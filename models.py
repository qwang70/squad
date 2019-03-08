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
    def __init__(self, word_vectors, char_vectors, hidden_size, drop_prob=0., enable_EM=True):
        super(BiDAF, self).__init__()
        self.embd_size = hidden_size
        self.d = self.embd_size * 2 # word_embedding + char_embedding
        self.enable_EM = enable_EM
        if enable_EM:
            self.d += 2                 # word_feature
        self.emb = layers.Embedding(word_vectors=word_vectors, char_vectors=char_vectors,
                                    hidden_size=self.embd_size,
                                    drop_prob=drop_prob)

        self.enc = layers.RNNEncoder(input_size=self.d,
                                     hidden_size=self.d,
                                     num_layers=1,
                                     drop_prob=drop_prob)

        self.att = layers.BiDAFAttention(hidden_size=2 * self.d,
                                         drop_prob=drop_prob)

        self.mod = layers.RNNEncoder(input_size=8 * self.d,
                                     hidden_size=self.d,
                                     num_layers=2,
                                     drop_prob=drop_prob)

        self.out = layers.BiDAFOutput(hidden_size=self.d,
                                      drop_prob=drop_prob)

    def forward(self, cc_idxs, qc_idxs, cw_idxs, qw_idxs, cwf=None, lemma_indicators=None):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        # char emb
        c_emb = self.emb(cc_idxs, cw_idxs)         # (batch_size, c_len, d)
        q_emb = self.emb(qc_idxs, qw_idxs)         # (batch_size, q_len, d)

        if self.enable_EM:
            # word feature
            cwf = torch.unsqueeze(cwf, dim = 2)
            cwf = cwf.float()
            lemma_indicators = torch.unsqueeze(lemma_indicators, dim = 2)
            lemma_indicators = lemma_indicators.float()
            c_emb = torch.cat((c_emb, cwf, lemma_indicators), dim = 2)

            s = q_emb.shape
            # 0 embedding for exact match and indicators
            # qf_emb = torch.zeros(s[0],s[1],2, device=q_emb.device)
            # -1 embedding for exact match and indicators
            qf_emb = torch.ones(s[0],s[1],2, device=q_emb.device)*-1
            q_emb = torch.cat((q_emb, qf_emb), dim = 2)
        assert c_emb.size(2) == self.d and q_emb.size(2) == self.d
        
        c_enc = self.enc(c_emb, c_len)    # (batch_size, c_len, 2 * d)
        q_enc = self.enc(q_emb, q_len)    # (batch_size, q_len, 2 * d)
        assert c_enc.size(2) == 2 * self.d and q_enc.size(2) == 2 * self.d

        att = self.att(c_enc, q_enc,
                       c_mask, q_mask)    # (batch_size, c_len, 8 * d)
        assert att.size(2) == 8 * self.d

        mod = self.mod(att, c_len)        # (batch_size, c_len, 2 * d)
        assert mod.size(2) == 2 * self.d

        out = self.out(att, mod, c_mask)  # 2 tensors, each (batch_size, c_len)

        return out
