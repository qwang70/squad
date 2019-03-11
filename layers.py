"""Assortment of layers for use in models.py.

Author:
    Chris Chute (chute@stanford.edu)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ujson import load as json_load

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from util import masked_softmax


class Embedding(nn.Module):
    """Embedding layer used by BiDAF, without the character-level component.

    Word-level embeddings are further refined using a 2-layer Highway Encoder
    (see `HighwayEncoder` class for details).

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations
    """
    def __init__(self, word_vectors, char_vectors, hidden_size, drop_prob):
        super(Embedding, self).__init__()
        self.drop_prob = drop_prob
        self.char_embed = CharEmbedding(char_vectors=char_vectors,
                                        hidden_size=hidden_size,
                                        drop_prob=drop_prob) 
        self.word_embed = nn.Embedding.from_pretrained(word_vectors)
        self.proj = nn.Linear(word_vectors.size(1), hidden_size, bias=False)
        self.hwy = HighwayEncoder(2, 2 * hidden_size)
        self.embed_size = hidden_size

    def forward(self, c_idxs, w_idxs):
        # word_embedding
        word_emb = self.word_embed(w_idxs)   # (batch_size, seq_len, embed_size)
        word_emb = F.dropout(word_emb, self.drop_prob, self.training)
        word_emb = self.proj(word_emb)  # (batch_size, seq_len, embed_size)
        assert word_emb.shape == (w_idxs.size(0), w_idxs.size(1), self.embed_size)
        # char_embedding
        char_emb = self.char_embed(c_idxs) # (batch_size, seq_len, embed_size)
        assert char_emb.shape == (w_idxs.size(0), w_idxs.size(1), self.embed_size)
        emb = torch.cat((char_emb, word_emb), 2) # (batch_size, seq_len, 2 * embed_size)
        assert emb.shape == (w_idxs.size(0), w_idxs.size(1), 2*self.embed_size)
        emb = self.hwy(emb)   # (batch_size, seq_len, 2 * embed_size)
        assert emb.shape == (w_idxs.size(0), w_idxs.size(1), 2*self.embed_size)

        return emb


class HighwayEncoder(nn.Module):
    """Encode an input sequence using a highway network.

    Based on the paper:
    "Highway Networks"
    by Rupesh Kumar Srivastava, Klaus Greff, Jürgen Schmidhuber
    (https://arxiv.org/abs/1505.00387).

    Args:
        num_layers (int): Number of layers in the highway encoder.
        hidden_size (int): Size of hidden activations.
    """
    def __init__(self, num_layers, hidden_size):
        super(HighwayEncoder, self).__init__()
        self.transforms = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                         for _ in range(num_layers)])
        self.gates = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                    for _ in range(num_layers)])

    def forward(self, x):
        for gate, transform in zip(self.gates, self.transforms):
            # Shapes of g, t, and x are all (batch_size, seq_len, hidden_size)
            g = torch.sigmoid(gate(x))
            t = F.relu(transform(x))
            x = g * t + (1 - g) * x

        return x


class RNNEncoder(nn.Module):
    """General-purpose layer for encoding a sequence using a bidirectional RNN.

    Encoded output is the RNN's hidden state at each position, which
    has shape `(batch_size, seq_len, hidden_size * 2)`.

    Args:
        input_size (int): Size of a single timestep in the input.
        hidden_size (int): Size of the RNN hidden state.
        num_layers (int): Number of layers of RNN cells to use.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers,
                 drop_prob=0.):
        super(RNNEncoder, self).__init__()
        self.drop_prob = drop_prob
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True,
                           bidirectional=True,
                           dropout=drop_prob if num_layers > 1 else 0.)

    def forward(self, x, lengths):
        # Save original padded length for use by pad_packed_sequence
        orig_len = x.size(1)

        # Sort by length and pack sequence for RNN
        lengths, sort_idx = lengths.sort(0, descending=True)
        x = x[sort_idx]     # (batch_size, seq_len, input_size)
        x = pack_padded_sequence(x, lengths, batch_first=True)

        # Apply RNN
        x, _ = self.rnn(x)  # (batch_size, seq_len, 2 * hidden_size)

        # Unpack and reverse sort
        x, _ = pad_packed_sequence(x, batch_first=True, total_length=orig_len)
        _, unsort_idx = sort_idx.sort(0)
        x = x[unsort_idx]   # (batch_size, seq_len, 2 * hidden_size)

        # Apply dropout (RNN applies dropout after all but the last layer)
        x = F.dropout(x, self.drop_prob, self.training)

        return x


class BiDAFAttention(nn.Module):
    """Bidirectional attention originally used by BiDAF.

    Bidirectional attention computes attention in two directions:
    The context attends to the query and the query attends to the context.
    The output of this layer is the concatenation of [context, c2q_attention,
    context * c2q_attention, context * q2c_attention]. This concatenation allows
    the attention vector at each timestep, along with the embeddings from
    previous layers, to flow through the attention layer to the modeling layer.
    The output has shape (batch_size, context_len, 8 * hidden_size).

    Args:
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, drop_prob=0.1):
        super(BiDAFAttention, self).__init__()
        self.drop_prob = drop_prob
        self.c_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.q_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.cq_weight = nn.Parameter(torch.zeros(1, 1, hidden_size))
        for weight in (self.c_weight, self.q_weight, self.cq_weight):
            nn.init.xavier_uniform_(weight)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, c, q, c_mask, q_mask):
        batch_size, c_len, _ = c.size()
        q_len = q.size(1)
        s = self.get_similarity_matrix(c, q)        # (batch_size, c_len, q_len)
        c_mask = c_mask.view(batch_size, c_len, 1)  # (batch_size, c_len, 1)
        q_mask = q_mask.view(batch_size, 1, q_len)  # (batch_size, 1, q_len)
        s1 = masked_softmax(s, q_mask, dim=2)       # (batch_size, c_len, q_len)
        s2 = masked_softmax(s, c_mask, dim=1)       # (batch_size, c_len, q_len)

        # (bs, c_len, q_len) x (bs, q_len, hid_size) => (bs, c_len, hid_size)
        a = torch.bmm(s1, q)
        # (bs, c_len, c_len) x (bs, c_len, hid_size) => (bs, c_len, hid_size)
        b = torch.bmm(torch.bmm(s1, s2.transpose(1, 2)), c)

        x = torch.cat([c, a, c * a, c * b], dim=2)  # (bs, c_len, 4 * hid_size)

        return x

    def get_similarity_matrix(self, c, q):
        """Get the "similarity matrix" between context and query (using the
        terminology of the BiDAF paper).

        A naive implementation as described in BiDAF would concatenate the
        three vectors then project the result with a single weight matrix. This
        method is a more memory-efficient implementation of the same operation.

        See Also:
            Equation 1 in https://arxiv.org/abs/1611.01603
        """
        c_len, q_len = c.size(1), q.size(1)
        c = F.dropout(c, self.drop_prob, self.training)  # (bs, c_len, hid_size)
        q = F.dropout(q, self.drop_prob, self.training)  # (bs, q_len, hid_size)

        # Shapes: (batch_size, c_len, q_len)
        s0 = torch.matmul(c, self.c_weight).expand([-1, -1, q_len])
        s1 = torch.matmul(q, self.q_weight).transpose(1, 2)\
                                           .expand([-1, c_len, -1])
        s2 = torch.matmul(c * self.cq_weight, q.transpose(1, 2))
        s = s0 + s1 + s2 + self.bias

        return s


class BiDAFOutput(nn.Module):
    """Output layer used by BiDAF for question answering.

    Computes a linear transformation of the attention and modeling
    outputs, then takes the softmax of the result to get the start pointer.
    A bidirectional LSTM is then applied the modeling output to produce `mod_2`.
    A second linear+softmax of the attention output and `mod_2` is used
    to get the end pointer.

    Args:
        hidden_size (int): Hidden size used in the BiDAF model.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, drop_prob):
        super(BiDAFOutput, self).__init__()
        self.att_linear_1 = nn.Linear(8 * hidden_size, 1)
        self.mod_linear_1 = nn.Linear(2 * hidden_size, 1)

        self.rnn = RNNEncoder(input_size=2 * hidden_size,
                              hidden_size=hidden_size,
                              num_layers=1,
                              drop_prob=drop_prob)

        self.att_linear_2 = nn.Linear(8 * hidden_size, 1)
        self.mod_linear_2 = nn.Linear(2 * hidden_size, 1)

    def forward(self, att, mod, mask):
        # Shapes: (batch_size, seq_len, 1)
        logits_1 = self.att_linear_1(att) + self.mod_linear_1(mod)
        mod_2 = self.rnn(mod, mask.sum(-1))
        logits_2 = self.att_linear_2(att) + self.mod_linear_2(mod_2)

        # Shapes: (batch_size, seq_len)
        log_p1 = masked_softmax(logits_1.squeeze(), mask, log_softmax=True)
        log_p2 = masked_softmax(logits_2.squeeze(), mask, log_softmax=True)

        return log_p1, log_p2

# https://github.com/jojonki/BiDAF/blob/master/layers/char_embedding.py
class CharEmbedding(nn.Module):
    '''
     In : (N, sentence_len, word_len, vocab_size_c)
     Out: (N, sentence_len, c_embd_size)
     '''
    def __init__(self, char_vectors, hidden_size, drop_prob):
        super(CharEmbedding, self).__init__()
        self.hidden_size = hidden_size
        self.drop_prob = drop_prob
        self.embedding = nn.Embedding.from_pretrained(char_vectors, freeze=False)
        self.c_embd_size = char_vectors.size(1)
        self.kernel_size = 5
        self.conv = nn.Conv1d(self.c_embd_size, hidden_size, self.kernel_size, stride=1, padding=0)
        """
        filters = [[1,5]]
        self.conv = nn.ModuleList([nn.Conv2d(1, hidden_size, (f[0], f[1])) for f in filters])
        """

    def forward(self, x):
        """
        # x: (N, seq_len, word_len)
        input_shape = x.size()
        word_len = x.size(2)
        x = x.view(-1, word_len) # (N*seq_len, word_len)
        x = self.embedding(x) # (N*seq_len, word_len, c_embd_size)
        x = x.view(*input_shape, -1) # (N, seq_len, word_len, c_embd_size)
        x = x.sum(2) # (N, seq_len, c_embd_size)
        # CNN
        x = x.unsqueeze(1) # (N, Cin, seq_len, c_embd_size), insert Channnel-In dim
        # Conv2d
        #    Input : (N,Cin, Hin, Win )
        #    Output: (N,Cout,Hout,Wout)
        x = [F.relu(conv(x)) for conv in self.conv] # (N, Cout, seq_len, c_embd_size-filter_w+1). stride == 1
        # [(N,Cout,Hout,Wout) -> [(N,Cout,Hout*Wout)] * len(filter_heights)

        # [(N, seq_len, c_embd_size-filter_w+1, Cout)] * len(filter_heights)
        x = [xx.view((xx.size(0), xx.size(2), xx.size(3), xx.size(1))) for xx in x]

        # maxpool like
        # [(N, seq_len, Cout)] * len(filter_heights)
        x = [torch.sum(xx, 2) for xx in x]
        # (N, seq_len, Cout==word_embd_size)
        x = torch.cat(x, 1)
        x = F.dropout(x, self.drop_prob, self.training)
        return x
        """
        # x: (N, seq_len, word_len)
        input_shape = x.size()
        word_len = x.size(2)
        x = x.view(-1, word_len) # (N*seq_len, word_len)
        assert x.shape == (input_shape[0]*input_shape[1], input_shape[2])
        x = self.embedding(x) # (N*seq_len, word_len, c_embd_size)
        assert x.shape == (input_shape[0]*input_shape[1], input_shape[2], self.c_embd_size)
        x = x.transpose(1,2) # (N * seq_len, c_embd_size, word_len)
        assert x.shape == (input_shape[0]*input_shape[1], self.c_embd_size, input_shape[2])
        # Dropout before conv
        # https://github.com/allenai/bi-att-flow/blob/master/my/tensorflow/nn.py#L163
        x = F.dropout(x, self.drop_prob, self.training)
        x = self.conv(x) #(N * seq_len, hidden, 64/filter_size)
        x = F.relu(x)
        # MaxPool simply takes the maximum across the second dimension
        # position 1 is the batch channel
        x = torch.max(x, 2)[0]
        x = x.view(input_shape[0], input_shape[1], self.hidden_size)
        return x 

# https://github.com/hengruo/RNet-pytorch/blob/master/models.py
# Input is question-aware passage representation
# Output is self-attention question-aware passage representation
class SelfMatcher(nn.Module):
    def __init__(self, in_size, drop_prob):
        super(SelfMatcher, self).__init__()
        self.hidden_size = in_size
        self.in_size = in_size
        self.gru = nn.GRUCell(input_size=in_size, hidden_size=self.hidden_size)
        self.Wp = nn.Linear(self.in_size, self.hidden_size, bias=False)
        self.Wp_ = nn.Linear(self.in_size, self.hidden_size, bias=False)
        self.out_size = self.hidden_size
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, v):
        (batch_size, l, _) = v.size()
        v.permute([1,0,2])
        h = torch.randn(batch_size, self.hidden_size).to(v.device)
        V = torch.randn(batch_size, self.hidden_size, 1).to(v.device)
        hs = torch.zeros(l, batch_size, self.out_size).to(v.device)
        
        for i in range(l):
            Wpv = self.Wp(v[i])
            Wpv_ = self.Wp_(v)
            x = F.tanh(Wpv + Wpv_)
            x = x.permute([1, 0, 2])
            s = torch.bmm(x, V)
            s = torch.squeeze(s, 2)
            a = F.softmax(s, 1).unsqueeze(1)
            c = torch.bmm(a, v.permute([1, 0, 2])).squeeze()
            h = self.gru(c, h)
            hs[i] = h
            # logger.gpu_mem_log("SelfMatcher {:002d}".format(i), ['x', 'Wpv', 'Wpv_', 's', 'c', 'hs'], [x.data, Wpv.data, Wpv_.data, s.data, c.data, hs.data])
            del Wpv, Wpv_, x, s, a, c
        hs = self.dropout(hs)
        del h, v
        return hs

# https://github.com/matthew-z/R-net/blob/master/modules/pair_encoder/attentions.py
class StaticDotAttention(nn.Module):
    def __init__(self, memory_size, input_size, attention_size,  drop_prob=0.2):
        super(StaticDotAttention, self).__init__()
        self.input_linear = nn.Sequential(
            RNNDropout(drop_prob, batch_first=True),
            nn.Linear(input_size, attention_size, bias=False),
            nn.ReLU()
        )

        self.memory_linear = nn.Sequential(
            RNNDropout(drop_prob, batch_first=True),
            nn.Linear(memory_size, attention_size, bias=False),
            nn.ReLU()
        )
        self.attention_size = attention_size

    def forward(self, inputs, memory, memory_mask):
        # if not self.batch_first:
        #     print("transposing")
        #     inputs = inputs.transpose(0, 1)
        #     memory = memory.transpose(0, 1)
        #     memory_mask = memory_mask.transpose(0, 1)

        input_ = self.input_linear(inputs)
        memory_ = self.memory_linear(memory)

        logits = torch.bmm(input_, memory_.transpose(2, 1)) / (self.attention_size ** 0.5)

        memory_mask = memory_mask.unsqueeze(1).expand(-1, inputs.size(1), -1)
        score = masked_softmax(logits, memory_mask, dim=-1)

        context = torch.bmm(score, memory)
        new_input = torch.cat([context, inputs], dim=-1)

        # if not self.batch_first:
        #     return new_input.transpose(0, 1)
        return new_input

# https://github.com/matthew-z/R-net/blob/master/modules/dropout.py
class RNNDropout(nn.Module):
    def __init__(self, p, batch_first=False):
        super().__init__()
        self.dropout = nn.Dropout(p)
        self.batch_first = batch_first

    def forward(self, inputs):

        if not self.training:
            return inputs
        if self.batch_first:
            mask = inputs.new_ones(inputs.size(0), 1, inputs.size(2), requires_grad=False)
        else:
            mask = inputs.new_ones(1, inputs.size(1), inputs.size(2), requires_grad=False)
        return self.dropout(mask) * inputs