import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.modules.normalization import LayerNorm
from functools import partial

""" This code is a slightly modified version of The Annotated Transformer."""


class TransformerModel(nn.Module):
    def __init__(self, encoder, src_embed, linear):
        super().__init__()
        self.encoder = encoder
        self.src_embed = src_embed
        self.linear = linear

    def forward(self, src, src_mask):
        output = F.relu(self.linear(
            self.encoder(self.src_embed(src), src_mask)))
        return output


def clones(module, N):
    """
    Produce N identical layers
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class TransformerEncoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn"
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super().__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """
        Apply residual connection to any sublayer with the same size.
        """
        return x + self.dropout(sublayer(self.norm(x)))


class TransformerEncoderLayer(nn.Module):
    """
    Encoder is made up of self-attn and feed foward
    """

    def __init__(self, size, self_attn, feed_forward, dropout):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


def attention(query, key, value, mask=None, dropout=0.0):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    p_attn = F.dropout(p_attn, p=dropout)
    return torch.matmul(p_attn, value), p_attn


# Performer code from performer-pytorch library
def generalized_kernel(data, projection_matrix, kernel_fn=nn.ReLU(), kernel_epsilon=0.001):
    b, h, *_ = data.shape
    if projection_matrix is None:
        return kernel_fn(data) + kernel_epsilon
    projection = projection_matrix.repeat(b, h, 1, 1)
    projection = projection.type_as(data)

    data_dash = torch.einsum('...id, ...jd->...ij', data, projection)
    data_prime = kernel_fn(data_dash) + kernel_epsilon
    return data_prime.type_as(data)


def orthogonal_matrix_chunk(cols):
    unstructured_block = torch.randn((cols, cols))
    q, _ = torch.linalg.qr(unstructured_block.cpu(), 'reduced')
    return q.t()


def gaussian_orthogonal_random_matrix(nb_rows, nb_columns, scaling=0):
    nb_full_blocks = int(nb_rows / nb_columns)
    block_list = []
    for _ in range(nb_full_blocks):
        q = orthogonal_matrix_chunk(nb_columns)
        block_list.append(q)

    remaining_rows = nb_rows - nb_full_blocks * nb_columns
    if remaining_rows > 0:
        q = orthogonal_matrix_chunk(nb_columns)
        block_list.append(q[:remaining_rows])

    final_matrix = torch.cat(block_list)

    if scaling == 0:
        multiplier = torch.randn((nb_rows, nb_columns)).norm(dim = 1)
    elif scaling == 1:
        multiplier = math.sqrt((float(nb_columns))) * torch.ones((nb_rows, ))
    else:
        raise ValueError(f"Invalid Scaling {scaling}")

    return torch.diag(multiplier) @ final_matrix


def fast_attention(q, k, v, mask=None, dropout=0.1):
    k_cumsum = k.sum(dim = -2)
    D_inv = 1. / torch.einsum('...nd,...d->...n', q, k_cumsum.type_as(q))
    context = torch.einsum('...nd,...ne->...de', k, v)
    out = torch.einsum('...de,...nd,...n->...ne', context, q, D_inv)
    if mask is not None:
        out  = out.masked_fill(mask == 0, -1e9)
    out = F.dropout(out, p=dropout)
    return out, context


class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1, fast_attention=False):
        """
        Take in model size and number of heads.
        """
        super().__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.p = dropout
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.fast_attention = fast_attention

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]
        if self.fast_attention:
            projection_matrix = gaussian_orthogonal_random_matrix(nb_rows=self.d_k, nb_columns=self.h, scaling=0)
            create_kernel = partial(generalized_kernel, projection_matrix=projection_matrix)
            query, key = map(create_kernel, (query, key))
        # 2) Apply attention on all the projected vectors in batch
        if self.fast_attention:
            x, self.attn = fast_attention(query, key, value, mask, dropout=self.p)
        else:
            x, self.attn = attention(query, key, value, mask=mask, dropout=self.p)
        # 3) "Concat" using a view and apply a final linear
        x = x.transpose(1, 2).contiguous().view(
            nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        # Torch linears have a "b" by default.
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class PositionalEncoding(nn.Module):
    """ Implement the PE function. """

    def __init__(self, d_model, dropout, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2)
                             * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(div_term * position)
        pe[:, 1::2] = torch.cos(div_term * position)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1), :], requires_grad=False)
        return self.dropout(x)


class Encoder(nn.Module):
    def __init__(self, in_seq_len, out_seq_len, d_model, dropout=0.1):
        super().__init__()
        self.in_seq_len = in_seq_len
        self.out_seq_len = out_seq_len
        self.d_model = d_model
        self.input_dims = self.in_seq_len * self.d_model
        self.output_dims = self.out_seq_len * self.d_model
        self.dims_1 = (self.input_dims - self.output_dims) // 4 * 3
        self.dims_2 = (self.input_dims - self.output_dims) // 4 * 2

        linear1 = nn.Linear(self.input_dims, self.dims_1)
        linear2 = nn.Linear(self.dims_1, self.dims_2)
        linear3 = nn.Linear(self.dims_2, self.output_dims)

        self.flatten = nn.Flatten()
        self.linears = nn.ModuleList([linear1, linear2, linear3])
        self.dropout = dropout
        self.unflatten = nn.Unflatten(1, (self.out_seq_len, self.d_model))

    def forward(self, x):
        x = self.flatten(x)
        for i, l in enumerate(self.linears):
            x = F.relu(l(x))
            x = nn.Dropout(p=self.dropout)(x)
        x = self.unflatten(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_seq_len, out_seq_len, d_model, dropout=0.1):
        super().__init__()
        self.in_seq_len = in_seq_len
        self.out_seq_len = out_seq_len
        self.d_model = d_model
        self.input_dims = self.in_seq_len * self.d_model
        self.output_dims =  self.out_seq_len * self.d_model
        self.dims_1 = (self.input_dims - self.output_dims) // 4 * 3
        self.dims_2 = (self.input_dims - self.output_dims) // 4 * 2

        linear1 = nn.Linear(self.output_dims, self.dims_2)
        linear2 = nn.Linear(self.dims_2, self.dims_1)
        linear3 = nn.Linear(self.dims_1, self.input_dims)

        self.linears = nn.ModuleList([linear1, linear2, linear3])
        self.dropout = dropout
        self.flatten = nn.Flatten()
        self.unflatten = nn.Unflatten(1, (self.in_seq_len, self.d_model))

    def forward(self, x):
        x = self.flatten(x)
        for i, l in enumerate(self.linears):
            x = F.relu(l(x))
            x = nn.Dropout(p=self.dropout)(x)
        x = self.unflatten(x)
        return x


class Autoencoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input):
        output = self.decoder(self.encoder(input))
        return output


def make_transformer_model(N, d_model, l_win, d_ff=0, h=8, dropout=0.1, fast_attention=False):
    if (d_ff == 0):
        d_ff = d_model * 4
    c = copy.deepcopy
    attn = MultiHeadAttention(h, d_model, dropout, fast_attention=fast_attention)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout, l_win)
    final_linear = nn.Linear(d_model, d_model)
    model = TransformerModel(
        TransformerEncoder(TransformerEncoderLayer(
            d_model, c(attn), c(ff), dropout), N),
        nn.Sequential(position),
        final_linear
    )

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


make_performer_model = partial(make_transformer_model, fast_attention=True)


def make_autoencoder_model(in_seq_len, out_seq_len, d_model, dropout=0.1):
    encoder = Encoder(in_seq_len, out_seq_len, d_model, dropout)
    decoder = Decoder(in_seq_len, out_seq_len, d_model, dropout)
    model = Autoencoder(encoder, decoder)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model
