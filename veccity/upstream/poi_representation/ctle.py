import math

import numpy as np
import torch
from torch import nn

from veccity.upstream.poi_representation.utils import weight_init
from veccity.upstream.abstract_model import AbstractModel


def gen_casual_mask(seq_len, include_self=True):
    """
    Generate a casual mask which prevents i-th output element from
    depending on any input elements from "the future".
    Note that for PyTorch Transformer model, sequence mask should be
    filled with -inf for the masked positions, and 0.0 else.

    :param seq_len: length of sequence.
    :return: a casual mask, shape (seq_len, seq_len)
    """
    if include_self:
        mask = 1 - torch.triu(torch.ones(seq_len, seq_len)).transpose(0, 1)
    else:
        mask = 1 - torch.tril(torch.ones(seq_len, seq_len)).transpose(0, 1)
    return mask.bool()


class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_len):
        super().__init__()
        pe = torch.zeros(max_len, embed_size).float()
        pe.requires_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, embed_size, 2).float() * -(math.log(10000.0) / embed_size)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x, **kwargs):
        return self.pe[:, :x.size(1)]


class TemporalEncoding(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.omega = nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, embed_size))).float(),
                                  requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(embed_size).float(), requires_grad=True)
        self.div_term = math.sqrt(1. / embed_size)

    def forward(self, x, **kwargs):
        timestamp = kwargs['timestamp']  # (batch, seq_len)
        time_encode = timestamp.unsqueeze(-1) * self.omega.reshape(1, 1, -1) + self.bias.reshape(1, 1, -1)
        time_encode = torch.cos(time_encode)
        return self.div_term * time_encode


class CTLEEmbedding(nn.Module):
    def __init__(self, encoding_layer, embed_size, num_vocab):
        super().__init__()
        self.embed_size = embed_size
        self.num_vocab = num_vocab
        self.encoding_layer = encoding_layer
        self.add_module('encoding', self.encoding_layer)

        self.token_embed = nn.Embedding(num_vocab + 2, embed_size, padding_idx=num_vocab)
        # self.token_embed.weight.data.uniform_(-0.5/embed_size, 0.5/embed_size)

    def forward(self, x, **kwargs):
        token_embed = self.token_embed(x)
        pos_embed = self.encoding_layer(x, **kwargs)
        return token_embed + pos_embed


class CTLE(AbstractModel):

    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        encoding_type = config.get('encoding_type', 'positional')
        num_layers = config.get('num_layers', 4)
        num_heads = config.get('cnum_heads', 8)
        detach = config.get('detach', False)
        ctle_objective = 'mlm'#config.get('objective', 'mlm')
        embed_size = config.get('embed_size', 128)
        init_param = config.get('init_param', False)
        hidden_size = embed_size * 4
        max_seq_len = data_feature.get('max_seq_len')
        num_loc = data_feature.get('num_loc')
        encoding_layer = PositionalEncoding(embed_size, max_seq_len)



        if encoding_type == 'temporal':
            encoding_layer = TemporalEncoding(embed_size)

        obj_models = [MaskedLM(embed_size, num_loc)]
        if ctle_objective == "mh":
            obj_models.append(MaskedHour(embed_size))
        self.obj_models = nn.ModuleList(obj_models)

        ctle_embedding = CTLEEmbedding(encoding_layer, embed_size, num_loc)
        self.embed_size = ctle_embedding.embed_size
        self.num_vocab = ctle_embedding.num_vocab

        self.embed = ctle_embedding
        self.add_module('embed', ctle_embedding)

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.embed_size, nhead=num_heads,
                                                   dim_feedforward=hidden_size, dropout=0.1)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers,
                                             norm=nn.LayerNorm(self.embed_size, eps=1e-6))
        self.detach = detach
        if init_param:
            self.apply(weight_init)

    def forward(self, x, **kwargs):
        """
        @param x: sequence of tokens, shape (batch, seq_len).
        """
        seq_len = x.size(1)

        src_key_padding_mask = (x == self.num_vocab)
        token_embed = self.embed(x, **kwargs)  # (batch_size, seq_len, embed_size)
        src_mask=None

        encoder_out = self.encoder(token_embed.transpose(0, 1), mask=src_mask,
                                   src_key_padding_mask=src_key_padding_mask).transpose(0,1)  # (batch_size, src_len, embed_size)
        
        return encoder_out

    def static_embed(self):
        return self.embed.token_embed.weight[:self.num_vocab].detach().cpu().numpy()
    
    def encode(self, inputs):
        seq=inputs['seq']

        ctle_out = self.forward(seq,timestamp=inputs['timestamp'])
        return ctle_out
        
    def calculate_loss(self, batch):
        origin_tokens, origin_hour, masked_tokens, src_t_batch, mask_index = batch
        ctle_out = self.forward(masked_tokens, timestamp=src_t_batch)  # (batch_size, src_len, embed_size)
        masked_out = ctle_out.reshape(-1, self.embed_size)[mask_index]  # (num_masked, embed_size)
        loss = 0.
        for obj_model in self.obj_models:
            loss += obj_model(masked_out, origin_tokens=origin_tokens, origin_hour=origin_hour)
        return loss


class MaskedLM(nn.Module):
    def __init__(self, input_size, vocab_size):
        super().__init__()
        self.linear = nn.Linear(input_size, vocab_size)
        self.dropout = nn.Dropout(0.1)
        self.loss_func = nn.CrossEntropyLoss()

        self.vocab_size = vocab_size

    def forward(self, x, **kwargs):
        """
        :param x: input sequence (batch, seq_len, embed_size).
        :param origin_tokens: original tokens, shape (batch, seq_len)
        :return: the loss value of MLM objective.
        """
        origin_tokens = kwargs['origin_tokens']
        origin_tokens = origin_tokens.reshape(-1)
        lm_pre = self.linear(self.dropout(x))  # (batch, seq_len, vocab_size)
        lm_pre = lm_pre.reshape(-1, self.vocab_size)  # (batch * seq_len, vocab_size)
        return self.loss_func(lm_pre, origin_tokens)


class MaskedHour(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.linear = nn.Linear(input_size, 24)
        self.dropout = nn.Dropout(0.1)
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, x, **kwargs):
        """
        @param x: input sequence (batch, seq_len, embed_size)
        @param original_hour: original hour indices, shape (batch, seq_len)
        @returns: the loss value of MH objective.
        """
        origin_hour = kwargs['origin_hour']
        origin_hour = origin_hour.reshape(-1)
        hour_pre = self.linear(self.dropout(x))
        hour_pre = hour_pre.reshape(-1, 24)
        return self.loss_func(hour_pre, origin_hour)


class MaskedWeekday(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.linear = nn.Linear(input_size, 7)
        self.dropout = nn.Dropout(0.1)
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, x, **kwargs):
        """
        @param x: input sequence (batch, seq_len, embed_size)
        @param original_hour: original hour indices, shape (batch, seq_len)
        @returns: the loss value of MH objective.
        """
        origin_weekday = kwargs['origin_weekday']
        origin_weekday = origin_weekday.reshape(-1)
        weekday_pre = self.linear(self.dropout(x))
        weekday_pre = weekday_pre.reshape(-1, 7)
        return self.loss_func(weekday_pre, origin_weekday)
