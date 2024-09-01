from itertools import zip_longest

import numpy as np
import torch
from torch import nn
from sklearn.utils import shuffle
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence

from veccity.upstream.poi_representation.utils import next_batch
from veccity.upstream.abstract_model import AbstractModel


class HierEmbedding(nn.Module):
    def __init__(self, token_embed_size, num_vocab, week_embed_size, hour_embed_size, duration_embed_size):
        super().__init__()
        self.num_vocab = num_vocab
        self.token_embed_size = token_embed_size
        self.embed_size = token_embed_size + week_embed_size + hour_embed_size + duration_embed_size

        self.token_embed = nn.Embedding(num_vocab, token_embed_size)
        self.token_embed.weight.data.uniform_(-0.5 / token_embed_size, 0.5 / token_embed_size)
        self.week_embed = nn.Embedding(7, week_embed_size)
        self.hour_embed = nn.Embedding(24, hour_embed_size)
        self.duration_embed = nn.Embedding(24, duration_embed_size)

        self.dropout = nn.Dropout(0.1)

    def add_unk(self):
        old_weight = self.token_embed.weight.data
        embed_dimension = old_weight.size(1)
        vocab_size = old_weight.size(0)
        self.token_embed=nn.Embedding(vocab_size+1, embed_dimension)
        initrange = 0.5 / embed_dimension
        self.token_embed.weight.data.uniform_(-initrange, initrange)
        for i in range(vocab_size):
            self.token_embed.weight.data[i] = old_weight[i]

    def forward(self, token, week, hour, duration):
        token = self.token_embed(token)
        week = self.week_embed(week)
        hour = self.hour_embed(hour)
        duration = self.duration_embed(duration)


        return self.dropout(torch.cat([token, week, hour, duration], dim=-1))
    



class Hier(AbstractModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        hier_week_embed_size = config.get('week_embed_size', 4)
        hier_hour_embed_size = config.get('hour_embed_size', 4)
        hier_duration_embed_size = config.get('duration_embed_size', 5)
        share = config.get('share', False)
        dropout = config.get('dropout', 0.1)
        num_layers = config.get('num_layers', 4)
        embed_size = config.get('embed_size', 128)
        hidden_size = embed_size
        num_loc = data_feature.get('num_loc')+1
        self.embed = HierEmbedding(embed_size, num_loc,
                                   hier_week_embed_size, hier_hour_embed_size, hier_duration_embed_size)
        self.add_module('embed', self.embed)
        self.encoder = nn.LSTM(self.embed.embed_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        if share:
            self.out_linear = nn.Sequential(nn.Linear(hidden_size, self.embed.token_embed_size), nn.LeakyReLU())
        else:
            self.out_linear = nn.Sequential(nn.Linear(hidden_size, self.embed.token_embed_size),
                                            nn.LeakyReLU(),
                                            nn.Linear(self.embed.token_embed_size, self.embed.num_vocab))
        self.share = share

    def forward(self, token, week, hour, duration, valid_len, **kwargs):
        """
        :param token: sequences of tokens, shape (batch, seq_len)
        :param week: sequences of week indices, shape (batch, seq_len)
        :param hour: sequences of visit time slot indices, shape (batch, seq_len)
        :param duration: sequences of duration slot indices, shape (batch, seq_len)
        :return: the output prediction of next vocab, shape (batch, seq_len, num_vocab)
        """
        embed = self.embed(token, week, hour, duration)  # (batch, seq_len, embed_size)
        packed_embed = pack_padded_sequence(embed, valid_len.cpu(), batch_first=True, enforce_sorted=False)
        encoder_out, hc = self.encoder(packed_embed)  # (batch, seq_len, hidden_size)
        out = self.out_linear(encoder_out.data)  # (batch, seq_len, token_embed_size)

        if self.share:
            out = torch.matmul(out, self.embed.token_embed.weight.transpose(0, 1))  # (total_valid_len, num_vocab)
        return out

    def static_embed(self):
        return self.embed.token_embed.weight[:self.embed.num_vocab].detach().cpu().numpy()
    
    def encode(self, inputs):
        token=inputs['seq']
        week, hour, duration= inputs['weekday'], inputs['hour'], inputs['duration']
        valid_len=inputs['length']
        embed = self.embed(token, week, hour, duration)  # (batch, seq_len, embed_size)
        packed_embed = pack_padded_sequence(embed, valid_len, batch_first=True, enforce_sorted=False)
        encoder_out, hc = self.encoder(packed_embed)  # (batch, seq_len, hidden_size)
        encoder_out = pad_packed_sequence(encoder_out, batch_first=True)
        return encoder_out[0]

    def calculate_loss(self, batch):
        src_token, src_weekday, src_hour, src_duration, src_len = batch
        hier_out = self.forward(token=src_token[:, :-1], week=src_weekday[:, :-1], hour=src_hour[:, :-1],
                                duration=src_duration, valid_len=src_len - 1)  # (batch, seq_len, num_vocab)
        trg_token = pack_padded_sequence(src_token[:, 1:], (src_len - 1).cpu(), batch_first=True,
                                         enforce_sorted=False).data
        loss_func = nn.CrossEntropyLoss()
        return loss_func(hier_out, trg_token)

    def add_unk(self):
        self.embed.add_unk()
    