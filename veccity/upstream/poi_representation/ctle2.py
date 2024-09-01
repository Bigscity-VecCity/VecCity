import math

import numpy as np
import torch
from torch import nn

from veccity.upstream.poi_representation.utils import weight_init
from veccity.upstream.abstract_model import AbstractModel
import torch.nn.functional as F


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
        ctle_objective = config.get('objective', 'mh')
        embed_size = config.get('embed_size', 128)
        init_param = config.get('init_param', False)
        hidden_size = embed_size * 4
        max_seq_len = data_feature.get('max_seq_len')
        num_loc = data_feature.get('num_loc')
        encoding_layer = PositionalEncoding(embed_size, max_seq_len)
        self.ablation = config.get('abl','mh')

        if encoding_type == 'temporal':
            encoding_layer = TemporalEncoding(embed_size)

        obj_models = [MaskedLM(embed_size, num_loc)]
        if self.ablation != "mh":
            obj_models.append(MaskedHour(embed_size))
        self.obj_models = nn.ModuleList(obj_models)

        ctle_embedding = CTLEEmbedding(encoding_layer, embed_size, num_loc)
        self.embed_size = ctle_embedding.embed_size
        self.num_vocab = ctle_embedding.num_vocab

        self.embed = ctle_embedding
        self.add_module('embed', ctle_embedding)

        if self.ablation=='lstm':
            self.encoder=nn.LSTM(self.embed_size,hidden_size//2,num_layers,batch_first=True,dropout=0.1,bidirectional=True)
        else:
            encoder_layer = nn.TransformerEncoderLayer(d_model=self.embed_size, nhead=num_heads,
                                                    dim_feedforward=hidden_size, dropout=0.1)
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers,
                                                norm=nn.LayerNorm(self.embed_size, eps=1e-6))
        self.detach = detach

        ##############################################
        self.user_emb_size=16
        self.loc_size=num_loc
        self.loc_noise_mean = 0
        self.loc_noise_sigma = 0.01
        self.tim_noise_mean = 0
        self.tim_noise_sigma = 0.01
        self.user_noise_mean = 0
        self.user_noise_sigma = 0.01

        self.dropout_rate_1 = 0.5
        self.dropout_rate_2 = 0.5
        self.tau = 4

        self.dropout_1 = nn.Dropout(self.dropout_rate_1)
        self.dropout_2 = nn.Dropout(self.dropout_rate_2)

        self.pos_eps = config.get('pos_eps',1e-6)
        self.neg_eps = config.get('neg_eps',1e-6)

        self.projection = nn.Sequential(
            nn.Linear(self.embed_size, hidden_size + self.user_emb_size),
            nn.ReLU())
        self.dense = nn.Linear(in_features=self.embed_size, out_features=self.loc_size)
        
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
        if self.ablation=='lstm':
            encoder_out=self.encoder(token_embed)
            encoder_out=encoder_out[0] # (batch_size, src_len, embed_size)
        else:
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
        origin_tokens, origin_hour, masked_tokens, src_t_batch, mask_index,src_len,user_ids,src_tokens = batch
        ctle_out = self.forward(masked_tokens, timestamp=src_t_batch)  # (batch_size, src_len, embed_size)
        masked_out = ctle_out.reshape(-1, self.embed_size)[mask_index]  # (num_masked, embed_size)
        loss = 0.
        for obj_model in self.obj_models:
            loss += obj_model(masked_out, origin_tokens=origin_tokens, origin_hour=origin_hour)
        if self.ablation=='contr':
            loss+=self.contra_loss(batch)
        return loss
    
    def contra_loss(self,batch):

        origin_tokens, origin_hour, masked_tokens, src_t_batch, mask_index,src_len,user_ids,src_tokens = batch
        batch_size=src_tokens.size(0)
        loc_emb=self.embed.token_embed(src_tokens)
        tim_emb=self.embed.encoding_layer(src_tokens,timestamp=src_t_batch)
        loc_noise = torch.normal(self.loc_noise_mean, self.loc_noise_sigma, loc_emb.shape).to(loc_emb.device)
        tim_noise = torch.normal(self.tim_noise_mean, self.tim_noise_sigma, tim_emb.shape).to(loc_emb.device)

        loc_emb_STNPos = loc_emb + loc_noise
        tim_emb_STNPos = tim_emb + tim_noise

        x_STNPos = loc_emb_STNPos+tim_emb_STNPos 
        src_key_padding_mask = (src_tokens == self.num_vocab)
        src_mask=None
        pool_mask = (1 - src_key_padding_mask.int()).unsqueeze(-1)

        encoder_out = self.encoder(x_STNPos.transpose(0, 1), mask=src_mask,
                                    src_key_padding_mask=src_key_padding_mask).transpose(0,1)  # (batch_size, src_len, embed_size)
        final_out_STNPos = (encoder_out * pool_mask).sum(1) / pool_mask.sum(1)
            
        origin_embed=self.embed(src_tokens,timestamp=src_t_batch)
        origin_hidden=self.encoder(origin_embed.transpose(0, 1), mask=src_mask,
                                    src_key_padding_mask=src_key_padding_mask).transpose(0,1)  # (batch_size, src_len, embed_size)
        
        final_out = (origin_hidden * pool_mask).sum(1) / pool_mask.sum(1)
        
        dense = self.dense(final_out)  # Batch * loc_size
        
        final_out_STNPos = self.dropout_1(final_out_STNPos)
        final_out = self.dropout_2(final_out)
            
        avg_STNPos = self.projection(final_out_STNPos)
        avg_Anchor = self.projection(final_out)

        cos = nn.CosineSimilarity(dim=-1)
        cont_crit = nn.CrossEntropyLoss()
        sim_matrix = cos(avg_STNPos.unsqueeze(1),
                            avg_Anchor.unsqueeze(0))
        
        adv_imposter = self.generate_adv(final_out, user_ids)

        avg_adv_imposter = self.projection(adv_imposter)

        adv_sim = cos(avg_STNPos, avg_adv_imposter).unsqueeze(1)  # [b,1]
        adv_disTarget = self.generate_cont_adv(final_out_STNPos,  # todo
                                                   final_out, dense,
                                                   self.tau, self.pos_eps)
        avg_adv_disTarget = self.projection(adv_disTarget)
        pos_sim = cos(avg_STNPos, avg_adv_disTarget).unsqueeze(-1)  # [b,1]
        logits = torch.cat([sim_matrix, adv_sim], 1) / self.tau
        
        identity = torch.eye(batch_size, device=final_out.device)
        pos_sim = identity * pos_sim
        neg_sim = sim_matrix.masked_fill(identity == 1, 0)
        new_sim_matrix = pos_sim + neg_sim
        new_logits = torch.cat([new_sim_matrix, adv_sim], 1)

        labels = torch.arange(batch_size,
                                device=final_out.device)

        cont_loss = cont_crit(logits, labels)
        new_cont_loss = cont_crit(new_logits, labels)

        cont_loss = 0.5 * (cont_loss + new_cont_loss)

        return cont_loss
        
    
    def generate_adv(self, Anchor_hiddens, lm_labels):
        Anchor_hiddens = Anchor_hiddens.detach()
        lm_labels = lm_labels.detach()
        Anchor_hiddens.requires_grad_(True)

        Anchor_logits = self.dense(Anchor_hiddens)

        Anchor_logits = F.log_softmax(Anchor_logits, -1)

        criterion = nn.CrossEntropyLoss()
        loss_adv = criterion(Anchor_logits,
                             lm_labels).requires_grad_()

        loss_adv.backward()
        dec_grad = Anchor_hiddens.grad.detach()
        l2_norm = torch.norm(dec_grad, dim=-1)

        dec_grad /= (l2_norm.unsqueeze(-1) + 1e-12)

        perturbed_Anc = Anchor_hiddens + self.neg_eps * dec_grad.detach()
        perturbed_Anc = perturbed_Anc  # [b,t,d]

        self.zero_grad()

        return perturbed_Anc

    def generate_cont_adv(self, STNPos_hiddens,
                          Anchor_hiddens, pred,
                          tau, eps):
        STNPos_hiddens = STNPos_hiddens.detach()
        Anchor_hiddens = Anchor_hiddens.detach()
        Anchor_logits = pred.detach()
        STNPos_hiddens.requires_grad = True
        Anchor_logits.requires_grad = True
        Anchor_hiddens.requires_grad = True


        avg_STNPos = self.projection(STNPos_hiddens)
        avg_Anchor = self.projection(Anchor_hiddens)

        cos = nn.CosineSimilarity(dim=-1)
        logits = cos(avg_STNPos.unsqueeze(1), avg_Anchor.unsqueeze(0)) / tau

        cont_crit = nn.CrossEntropyLoss()
        labels = torch.arange(avg_STNPos.size(0),
                              device=STNPos_hiddens.device)
        loss_cont_adv = cont_crit(logits, labels)
        loss_cont_adv.backward()

        dec_grad = Anchor_hiddens.grad.detach()
        l2_norm = torch.norm(dec_grad, dim=-1)
        dec_grad /= (l2_norm.unsqueeze(-1) + 1e-12)

        perturb_Anchor_hidden = Anchor_hiddens + eps * dec_grad
        perturb_Anchor_hidden = perturb_Anchor_hidden.detach()
        perturb_Anchor_hidden.requires_grad = True
        perturb_logits = self.dense(perturb_Anchor_hidden)
        # perturb_logits = nn.LogSoftmax(dim=1)(perturb_logits)

        true_probs = F.softmax(Anchor_logits, -1)
        # true_probs = true_probs * dec_mask.unsqueeze(-1).float()

        perturb_log_probs = F.log_softmax(perturb_logits, -1)

        kl_crit = nn.KLDivLoss(reduction="sum")
        vocab_size = Anchor_logits.size(-1)

        kl = kl_crit(perturb_log_probs.view(-1, vocab_size),
                     true_probs.view(-1, vocab_size))
        kl = kl / torch.tensor(true_probs.shape[0]).float()
        kl.backward()

        kl_grad = perturb_Anchor_hidden.grad.detach()

        l2_norm = torch.norm(kl_grad, dim=-1)

        kl_grad /= (l2_norm.unsqueeze(-1) + 1e-12)

        perturb_Anchor_hidden = perturb_Anchor_hidden - eps * kl_grad

        return perturb_Anchor_hidden


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
