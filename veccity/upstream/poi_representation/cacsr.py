
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
import torch.nn.functional as F
import copy
import math
from torch.distributions import Uniform
from veccity.upstream.abstract_model import AbstractModel


class DotDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def clamp_preserve_gradients(x, min, max):
    """Clamp the tensor while preserving gradients in the clamped region."""
    return x + (x.clamp(min, max) - x).detach()


def KLD(mu, logvar):
    '''
    the KL divergency of  Gaussian distribution with a standard normal distribution
    https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Multivariate_normal_distributions
    :param mu: the mean (batch, dim)
    :param logvar: the log of variance (batch, dim)
    :return:
    '''
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.shape[0]

def KLD_category(w):
    '''
    the KL divergency of category distribution with param=w and the uniform category distribution
    :param w: (batch, nClass)
    :return:
    '''
    nClass = w.shape[1]
    p = torch.ones_like(w)/nClass  # (batch, nClass)
    # print(p[0])
    return torch.sum(w * torch.log(w/p)) / w.shape[0]


class MLP2(nn.Module):
    """
    MLP with two outputs， one for mu, one for log(var)
    """
    def __init__(self, input_size, output_size,
                 dropout=.0, hidden_size=128, use_selu=True):
        super(MLP2, self).__init__()
        self.hidden_size = hidden_size
        if self.hidden_size > 0:
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc21 = nn.Linear(hidden_size, output_size)
            self.fc22 = nn.Linear(hidden_size, output_size)
            self.nonlinear_f = F.selu if use_selu else F.relu
            self.dropout = nn.Dropout(dropout)
        else:
            self.fc21 = nn.Linear(input_size, output_size)
            self.fc22 = nn.Linear(input_size, output_size)
            self.nonlinear_f = F.selu if use_selu else F.relu
            self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        '''

        :param x: (batch, dim)
        :return:
        '''
        # print('mlp x:', x[:3,:])
        # print('mpl self.fc1(x):', self.fc1(x)[:3, :])
        # print('mpl self.nonlinear_f(self.fc1(x)):', self.nonlinear_f(self.fc1(x))[:3, :])
        if self.hidden_size > 0:
            h1 = self.dropout(self.nonlinear_f(self.fc1(x)))
            return self.fc21(h1), self.fc22(h1)
        else:
            return self.fc21(x), self.fc22(x)


class MLP(nn.Module):
    """
    MLP with one output (not normalized) for multinomial distribution
    """
    def __init__(self, input_size, hidden_size=64, output_size=1, dropout=0.0, use_selu=True):
        '''

        :param input_size:
        :param hidden_size:
        :param output_size: the num of cluster
        :param dropout:
        :param use_selu:
        '''
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.nonlinear_f = F.selu if use_selu else F.leaky_relu
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h1 = self.dropout(self.nonlinear_f(self.fc1(x)))
        return self.fc2(h1)


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


def attention(query, key, value, mask=None, dropout=None):
    '''

    :param query: （B, <h>, max_length, d_k）
    :param key: （B, <h>, max_length, d_k）
    :param value: （B, <h>, max_length, d_k）
    :param mask:  (B, <1>, max_length, max_length), true/false matrix, and true means paddings
    :param dropout:
    :return: outputs:(B, <h>, max_length, d_k), att_scores:(B, <h>, max_length, max_length)
    '''
    "Compute 'Scaled Dot Product Attention'"
    # print('query:', query.shape)
    # print('key:', key.shape)
    # print('value:', value.shape)
    d_k = query.size(-1)
    # print('start 4 query:', query[-1,0])
    # print('start 4 key:', key[-1,0])
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # print('start 4 scores:', scores.shape, scores[-1, 0, :, :])
    if mask is not None:
        scores = scores.masked_fill(mask, -1e9)  # true->-1e9
    # print('mask:', mask.shape, mask[-1,0,0,:])
    # print('start 5 scores:', scores.shape, scores[-1,0,:,:])
    p_attn = F.softmax(scores, dim=-1)  # 每行和为1
    # print('start 5 p_attn:', p_attn.shape, p_attn[-1, 0, :, :])
    # print('----')
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)  # for query, key, value, output
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None, distance=None):
        '''

        :param query: (B, max_length, d_model)
        :param key: (B, max_length, d_model)
        :param value: (B, max_length, d_model)
        :param mask: (B, max_length, max_length)
        :return: (B, max_length, d_model)
        '''
        # print('start 3 MHA query:', query[0])
        # print('start 3 MHA key:', key[0])
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)  # mask (B, 1, max_length)->(B, 1, 1, max_length)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


def spatial_aware_attention(query, key, value, distance, mask=None, dropout=None):
    '''

    :param query: （B, h, max_length, d_k）
    :param key: （B, h, max_length, d_k）
    :param value: （B, h, max_length, d_k）
    :param distance: (B, h, max_length, max_length)
    :param mask:  (B, 1, max_length, max_length), true/false matrix, and true means paddings
    :param dropout:
    :return: outputs:(B, h, max_length, d_k), att_scores:(B, h, max_length, max_length)
    '''
    "Compute 'Scaled Dot Product Attention'"
    # print('query:', query.shape)
    # print('key:', key.shape)
    # print('value:', value.shape)
    d_k = query.size(-1)
    # print('start 4 query:', query[-1,0])
    # print('start 4 key:', key[-1,0])
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # （B, h, max_length, max_length）
    # print('start 4 scores:', scores.shape, scores[-1, 0, :, :])
    scores = scores - distance  # （B, h, max_length, max_length）
    if mask is not None:
        scores = scores.masked_fill(mask, -1e9)  # true->-1e9
    # print('mask:', mask.shape, mask[-1,0,0,:])
    # print('start 5 scores:', scores.shape, scores[-1,0,:,:])
    p_attn = F.softmax(scores, dim=-1)  # 每行和为1
    # print('start 5 p_attn:', p_attn.shape, p_attn[-1, 0, :, :])
    # print('----')
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class SpatialAwareMultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(SpatialAwareMultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)  # for query, key, value, output
        self.logwb = nn.Parameter(Uniform(0.0, 1.0).sample((self.h,)))  # (h,)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None, distance=None):
        '''

        :param query: (B, max_length, d_model)
        :param key: (B, max_length, d_model)
        :param value: (B, max_length, d_model)
        :param distance: (B, max_length, max_length)
        :param mask: (B, max_length, max_length)
        :return: (B, max_length, d_model)
        '''
        # print('start 3 MHA query:', query[0])
        # print('start 3 MHA key:', key[0])
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)  # mask (B, 1, max_length)->(B, 1, 1, max_length)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        # distance to multi-head: (B, max_length, max_length) --> (B, h, max_length, max_length)
        wb = torch.exp(self.logwb).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # (1,h,1,1)
        mh_distance = distance.unsqueeze(1).repeat(1, self.h, 1, 1) * wb  # (B, h, max_length, max_length)*(1,h,1,1)
        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = spatial_aware_attention(query, key, value, mh_distance, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        '''

        :param x: (B, max_length, d_model)
        :return: (B, max_length, d_model)
        '''
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class TransformerEncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask=None, distance=None):
        '''

        :param x: (B, max_length, d_model)
        :param mask: (B, 1, max_length)
        :return: (B, max_length, d_model)
        '''
        # print('start 2 x:', x[0])
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask=mask, distance=distance))
        return self.sublayer[1](x, self.feed_forward)


class TransformerEncoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(TransformerEncoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size)

    def forward(self, x, padding_mask=None, session_mask=None, subsequent_mask=None, distance=None):
        mask = torch.zeros(x.size(0), x.size(1), x.size(1), device=x.device).bool()  # .type(torch.uint8)  # (B, max_length, max_length)
        # torch.set_printoptions(threshold=1000000)
        if padding_mask is not None:
            padding_mask = padding_mask.repeat(1, x.size(1), 1).bool()  # (B, max_length, max_length)
            # print('in padding_mask:', padding_mask)
            mask = mask | padding_mask
        if session_mask is not None:
            # print('in session_mask:', session_mask)
            mask = mask | session_mask
        if subsequent_mask is not None:
            # print('in subsequent_mask:', subsequent_mask)
            mask = mask | subsequent_mask
        # print('in mask', mask)
        for layer in self.layers:
            x = layer(x, mask=mask, distance=distance)
        return self.norm(x)

class Hypernet(nn.Module):
    """
        Hypernetwork deals with decoder input and generates params for mu, sigma, w

    Args:
        config: Model configuration.
        hidden_sizes: Sizes of the hidden layers. [] corresponds to a linear layer.
        param_sizes: Sizes of the output parameters. [n_components, n_components, n_components] 分别指定w,mu,s的维度/components
        activation: Activation function.
    """
    def __init__(self, config, hidden_sizes=[], param_sizes=[1, 1, 1], activation=nn.Tanh()):
        super().__init__()
        self.decoder_input_size = config.decoder_input_size
        self.activation = activation

        # print("hidden_sizes:", hidden_sizes)  # []
        # print("param_sizes:", param_sizes)  # [64, 64, 64]
        # Indices for unpacking parameters
        ends = torch.cumsum(torch.tensor(param_sizes), dim=0)
        starts = torch.cat((torch.zeros(1).type_as(ends), ends[:-1]))
        self.param_slices = [slice(s.item(), e.item()) for s, e in zip(starts, ends)]
        # self.param_slices.shape =  [slice(0, 64, None), slice(64, 128, None), slice(128, 192, None)]

        self.output_size = sum(param_sizes)  
        layer_sizes = list(hidden_sizes) + [self.output_size]
        # print("Hypernet layer_sizes:", layer_sizes)  # [192]
        # Bias used in the first linear layer
        self.first_bias = nn.Parameter(torch.empty(layer_sizes[0]).uniform_(-0.1, 0.1))  
        self.first_linear = nn.Linear(self.decoder_input_size, layer_sizes[0], bias=False)

        # Remaining linear layers
        self.linear_layers = nn.ModuleList()
        for idx, size in enumerate(layer_sizes[:-1]):
            self.linear_layers.append(nn.Linear(size, layer_sizes[idx + 1]))

    def reset_parameters(self):

        self.first_bias.data.fill_(0.0)
        self.first_linear.reset_parameters()
        nn.init.orthogonal_(self.first_linear.weight)
        for layer in self.linear_layers:
            layer.reset_parameters()
            nn.init.orthogonal_(layer.weight)

    def forward(self, decoder_input):
        """Generate model parameters from the embeddings.

        Args:
            input: decoder input, shape (batch, decoder_input_size)

        Returns:
            params: Tuple of model parameters.
        """
        # Generate the output based on the input
        hidden = self.first_bias
        hidden = hidden + self.first_linear(decoder_input)
        for layer in self.linear_layers:
            hidden = layer(self.activation(hidden))

        # # Partition the output
        # if len(self.param_slices) == 1:
        #     return hidden
        # else:
        return tuple([hidden[..., s] for s in self.param_slices])

class CACSR_ModelConfig(DotDict):
    '''
    configuration of the CACSR
    '''

    def __init__(self, loc_size=None, tim_size=None, uid_size=None, tim_emb_size=None, loc_emb_size=None,
                 hidden_size=None, user_emb_size=None, device=None,
                 loc_noise_mean=None, loc_noise_sigma=None, tim_noise_mean=None, tim_noise_sigma=None,
                 user_noise_mean=None, user_noise_sigma=None, tau=None,
                 pos_eps=None, neg_eps=None, dropout_rate_1=None, dropout_rate_2=None, rnn_type='BiLSTM',
                 num_layers=3, downstream='POI_RECOMMENDATION'):
        super().__init__()
        self.loc_size = loc_size  # 
        self.uid_size = uid_size  # 
        self.tim_size = tim_size  # 
        self.loc_emb_size = loc_emb_size
        self.tim_emb_size = tim_emb_size
        self.user_emb_size = user_emb_size
        self.hidden_size = hidden_size  # RNN hidden_size
        self.device = device
        self.rnn_type = rnn_type
        self.num_layers = num_layers

        self.loc_noise_mean = loc_noise_mean
        self.loc_noise_sigma = loc_noise_sigma
        self.tim_noise_mean = tim_noise_mean
        self.tim_noise_sigma = tim_noise_sigma
        self.user_noise_mean = user_noise_mean
        self.user_noise_sigma = user_noise_sigma
        self.tau = tau
        self.pos_eps = pos_eps
        self.neg_eps = neg_eps
        self.dropout_rate_1 = dropout_rate_1
        self.dropout_rate_2 = dropout_rate_2
        self.downstream = downstream


class CACSR(AbstractModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        # initialize parameters
        # print(config['dataset_class'])
        self.loc_size = data_feature.get("num_loc",0)+1
        self.loc_emb_size = config.get('loc_emb_size',128)
        self.tim_size = config.get('tim_size',25)
        self.tim_emb_size = config.get('tim_emb_size',128)
        self.user_size = data_feature.get("num_user",0)+1
        self.user_emb_size = config.get('user_emb_size',128)
        self.hidden_size = config.get('hidden_size',128)
        # add by Tianyi (rnn_type & num_layers)
        self.rnn_type = config.get('rnn_type','BiLSTM')
        self.num_layers = config['num_layers']
        self.device = config['device']
        self.downstream = config['downstream']
        self.adv = config.get("adv",1)
        self.weight = config.get("weight",0.05)
        if self.rnn_type == 'BiLSTM':
            self.bi = 2
        else:
            self.bi = 1

        ##############################################
        self.loc_noise_mean = config['loc_noise_mean']
        self.loc_noise_sigma = config['loc_noise_sigma']
        self.tim_noise_mean = config['tim_noise_mean']
        self.tim_noise_sigma = config['tim_noise_sigma']
        self.user_noise_mean = config['user_noise_mean']
        self.user_noise_sigma = config['user_noise_sigma']

        self.tau = config['tau']
        self.pos_eps = config['pos_eps']
        self.neg_eps = config['neg_eps']
        self.dropout_rate_1 = config['dropout_rate_1']
        self.dropout_rate_2 = config['dropout_rate_2']

        self.dropout_1 = nn.Dropout(self.dropout_rate_1)
        self.dropout_2 = nn.Dropout(self.dropout_rate_2)
        ################################################

        # Embedding layer
        self.emb_loc = nn.Embedding(num_embeddings=self.loc_size, embedding_dim=self.loc_emb_size)
        self.emb_tim = nn.Embedding(num_embeddings=self.tim_size, embedding_dim=self.tim_emb_size)
        self.emb_user = nn.Embedding(num_embeddings=self.user_size, embedding_dim=self.user_emb_size)

        # lstm layer
        # modified by Tianyi (3 kinds of rnn)
        if self.rnn_type == 'GRU':
            self.lstm = nn.GRU(self.loc_emb_size + self.tim_emb_size, self.hidden_size, num_layers=self.num_layers,
                               batch_first=True)
        elif self.rnn_type == 'LSTM':
            self.lstm = nn.LSTM(self.loc_emb_size + self.tim_emb_size, self.hidden_size, num_layers=self.num_layers,
                                batch_first=True)
        elif self.rnn_type == 'BiLSTM':
            self.lstm = nn.LSTM(self.loc_emb_size + self.tim_emb_size, self.hidden_size, num_layers=self.num_layers,
                                batch_first=True, bidirectional=True)
        else:
            raise ValueError("rnn_type should be ['GRU', 'LSTM', 'BiLSTM']")


        self.projection = nn.Sequential(
            nn.Linear(self.hidden_size * self.bi, self.hidden_size * self.bi + self.user_emb_size),
            nn.ReLU())
        self.dense = nn.Linear(in_features=self.hidden_size * self.bi , out_features=self.loc_size)
        self.apply(self._init_weight)

    def _init_weight(self, module):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

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

    def avg_pool(self, hidden_states, mask):
        length = torch.sum(mask, 1, keepdim=True).float()
        mask = mask.unsqueeze(2)
        hidden = hidden_states.masked_fill(mask == 0, 0.0)
        avg_hidden = torch.sum(hidden, 1) / length

        return avg_hidden

    def forward(self, batch, mode='test', adv=1):
        # downstream shouldn't affect the train process
        # X_all_loc, X_all_tim, X_all_text, Y_location, target_lengths, X_lengths, X_users
        loc = batch[0]#.X_all_loc
        tim = batch[1]#.X_all_tim
        all_len = batch[2]#.X_lengths  
        user = batch[3]#.X_users
        batch_size = batch[0].shape[0]#.X_all_loc.shape[0]

        loc_emb = self.emb_loc(loc)
        tim_emb = self.emb_tim(tim)
        
        if mode == 'train' and adv == 1:
            loc_noise = torch.normal(self.loc_noise_mean, self.loc_noise_sigma, loc_emb.shape).to(loc_emb.device)
            tim_noise = torch.normal(self.tim_noise_mean, self.tim_noise_sigma, tim_emb.shape).to(loc_emb.device)

            loc_emb_STNPos = loc_emb + loc_noise
            tim_emb_STNPos = tim_emb + tim_noise
            x_STNPos = torch.cat([loc_emb_STNPos, tim_emb_STNPos], dim=2).permute(1, 0, 2)  # batch_first=False
            pack_x_STNPos = pack_padded_sequence(x_STNPos, lengths=all_len.cpu(), enforce_sorted=False)
            # modified by Tianyi
            if self.rnn_type == 'GRU':
                lstm_out_STNPos, h_n_STNPos = self.lstm(pack_x_STNPos)  # max_len*batch*hidden_size
            elif self.rnn_type == 'LSTM':
                lstm_out_STNPos, (h_n_STNPos, c_n_STNPos) = self.lstm(pack_x_STNPos)  # max_len*batch*hidden_size
            elif self.rnn_type == 'BiLSTM':
                lstm_out_STNPos, (h_n_STNPos, c_n_STNPos) = self.lstm(pack_x_STNPos)  # max_len*batch*hidden_size
            else:
                raise ValueError('rnn_type is not in [GRU, LSTM, BiLSTM]!')

            lstm_out_STNPos, out_len_STNPos = pad_packed_sequence(lstm_out_STNPos, batch_first=True)
            final_out_STNPos = lstm_out_STNPos[0, (all_len[0] - 1): all_len[0], :]
            for i in range(1, batch_size): 
                final_out_STNPos = torch.cat(
                    [final_out_STNPos, lstm_out_STNPos[i, (all_len[i] - 1): all_len[i], :]], dim=0)

        # concatenate and permute
        x = torch.cat([loc_emb, tim_emb], dim=2).permute(1, 0, 2)  # batch_first=False
        # pack
        pack_x = pack_padded_sequence(x, lengths=all_len.cpu(), enforce_sorted=False)

        # modified by Tianyi
        if self.rnn_type == 'GRU':
            lstm_out, h_n = self.lstm(pack_x)  # max_len*batch*hidden_size
        elif self.rnn_type == 'LSTM':
            lstm_out, (h_n, c_n) = self.lstm(pack_x)  # max_len*batch*hidden_size
        elif self.rnn_type == 'BiLSTM':
            lstm_out, (h_n, c_n) = self.lstm(pack_x)  # max_len*batch*hidden_size

        # unpack
        lstm_out, out_len = pad_packed_sequence(lstm_out, batch_first=True)

        final_out = lstm_out[0, (all_len[0] - 1): all_len[0], :]
        for i in range(1, batch_size):  
            final_out = torch.cat([final_out, lstm_out[i, (all_len[i] - 1): all_len[i], :]], dim=0)
        dense = self.dense(final_out)  # Batch * loc_size


        ####################   adv  start    #####################
        if mode == 'train' and adv == 1:
            final_out_STNPos = self.dropout_1(final_out_STNPos)
            final_out = self.dropout_2(final_out)

            avg_STNPos = self.projection(final_out_STNPos)
            avg_Anchor = self.projection(final_out)

            cos = nn.CosineSimilarity(dim=-1)
            cont_crit = nn.CrossEntropyLoss()
            sim_matrix = cos(avg_STNPos.unsqueeze(1),
                             avg_Anchor.unsqueeze(0))

            adv_imposter = self.generate_adv(final_out, user)  # [n,b,t,d] or [b,t,d]  

            batch_size = final_out.size(0)

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

    def calculate_loss(self, batch):
        return self(batch, mode='train', adv=self.adv)
    
    def static_embed(self):
        return self.emb_loc.weight[:self.loc_size].data.cpu().numpy()

    #@torch.no_grad()
    def encode(self, inputs):
        # encode for downstream with no grad
        # user_embed is not train, so we remove the user embed
        loc = inputs['seq']#.X_all_loc
        tim = inputs['hour']#.X_all_tim
        all_len=inputs['length']

        batch_size = loc.shape[0]#.X_all_loc.shape[0]

        loc_emb = self.emb_loc(loc)
        tim_emb = self.emb_tim(tim)

        # concatenate and permute
        x = torch.cat([loc_emb, tim_emb], dim=2).permute(1, 0, 2)  # batch_first=False
        # pack
        pack_x = pack_padded_sequence(x, lengths=all_len, enforce_sorted=False)


        lstm_out, (h_n, c_n) = self.lstm(pack_x)  # max_len*batch*hidden_size

        # unpack
        lstm_out, out_len = pad_packed_sequence(lstm_out, batch_first=True)
        return lstm_out

# class CacsrData:
#     def __init__(self,config,data_feature):
#         self.distance_theta = data_feature.get('distance_theta',1)
#         self.gaussian_beta = data_feature.get('gaussian_beta',10)
#         self.max_his_period_days = data_feature.get('max_his',120)
#         self.max_merge_seconds_limit = data_feature.get('max_merge_seconds_limit',10800)
#         self.max_delta_mins = data_feature.get('max_delta_mins',1440)
#         self.min_session_mins = data_feature.get('min_session_mins',1440)
#         self.latN = data_feature.get('latN',50)
#         self.lngN = data_feature.get('lngN',50)
#         dirname = os.path.join(os.getcwd(), 'veccity','cache', 'dataset_cache', config['dataset'],'cacsr')
#         if not os.path.exists(dirname):
#             os.makedirs(dirname)
#         train_save_filename = os.path.join(os.getcwd(), 'veccity','cache', 'dataset_cache', config['dataset'],'cacsr','train.npz')
#         ss_save_filename = os.path.join(os.getcwd(), 'veccity','cache', 'dataset_cache', config['dataset'],'cacsr','ss.npz')
#         venue_cnt = data_feature['num_loc']
#         venue_lat = [coord[1] for coord in data_feature['coor_mat']]
#         venue_lng = [coord[2] for coord in data_feature['coor_mat']]
#         venue_category = data_feature['coor_df']['category'].values
#         category_cnt = len(set(venue_category))
#         venueId_lidx = {venue: idx for idx, venue in enumerate(data_feature['coor_df']['geo_uid'])}
#         if os.path.exists(ss_save_filename):
#             ss = np.load(ss_save_filename)
#             SS_distance = ss['SS_distance']
#             SS_proximity = ss['SS_proximity']
#             SS_gaussian_distance = ss['SS_gaussian_distance']
#         else:
#             print('Constructing spatial matrix...')
#             SS_distance, SS_proximity, SS_gaussian_distance = construct_spatial_matrix_accordingDistance(self.distance_theta, venue_cnt, venue_lng, venue_lat, gaussian_beta=self.gaussian_beta)
            
#             np.savez_compressed(ss_save_filename, SS_distance=SS_distance, SS_proximity=SS_proximity, SS_gaussian_distance=SS_gaussian_distance)
#         if os.path.exists(train_save_filename):
#             print('Loading data from cache...')
            
#         else :
#             max_lat = max(venue_lat)
#             min_lat = min(venue_lat)
#             max_lng = max(venue_lng)
#             min_lng = min(venue_lng)

#             lats = []
#             lngs = []
#             for i in range(venue_cnt):
#                 lats.append(venue_lat[i])
#                 lngs.append(venue_lng[i])

#             venue_latidx = {}
#             venue_lngidx = {}
#             for i in range(venue_cnt):
#                 eps = 1e-7
#                 latidx = int((venue_lat[i]-min_lat)*self.latN/(max_lat - min_lat + eps)) 
#                 lngidx = int((venue_lng[i]-min_lng)*self.lngN/(max_lng - min_lng + eps))
#                 venue_latidx[i]= latidx if latidx < self.latN else (self.latN-1)
#                 venue_lngidx[i]= lngidx if lngidx < self.lngN else (self.lngN-1)

#             feature_category = []
#             feature_lat = []
#             feature_lng = []
#             feature_lat_ori = []
#             feature_lng_ori = []
#             for i in range(venue_cnt):
#                 feature_category.append(venue_category[i])
#                 feature_lat.append(venue_latidx[i])
#                 feature_lng.append(venue_lngidx[i])
#                 feature_lat_ori.append(venue_lat[i])
#                 feature_lng_ori.append(venue_lng[i])

#             sample_constructor = cacsr_sample(venueId_lidx, SS_distance=SS_distance, SS_gaussian_distance=SS_gaussian_distance, max_his_period_days=self.max_his_period_days, 
#                                               max_merge_seconds_limit=self.max_merge_seconds_limit, max_delta_mins=self.max_delta_mins, min_session_mins=self.min_session_mins)
#             checkins_filter = pd.DataFrame(data_feature['df'], copy=True)
#             checkins_filter['local time'] = pd.to_datetime(checkins_filter["datetime"])
            
#             checkins_filter['timestamp'] = checkins_filter['local time'].apply(lambda x: x.timestamp()) 
#             checkins_filter['local weekday'] = checkins_filter['local time'].apply(lambda x: x.weekday())
#             checkins_filter['local hour'] = checkins_filter['local time'].apply(lambda x: x.hour)
#             checkins_filter['local minute'] = checkins_filter['local time'].apply(lambda x: x.minute)
#             checkins_filter['lid'] = checkins_filter['loc_index'].apply(lambda x: venueId_lidx[x]) 
#             checkins_filter.rename(columns={'user_index': 'userId'}, inplace=True)
#             #print('after filtering, %d check-ins points are left.' % len(checkins_filter), flush=True)
#             #print("checkins_filter's columns: ", checkins_filter.columns, checkins_filter.dtypes)           
#             userId_checkins = checkins_filter.groupby('userId')

#             all_drops = []
#             all_drops_ratio = []
#             for userId, checkins in userId_checkins:
#                 #print('userId:', userId, flush=True)
#                 uid = sample_constructor.user_cnt
#                 #print('uid:', uid, flush=True)
#                 checkins = checkins.sort_values(by=['timestamp'])  
#                 checkins = checkins.reset_index(drop=True)  

#                 tmp_len = len(checkins)
#                 checkins, drops = sample_constructor.deal_cluster_sequence_for_each_user(checkins) 
#                 all_drops.append(drops) 
#                 all_drops_ratio.append(drops / tmp_len) 

#                 total = len(checkins)
#                 user_lidFreq = (checkins[:].groupby(['lid']).count()).iloc[:, [0]]/total
#                 lid_visitFreq = venue_cnt * [0]
#                 for index, row in user_lidFreq.iterrows(): 
#                     lid_visitFreq[index] = row['userId']

#                 flag = sample_constructor.construct_sample_seq2seq(checkins, uid)
#                 if flag:
#                     sample_constructor.userId2uid[userId] = uid
#                     sample_constructor.user_cnt += 1
#                     sample_constructor.user_lidfreq.append(lid_visitFreq)
#             np.savez_compressed(train_save_filename,
#                                 trainX_target_lengths=sample_constructor.trainX_target_lengths,
#                                 trainX_arrival_times=sample_constructor.trainX_arrival_times,
#                                 trainX_duration2first=sample_constructor.trainX_duration2first,
#                                 trainX_session_arrival_times=sample_constructor.trainX_session_arrival_times,
#                                 trainX_local_weekdays=sample_constructor.trainX_local_weekdays,
#                                 trainX_session_local_weekdays=sample_constructor.trainX_session_local_weekdays,
#                                 trainX_local_hours=sample_constructor.trainX_local_hours,
#                                 trainX_session_local_hours=sample_constructor.trainX_session_local_hours,
#                                 trainX_local_mins=sample_constructor.trainX_local_mins,
#                                 trainX_session_local_mins=sample_constructor.trainX_session_local_mins,
#                                 trainX_delta_times=sample_constructor.trainX_delta_times,
#                                 trainX_session_delta_times=sample_constructor.trainX_session_delta_times,
#                                 trainX_locations=sample_constructor.trainX_locations,
#                                 trainX_session_locations=sample_constructor.trainX_session_locations,
#                                 trainX_last_distances=sample_constructor.trainX_last_distances,
#                                 trainX_users=sample_constructor.trainX_users, trainX_lengths=sample_constructor.trainX_lengths,
#                                 trainX_session_lengths=sample_constructor.trainX_session_lengths,
#                                 trainX_session_num=sample_constructor.trainX_session_num,
#                                 trainY_arrival_times=sample_constructor.trainY_arrival_times,
#                                 trainY_delta_times=sample_constructor.trainY_delta_times,
#                                 trainY_locations=sample_constructor.trainY_locations,
#                                 user_lidfreq=sample_constructor.user_lidfreq,
#                                 us=sample_constructor.us, vs=sample_constructor.vs,

#                                 feature_category=feature_category, feature_lat=feature_lat, feature_lng=feature_lng,
#                                 feature_lat_ori=feature_lat_ori, feature_lng_ori=feature_lng_ori,

#                                 latN=self.latN, lngN=self.lngN, category_cnt=category_cnt,

#                                 user_cnt=sample_constructor.user_cnt, venue_cnt=sample_constructor.venue_cnt,

#                                 SS_distance=sample_constructor.SS_distance, SS_guassian_distance=sample_constructor.SS_gaussian_distance)

            
#         loader = np.load(train_save_filename,allow_pickle=True)
#         user_cnt = loader['user_cnt']
#         venue_cnt = loader['venue_cnt']
#         feature_category = loader['feature_category']
#         feature_lat = loader['feature_lat']  # index
#         feature_lng = loader['feature_lng']  # index

#         # put spatial point features into tensor
#         self.feature_category = torch.LongTensor(feature_category)
#         self.feature_lat = torch.LongTensor(feature_lat)
#         self.feature_lng = torch.LongTensor(feature_lng)

#         self.latN, self.lngN = loader['latN'], loader['lngN']
#         self.category_cnt = loader['category_cnt']

#         # ----- load train / val / test to get dataset -----
#         self.data_train = self.load_data_from_dataset('train', loader, user_cnt, venue_cnt)
#         self.collate = collate_session_based
   
#     def load_data_from_dataset(self,set_name, loader, user_cnt, venue_cnt) :
#         X_target_lengths = loader[f'{set_name}X_target_lengths']
#         X_arrival_times = loader[f'{set_name}X_arrival_times']
#         X_users = loader[f'{set_name}X_users']
#         X_locations = loader[f'{set_name}X_locations']
#         Y_location = loader[f'{set_name}Y_locations']

#         X_all_loc = []
#         X_all_tim = []
#         X_lengths = []

#         for i in range(len(X_arrival_times)):
#             tim = X_arrival_times[i]
#             loc = X_locations[i]

#             len_ = len(tim)
#             for j in range(len_):
#                 tim[j] = tid_list_48(tim[j]) 

#             X_all_loc.append(loc)
#             X_all_tim.append(tim)
#             X_lengths.append(len_)

#         #print("X_all_loc: ", len(X_all_loc), X_all_loc[0])
#         #print("X_all_tim: ", len(X_all_tim), X_all_tim[0])
#         #print("X_target_lengths: ", len(X_target_lengths), X_target_lengths[0])
#         #print("X_lengths: ", len(X_lengths), X_lengths[0])
#         #print("X_users:", len(X_users), X_users)
#         #print("Y_location:", len(Y_location), Y_location[0])

#         dataset = SessionBasedSequenceDataset(user_cnt, venue_cnt, X_users, X_all_loc,
#                                               X_all_tim, Y_location, X_target_lengths, X_lengths, None)
#         #print(f'samples cnt of data_{set_name}:', dataset.real_length())

#         return dataset

# class SessionBasedSequenceDataset(data_utils.Dataset):
#     """Dataset class containing variable length sequences.
#     """

#     def __init__(self, user_cnt, venue_cnt, X_users, X_all_loc,
#                  X_all_tim, Y_location, target_lengths, X_lengths, X_all_text):
#         # torch.set_default_tensor_type(torch.cuda.FloatTensor)
#         self.user_cnt = user_cnt
#         self.venue_cnt = venue_cnt
#         self.X_users = X_users
#         self.X_all_loc = X_all_loc
#         self.X_all_tim = X_all_tim
#         self.target_lengths = target_lengths
#         self.X_lengths = X_lengths
#         self.Y_location = Y_location
#         self.X_all_text = X_all_text
#         self.validate_data()

#     @property
#     def num_series(self):
#         return len(self.Y_location)

#     def real_length(self):  
#         res = 0
#         n = len(self.Y_location)
#         for i in range(n):
#             res += len(self.Y_location[i])
#         return res

#     def validate_data(self):
#         if len(self.X_all_loc) != len(self.Y_location) or len(self.X_all_tim) != len(self.Y_location):
#             raise ValueError("Length of X_all_loc, X_all_tim, Y_location should match")

#     def __getitem__(self, key):
#         '''
#         the outputs are feed into collate()
#         :param key:
#         :return:
#         '''
#         return self.X_all_loc[key], self.X_all_tim[key], None, self.Y_location[key], self.target_lengths[key], \
#                self.X_lengths[key], self.X_users[key]

#     def __len__(self):
#         return self.num_series

#     def __repr__(self):  
#         pass

# def collate_session_based(batch,device):
#     '''
#     get the output of dataset.__getitem__, and perform padding
#     :param batch:
#     :return:
#     '''

#     batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)  

#     X_all_loc = [item[0] for item in batch]
#     X_all_tim = [item[1] for item in batch]
#     X_all_text = [item[2] for item in batch]
#     Y_location = [lid for item in batch for lid in item[3]] 
#     target_lengths = [item[4] for item in batch]
#     X_lengths = [item[5] for item in batch]
#     X_users = [item[6] for item in batch]

#     padded_X_all_loc = pad_session_data_one(X_all_loc)
#     padded_X_all_tim = pad_session_data_one(X_all_tim)
#     padded_X_all_loc = torch.tensor(padded_X_all_loc).long().to(device)
#     padded_X_all_tim = torch.tensor(padded_X_all_tim).long().to(device)

#     return session_Batch(padded_X_all_loc, padded_X_all_tim, X_all_text, Y_location, target_lengths, X_lengths, X_users,
#                          device)


# class session_Batch():
#     def __init__(self, padded_X_all_loc, padded_X_all_tim, X_all_text, Y_location, target_lengths, X_lengths, X_users,
#                  device):
#         self.X_all_loc = torch.LongTensor(padded_X_all_loc).to(device)  # (batch, max_all_length)
#         self.X_all_tim = torch.LongTensor(padded_X_all_tim).to(device)  # (batch, max_all_length)
#         self.X_all_text = X_all_text 
#         self.Y_location = torch.Tensor(Y_location).long().to(device)  # (Batch,) 
#         self.target_lengths = target_lengths 
#         self.X_lengths = X_lengths 
#         self.X_users = torch.Tensor(X_users).long().to(device)
        
        
# def pad_session_data_one(data):
#     fillvalue = 0
#     # zip_longest
#     data = list(zip(*itertools.zip_longest(*data, fillvalue=fillvalue)))
#     res = []
#     res.extend([list(data[i]) for i in range(len(data))])

#     return res