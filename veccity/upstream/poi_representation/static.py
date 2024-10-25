import numpy as np
import torch
from torch import nn

from veccity.upstream.abstract_model import AbstractModel


class DownstreamEmbed(AbstractModel):

    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        num_loc = data_feature.get('num_loc')
        embed_size = config.get('embed_size', 128)
        self.embed = nn.Embedding(num_loc + 1, embed_size, padding_idx=num_loc)
        self.embed.weight.data.uniform_(-0.5/embed_size, 0.5/embed_size)

    def forward(self, token, **kwargs):
        return self.embed(token)


class StaticEmbed(nn.Module):
    def __init__(self, embed_mat):
        """
        @param embed_mat: embed matrix of shape (num_loc, embed_size)
        """
        super().__init__()
        embed_size = embed_mat.shape[1]

        embed_mat = np.concatenate([embed_mat, np.zeros((1, embed_size))], axis=0)
        self.embed = nn.Parameter(torch.from_numpy(embed_mat).float(),requires_grad=True)

    def forward(self, token, **kwargs):
        """
        @param token: input token index.
        """
        return self.embed[token]
    
    def encode(self,inputs):
        token=inputs['seq']
        return self.embed[token]
    

