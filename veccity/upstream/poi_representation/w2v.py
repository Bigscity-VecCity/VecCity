from collections import Counter
import math
import heapq
from itertools import zip_longest

import numpy as np
import torch
from logging import getLogger
from torch import nn
from torch.nn import functional as F
from sklearn.utils import shuffle

from veccity.upstream.poi_representation.utils import next_batch
from veccity.upstream.abstract_model import AbstractModel


def gen_token_freq(sentences):
    freq = Counter()
    for sentence in sentences:
        freq.update(sentence)
    freq = np.array(sorted(freq.items()))
    return freq


def gen_neg_sample_table(freq_array, sample_table_size=1e8, clip_ratio=1e-3):
    sample_table = []
    pow_freq = freq_array[:, 1] ** 0.75

    words_pow = pow_freq.sum()
    ratio = pow_freq / words_pow
    ratio = np.clip(ratio, a_min=0, a_max=clip_ratio)
    ratio = ratio / ratio.sum()

    count = np.round(ratio * sample_table_size)
    for word_index, c in enumerate(count):
        sample_table += [freq_array[word_index, 0]] * int(c)
    sample_table = np.array(sample_table)
    return sample_table


class W2VData:
    def __init__(self, sentences):
        self.word_freq = gen_token_freq(sentences)  # (num_vocab, 2)


class SkipGramData(W2VData):
    def __init__(self, sentences, sample=1e-3):
        super().__init__(sentences)
        self.sentences = sentences

        # Initialize negative sampling table.
        self.sample_table = gen_neg_sample_table(self.word_freq, clip_ratio=sample)

    def gen_pos_pairs(self, window_size):
        pos_pairs = []
        for sentence in self.sentences:
            for i in range(0, len(sentence) - (2 * window_size + 1)):
                target = sentence[i + window_size]
                context = sentence[i:i + window_size] + sentence[i + window_size + 1:i + 2 * window_size + 1]
                # pos_pairs += [[target, [c]] for c in context]
                pos_pairs.append([target, context])
        return pos_pairs

    def get_neg_v_sampling(self, batch_size, num_neg):
        neg_v = np.random.choice(self.sample_table, size=(batch_size, num_neg))
        return neg_v


class HSData(W2VData):
    def __init__(self, sentences):
        super().__init__(sentences)
        self.sentences = sentences
        self.huffman_tree = HuffmanTree(self.word_freq)

    def get_path_pairs(self, window_size):
        path_pairs = []
        for sentence in self.sentences:
            for i in range(0, len(sentence) - (2 * window_size + 1)):
                target = sentence[i + window_size]
                pos_path = self.huffman_tree.id2pos[target]
                neg_path = self.huffman_tree.id2neg[target]
                context = sentence[i:i + window_size] + sentence[i + window_size + 1:i + 2 * window_size + 1]
                # path_pairs += [[[c], pos_path, neg_path] for c in context]
                path_pairs.append([context, pos_path, neg_path])
        return path_pairs


class HuffmanNode:
    def __init__(self, id, frequency):
        """
        :param id: index of word (leaf nodes) or inner nodes.
        :param frequency: frequency of word.
        """
        self.id = id
        self.frequency = frequency

        self.left = None
        self.right = None
        self.father = None
        self.huffman_code = []
        self.path = []  # (path from root node to leaf node)

    def __str__(self):
        return 'HuffmanNode#{},freq{}'.format(self.id, self.frequency)

    def __lt__(self, other):
        return self.frequency < other.frequency

    def __eq__(self, other):
        return self.frequency == other.frequency


class HuffmanTree:
    def __init__(self, freq_array):
        """
        :param freq_array: numpy array containing all words' frequencies, format {id: frequency}.
        """
        self.num_words = freq_array.shape[0]
        self.id2code = {}
        self.id2path = {}
        self.id2pos = {}
        self.id2neg = {}
        self.root = None  # Root node of this tree.
        self.num_inner_nodes = 0  # Records the number of inner nodes of this tree.

        unmerged_node_list = [HuffmanNode(id, frequency) for id, frequency in freq_array]
        self.tree = {node.id: node for node in unmerged_node_list}
        self.id_offset = max(self.tree.keys())  # Records the starting-off ID of this tree.
        # Because the ID of leaf nodes will not be needed during calculation, you can minus this value to all inner nodes' IDs to save some space in output embeddings.

        self._offset = self.id_offset
        self._build_tree(unmerged_node_list)
        self._gen_path()
        self._get_all_pos_neg()

    def _merge_node(self, node1: HuffmanNode, node2: HuffmanNode):
        """
        Merge two nodes into one, adding their frequencies.
        """
        sum_freq = node1.frequency + node2.frequency
        self._offset += 1
        mid_node_id = self._offset
        father_node = HuffmanNode(mid_node_id, sum_freq)
        if node1.frequency >= node2.frequency:
            father_node.left, father_node.right = node1, node2
        else:
            father_node.left, father_node.right = node2, node1
        self.tree[mid_node_id] = father_node
        self.num_inner_nodes += 1
        return father_node

    def _build_tree(self, node_heap):
        heapq.heapify(node_heap)
        while len(node_heap) > 1:
            n1 = heapq.heappop(node_heap)
            n2 = heapq.heappop(node_heap)
            father_node = self._merge_node(n1, n2)
            heapq.heappush(node_heap, father_node)
        self.root = heapq.heappop(node_heap)
        # while len(node_list) > 1:
        #     i1, i2 = 0, 1
        #     if node_list[i2].frequency < node_list[i1].frequency:
        #         i1, i2 = i2, i1
        #     for i in range(2, len(node_list)):
        #         if node_list[i].frequency < node_list[i2].frequency:
        #             i2 = i
        #             if node_list[i2].frequency < node_list[i1].frequency:
        #                 i1, i2 = i2, i1
        #     father_node = self._merge_node(node_list[i1], node_list[i2])
        #     assert not i1 == i2
        #     if i1 < i2:
        #         node_list.pop(i2)
        #         node_list.pop(i1)
        #     else:
        #         node_list.pop(i1)
        #         node_list.pop(i2)
        #     node_list.insert(0, father_node)
        # self.root = node_list[0]

    def _gen_path(self):
        stack = [self.root]
        while len(stack) > 0:
            node = stack.pop()
            while node.left or node.right:
                code = node.huffman_code
                path = node.path
                node.left.huffman_code = code + [1]
                node.right.huffman_code = code + [0]
                node.left.path = path + [node.id]
                node.right.path = path + [node.id]
                stack.append(node.right)
                node = node.left
            id = node.id
            code = node.huffman_code
            path = node.path
            self.tree[id].huffman_code, self.tree[id].path = code, path
            self.id2code[id], self.id2path[id] = code, path

    def _get_all_pos_neg(self):
        for id in self.id2code.keys():
            pos_id = []
            neg_id = []
            for i, code in enumerate(self.tree[id].huffman_code):
                if code == 1:
                    pos_id.append(self.tree[id].path[i] - self.id_offset)
                    # This will make the generated inner node IDs starting from 1.
                else:
                    neg_id.append(self.tree[id].path[i] - self.id_offset)
            self.id2pos[id] = pos_id
            self.id2neg[id] = neg_id


class HS(AbstractModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        num_vocab = data_feature.get('num_loc')
        embed_dimension = config.get('embed_size', 128)
        self.num_vocab = num_vocab
        self.embed_dimension = embed_dimension

        # Input embedding.
        self.u_embeddings = nn.Embedding(num_vocab, embed_dimension, sparse=True)
        # Output embedding. Here is actually the embedding of inner nodes.
        self.w_embeddings = nn.Embedding(num_vocab, embed_dimension, padding_idx=0, sparse=True)

        initrange = 0.5 / self.embed_dimension
        self.u_embeddings.weight.data.uniform_(-initrange, initrange)
        self.w_embeddings.weight.data.uniform_(-0, 0)

    def forward(self, pos_u, pos_w, neg_w, **kwargs):
        """
        @param pos_u: positive input tokens, shape (batch_size, window_size * 2)
        @param pos_w: positive output tokens, shape (batch_size, num_pos)
        @param neg_w: negative output tokens, shape (batch_size, num_neg)
        @param sum: whether to sum up all scores.
        """
        pos_u_embed = self.u_embeddings(pos_u)  # (batch_size, window_size * 2, embed_size)
        pos_u_embed = pos_u_embed.sum(1, keepdim=True)  # (batch_size, 1, embed_size)

        pos_w_mask = torch.where(pos_w == 0, torch.ones_like(pos_w),
                                 torch.zeros_like(pos_w)).bool()  # (batch_size, num_pos)
        pos_w_embed = self.w_embeddings(pos_w)  # (batch_size, num_pos, embed_size)
        score = torch.mul(pos_u_embed, pos_w_embed).sum(dim=-1)  # (batch_size, num_pos)
        score = F.logsigmoid(-1 * score)  # (batch_size, num_pos)
        score = score.masked_fill(pos_w_mask, torch.tensor(0.0).to(pos_u.device))

        neg_w_mask = torch.where(neg_w == 0, torch.ones_like(neg_w), torch.zeros_like(neg_w)).bool()
        neg_w_embed = self.w_embeddings(neg_w)
        neg_score = torch.mul(pos_u_embed, neg_w_embed).sum(dim=-1)  # (batch_size, num_neg)
        neg_score = F.logsigmoid(neg_score)
        neg_score = neg_score.masked_fill(neg_w_mask, torch.tensor(0.0).to(pos_u.device))
        if kwargs.get('sum', True):
            return -1 * (torch.sum(score) + torch.sum(neg_score))
        else:
            return score, neg_score


class SkipGram(AbstractModel):

    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        num_vocab = data_feature.get('num_loc')
        embed_dimension = config.get('embed_size', 128)
        self.num_vocab = num_vocab
        self.embed_dimension = embed_dimension
        self.cbow = config.get('model') == 'CBOW'

        # Input embedding.
        self.u_embeddings = nn.Embedding(num_vocab, embed_dimension, sparse=True)
        # Output embedding.
        self.v_embeddings = nn.Embedding(num_vocab, embed_dimension, sparse=True)

        initrange = 0.5 / self.embed_dimension
        self.u_embeddings.weight.data.uniform_(-initrange, initrange)
        self.v_embeddings.weight.data.uniform_(-0, 0)

    def forward(self, pos_u, pos_v, neg_v):
        if self.cbow:
            embed_u = self.u_embeddings(pos_v).sum(1, keepdim=True)  # (batch_size, 1, embed_size)
            embed_v = self.v_embeddings(pos_u)  # (batch_size, embed_size)
            score = torch.mul(embed_u, embed_v.unsqueeze(1)).squeeze()  # (batch_size, embed_size)
            score = score.sum(dim=-1)  # (batch_size)
            score = F.logsigmoid(score)

            neg_embed_v = self.v_embeddings(neg_v)  # (batch_size, num_neg, embed_size)
            neg_score = torch.mul(embed_u, neg_embed_v)  # (batch_size, num_neg, embed_size)
            neg_score = neg_score.sum(-1)  # (batch_size, num_neg)
            neg_score = F.logsigmoid(-1 * neg_score)
            return -1 * (torch.sum(score) + torch.sum(neg_score))
        else:
            embed_u = self.u_embeddings(pos_u)  # (batch_size, embed_size)
            embed_v = self.v_embeddings(pos_v)  # (batch_size, N, embed_size)
            score = torch.mul(embed_u.unsqueeze(1), embed_v).squeeze()  # (batch_size, N, embed_size)
            score = torch.sum(score, dim=-1)  # (batch_size, N)
            score = F.logsigmoid(score)

            neg_embed_v = self.v_embeddings(neg_v)  # (batch_size, num_neg, embed_size)
            neg_score = torch.bmm(neg_embed_v, embed_u.unsqueeze(2)).squeeze()
            neg_score = F.logsigmoid(-1 * neg_score)
            return -1 * (torch.sum(score) + torch.sum(neg_score))

    def static_embed(self):
        return self.u_embeddings.weight.detach().cpu().numpy()

    def calculate_loss(self, batch):
        batch_count, pos_u, pos_v, neg_v = batch
        return self.forward(pos_u, pos_v, neg_v)
    
    def encode(self, tokens,**kwargs):
        return self.u_embeddings(tokens)
