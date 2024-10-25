import numpy as np
import torch
import random
from torch.utils.data import Dataset

PAD=0
MASK=1

class DataPipeline:
    def __init__(self, data, vocabs, id2node, road_lengths, type_num, pos_index=0, sen_index=0):
        self.data = data
        self.center_pos = pos_index
        self.sen_index = sen_index
        self.unigram_table = vocabs
        self.id2node = id2node
        self.num_sen = len(data)
        self.type_num = type_num
        
        self.road_lengths = road_lengths

    def get_neg_data(self, num, target_inputs):

        batch_size = len(target_inputs)
        neg = np.zeros((num))
        for i in range(batch_size):
            delta = random.sample(self.unigram_table, num)
            while target_inputs[i] in delta:
                delta = random.sample(self.unigram_table, num)
            neg = np.vstack([neg, delta])
        return neg[1: batch_size + 1]

    def generate_batch_one(self, skip_window=5):

        batch, labels = [],[]
        # span = 2 * skip_window + 1  # [ skip_window, target, skip_window ]
        # buffer = collections.deque(maxlen=span)

        sentence = self.data[self.sen_index]
        while len(sentence) == 1:
            self.sen_index += 1
            sentence = self.data[self.sen_index]
        # print(sentence)
        end_pos = min(len(sentence), self.center_pos+skip_window+1)
        start_pos = max(0, self.center_pos-skip_window)

        target = sentence[self.center_pos]
        context = sentence[start_pos:self.center_pos] + sentence[self.center_pos + 1: end_pos]
        # print(target, context)

        for i in range(len(context)):
            batch.append(target)
            labels.append(context[i])

        if self.center_pos + 1 != len(sentence):
            self.center_pos += 1
        else:
            self.center_pos = 0
            self.sen_index = self.sen_index + 1

        return batch, labels

    def generate_batch(self, skip_window=10):


        batch, input_type, labels, types, type_mask = [],[],[],[],[]
        count = 0
        min_skip_window = 4
        while count < 6 and self.sen_index < self.num_sen:
            data = self.data[self.sen_index]
            sentence, tp = data[0], data[1]
            # print(sentence, tp)
            while len(sentence) == 1:
                self.sen_index += 1
                self.center_pos = 0
                sentence, tp = self.data[self.sen_index]
            end_pos = min(len(sentence), self.center_pos + skip_window + 1)
            start_pos = max(0, self.center_pos - skip_window)


            target, t_tp = sentence[self.center_pos], tp[self.center_pos]
            left_context = sentence[start_pos:self.center_pos]
            right_context = sentence[self.center_pos + 1: end_pos]
            c_left_tp = tp[start_pos:self.center_pos]
            c_right_tp = tp[self.center_pos + 1: end_pos]

            # process type input
            tp_ = np.zeros(self.type_num, dtype=np.int32)
            for i in range(len(left_context)):
                tp_[c_left_tp[i]] = 1
            for i in range(len(right_context)):
                tp_[c_right_tp[i]] = 1

            total_length = 0
            for i in range(len(left_context)):
                total_length += self.road_lengths[self.id2node[left_context[i] - 2]]
                if total_length < 1000 or i == 0:
                    batch.append(target)
                    labels.append(left_context[i])
                    types.append(tp_)
                    input_type.append(t_tp)
                    if len(left_context) < min_skip_window:
                        type_mask.append([1] * self.type_num)
                    else:
                        type_mask.append([0] * self.type_num)
                else:
                    break

            total_length = 0
            for i in range(len(right_context)):
                total_length += self.road_lengths[self.id2node[right_context[i] - 2]]
                if total_length < 1000 or i == 0:
                    batch.append(target)
                    labels.append(right_context[i])
                    types.append(tp_)
                    input_type.append(t_tp)
                    if len(right_context) < min_skip_window:
                        type_mask.append([1] * self.type_num)
                    else:
                        type_mask.append([0] * self.type_num)
                else:
                    break

            if self.center_pos + 1 != len(sentence):
                self.center_pos += 1
            else:
                self.center_pos = 0
                self.sen_index = self.sen_index + 1

            count += 1

        return batch, input_type, types, labels, type_mask




class RandomWalker():
    def __init__(self, config, data_feature):

        self.G = data_feature.get('adj_mx',None)
        self.node2id = data_feature.get('node2id')

        self.node2type = data_feature.get('node2type')
        self.sentences = []

    def generate_sentences_bert(self, num_walks=24):
        sts = []
        for _ in range(num_walks):
            sts.extend(self.random_walks())
        return sts

    def generate_sentences_bert_type(self, num_walks=24):
        sts = []
        for _ in range(num_walks):
            sts.extend(self.random_walks_type())
        return sts

    def generate_sentences_dw(self, num_walks=24):
        sts = []
        for _ in range(num_walks):
            sts.extend(self.random_walks_dw())
        return sts

    def random_walks_type(self):
        # random walk with every node as start point once

        walks = []
        nodes = list(range(self.G.shape[0]))
        random.shuffle(nodes)
        for node in nodes:
            walk = [self.node2id[node]+2]
            tp = []
            if node in self.node2type:
                tp.append(self.node2type[node])
            else:
                tp.append(random.randint(0, 4))
            v = node
            length_walk = random.randint(5, 100)
            for _ in range(length_walk):
                nbs = list(np.where(self.G[v]==1)[0].tolist())
                if len(nbs) == 0:
                    break
                v = random.choice(nbs)
                walk.append(self.node2id[v] + 2)
                if v in self.node2type:
                    tp.append(self.node2type[v])
                else:
                    tp.append(random.randint(0, 4))
            walks.append([walk,tp])
        return walks

    def random_walks(self):
        # random walk with every node as start point once
        walks = []
        nodes = list(range(self.G.shape[0]))
        random.shuffle(nodes)
        for node in nodes:
            walk = [self.node2id[node] + 2]
            v = node
            length_walk = random.randint(5, 100)
            for _ in range(length_walk):
                nbs = list(np.where(self.G[v]==1)[0].tolist())
                if len(nbs) == 0:
                    break
                v = random.choice(nbs)
                walk.append(self.node2id[v] + 2)
            walks.append(walk)
        return walks

    def random_walks_dw(self):
        # random walk with every node as start point once
        walks = []
        nodes = list(range(self.G.shape[0]))
        random.shuffle(nodes)
        for node in nodes:
            walk = [str(node)]
            v = node
            length_walk = random.randint(5, 100)
            for _ in range(length_walk):
                nbs = list(np.where(self.G[v]==1)[0].tolist())
                if len(nbs) == 0:
                    break
                v = random.choice(nbs)
                walk.append(str(v))
            walks.append(walk)
        return walks
