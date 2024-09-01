from veccity.upstream.poi_representation.w2v import *
from logging import getLogger


def gen_all_slots(minute, time_slice_length, influence_span_length):
    """
    @param minute: UTC timestamp in minute.
    @param time_slice_length: length of one slot in seconds.
    @param influence_span_length: length of influence span in seconds.
    """

    def _cal_slice(x):
        return int((x % (24 * 60)) / time_slice_length)

    # max_num_slots = math.ceil(time_slice_length / influence_span_length) + 1
    if influence_span_length == 0:
        slices, props = [_cal_slice(minute)], [1.0]

    else:
        minute_floors = list({minute - influence_span_length / 2, minute + influence_span_length / 2} |
                             set(range((int((
                                                        minute - influence_span_length / 2) / time_slice_length) + 1) * time_slice_length,
                                       int(minute + influence_span_length / 2), time_slice_length)))
        minute_floors.sort()

        slices = [_cal_slice(time_minute) for time_minute in minute_floors[:-1]]
        props = [(minute_floors[index + 1] - minute_floors[index]) / influence_span_length
                 for index in range(len(minute_floors) - 1)]

        # mask_length = max_num_slots - len(slices)
        # slices += [slices[-1]] * mask_length
        # props += [0.0] * mask_length

    return slices, props


num_temp_vocab = 0


class TaleData(W2VData):
    def __init__(self, sentences, timestamps, time_slice_len, influence_span_length, indi_context=True):
        """
        @param sentences: sequences of locations.
        @param minutes: UTC minutes corresponding to sentences.
        @param time_slice_len: length of one time slice, in minute.
        """
        temp_sentence = []
        slices, props = [], []
        for sentence, timestamp in zip(sentences, timestamps):
            slice_row, prop_row = [], []
            minute = list(map(lambda x: x / 60, timestamp))
            for token, one_minute in zip(sentence, minute):
                slice, prop = gen_all_slots(one_minute, time_slice_len, influence_span_length)
                temp_sentence += ['{}-{}'.format(token, s) for s in slice]
                slice_row.append(slice)
                prop_row.append(prop)
            slices.append(slice_row)
            props.append(prop_row)

        super().__init__([temp_sentence])

        self.id2index = {id: index for index, id in enumerate(self.word_freq[:, 0])}
        global num_temp_vocab
        num_temp_vocab = len(self.id2index)
        self.word_freq[:, 0] = np.array([self.id2index[x] for x in self.word_freq[:, 0]])
        self.word_freq = self.word_freq.astype(int)
        self.huffman_tree = HuffmanTree(self.word_freq)

        self.sentences = sentences
        self.slices = slices
        self.props = props
        self.indi_context = indi_context

    def get_path_pairs(self, window_size):
        path_pairs = []
        for sentence, slice, prop in zip(self.sentences, self.slices, self.props):
            for i in range(0, len(sentence) - (2 * window_size + 1)):
                temp_targets = ['{}-{}'.format(sentence[i + window_size], s) for s in slice[i + window_size]]
                target_indices = [self.id2index[t] for t in temp_targets]  # (num_overlapping_slices)
                pos_paths = [self.huffman_tree.id2pos[t] for t in target_indices]
                neg_paths = [self.huffman_tree.id2neg[t] for t in target_indices]
                context = sentence[i:i + window_size] + sentence[i + window_size + 1:i + 2 * window_size + 1]
                if self.indi_context:
                    path_pairs += [[[c], pos_paths, neg_paths, prop[i + window_size]] for c in context]
                else:
                    path_pairs.append([context, pos_paths, neg_paths, prop[i + window_size]])
        return path_pairs


class Tale(HS):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        embed_dimension = config.get('embed_size', 128)
        self.w_embeddings = nn.Embedding(num_temp_vocab, embed_dimension, padding_idx=0)

    def forward(self, pos_u, pos_w, neg_w, **kwargs):
        """
        @param pos_u: positive input tokens, shape (batch_size, window_size * 2)
        @param pos_w: positive output tokens, shape (batch_size, pos_path_len)
        @param neg_w: negative output tokens, shape (batch_size, neg_path_len)
        """
        pos_score, neg_score = super().forward(pos_u, pos_w, neg_w, sum=False)  # (batch_size, pos_path_len)
        prop = kwargs['prop']
        pos_score, neg_score = (-1 * (item.sum(axis=1) * prop).sum() for item in (pos_score, neg_score))
        return pos_score + neg_score

    def static_embed(self):
        return self.u_embeddings.weight.detach().cpu().numpy()

    def encode(self, context,**kwargs):
        return self.u_embeddings(context)

    def calculate_loss(self, batch):
        batch_count, context, pos_pairs, neg_pairs, prop = batch
        return self.forward(context, pos_pairs, neg_pairs, prop=prop)
    
    def add_unk(self):
        old_weight = self.u_embeddings.weight.data
        embed_dimension = old_weight.size(1)
        vocab_size = old_weight.size(0)
        self.u_embeddings=nn.Embedding(vocab_size+1, embed_dimension, sparse=True)
        initrange = 0.5 / self.embed_dimension
        self.u_embeddings.weight.data.uniform_(-initrange, initrange)
        for i in range(vocab_size):
            self.u_embeddings.weight.data[i] = old_weight[i]
