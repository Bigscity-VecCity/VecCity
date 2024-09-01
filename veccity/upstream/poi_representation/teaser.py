import numpy as np
from veccity.upstream.poi_representation.w2v import *
from veccity.upstream.abstract_model import AbstractModel
from logging import getLogger
from math import sin, cos, radians, acos


class Teaser(AbstractModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        teaser_num_ne = config.get('num_ne', 3)  # (number of unvisited locations)
        teaser_num_nn = config.get('num_nn', 3)  # (number of non-neighbor locations)
        teaser_indi_context = config.get('indi_context', False)
        self.alpha = config.get('alpha', 1.0)
        self.beta = config.get('beta', 0.0)
        week_embed_dimension = config.get('week_embed_size', 2)
        coor_mat = data_feature.get('coor_mat')
        num_vocab = data_feature.get('num_loc')
        num_user = data_feature.get('num_user')
        embed_dimension = config.get('embed_size', 128)
        self.__dict__.update(locals())

        self.u_embeddings = nn.Embedding(num_vocab+1, embed_dimension, sparse=True)
        self.v_embeddings = nn.Embedding(num_vocab, embed_dimension + week_embed_dimension, sparse=True)
        self.user_embeddings = nn.Embedding(num_user, embed_dimension + week_embed_dimension, sparse=True)
        self.week_embeddings = nn.Embedding(2, week_embed_dimension, sparse=True)

        initrange = 0.5 / self.embed_dimension
        self.u_embeddings.weight.data.uniform_(-initrange, initrange)
        self.user_embeddings.weight.data.uniform_(-initrange, initrange)
        self.week_embeddings.weight.data.uniform_(-initrange, initrange)
        self.v_embeddings.weight.data.uniform_(-0, 0)

    def forward(self, pos_u, pos_v, neg_v, user, weekday, neg_ne, neg_nn):
        """
        @param pos_u: positive input tokens, shape (batch_size). target
        @param pos_v: positive output tokens, shape (batch_size, window_size * 2). context
        @param neg_v: negative output tokens, shape (batch_size, num_neg).
        @param user: user indices corresponding to input tokens, shape (batch_size)
        @param weekday: weekday indices corresponding to input tokens, shape (batch_size)
        @param neg_ne: negative unvisited locations, shape (batch_size, num_ne_neg)
        @param neg_nn: negative non-neighborhood locations, shape (batch_size, num_nn_neg)
        """
        embed_u = self.u_embeddings(pos_u)  # (batch_size, embed_size)
        embed_week = self.week_embeddings(weekday)  # (batch_size, week_embed_size)
        embed_cat = torch.cat([embed_u, embed_week], dim=-1)  # (batch_size, embed_size + week_embed_size)

        embed_v = self.v_embeddings(pos_v)  # (batch_size, window_size * 2, embed_size + week_embed_size)
        score = torch.mul(embed_cat.unsqueeze(1), embed_v).squeeze()
        # (batch_size, window_size * 2, embed_size + week_embed_size)
        score = torch.sum(score, dim=-1)  # (batch_size, window_size * 2)
        score = F.logsigmoid(score)

        neg_embed_v = self.v_embeddings(neg_v)  # (batch_size, num_neg, embed_size + week_embed_size)
        neg_score = torch.bmm(neg_embed_v, embed_cat.unsqueeze(2)).squeeze()  # (batch_size, num_neg)
        neg_score = F.logsigmoid(-1 * neg_score)

        embed_user = self.user_embeddings(user)  # (batch_size, embed_size + week_embed_size)
        neg_embed_ne = self.v_embeddings(neg_ne)  # (batch_size, N, embed_size + week_embed_size)
        neg_embed_nn = self.v_embeddings(neg_nn)

        neg_ne_score = torch.bmm(embed_cat.unsqueeze(1) - neg_embed_ne, embed_user.unsqueeze(2)).squeeze()
        # (batch_size, N)
        neg_ne_score = F.logsigmoid(neg_ne_score)
        neg_nn_score = torch.bmm(embed_cat.unsqueeze(1) - neg_embed_nn, embed_user.unsqueeze(2)).squeeze()
        neg_nn_score = F.logsigmoid(neg_nn_score)

        return -1 * (self.alpha * (torch.sum(score) + torch.sum(neg_score)) +
                     self.beta * (torch.sum(neg_ne_score) + torch.sum(neg_nn_score)))

    def static_embed(self):
        return self.u_embeddings.weight[:self.num_vocab].detach().cpu().numpy()

    def calculate_loss(self, batch):
        batch_count, pos_u, pos_v, neg_v, user, week, neg_ne, neg_nn = batch
        return self.forward(pos_u, pos_v, neg_v, user, week, neg_ne, neg_nn)
    
    def encode(self, pos_u, week):
        embed_u = self.u_embeddings(pos_u)
        embed_week = self.week_embeddings(week)
        embed_cat = torch.cat([embed_u, embed_week], dim=-1)
        return embed_cat


def dis(lat1, lng1, lat2, lng2):
    lng1, lat1, lng2, lat2 = map(radians, [lng1, lat1, lng2, lat2])
    dlng = lng2 - lng1

    c = sin(lat1) * sin(lat2) + cos(lat1) * cos(lat2) * cos(dlng)
    r = 6371  # km
    if c > 1:
        c = 1
    return int(r * acos(c))


class TeaserData(SkipGramData):
    def __init__(self, users, sentences, weeks, coordinates, num_ne, num_nn, indi_context, distance_threshold=5,
                 sample=1e-3):
        """
        @param sentences: all users' full trajectories, shape (num_users, seq_len)
        @param weeks: weekday indices corresponding to the sentences.
        @param coordinates: coordinates of all locations, shape (num_locations, 3), each row is (loc_index, lat, lng)
        """
        super().__init__(sentences, sample)
        self.num_ne = num_ne
        self.num_nn = num_nn
        self.indi_context = indi_context
        self.users = users
        self.weeks = weeks
        all_locations = set(coordinates[:, 0].astype(int).tolist())
        self.location_num = len(all_locations)
        lat_sin = np.sin(np.radians(coordinates[:, 1]))
        lat_cos = np.cos(np.radians(coordinates[:, 1]))
        lng_radians = np.radians(coordinates[:, 2])
        # logger = getLogger()
        # logger.info('Total {}'.format(len(coordinates)))
        # counter = 0
        # A dict mapping one location index to its non-neighbor locations.
        self.non_neighbors = {}
        for coor_row in coordinates:

            loc_index = int(coor_row[0])
            # logger.info(loc_index)
            # distance = coor_row[1:].reshape(1, 2) - coordinates[:, 1:]  # (num_loc, 2)
            # distance = np.sqrt(np.power(distance[:, 0], 2) + np.power(distance[:, 1], 2))  # (num_loc)

            lat = np.full(len(coordinates), np.radians(coor_row[1]))
            lng = np.full(len(coordinates), np.radians(coor_row[2]))
            distance = np.arccos(np.minimum(np.sin(lat) * lat_sin +
                                            np.cos(lat) * lat_cos * np.cos(lng - lng_radians), 1.)) * 6371.

            non_neighbor_indices = coordinates[:, 0][np.argwhere(distance > distance_threshold)].reshape(-1).astype(int)
            if non_neighbor_indices.shape[0] == 0:
                non_neighbor_indices = np.array([len(all_locations)], dtype=int)
            self.non_neighbors[loc_index] = non_neighbor_indices

            # counter += 1
            # if counter % 1000 == 0:
            #     logger.info('Finish {}'.format(counter))

        # logger.info('Total {}'.format(min(len(users), len(sentences))))
        # counter = 0
        # A dict mapping one user index to its all unvisited locations.
        # self.unvisited = {}
        # for user, visited in zip(users, sentences):
        #     user = int(user)
        #     user_unvisited = all_locations - set(visited)
        #     self.unvisited[user] = user_unvisited & self.unvisited.get(user, all_locations)
        #     counter += 1
        #     if counter % 1000 == 0:
        #         logger.info('Finish {}'.format(counter))

        self.visited = {}
        for user, visited in zip(users, sentences):
            user = int(user)
            self.visited[user] = set(visited) & self.visited.get(user, set())
            # counter += 1
            # if counter % 1000 == 0:
            #     logger.info('Finish {}'.format(counter))

    def gen_pos_pairs(self, window_size):
        pos_pairs = []
        # logger = getLogger()
        # logger.info('Total {}'.format(min(len(self.users), len(self.sentences), len(self.weeks))))
        # counter = 0
        for user, sentence, week in zip(self.users, self.sentences, self.weeks):
            for i in range(0, len(sentence) - (2 * window_size + 1)):
                target = sentence[i + window_size]
                target_week = 0 if week[i + window_size] in range(5) else 1
                context = sentence[i:i + window_size] + sentence[i + window_size + 1:i + 2 * window_size + 1]
                sample_ne = self.sample_unvisited(user, num_neg=self.num_ne)
                sample_nn = self.sample_non_neighbor(target, num_neg=self.num_nn)
                if self.indi_context:
                    pos_pairs += [[user, target, target_week, [c], sample_ne, sample_nn] for c in context]
                else:
                    pos_pairs.append([user, target, target_week, context, sample_ne, sample_nn])
            # counter += 1
            # if counter % 1000 == 0:
            #     logger.info('Finish {}'.format(counter))
        return pos_pairs

    def sample_unvisited(self, user, num_neg):
        # return np.random.choice(np.array(list(self.unvisited[user])), size=num_neg).tolist()
        unvisited = []
        for i in range(num_neg):
            location = np.random.randint(0, self.location_num)
            while location in self.visited[user]:
                location = np.random.randint(0, self.location_num)
            unvisited.append(location)
        return unvisited

    def sample_non_neighbor(self, target, num_neg):
        return np.random.choice(self.non_neighbors[target], size=num_neg).tolist()
