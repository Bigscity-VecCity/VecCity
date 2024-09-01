import itertools
import math
import random

import pandas as pd
from joblib import Parallel, delayed

from veccity.upstream.poi_representation.utils import partition_num


class RandomWalker:
    def __init__(self, G_dict):
        """
        :param G:
        """
        self.G_dict = G_dict
        self.count = 0

    def deepwalk_walk(self, walk_length, start_node):

        walk = [int(start_node)]

        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = list(self.G_dict[str(cur)])
            if len(cur_nbrs) > 0:
                walk.append(random.choice(cur_nbrs))
            else:
                break
        return walk

    def simulate_walks(self, num_walks, walk_length, workers=8, verbose=0):

        G_dict = self.G_dict

        nodes = list(G_dict)

        # parallel process
        results = Parallel(n_jobs=workers, verbose=verbose, )(
            delayed(self._simulate_walks)(nodes, num, walk_length) for num in
            partition_num(num_walks, workers))

        walks = list(itertools.chain(*results))

        return walks

    def _simulate_walks(self, nodes, num_walks, walk_length, ):
        walks = []
        for _ in range(num_walks):
            random.shuffle(nodes)
            for v in nodes:
                walks.append(self.deepwalk_walk(
                    walk_length=walk_length, start_node=v))
        return walks
