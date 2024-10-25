import sys
sys.path.append('..')
import numpy as np
import networkx as nx
import dgl

class EdgeIndex:
    def __init__(self, g: nx.DiGraph, seg_id_to_idx):
        self.nnode = len(g) # int
        edges = g.edges(data = True)
        edges = list(filter(lambda x: 'weight' in x[2] or 'spatial_weight' in x[2], edges)) # [(u_nodeid, v_nodeid, dict), (), ..]
        self.edges = np.array(list(map(lambda x: (seg_id_to_idx[x[0]], seg_id_to_idx[x[1]]), edges)), np.int64) # [num_edges, 2], np.array
        self.tweight = np.array( list(map(lambda x: x[2].get('weight', 0), edges)) ) # topo_weight [num_edges], np.array
        self.sweight = np.array( list(map(lambda x: x[2].get('spatial_weight', 0), edges)) ) # spatial_weight [num_edges], np.array

        assert self.edges.shape[0] == self.tweight.shape[0] == self.sweight.shape[0]
        
        self.node_neighbours = None # []


    def length(self):
        return self.edges.shape[0]


    def remove_edges(self, idxs: list):
        # idxs: 1D list
        self.edges = np.delete(self.edges, idxs, axis = 0)
        self.tweight = np.delete(self.tweight, idxs)
        self.sweight = np.delete(self.sweight, idxs)
        

    def create_adj_index(self):

        self.node_neighbours = [ ([],[]) for _ in range(self.nnode) ] # [([ingress], [egress]), (), ..]

        for x, y in self.edges:
            self.node_neighbours[x][1].append(y)
            self.node_neighbours[y][0].append(x)


    def sub_edge_index(self, sub_idx):
        # see GAT_pyG.py->create_sub_adj()
        # idxL: 1D list

        if self.node_neighbours == None:
            self.create_adj_index()
        
        idx = sorted(list(set(sub_idx)))
        idx1 = []

        sub_edge_index = [] # [, 2]
        for _i in idx:
            idx1.extend(self.node_neighbours[_i][0])
            sub_edge_index.append((_i, _i))
            sub_edge_index.extend( list(map(lambda x: (x, _i), self.node_neighbours[_i][0])) )

        idx1 = idx1 + idx
        idx1 = sorted(list(set(idx1))) # all involved nodes

        idx1_to_newidx = [-1] * self.nnode # dont use dict that runs very slow. use array for indexing
        for i, v in enumerate(idx1):
            idx1_to_newidx[v] = i

        sub_edge_index = [(idx1_to_newidx[i], idx1_to_newidx[j]) for (i,j) in sub_edge_index]
        sub_edge_index = np.array(sub_edge_index, np.int64).T
        new_x_idx = idx1
        mapping_to_origin_idx = [idx1_to_newidx[_i] for _i in sub_idx]
        return sub_edge_index, new_x_idx, mapping_to_origin_idx
