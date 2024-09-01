import pandas as pd
import numpy as np
import torch
import dgl
import numpy_indexed as npi
from logging import getLogger

def build_graph_from_matrix(adjm, node_feats, device='cpu'):
    '''
    Build graph using DGL library from adjacency matrix.

    Inputs:
    -----------------------------------
    adjm: graph adjacency matrix of which entries are either 0 or 1.
    node_feats: node features

    Returns:
    -----------------------------------
    g: DGL graph object
    '''
    # get edge nodes' tuples [(src, dst)]
    dst, src = adjm.nonzero()
    # get edge weights
    d = adjm[adjm.nonzero()]
    # create a graph
    g = dgl.DGLGraph()
    # add nodes
    g.add_nodes(adjm.shape[0])
    # add edges and edge weights
    #g.add_edges(src, dst, {'d': torch.tensor(d).float().view(-1, 1).to(device)})
    g.add_edges(src, dst, {'d': torch.tensor(d).float().view(-1, 1)})
    # add node attribute, i.e. the geographical features of census tract
    #g.ndata['attr'] = torch.from_numpy(node_feats).to(device)
    g.ndata['attr'] = torch.from_numpy(node_feats)
    # compute the degree norm
    norm = comp_deg_norm(g)
    # add nodes norm
    #g.ndata['norm'] = torch.from_numpy(norm).view(-1,1).to(device) # column vector
    g.ndata['norm'] = torch.from_numpy(norm).view(-1, 1)
    g = g.to(device)
    # return
    return g

def comp_deg_norm(g):
    '''
    compute the degree normalization factor which is 1/in_degree
    '''
    in_deg = g.in_degrees(range(g.number_of_nodes())).float().cpu().numpy()
    norm = 1.0 / in_deg
    norm[np.isinf(norm)] = 0
    return norm


def mini_batch_gen(train_data, mini_batch_size, num_nodes, negative_sampling_rate = 0):
    '''
    generator of mini-batch samples
    '''
    logger = getLogger()
    # positive data
    pos_samples = train_data
    # negative sampling to get negative data
    neg_samples = negative_sampling(pos_samples, num_nodes, negative_sampling_rate)
    # binding together
    if neg_samples is not None:
        samples = torch.cat((pos_samples, neg_samples), dim=0)
    else:
        samples = pos_samples
    # shuffle
    samples = samples[torch.randperm(samples.shape[0])]
    # cut to mini-batches and wrap them by a generator
    logger.info('Number of samples: {}'.format(samples.shape[0]//mini_batch_size))
    for i in range(0, samples.shape[0], mini_batch_size):
        yield samples[i:i+mini_batch_size]

def negative_sampling(pos_samples, num_nodes, negative_sampling_rate = 0):
    '''
    perform negative sampling by perturbing the positive samples
    '''
    # if do not require negative sampling
    if negative_sampling_rate == 0:
        return None
    # else, let's do negative sampling
    # number of negative samples
    size_of_batch = len(pos_samples)
    num_to_generate = size_of_batch * negative_sampling_rate
    # create container for negative samples
    neg_samples = np.tile(pos_samples, [negative_sampling_rate, 1])
    neg_samples[:, -1] = 0 # set trip volume to be 0
    # perturbing the edge
    sample_nid = np.random.randint(num_nodes, size = num_to_generate) # randomly sample nodes
    pos_choices = np.random.uniform(size = num_to_generate) # randomly sample position
    subj = pos_choices > 0.5
    obj = pos_choices <= 0.5
    neg_samples[subj, 0] = sample_nid[subj]
    neg_samples[obj, 1] = sample_nid[obj]
    # sanity check
    while(True):
        # check overlap edges
        overlap = npi.contains(pos_samples[:, :2], neg_samples[:, :2]) # True means overlap
        if overlap.any(): # if there is any overlap edge, resample for these edges
            # get the overlap subset
            neg_samples_overlap = neg_samples[overlap]
            # resample
            sample_nid = np.random.randint(num_nodes, size = overlap.sum())
            pos_choices = np.random.uniform(size = overlap.sum())
            subj = pos_choices > 0.5
            obj = pos_choices <= 0.5
            neg_samples_overlap[subj, 0] = sample_nid[subj]
            neg_samples_overlap[obj, 1] = sample_nid[obj]
            # reassign the subset
            neg_samples[overlap] = neg_samples_overlap
        else: # if no overlap, just break resample loop
            break
    # return negative samples
    return torch.from_numpy(neg_samples)

def evaluate(model, g, trip_od, trip_volume):
    '''
    evaluate trained model.
    '''
    with torch.no_grad():
        # get embedding
        src_embedding = model(g)
        dst_embedding = model.forward2(g)
        # get prediction
        prediction = model.predict_edge(src_embedding, dst_embedding, trip_od)
        # get ground-truth label
        y = trip_volume.float().view(-1, 1)
        # get metric
        rmse = RMSE(prediction, y)
        mae = MAE(prediction, y)
        mape = MAPE(prediction, y)
        cpc = CPC(prediction, y)
        cpl = CPL(prediction, y)
    # return
    return rmse.item(), mae.item(), mape.item(), cpc.item(), cpl.item()


def scale(y):
    '''
    scale the target variable
    '''
    return torch.sqrt(y)


def scale_back(scaled_y):
    '''
    scale back the target varibale to normal scale
    '''
    return scaled_y ** 2


def RMSE(y_hat, y):
    '''
    Root Mean Square Error Metric
    '''
    return torch.sqrt(torch.mean((y_hat - y) ** 2))


def MAE(y_hat, y):
    '''
    Mean Absolute Error Metric
    '''
    abserror = torch.abs(y_hat - y)
    return torch.mean(abserror)


def MAPE(y_hat, y):
    '''
    Mean Absolute Error Metric
    '''
    abserror = torch.abs(y_hat - y)
    return torch.mean(abserror / y)


def CPC(y_hat, y):
    '''
    Common Part of Commuters Metric
    '''
    return 2 * torch.sum(torch.min(y_hat, y)) / (torch.sum(y_hat) + torch.sum(y))


def CPL(y_hat, y):
    '''
    Common Part of Links Metric.

    Check the topology.
    '''
    yy_hat = y_hat > 0
    yy = y > 0
    return 2 * torch.sum(yy_hat * yy) / (torch.sum(yy_hat) + torch.sum(yy))