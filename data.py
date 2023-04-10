import torch
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset, BitcoinOTCDataset, RedditDataset
from dgl.partition import metis_partition_assignment as min_cut
from dgl.random import choice as random_choice
from dgl import node_subgraph, remove_self_loop, add_self_loop
from utils import *
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms.community import asyn_fluidc


def data_load(args):
    # load and preprocess dataset
    if args.dataset == 'cora':
        data = CoraGraphDataset()
    elif args.dataset == 'citeseer':
        data = CiteseerGraphDataset()
    elif args.dataset == 'pubmed':
        data = PubmedGraphDataset()
    elif args.dataset == 'bitcoin':
        data = BitcoinOTCDataset()
    elif args.dataset == 'reddit':
        data = RedditDataset()
    else:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))
    g = data[0]
    args.device = torch.device(
        args.device if torch.cuda.is_available() else 'cpu')
    g.int().to(args.device)
    args.in_feats = g.ndata['feat'].shape[1]
    args.num_classes = data.num_classes
    return g


def major_connected_graph(G):
    major = max(nx.connected_components(G), key=len)
    r = G.nodes()-major
    G.remove_nodes_from(r)


def data_sample(g, args):
    if args.split_method == 'b_min_cut':
        # balance min-cut
        labels = g.ndata['label']
        assign = min_cut(g, args.split, balance_ntypes=labels).tolist()
        index = [[] for i in range(args.split)]
        [index[ind].append(i) for i, ind in enumerate(assign)]
    elif args.split_method == 'random_choice':
        # random_choice
        assign = random_choice(args.split, g.number_of_nodes()).tolist()
        index = [[] for i in range(args.split)]
        [index[ind].append(i) for i, ind in enumerate(assign)]
    elif args.split_method == 'ub_min_cut':
        # unbalance min-cut
        assign = min_cut(g, args.split).tolist()
        index = [[] for i in range(args.split)]
        [index[ind].append(i) for i, ind in enumerate(assign)]
    elif args.split_method == 'community_detection':
        # community detection
        G = g.to_networkx().to_undirected()
        major_connected_graph(G)
        index = list(asyn_fluidc(G, args.split, seed=0))
        index = [list(k) for k in index]
    else:
        raise ValueError('Unknown split_method: {}'.format(args.split_method))

    # mini_batch

    return index


def data_split(g, args):

    index = data_sample(g, args)
    # num_nodes = g.number_of_nodes()
    # train_mask = g.ndata['train_mask']
    # val_mask = g.ndata['val_mask']
    # test_mask = g.ndata['test_mask']
    # train_ind = torch.where(train_mask == True)[0].tolist()
    # val_ind = torch.where(val_mask == True)[0].tolist()
    # test_ind = torch.where(test_mask == True)[0].tolist()
    # else_ind = list(set(range(num_nodes)) -
    #                 set(train_ind) - set(val_ind) - set(test_ind))
    # shuffle(train_ind, val_ind, test_ind, else_ind)
    # shard = [int(len(train_ind)/args.split), int(len(val_ind)/args.split),
    #          int(len(test_ind)/args.split), int(len(else_ind)/args.split)]
    # ind = [train_ind[i*shard[0]:(i+1)*shard[0]] + val_ind[i*shard[1]:(i+1)*shard[1]] + test_ind[i*shard[2]:(
    #     i+1)*shard[2]] + else_ind[i*shard[3]:(i+1)*shard[3]] for i in range(args.split)]
    graphs = [node_subgraph(g, index[i]) for i in range(args.split)]

    for i in range(len(graphs)):
        # graphs[i].int().to(args.device)
        # add self loop
        graphs[i] = remove_self_loop(graphs[i])
        graphs[i] = add_self_loop(graphs[i])

    return graphs
