import torch
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from dgl import node_subgraph, remove_self_loop, add_self_loop
from utils import *


def data_load(args):
    # args.dataset = 'pubmed'
    # load and preprocess dataset
    if args.dataset == 'cora':
        data = CoraGraphDataset()
    elif args.dataset == 'citeseer':
        data = CiteseerGraphDataset()
    elif args.dataset == 'pubmed':
        data = PubmedGraphDataset()
    else:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))
    g = data[0]
    args.in_feats = g.ndata['feat'].shape[1]
    args.num_classes = data.num_classes
    return g


def data_split(g, args):
    num_nodes = g.number_of_nodes()
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    train_ind = torch.where(train_mask == True)[0].tolist()
    val_ind = torch.where(val_mask == True)[0].tolist()
    test_ind = torch.where(test_mask == True)[0].tolist()
    else_ind = list(set(range(num_nodes)) -
                    set(train_ind) - set(val_ind) - set(test_ind))
    shuffle(train_ind, val_ind, test_ind, else_ind)
    shard = [int(len(train_ind)/args.split), int(len(val_ind)/args.split),
             int(len(test_ind)/args.split), int(len(else_ind)/args.split)]
    ind = [train_ind[i*shard[0]:(i+1)*shard[0]] + val_ind[i*shard[1]:(i+1)*shard[1]] + test_ind[i*shard[2]:(
        i+1)*shard[2]] + else_ind[i*shard[3]:(i+1)*shard[3]] for i in range(args.split)]
    graphs = [node_subgraph(g, ind[i]) for i in range(args.split)]
    # sg1 = node_subgraph(g, range(0, 6000))
    # sg2 = node_subgraph(g, range(6000, 12000))
    # sg3 = node_subgraph(g, range(12000, 18000))
    # sg4 = node_subgraph(g, range(18000, 19717))
    # graphs = [sg1, sg2, sg3, sg4]

    # cg = node_subgraph(g, range(0, 18000))
    # clabels = cg.ndata['label']
    # c1 = torch.where(clabels == 0)
    # c2 = torch.where(clabels == 1)
    # c3 = torch.where(clabels == 2)
    # cg1 = node_subgraph(cg, c1)
    # cg2 = node_subgraph(cg, c2)
    # cg3 = node_subgraph(cg, c3)
    args.device = torch.device(
        args.device if torch.cuda.is_available() else 'cpu')
    g.int().to(args.device)
    for i in range(len(graphs)):
        graphs[i].int().to(args.device)
        # # add mask
        # graphs[i].ndata['train_mask'][0:60] = True
        # graphs[i].ndata['val_mask'][0:500] = True
        # graphs[i].ndata['test_mask'][0:1000] = True
        # add self loop
        graphs[i] = remove_self_loop(graphs[i])
        graphs[i] = add_self_loop(graphs[i])
    return graphs
