import argparse
import time
import torch
import torch.nn.functional as F
from dgl import node_subgraph, remove_self_loop, add_self_loop
from dgl.data import register_data_args
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from models import GCN, MLP
import copy
#from gcn_mp import GCN
#from gcn_spmv import GCN


def weights_zero(model):
    for p in model.parameters():
        if p.data is not None:
            p.data.detach_()
            p.data.zero_()


def setup_seed(seed):
    import numpy as np
    import random
    from torch.backends import cudnn
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True


def evaluate(model, g, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
        logits = logits[mask]
        labels = labels[mask]
        # _, indices = torch.max(logits, dim=1)
        # correct = torch.sum(indices == labels)
        pred = logits.argmax(dim=1)
        correct = torch.sum(pred == labels)
        return correct.item() * 1.0 / len(labels)


def main(args):
    setup_seed(2020)
    args.dataset = 'pubmed'
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

    sg1 = node_subgraph(g, range(0, 6000))
    sg2 = node_subgraph(g, range(6000, 12000))
    sg3 = node_subgraph(g, range(12000, 18000))
    sg4 = node_subgraph(g, range(18000, 19717))
    sg = [sg1, sg2, sg3, sg4]

    # cg = node_subgraph(g, range(0, 18000))
    # clabels = cg.ndata['label']
    # c1 = torch.where(clabels == 0)
    # c2 = torch.where(clabels == 1)
    # c3 = torch.where(clabels == 2)
    # cg1 = node_subgraph(cg, c1)
    # cg2 = node_subgraph(cg, c2)
    # cg3 = node_subgraph(cg, c3)

    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        g = g.int().to(args.gpu)
        for i in range(4):
            sg[i] = sg[i].int().to(args.gpu)

    allfeatures = g.ndata['feat']
    alllabels = g.ndata['label']
    alltrain_mask = g.ndata['train_mask']
    allval_mask = g.ndata['val_mask']
    alltest_mask = g.ndata['test_mask']
    allin_feats = allfeatures.shape[1]
    alln_classes = data.num_classes
    alln_edges = g.number_of_edges()

    features = []
    labels = []
    train_mask = []
    val_mask = []
    test_mask = []
    in_feats = []
    n_classes = []
    n_edges = []
    for i in range(4):
        features.append(sg[i].ndata['feat'])
        labels.append(sg[i].ndata['label'])
        train_mask.append(sg[i].ndata['train_mask'])
        val_mask.append(sg[i].ndata['val_mask'])
        test_mask.append(sg[i].ndata['test_mask'])
        in_feats.append(features[i].shape[1])
        n_classes.append(data.num_classes)
        n_edges.append(sg[i].number_of_edges())

    for i in range(3):
        train_mask[i][0:60] = True
        val_mask[i][0:500] = True
        test_mask[i][0:1000] = True
    # features = g.ndata['feat']
    # labels = g.ndata['label']
    # train_mask = g.ndata['train_mask']
    # val_mask = g.ndata['val_mask']
    # test_mask = g.ndata['test_mask']
    # in_feats = features.shape[1]
    # n_classes = data.num_classes
    # n_edges = g.number_of_edges()

    # num_classes = labels.shape[0]
    # num = in_feats = features.shape[0]
    # features = features[0:int(num/3), :]
    # in_feats = int(in_feats/3)

    # print("""----Data statistics------'
    #   #Edges %d
    #   #Classes %d
    #   #Train samples %d
    #   #Val samples %d
    #   #Test samples %d""" %
    #       (n_edges, n_classes,
    #           train_mask.int().sum().item(),
    #           val_mask.int().sum().item(),
    #           test_mask.int().sum().item()))

    # add self loop
    args.self_loop = True
    for i in range(4):
        if args.self_loop:
            sg[i] = remove_self_loop(sg[i])
            sg[i] = add_self_loop(sg[i])
        n_edges[i] = sg[i].number_of_edges()

    # normalization
    # degs = g.in_degrees().float()
    # norm = torch.pow(degs, -0.5)
    # norm[torch.isinf(norm)] = 0
    # if cuda:
    #     norm = norm.cuda()
    # g.ndata['norm'] = norm.unsqueeze(1)

    # create GCN model
    model = []
    gl = GCN(
        in_feats[0],
        args.n_hidden,
        n_classes[0],
        args.n_layers,
        args.dropout)
    for i in range(3):
        model.append(GCN(
            in_feats[i],
            args.n_hidden,
            n_classes[i],
            args.n_layers,
            args.dropout))
    # model = MLP()
    if cuda:
        gl.cuda()
        for i in range(3):
            model[i].cuda()
    loss_fcn = torch.nn.CrossEntropyLoss()

    # use optimizer
    optimizer = []
    for i in range(3):
        optimizer.append(torch.optim.Adam(model[i].parameters(),
                                          lr=args.lr,
                                          weight_decay=args.weight_decay))

    # initialize graph
    E = 3
    dur = []
    for epoch in range(int(args.n_epochs/E)):
        for k in range(E):
            for i in range(3):
                # print('node{}'.format(i))

                # Fork
                model[i] = copy.deepcopy(gl)
                optimizer[i] = torch.optim.Adam(model[i].parameters(),
                                                lr=args.lr,
                                                weight_decay=args.weight_decay)

                model[i].train()
                if epoch >= 3:
                    t0 = time.time()

                # forward
                logits = model[i](sg[i], features[i])
                a = logits[train_mask[i]]
                b = labels[i][train_mask[i]]
                loss = loss_fcn(a, b)

                optimizer[i].zero_grad()
                loss.backward()
                optimizer[i].step()

                if epoch >= 3:
                    dur.append(time.time() - t0)

                acc = evaluate(model[i], sg[i], features[i],
                               labels[i], val_mask[i])
                # print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
                #       "ETputs(KTEPS) {:.2f}". format(epoch, np.mean(dur), loss.item(),
                #                                      acc, n_edges[i] / np.mean(dur) / 1000))

        # Merge
        weights_zero(gl)
        Dict = gl.state_dict()
        Node_State_List = [copy.deepcopy(
            model[i].state_dict()) for i in range(len(model))]
        for key in Dict.keys():
            for i in range(len(model)):
                Dict[key] += Node_State_List[i][key]
            Dict[key] /= len(model)

    for i in range(3):
        acc = evaluate(model[i], sg[i], features[i], labels[i], test_mask[i])
        # print("Test accuracy {:.2%}".format(acc))
        print("{:.2%}".format(acc))

    inacc = evaluate(gl, sg[-1], features[-1], labels[-1], test_mask[-1])
    # print("Test accuracy {:.2%}".format(inacc))
    print("{:.2%}".format(inacc))
    allacc = evaluate(gl, g, allfeatures, alllabels, alltest_mask)
    # print("Test accuracy {:.2%}".format(allacc))
    print("{:.2%}".format(allacc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    register_data_args(parser)
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="dropout probability")
    parser.add_argument("--gpu", type=int, default=0,
                        help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=200,
                        help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=16,
                        help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=1,
                        help="number of hidden gcn layers")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
                        help="Weight for L2 loss")
    parser.add_argument("--self-loop", action='store_true',
                        help="graph self-loop (default=False)")
    # parser.add_argument("--dataset", type=str, default='cora',
    #                     help="dataset")
    parser.set_defaults(self_loop=False)
    args = parser.parse_args()
    print(args)

    main(args)
