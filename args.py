from configargparse import ArgumentParser
from torch._C import default_generator


def fl_args(parser):
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='device: {cuda, cpu}')
    parser.add_argument("--E", type=int, default=3,
                        help="dropout probability")
    parser.add_argument("--num_clients", type=int, default=3,
                        help="dropout probability")
    parser.add_argument("--n_epochs", type=int, default=200,
                        help="number of training epochs")

def data_args(parser):
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--in_feats", type=int)
    parser.add_argument("--num_classes", type=int)
    parser.add_argument('--split', type=int, default=4)
    parser.add_argument('--split_method', type=str, default='b_min-cut')
    # parser.add_argument("--self-loop", type=bool, default=True)


def model_args(parser):
    parser.add_argument("--model_type", type=str, default='GCN',
                        help="GCN/MLP")
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="dropout probability")
    parser.add_argument("--n-hidden", type=int, default=16,
                        help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=1,
                        help="number of hidden gcn layers")
                    


def optim_args(parser):
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="learning rate")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
                        help="Weight for L2 loss")


def args():
    parser = ArgumentParser(default_config_files=['./config.yml'])
    fl_args(parser)
    data_args(parser)
    model_args(parser)
    optim_args(parser)
    args = parser.parse_args()
    if args.split_method == 'b_min-cut' or 'random_choice':
        pass
    else:
        args.n_epochs = int(args.n_epochs*2.5)
    return args
