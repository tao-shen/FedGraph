from configargparse import ArgumentParser
# import configargparse


def args():
    parser = ArgumentParser(default_config_files=['./config.yml'])
    parser.add_argument('--dataset', type=str, default='pubmed',
                        help='pubmed,cora,citeseer')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='device: {cuda, cpu}')
    parser.add_argument("--E", type=int, default=3,
                        help="dropout probability")
    parser.add_argument("--num_clients", type=int, default=3,
                        help="dropout probability")
    parser.add_argument("--in_feats", type=int, default=500,
                        help="dropout probability")
    parser.add_argument("--n_classes", type=int, default=3,
                        help="dropout probability")
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
    parser.set_defaults(self_loop=False)

    args = parser.parse_args()

    return args
