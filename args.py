from configargparse import ArgumentParser


def data_args(parser):
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--in_feats", type=int)
    parser.add_argument("--num_classes", type=int)
    parser.add_argument('--split', type=int, default=4)
    # parser.add_argument("--self-loop", type=bool, default=True)


def args():
    parser = ArgumentParser(default_config_files=['./config.yml'])

    data_args(parser)
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='device: {cuda, cpu}')
    parser.add_argument("--E", type=int, default=3,
                        help="dropout probability")
    parser.add_argument("--num_clients", type=int, default=3,
                        help="dropout probability")
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="dropout probability")
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

    args = parser.parse_args()

    return args
