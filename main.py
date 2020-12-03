import argparse
from dgl.data import register_data_args
from models import *
from utils import *
from fl import *
from data import *


def main(args):
    setup_seed(2020)
    g = data_load(args)
    graphs = data_split(g, args)
    server = Server(g, args)
    clients = [Client(k, graphs[k], args) for k in range(args.num_clients)]
    new_client = Client(-1, graphs[-1], args)

    for _ in range(int(args.n_epochs/args.E)):
        for k in range(3):
            # Fork
            clients[k].fork(server)
            # Local_update
            clients[k].local_update()
        # Merge
        server.merge(clients)

    for k in range(3):
        acc = evaluate(clients[k].model, clients[k], mask='test')
        print("{:.2%}".format(acc))
    # inacc = evaluate(new_client)
    # print("{:.2%}".format(inacc))
    allacc = evaluate(server.model, server)
    print("{:.2%}".format(allacc))
    acc = evaluate(server.model, new_client)
    print("{:.2%}".format(acc))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fedgraph')
    register_data_args(parser)
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='device: {cuda, cpu}')
    parser.add_argument("--E", type=float, default=3,
                        help="dropout probability")
    parser.add_argument("--num_clients", type=float, default=3,
                        help="dropout probability")
    parser.add_argument("--in_feats", type=float, default=500,
                        help="dropout probability")
    parser.add_argument("--n_classes", type=float, default=3,
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
    print(args)
    main(args)
