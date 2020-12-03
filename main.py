from models import *
from utils import *
from fl import *
from data import *
from args import *

# Initialize
args = args()
setup_seed(2020)
# Data load & split
g = data_load(args)
graphs = data_split(g, args)
# FL initialize
server = Server(g, args)
clients = [Client(k, graphs[k], args) for k in range(args.num_clients)]
new_client = Client(-1, graphs[-1], args)
# FLearning
for _ in range(int(args.n_epochs/args.E)):
    for k in range(len(clients)):
        # Fork
        clients[k].fork(server)
        # Local_update
        clients[k].local_update()
    # Merge
    server.merge(clients)
# Evaluate
for k in range(len(clients)):
    acc = evaluate(clients[k].model, clients[k], mask='test')
    print("Client{}: {:.2%}".format(clients[k].id, acc))
acc = evaluate(server.model, server)
print("Server: {:.2%}".format(acc))
acc = evaluate(server.model, new_client)
print("Client{}: {:.2%}".format(new_client.id, acc))
