import dgl
from models import *
from utils import *
from fl import *
from data import *
from args import *
from plot import *
import os

# Initialize
os.makedirs('saves', exist_ok=True)
args = args()
setup_seed(args.seed)
recorder = {'train_loss': {'clients': [[] for k in range(args.split)], 'server': []},
            'val_loss': {'clients': [[] for k in range(args.split)], 'server': []},
            'test_loss': {'clients': [[] for k in range(args.split)], 'server': []},
            'train_acc': {'clients': [[] for k in range(args.split)], 'server': []},
            'val_acc': {'clients': [[] for k in range(args.split)], 'server': []},
            'test_acc': {'clients': [[] for k in range(args.split)], 'server': []}}
# Data load & split
g = data_load(args)
e = g.edges(form='eid')
graphs = data_split(g, args)
# subfigs(graphs,args)
# FL initialize
server = Server(g, args)
clients = [Client(k, graphs[k], args) for k in range(args.num_clients)]
for i in clients:
    i.g.remove_edges(i.g.edges(form='eid'))
    # i.g = add_self_loop(i.g)
    add_edges(i.g)
new_client = Client(-1, graphs[-1], args)
# FLearning
for _ in range(int(args.n_epochs)):
    for k in range(len(clients)):
        # Fork
        clients[k].fork(server)
        # Evaluate
        acc = evaluate(clients[k].model, clients[k], mask='test')
        recorder['test_acc']['clients'][k+1].append(acc)
        # Local_update
        clients[k].local_update()
        # # Evaluate
        # acc = evaluate(clients[k].model, clients[k], mask='test')
        # recorder['test_acc']['clients'][k+1].append(acc)
    # Merge
    server.merge(clients)
    acc = evaluate(server.model, server)
    recorder['test_acc']['server'].append(acc)
    acc = evaluate(server.model, new_client)
    recorder['test_acc']['clients'][0].append(acc)
# Evaluate
for k in range(len(clients)):
    acc = evaluate(clients[k].model, clients[k], mask='test')
    print("Client{}: {:.2%}".format(clients[k].id, acc))
acc = evaluate(server.model, server)
print("Server: {:.2%}".format(acc))
server.g.remove_edges(server.g.edges(form='eid'))
server.g = add_self_loop(server.g)
acc = evaluate(server.model, server)
print("Server: {:.2%}".format(acc))
acc = evaluate(server.model, new_client)
print("Client{}: {:.2%}".format(new_client.id, acc))
new_client.g.remove_edges(new_client.g.edges(form='eid'))
new_client.g = add_self_loop(new_client.g)
acc = evaluate(server.model, new_client)
print("Client{}: {:.2%}".format(new_client.id, acc))
torch.save(recorder, './saves/recorder.pt')
plotfig(recorder, args)
