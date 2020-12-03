import copy
from utils import *


class Client(object):
    def __init__(self, id, graph, args):
        self.args = args
        self.id = id + 1
        self.g = graph.int().to(args.device)
        self.feats = self.g.ndata['feat']
        self.labels = self.g.ndata['label']
        self.train_mask = self.g.ndata['train_mask']
        self.val_mask = self.g.ndata['val_mask']
        self.test_mask = self.g.ndata['test_mask']
        self.model = init_model(self.args).to(args.device)
        self.optimizer = init_optimizer(self.model, self.args)

    def fork(self, server):
        self.model = copy.deepcopy(server.model)
        self.optimizer = init_optimizer(self.model, self.args)

    def local_update(self):
        for E in range(self.args.E):
            train(self)


class Server(object):
    def __init__(self, graph, args):
        self.args = args
        self.g = graph.int().to(args.device)
        self.feats = self.g.ndata['feat']
        self.labels = self.g.ndata['label']
        self.model = init_model(self.args).to(args.device)
        self.train_mask = self.g.ndata['train_mask']
        self.val_mask = self.g.ndata['val_mask']
        self.test_mask = self.g.ndata['test_mask']
        self.dict = self.model.state_dict()

    def merge(self, clients):
        weights_zero(self.model)
        clients_states = [copy.deepcopy(
            clients[k].model.state_dict()) for k in range(len(clients))]
        for key in self.dict.keys():
            for i in range(len(clients)):
                self.dict[key] += clients_states[i][key]
            self.dict[key] /= len(clients)
