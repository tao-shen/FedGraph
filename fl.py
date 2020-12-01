import copy
import torch
from utils import *


class Client(object):
    def __init__(self, id, data, mask, args):
        self.args = args
        self.id = id + 1
        self.data = data
        self.train_data = data[mask['train']]
        self.val_data = data[mask['val']]
        self.model = init_model(self.args).cuda()
        self.optimizer = init_optimizer(self.model, self.args)

    def fork(self, server):
        self.model = copy.deepcopy(server.model)
        self.optimizer = init_optimizer(self.model, self.args)


class Server(object):
    def __init__(self, test_data, args):
        self.id = 0
        self.args = args
        self.model = init_model(self.args).cuda()
        self.test_data = test_data
        self.dict = self.model.state_dict()

    def merge(self, clients):
        weights_zero(self.model)
        clients_states = [copy.deepcopy(
            clients[i].meme.state_dict()) for i in range(len(clients))]
        for key in self.dict.keys():
            for i in range(len(clients)):
                self.dict[key] += clients_states[i][key]
            self.dict[key] /= len(clients)
