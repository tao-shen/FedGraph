"""GCN using DGL nn package

References:
- Semi-Supervised Classification with Graph Convolutional Networks
- Paper: https://arxiv.org/abs/1609.02907
- Code: https://github.com/tkipf/gcn
"""
import torch
import torch.nn as nn
from dgl.nn.pytorch import GraphConv
import torch.nn.functional as F


class MLP(nn.Module):  # mlp用来跑mnist
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(500, 500)
        self.fc2 = nn.Linear(500, 300)
        self.fc3 = nn.Linear(300, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class GCN(nn.Module):
    def __init__(self,

                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super(GCN, self).__init__()
        # self.g = g
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(
            GraphConv(in_feats, n_hidden, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(
                GraphConv(n_hidden, n_hidden, activation=activation))
        # output layer
        self.layers.append(GraphConv(n_hidden, n_classes))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, g, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)
        return h
