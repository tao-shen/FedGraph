import torch.nn as nn
from dgl.nn.pytorch import GraphConv
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_feats, n_hidden, num_classes, n_layers, dropout):
        super(MLP, self).__init__()
        self.activation = F.relu
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_feats, n_hidden))
        for i in range(n_layers - 1):
            self.layers.append(nn.Linear(n_hidden, n_hidden))
        self.layers.append(nn.Linear(n_hidden, num_classes))

    def forward(self, h):
        for i, layer in enumerate(self.layers):
            # if i != 0:
            # h = self.dropout(h)
            h = self.activation(layer(h))
        return h


class GCN(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 num_classes,
                 n_layers,
                 dropout):
        super(GCN, self).__init__()
        # self.g = g
        self.layers = nn.ModuleList()
        self.activation = F.relu
        # input layer
        self.layers.append(
            GraphConv(in_feats, n_hidden, activation=self.activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(
                GraphConv(n_hidden, n_hidden, activation=self.activation))
        # output layer
        self.layers.append(GraphConv(n_hidden, num_classes))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, g, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)
        return h
