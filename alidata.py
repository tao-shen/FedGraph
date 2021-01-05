import csv
import torch
import dgl
from plot import *
from data import *
from networkx import strongly_connected_components
from args import *
args = args()
l = [[] for _ in range(1144)]
al = []
ala = []
alb = []
for day in range(1, 2):
    with open('ali/day{}.csv'.format(day), 'rt') as f:
        cr = csv.DictReader(f)
        for row in cr:
            l[int(row['user_id'])].append(int(row['item_id']))
# torch.where(l='user_id')
for i in range(len(l)):
    a = l[i][0:-1]
    b = l[i][1:]
    ala.extend(a)
    alb.extend(b)
g = dgl.graph((ala, alb))
# graphs = data_split(g, args)
# subfigs(graphs, args)
G = g.to_networkx()
a = strongly_connected_components(G)
c = [len(c) for c in sorted(
    nx.strongly_connected_components(G), key=len, reverse=True)]
# plotg(g)
a = 1
