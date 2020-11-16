import dgl
import torch as th
g = dgl.graph(([0, 0, 1, 5], [1, 2, 2, 0]))  # 6 nodes, 4 edges
print(g)
# node feature of length 3
g.ndata['x'] = th.ones(g.num_nodes(), 3)
g.edata['x'] = th.ones(g.num_edges(), dtype=th.int32)  # scalar integer feature
print(g)
# different names can have different shapes
g.ndata['y'] = th.randn(g.num_nodes(), 5)
print(g)
print(g.ndata['x'][1])                  # get node 1's feature
print(g.edata['x'][th.tensor([0, 3])])  # get features of edge 0 and 3
