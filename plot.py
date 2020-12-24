import networkx as nx
import matplotlib.pyplot as plt
import torch


def plotfig(recorder):
    recorder = torch.load('./saves/recorder.pt',
                          map_location=torch.device('cpu'))
    plt.style.use('seaborn')
    d = recorder['test_acc']['clients']
    # mpl.rcParams['lines.linewidth'] = 2
    # mpl.rcParams['lines.color'] = 'r'
    plt.plot(recorder['test_acc']['server'])
    for k in range(len(d)):
        plt.plot(d[k])
    plt.savefig('./saves/test_acc.pdf')


def visualize(g):
    nx_G = g.to_networkx()
    labels = g.ndata['label']
    pos = nx.spring_layout(nx_G)
    plt.figure(figsize=(8, 8))
    plt.axis('off')
    nx.draw_networkx(nx_G, pos=pos, node_size=50, cmap=plt.get_cmap('coolwarm'),
                     node_color=labels, edge_color='k',
                     arrows=False, width=0.5, style='dotted', with_labels=False)
    plt.savefig('./graph.pdf')
