import networkx as nx
import matplotlib.pyplot as plt
import torch


def plotfig(recorder, args):
    recorder = torch.load('./saves/recorder.pt',
                          map_location=torch.device('cpu'))
    plt.style.use('seaborn')
    d = recorder['test_acc']['clients']
    # mpl.rcParams['lines.linewidth'] = 2
    # mpl.rcParams['lines.color'] = 'r'
    plt.plot(recorder['test_acc']['server'], label='server')
    for k in range(len(d)):
        plt.plot(d[k], label='client{}'.format(k))
    plt.legend()
    plt.title('test_acc with {}'.format(args.split_method))
    plt.savefig('./saves/test_acc.pdf')


def visualize(g):
    nx_G = g.to_networkx()
    labels = g.ndata['label']
    pos = nx.spring_layout(nx_G)
    # plt.figure(figsize=(8, 8))
    # plt.axis('off')
    nx.draw_networkx(nx_G, pos=pos, node_size=5, cmap=plt.get_cmap('coolwarm'),
                     node_color=labels, edge_color='k',
                     arrows=False, width=0.5, style='dotted', with_labels=False)
    # plt.savefig('./graph.pdf')


def plotg(g):
    nx_G = g.to_networkx()
    # labels = g.ndata['label']
    # pos = nx.spring_layout(nx_G)
    # plt.figure(figsize=(8, 8))
    # plt.axis('off')
    nx.draw_networkx(nx_G, node_size=5, cmap=plt.get_cmap('coolwarm'),
                     edge_color='k',
                     arrows=False, width=0.5, style='dotted', with_labels=False)
    plt.savefig('./graph.pdf')

def subfigs(graphs, args):
    plt.subplot(221)
    # visualize(graphs[0])
    plotg(graphs[0])
    plt.title('client1')
    plt.subplot(222)
    # visualize(graphs[1])
    plotg(graphs[1])
    plt.title('client2')
    plt.subplot(223)
    # visualize(graphs[2])
    plotg(graphs[2])
    plt.title('client3')
    plt.subplot(224)
    # visualize(graphs[3])
    plotg(graphs[3])
    plt.title('client0')
    plt.suptitle(args.split_method)
    plt.savefig('./allgraphs.pdf')
