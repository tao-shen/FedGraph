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
