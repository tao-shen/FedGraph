from tqdm import tqdm
import torch.nn as nn
import torch


def local_update(client):
    client.model.cuda().train()
    client.g.cuda()
    logits = client.model(client.g, client.features)
    a = logits[client.train_mask]
    b = client.labels[client.train_mask]
    loss = loss_fcn(a, b)
    client.optimizer.zero_grad()
    loss.backward()
    client.optimizer.step()


class Trainer(object):

    def __init__(self, args):
        if args.algorithm == 'fed_mutual':
            self.train = train_mutual
        elif args.algorithm == 'fed_avg':
            self.train = train_avg
        elif args.algorithm == 'normal':
            self.train = train

    def __call__(self, node):
        self.train(node)
