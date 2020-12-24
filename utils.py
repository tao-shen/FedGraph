import torch
from models import *
from random import shuffle as sf

loss_fcn = torch.nn.CrossEntropyLoss()


def shuffle(*args):
    import random
    random.seed(0)
    for i in args:
        sf(i)


def init_model(args):
    if args.model_type == 'GCN':
        return GCN(args.in_feats, args.n_hidden,
                   args.num_classes, args.n_layers, args.dropout).to(args.device)
    elif args.model_type == 'MLP':
        return MLP(args.in_feats, args.n_hidden,
                   args.num_classes, args.n_layers, args.dropout).to(args.device)
    else:
        raise ValueError('Unknown model type {}'.format(args.model_type))


def init_optimizer(model, args):
    return torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def weights_zero(model):
    for p in model.parameters():
        if p.data is not None:
            p.data.detach_()
            p.data.zero_()


def setup_seed(seed):
    import numpy as np
    import random
    from torch.backends import cudnn
    import dgl
    dgl.seed(seed)
    dgl.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.seed_all()
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def train(client):
    client.model.train()
    logits = client.model(client.g, client.feats)
    logits = logits[client.train_mask]
    labels = client.labels[client.train_mask]
    loss = loss_fcn(logits, labels)
    client.optimizer.zero_grad()
    loss.backward()
    client.optimizer.step()


def evaluate(model, target, mask='test'):
    if mask == 'test':
        mask = target.test_mask
    elif mask == 'val':
        mask = target.val_mask
    elif mask == 'train':
        mask = target.train_mask
    else:
        raise ValueError('Unknown mask {}'.format(mask))
    model.eval()
    with torch.no_grad():
        logits = model(target.g, target.feats)
        logits = logits[mask]
        labels = target.labels[mask]
        pred = logits.argmax(dim=1)
        correct = torch.sum(pred == labels)
        return correct.item() * 1.0 / len(labels)
