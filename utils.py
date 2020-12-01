import torch
from models import GCN


def init_model(args):
    return GCN(args.in_feats, args.n_hidden,
               args.n_classes, args.n_layers, args.dropout)


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
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True


def evaluate(model, g, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
        logits = logits[mask]
        labels = labels[mask]
        pred = logits.argmax(dim=1)
        correct = torch.sum(pred == labels)
        return correct.item() * 1.0 / len(labels)
