import copy
import torch


def weights_zero(model):
    for p in model.parameters():
        if p.data is not None:
            p.data.detach_()
            p.data.zero_()


def fork(gl, model, optimizer, args):
    model = copy.deepcopy(gl)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)


def merge(gl, model):
    weights_zero(gl)
    Dict = gl.state_dict()
    Node_State_List = [copy.deepcopy(
        model[i].state_dict()) for i in range(len(model))]
    for key in Dict.keys():
        for i in range(len(model)):
            Dict[key] += Node_State_List[i][key]
        Dict[key] /= len(model)
