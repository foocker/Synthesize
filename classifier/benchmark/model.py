from torchvision import models
import torch.nn as nn
import torch


def set_parameter_requires_grad(model, layer_num=-1):
    for child in list(model.children())[:layer_num]:
        for param in child.parameters():
            param.requires_grad = False


def create_model(num_classes):
    model = models.resnet18(pretrained=True)
    set_parameter_requires_grad(model, layer_num=-3)
    num_fea = model.fc.in_features
    model.fc = nn.Linear(num_fea, num_classes)

    return model



def get_best_epoch(weight_path):
    filename = weight_path.split('/')[-1]
    history_best_epoch = 0 
    name_list = filename.split('_')
    for i, x in enumerate(name_list):
        if x == 'epoch':
            history_best_epoch = int(name_list[i+1])
            break
    return history_best_epoch


def reusemodel(model, weight_path):
    state_dict = torch.load(weight_path, map_location=torch.device('cpu'))
    if 'model_state_dict' in state_dict.keys():
        print("load from model_state_dict!!!")
        model.load_state_dict(state_dict['model_state_dict'])
        best_epoch = state_dict['epoch']
    else:
        print("not load from model_state_dict!!!")
        model.load_state_dict(state_dict)
        best_epoch = get_best_epoch(weight_path)
    return model, best_epoch


def progressive_model(model, weight_path, new_numclass):
    model, best_epoch = reusemodel(model, weight_path)
    num_fea = model.fc.in_features
    model.fc = nn.Linear(num_fea, new_numclass)
    return model, best_epoch


def get_params_to_update(model):
    params_to_update = []
    for _, param in model.named_parameters():
        if param.requires_grad:
            params_to_update.append(param)
    return params_to_update

