import os,sys, copy, json
import torch
import torch.nn as nn
import random
import numpy as np


def read_json(path):
    with open(path, "r") as st_json:
        json_file = json.load(st_json)
    return json_file


class RMSELoss(nn.Module):
    """ Simple torch implementation of the RMSE loss. """
    def __init__(self, reduction='none', eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = torch.tensor(eps)

    def forward(self, y_pred, y):
        loss = torch.sqrt(self.mse(y_pred, y) + self.eps.to(y.device))
        return loss

class BiLoss(nn.Module):
    """ Simple torch implementation of the RMSE loss. """
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, y_pred, y):
        loss = self.bce(y_pred.squeeze(), y.float())
        return loss

def set_loss(task_type):
    if task_type == "multi":
        return nn.CrossEntropyLoss()
    elif task_type == "bi":
        return BiLoss()
    else:
        return RMSELoss()


def fix_random_seed(random_num):
    torch.manual_seed(random_num)
    torch.cuda.manual_seed(random_num)
    torch.cuda.manual_seed_all(random_num) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_num)
    random.seed(random_num)

def save_score_model(model, score_info_dic, model_dic, score_names, val_scores, test_scores, epoch):
    change = False
    for name, val in zip(score_names, val_scores):
        if name not in score_info_dic:
            score_info_dic[name] = (val,dict(zip(score_names, test_scores)),epoch)
            model_dic["model"][name] = copy.deepcopy(model.state_dict())
            change = True
        elif score_info_dic[name][0] < val:
            score_info_dic[name] = (val,dict(zip(score_names, test_scores)),epoch)
            model_dic["model"][name] = copy.deepcopy(model.state_dict())
            change = True
    return change

def mkdir(save_loc,data_name):
    # os.makedirs(save_loc+"/param/" + data_name, exist_ok=True)
    os.makedirs(save_loc+"/tensorboard/" + data_name, exist_ok=True)
    # os.makedirs(save_loc+"/save_model/" + data_name, exist_ok=True)
    os.makedirs(save_loc+"/score/" + data_name, exist_ok=True)
