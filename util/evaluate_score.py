import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import explained_variance_score, mean_squared_error, mean_absolute_error, r2_score
import torch


def cal_biclass_score(y_hat, y):
    """
    y_hat: (Batch * feature) -> prob matrix
    y : (Batch) -> matrix
    """
    if not isinstance(y_hat, np.ndarray):
        # y_hat = torch.nn.functional.softmax(y_hat,1)
        y_hat = y_hat.detach().cpu().numpy()
    if not isinstance(y,np.ndarray):
        y = y.cpu().numpy()
    y_hat_n = y_hat > 0
    # y_hat_n = np.argmax(y_hat, axis = 1)

    acc = accuracy_score(y, y_hat_n)
    mac_f1 = f1_score(y, y_hat_n)
    mic_f1 = f1_score(y, y_hat_n, average="micro")
    wei_f1 = f1_score(y, y_hat_n, average="weighted")
    # sam_f1 = f1_score(y, y_hat_n, average="samples")
    roc_auc = roc_auc_score(y, y_hat)
    
    return (["acc", "mac_f1", "mic_f1", "wei_f1", "roc_auc"], [acc, mac_f1, mic_f1, wei_f1, roc_auc])


def cal_multiclass_score(y_hat, y):
    """
    y_hat: (Batch * feature) -> prob matrix
    y : (Batch) -> matrix
    """
    if not isinstance(y_hat, np.ndarray):
        y_hat = torch.nn.functional.softmax(y_hat,1)
        y_hat = y_hat.detach().cpu().numpy()
    if not isinstance(y,np.ndarray):
        y = y.cpu().numpy()
    y_hat_n = np.argmax(y_hat, axis = 1)

    acc = accuracy_score(y, y_hat_n)
    mac_f1 = f1_score(y, y_hat_n, average="macro")
    mic_f1 = f1_score(y, y_hat_n, average="micro")
    wei_f1 = f1_score(y, y_hat_n, average="weighted")
    # sam_f1 = f1_score(y, y_hat_n, average="samples")
    try:
        roc_auc = roc_auc_score(np.eye(y_hat.shape[-1])[y], y_hat)
    except:
        roc_auc = 0
    return (["acc", "mac_f1", "mic_f1", "wei_f1", "roc_auc"], [acc, mac_f1, mic_f1, wei_f1, roc_auc])

def cal_reg_score(y_hat, y):
    """
    y_hat: (Batch * feature) -> prob matrix
    y : (Batch) -> matrix
    """
    if not isinstance(y_hat, np.ndarray):
        y_hat = y_hat.detach().cpu().numpy()
    if not isinstance(y,np.ndarray):
        y = y.cpu().numpy()
    y = y.reshape(-1)
    y_hat = y_hat.reshape(-1)
    r2 = r2_score(y, y_hat)
    explained_variance = explained_variance_score(y, y_hat)
    mean_squared = -mean_squared_error(y, y_hat)
    mean_absolute = -mean_absolute_error(y, y_hat)
    return ["r2", "explained_variance", "mean_squared", "mean_absolute"], list(map(float,[r2, explained_variance, mean_squared, mean_absolute]))

def cal_score(y_hat, y, types):
    # print(types)
    if types == "bi":
        return cal_biclass_score(y_hat, y)
    elif types == "multi":
        return cal_multiclass_score(y_hat, y)
    else:
        return cal_reg_score(y_hat, y)


if __name__ == "__main__":
    import torch
    arr = np.array([[5,6],[-1,4],[1,2]])
    ten = torch.from_numpy(arr).float()
    print(ten)
    print(torch.nn.functional.softmax(ten,1))

    # print(np.argmax(arr, axis = 1))