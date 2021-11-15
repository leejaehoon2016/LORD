import torch
from torch.utils.data import Dataset, DataLoader

class DatasetBase(Dataset): 
    def __init__(self, x, y):
        self.x, self.y = x, y
        self.x_tuple = type(self.x) is not torch.Tensor
    def __len__(self): 
        return len(self.y)
    def __getitem__(self, idx): 
        if self.x_tuple:
            return tuple(each_x[idx] for each_x in self.x), self.y[idx]
        else:
            return self.x[idx], self.y[idx]

def make_base_dataset(dl, batch_size, ds_folder = None, mean_y = None, std_y = None):
    check = False
    for data in dl:
        if check:
            assert 0
        check = True
        x, y = data
    if ds_folder == "TSR":
        if mean_y is None:
            mean_y = y.mean() 
        if std_y is None:
            std_y = y.std() 
        y = (y - mean_y) / std_y
    dataset = DatasetBase(x, y)
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dl, mean_y, std_y


class DatasetFromTo(Dataset): 
    def __init__(self, init, all_data, todepth_logsig, fromdepth_logsig, y):
        self.init, self.all_data, self.todepth_logsig, self.fromdepth_logsig, self.y = \
            init, all_data, todepth_logsig, fromdepth_logsig, y
    def __len__(self): 
        return len(self.y)
    def __getitem__(self, idx): 
        return (self.init[idx], self.all_data[idx], self.fromdepth_logsig[idx], self.todepth_logsig[idx]), self.y[idx]
    
def make_new_dataset(dl_todepth, dl_fromdepth, dl_alldata, batch_size, ds_folder = None, mean_y = None, std_y = None):
    check = False
    for todepth, fromdepth, all_data in zip(dl_todepth, dl_fromdepth, dl_alldata):
        if check:
            assert 0
        check = True
        todepth_init, todepth_logsig, todepth_y = todepth[0][0], todepth[0][1], todepth[1]
        fromdepth_init, fromdepth_logsig, fromdepth_y = fromdepth[0][0], fromdepth[0][1], fromdepth[1]
        all_data, all_y = all_data[0], all_data[1]
        if (todepth_y  !=  fromdepth_y).sum() + (fromdepth_y  !=  all_y).sum(): 
            assert 0, "data is different"
    if ds_folder == "TSR":
        if mean_y is None:
            mean_y = todepth_y.mean() 
        if std_y is None:
            std_y = todepth_y.std() 
        todepth_y = (todepth_y - mean_y) / std_y
    dataset = DatasetFromTo(todepth_init, all_data, todepth_logsig, fromdepth_logsig, todepth_y)
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dl, mean_y, std_y
