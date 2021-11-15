import torch.nn as nn
import numpy as np

class Base(nn.Module):
    def __init__(self):
        super().__init__()
    def get_total_param_num(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return params