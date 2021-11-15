
from abc import abstractmethod
import pathlib
import sys
import torch
import numpy as np
import os
import time 
here = pathlib.Path(__file__).resolve().parent
sys.path.append(str(here / ".." / ".."))
from .base import Base
from . import ancde_m as ancde
import bisect
# from ancde import 

## we need this functions
# only for ancde
class ancde_f(torch.nn.Module):
    
    def __init__(self,input_channels, hidden_atten_channels, hidden_hidden_atten_channels, num_hidden_layers):
        super(ancde_f, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_atten_channels
        self.hidden_hidden_channels = hidden_hidden_atten_channels
        self.num_hidden_layers = num_hidden_layers
        
        
        self.linear_in = torch.nn.Linear(input_channels, hidden_hidden_atten_channels)
        self.linear_test = torch.nn.Linear(hidden_hidden_atten_channels, hidden_atten_channels)
        self.linears = torch.nn.ModuleList(torch.nn.Linear(hidden_atten_channels, hidden_atten_channels)
                                           for _ in range(num_hidden_layers - 1))
        self.linear_out = torch.nn.Linear(hidden_atten_channels, input_channels * input_channels)
        # self.linear_out = torch.nn.Linear(hidden_hidden_channels,input_channels*input_channels)

    def extra_repr(self):
        return "input_channels: {}, hidden_channels: {}, hidden_hidden_channels: {}, num_hidden_layers: {}" \
               "".format(self.input_channels, self.hidden_channels, self.hidden_hidden_channels, self.num_hidden_layers)

    def forward(self, z):
        
        z = self.linear_in(z)
        z = z.relu()
        z = self.linear_test(z)
        z = z.relu()
        
        for linear in self.linears:
            z = linear(z)
            z = z.relu()
        z = self.linear_out(z).view(*z.shape[:-1], self.input_channels, self.input_channels)
        
        z= z.tanh()
        return z


class _GetLogsignature:
    """Given a time value, gets the corresponding piece of the log-signature.

    When performing a forward solve, torchdiffeq will give us the time value that it is solving the ODE on, and we need
    to return the correct piece of the log-signature corresponding to that value. For example, let our intervals ends
    be the integers from 0 to 10. Then if the time value returned by torchdiffeq is 5.5, we need to return the
    logsignature on [5, 6]. This function simply holds the logsignature, and interval end times, and returns the
    correct logsignature given any time.
    """
    def __init__(self, logsig):
        self.knots = range(logsig.size(1))
        self.logsig = logsig

    def __getitem__(self, t):
        index = bisect.bisect(self.knots, t) - 1
        return self.logsig[:, index]


class ancde_g(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, hidden_hidden_channels, num_hidden_layers):
        super(ancde_g, self).__init__()
        # import pdb ; pdb.set_trace()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.hidden_hidden_channels = hidden_hidden_channels
        self.num_hidden_layers = num_hidden_layers

        self.linear_in = torch.nn.Linear(hidden_channels, hidden_hidden_channels)
        
        self.linears = torch.nn.ModuleList(torch.nn.Linear(hidden_hidden_channels, hidden_hidden_channels)
                                           for _ in range(num_hidden_layers - 1))
        self.linear_out = torch.nn.Linear(hidden_hidden_channels, input_channels * hidden_channels) #32,32*4  -> # 32,32,4 

    def extra_repr(self):
        return "input_channels: {}, hidden_channels: {}, hidden_hidden_channels: {}, num_hidden_layers: {}" \
               "".format(self.input_channels, self.hidden_channels, self.hidden_hidden_channels, self.num_hidden_layers)

    def forward(self, z):
        
        
        z = self.linear_in(z)
        z = z.relu()
        
        for linear in self.linears:
            z = linear(z)
            z = z.relu()
        z = self.linear_out(z).view(*z.shape[:-1], self.hidden_channels, self.input_channels)
    
        z = z.tanh()  
        
        return z
       
class Hardsigmoid(torch.nn.Module):

    def __init__(self):
        super(Hardsigmoid, self).__init__()
        self.act = torch.nn.Hardtanh()

    def forward(self, x):
        return ((self.act(x) + 1.0) / 2.0)
        
class RoundFunctionST(torch.autograd.Function):
    """Rounds a tensor whose values are in [0, 1] to a tensor with values in {0, 1}"""

    @staticmethod
    def forward(ctx, input):

        return torch.round(input)

    @staticmethod
    def backward(ctx, grad_output):

        return grad_output

RoundST = RoundFunctionST.apply
class ANCDE(Base):
    
    # #  model = ANCDE(initial_dim, hidden_dim,output_dim,attention_channel=hidden_hidden_dim,slope_check = False,soft = True,timewise=False, initial=False,return_sequences=return_sequences)
        
    def __init__(self, initial_dim, hidden_dim, output_dim, num_layers, attention_channel, slope_check, soft, timewise,return_sequences,initial=True):
        super(ANCDE, self).__init__()
        self.input_channels = initial_dim
        self.hidden_channels = hidden_dim
        # self.output_channels = output_dim
        self.return_sequences = return_sequences
        self.func_f = ancde_f(input_channels=initial_dim,hidden_atten_channels=attention_channel,hidden_hidden_atten_channels=attention_channel,num_hidden_layers=num_layers)
        self.func_g = ancde_g (input_channels=initial_dim,hidden_channels=hidden_dim,hidden_hidden_channels=attention_channel,num_hidden_layers=num_layers)
        self.initial = initial
        self.attention_channel = attention_channel
        self.slope_check = slope_check
        self.soft = soft
        self.STE = Hardsigmoid()
        self.binarizer = RoundST 
        
        self.feature_extractor = torch.nn.Linear(initial_dim,hidden_dim)
        # self.linear = torch.nn.Linear(hidden_dim, initial_dim-1) # hidden state -> prediction
        self.time_attention = torch.nn.Linear(initial_dim,1)
        self.timewise = timewise

        self.initial_linear = torch.nn.Linear(initial_dim, initial_dim)

        # Linear classifier to apply to final layer
        self.final_linear = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self,x):
        x0, logsig = x
        z0 = self.initial_linear(x0) # aa0 32,32
        dX_dt = _GetLogsignature(logsig)
        
        # times = torch.arange(0, length, dtype=torch.float).to(x.device)
        # coeffs = ancde.natural_cubic_spline_coeffs(times, x)
        # cubic_spline = ancde.NaturalCubicSpline(times, coeffs) # interpolated values
        # XX = cubic_spline.evaluate(times[0]).float() # 32,4    
        

        length = logsig.size(1) + 1
        times = torch.arange(0, length, dtype=torch.float).to(x0.device)    

        kwargs = {}
        kwargs['method'] = 'rk4'
        kwargs['options'] = {}

        sigmoid = torch.nn.Sigmoid()

        self.atten_in = self.hidden_channels
        
        attention = ancde.cdeint(dX_dt=dX_dt.__getitem__,
                                 z0=z0,
                                 func=self.func_f,
                                 t=times,
                                 **kwargs)
        if self.timewise:
            attention = self.time_attention(attention)
        if self.soft :
            attention = sigmoid(attention)
        else:
            slope = 0 
            if self.slope_check :
                attention = self.STE(slope * attention)
                attention = self.binarizer(attention)
            else :
                
                attention = sigmoid(attention) # sigmoid -> 다른 것 
                attention = self.binarizer(attention)
        
        # x0 = cubic_spline.evaluate(times[0]).float()
        a0 =  attention[0,:,:]
        y0 = torch.mul(x0,a0)
        y0 = self.feature_extractor(y0) 
        length = logsig.size(1) + 1
        if self.return_sequences:
            times = torch.arange(0, length, dtype=torch.float).to(x0.device)    
        else:
            times = torch.Tensor([0, length-1]).to(x0.device)

        z_t = ancde.ancde(dX_dt=dX_dt.__getitem__,
                          attention=attention,
                          z0 =y0,
                          func_g = self.func_g,
                          t=times,
                          timewise=self.timewise,
                          **kwargs)
            
        outputs = self.final_linear(z_t[-1, :, :]) if not self.return_sequences else self.final_linear(z_t)

        return outputs

