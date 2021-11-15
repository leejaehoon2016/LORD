import torch
import torchdiffeq
# from torch.autograd.functional import jacobian
# from TorchDiffEqPack import odesolve_adjoint_sym12 as odesolve
import numpy as np
import os
import time 




class AttentiveVectorField(torch.nn.Module):
    
    def __init__(self,dX_dt,func_g,X_s,time,attention):
        super(AttentiveVectorField, self).__init__()
        if not isinstance(func_g, torch.nn.Module):
            raise ValueError("func must be a torch.nn.Module.")
        self.sigmoid = torch.nn.Sigmoid()
        self.dX_dt = dX_dt
        
        self.func_g =func_g
        self.X_s = X_s  # interpolated_value
        self.feature_Extrator = torch.nn.Linear(69,49)
        
        self.timewise = time
        self.attention=attention
        

    def __call__(self, t, z):
        z = self.func_g(z) # attention * x 
        
        control_gradient = self.dX_dt(t).float() # 32,4 # 32,12
        if self.timewise:
            a_t = self.attention[int(np.floor(t.item())),:,:]
        else:
            a_t = self.attention[int(np.floor(t.item())),:,:].squeeze()
        
        Xt = self.dX_dt(t)
        dY_dt_1 = torch.mul(control_gradient,a_t)
        dY_dt_2 = torch.mul(torch.mul(a_t,(1-a_t)),Xt)
        
        dY_dt = (dY_dt_1+dY_dt_2).float()
         
        
        out = (z@dY_dt.unsqueeze(-1)).squeeze(-1)
        
        return out


class VectorField(torch.nn.Module):
    def __init__(self, dX_dt, func):
        """Defines a controlled vector field.

        Arguments:
            dX_dt: As cdeint.
            func: As cdeint.
        """
        super(VectorField, self).__init__()
        if not isinstance(func, torch.nn.Module):
            raise ValueError("func must be a torch.nn.Module.")

        self.dX_dt = dX_dt
        self.func = func

    def __call__(self, t, z):
        # control_gradient is of shape (..., input_channels)
        # start = time.time()
        control_gradient = self.dX_dt(t)
        # vector_field is of shape (..., hidden_channels, input_channels)
        
        vector_field = self.func(z)
        
        # out is of shape (..., hidden_channels)
        # (The squeezing is necessary to make the matrix-multiply properly batch in all cases)
        out = (vector_field @ control_gradient.unsqueeze(-1)).squeeze(-1)
        
        return out

def cdeint(dX_dt, z0, func, t, adjoint=True, **kwargs):
    if 'method' not in kwargs:
        kwargs['method'] = 'rk4'
    # import pdb; pdb.set_trace()        
    

    elif kwargs['method'] == 'dopri5':
        if 'atol' not in kwargs:
            kwargs['atol'] = 1e-12
        if 'rtol' not in kwargs:
            kwargs['rtol'] = 1e-6

    elif kwargs['method'] == 'rk4':
        if 'options' not in kwargs:
            kwargs['options'] = {}
        options = kwargs['options']
        if 'step_size' not in options and 'grid_constructor' not in options:
            time_diffs = 1.0
            options['step_size'] = time_diffs
    
    control_gradient = dX_dt(torch.zeros(1, dtype=z0.dtype, device=z0.device))
    if control_gradient.shape[:-1] != z0.shape[:-1]:
        raise ValueError("dX_dt did not return a tensor with the same number of batch dimensions as z0. dX_dt returned "
                         "shape {} (meaning {} batch dimensions)), whilst z0 has shape {} (meaning {} batch "
                         "dimensions)."
                         "".format(tuple(control_gradient.shape), tuple(control_gradient.shape[:-1]), tuple(z0.shape),
                                   tuple(z0.shape[:-1])))
    vector_field = func(z0)
    if vector_field.shape[:-2] != z0.shape[:-1]:
        raise ValueError("func did not return a tensor with the same number of batch dimensions as z0. func returned "
                         "shape {} (meaning {} batch dimensions)), whilst z0 has shape {} (meaning {} batch"
                         " dimensions)."
                         "".format(tuple(vector_field.shape), tuple(vector_field.shape[:-2]), tuple(z0.shape),
                                   tuple(z0.shape[:-1])))
    if vector_field.size(-2) != z0.shape[-1]:
        raise ValueError("func did not return a tensor with the same number of hidden channels as z0. func returned "
                         "shape {} (meaning {} channels), whilst z0 has shape {} (meaning {} channels)."
                         "".format(tuple(vector_field.shape), vector_field.size(-2), tuple(z0.shape),
                                   z0.shape.size(-1)))
    if vector_field.size(-1) != control_gradient.size(-1):
        raise ValueError("func did not return a tensor with the same number of input channels as dX_dt returned. "
                         "func returned shape {} (meaning {} channels), whilst dX_dt returned shape {} (meaning {}"
                         " channels)."
                         "".format(tuple(vector_field.shape), vector_field.size(-1), tuple(control_gradient.shape),
                                   control_gradient.size(-1)))
    if control_gradient.requires_grad and adjoint:
        raise ValueError("Gradients do not backpropagate through the control with adjoint=True. (This is a limitation "
                         "of the underlying torchdiffeq library.)")

    odeint = torchdiffeq.odeint_adjoint if adjoint else torchdiffeq.odeint
    vector_field = VectorField(dX_dt=dX_dt, func=func)
    
    
    out = odeint(func=vector_field, y0=z0, t=t, **kwargs)
    real_output = out
    

    return real_output




def ancde(dX_dt,attention,z0, func_g, t,timewise,adjoint=True, X_s = None, **kwargs):

    
    # control_gradient = dX_dt(torch.zeros(1, dtype=z0.dtype, device=z0.device))
    
    vector_field = func_g(z0) 
    
    odeint = torchdiffeq.odeint_adjoint if adjoint else torchdiffeq.odeint
    # vector_field = AttentiveTimeVectorField(dX_dt=dX_dt, func_f=func_f,func_g =func_g,X_s=X_s,linear_f=linear_f,atten_in=atten_in)
    vector_field = AttentiveVectorField(dX_dt=dX_dt, func_g =func_g,X_s=X_s,time=timewise,attention = attention)

    # import pdb ; pdb.set_trace()
    out = odeint(func=vector_field.cuda(), y0=z0, t=t, **kwargs)
    return out

