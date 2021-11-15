import torch
from torch import nn
from torchdiffeq import odeint, odeint_adjoint
from .TorchDiffEqPack import odesolve as odesolve
from .TorchDiffEqPack import odesolve_adjoint_sym12 as ad_odesolve
from .TorchDiffEqPack import odesolve_adjoint

def rdeint(logsig, h0, func, rtol, atol, method, adjoint, return_sequences, max_eval, time_length, step_size, time_seq = None):
    t, options, ok0 = set_options(logsig, method, time_length, max_eval, step_size, return_sequences, time_seq)
    if (method == "sym12async") and adjoint and not return_sequences:
        options["method"] = method
        options["atol"] = atol
        options["rtol"] = rtol
        options["t0"]  = t[0]
        options["t1"]  = t[-1]
        output = ad_odesolve(func, h0, options)
        output = tuple([i,j] for i,j in zip(h0, output))
    elif (method == "sym12async") and adjoint and return_sequences:
        options["method"] = method
        options["atol"] = atol
        options["rtol"] = rtol
        output = [[i.unsqueeze(0)] for i in h0]
        for i in range(len(t)-1):
            options["t0"]  = t[i]
            options["t1"]  = t[i+1]
            output_mid = ad_odesolve(func, h0, options)
            h0 = output_mid
            for i, each_h0 in enumerate(h0):
                output[i].append(each_h0.unsqueeze(0))
        output = tuple(torch.cat(i,dim=0) for i in output)
    elif method in ["rk4", "euler"]: # ["dopri8", "dopri5", "bosh3", "adaptive_heun", "sym12async"]:
        if adjoint:
            ode_func = odeint_adjoint
        else:
            ode_func = odeint
        output = ode_func(func=func, y0=h0, t=t, method=method, options=options, rtol = rtol, atol = atol)
    elif method == "dopri5" and adjoint:
        # del options["neval_max"]
        rtol = rtol
        atol = atol
        ode_func = odeint_adjoint
        output = ode_func(func=func, y0=h0, t=t, method=method, options=options, rtol = rtol, atol = atol) 
    elif method in ["sym12async", "dopri5"]:
        if adjoint:
            assert 0 
            ode_func = odesolve_adjoint
        else:
            ode_func = odesolve
        options["method"] = method
        options["atol"] = atol
        options["rtol"] = rtol
        options["t0"]  = t[0]
        options["t1"]  = t[-1]
        options['t_eval'] = t
        output = ode_func(func, h0, options)
    if type(output[1]) is not str and not ok0:
        output = (output[0][1:], output[1][1:])
    return output

def set_options(logsig, method, int_time_length, max_eval, step_size = 1, return_sequences=False, time_seq = None, eps=1e-5):
    length = logsig.size(1) + 1
    options = {'eps': eps}
    ok0 = True
    if time_seq is not None:
        t = time_seq
        ok0 = 0 in t
        if not ok0:
            t = torch.cat([torch.arange(1).to(logsig.device),t])
    else:
        if return_sequences:
            t = torch.arange(0, length, dtype=torch.float).to(logsig.device)
        else:
            t = torch.Tensor([0, length]).to(logsig.device)
    if method in ["dopri8", "dopri5", "bosh3", "adaptive_heun", "sym12async"]:
        # options.update({'neval_max': max(logsig.size(1) * 1.1, max_eval)})
        pass
    else:
        options.update({'step_size': step_size})
    if int_time_length is not None:
        t = t / length * int_time_length

    return t, options, ok0