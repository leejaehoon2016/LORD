"""
model.py
===========================
This contains a model class for NeuralRDEs that wraps `rdeint` as a `nn.Module` that will act similarly to an RNN.
"""
from kinetic_wrapper import KineticWrapper
import bisect
import numpy as np
import torch
from torch import nn
from rdeint.emb_rdeint import rdeint
from util.train_util import RMSELoss

def get_term_logsig_timeseq(term, now_idx, logsig):
    time_seq = torch.arange(0, len(logsig) + 1, dtype=torch.float).to(logsig.device)
    length = len(logsig) - now_idx
    length = (length // term) * term + now_idx
    logsig = logsig[now_idx:length]
    time_seq = time_seq[now_idx:length + 1:term]
    # dec_output = dec_output[now_idx:length + 1]
    new_logisig = 0
    for i in range(term):
        new_logisig += logsig[i::term]
    # dec_logsig = dec_output[term::term] - dec_output[:-term:term]
    return new_logisig, time_seq, (now_idx + 1) % term

class EmbRDE(nn.Module):
    def __init__(self, initial_dim, func_inputdim, from_logsig_dim, to_logsig_dim, emb_dim_factor, output_dim, decoutput_dim, apply_final_dec,
                 ienc_param, idec_param, ifunc_param, finalfunc_param, finaldec_param, finaldecdata_param, enc_param, dec_param, func_param,
                 term, time_length, return_sequences, adjoint, solver, emb_adjoint, emb_solver, rtol, atol, max_eval, step_size, emb_loss_type,
                 emblogsig_mean, emblogsig_std, apply_kinetic_encoder, apply_kinetic_decoder, apply_kinetic_func, is_recon_folddata, data_step_size):
        super().__init__()
        self.now_idx = 0
        self.term = term
        self.apply_kinetic_encoder, self.apply_kinetic_decoder, self.apply_kinetic_func = apply_kinetic_encoder, apply_kinetic_decoder, apply_kinetic_func
        self.emblogsig_mean, self.emblogsig_std = emblogsig_mean, emblogsig_std
        self.time_length = time_length
        self.return_sequences = return_sequences
        self.adjoint, self.solver, self.emb_adjoint = adjoint, solver, emb_adjoint
        self.emb_solver, self.rtol, self.atol, self.max_eval, self.step_size = emb_solver, rtol, atol, max_eval, step_size
        if emb_dim_factor is None:
            self.emb_dim = from_logsig_dim 
        else:
            self.emb_dim = emb_dim_factor if type(emb_dim_factor) is int else int(to_logsig_dim * emb_dim_factor)

        print("emb_size : {}".format(self.emb_dim))
        decoutput_dim = to_logsig_dim if decoutput_dim is None else decoutput_dim
        # Initial to hidden
        self.initial_function = _Initial(initial_dim, None, self.emb_dim, func_inputdim, **ifunc_param)
        self.initial_encoder =  _Initial(initial_dim, from_logsig_dim, self.emb_dim, self.emb_dim, **ienc_param)
        self.initial_decoder =  _Initial(initial_dim, to_logsig_dim, self.emb_dim, decoutput_dim, **idec_param)
        if apply_final_dec or decoutput_dim < to_logsig_dim:
            self.final_decoder_layer = _Initial_part(decoutput_dim, to_logsig_dim, **finaldec_param)
        else:
            self.final_decoder_layer = lambda x : x
        if is_recon_folddata:
            self.final_decoder_data = _Initial_part(to_logsig_dim, initial_dim * data_step_size, **finaldecdata_param)
        else:
            self.final_decoder_data = _Initial_part(to_logsig_dim, initial_dim, **finaldecdata_param)
        self.final_linear = _Initial_part(func_inputdim, output_dim, **finalfunc_param)
        self.main_func, self.encoder_decoder = get_EmbRDEFunc(initial_dim, func_inputdim, from_logsig_dim, decoutput_dim, self.emb_dim, enc_param, dec_param, func_param)
        if self.apply_kinetic_decoder or self.apply_kinetic_encoder:
            self.encoder_decoder = KineticWrapper(self.encoder_decoder, self.apply_kinetic_encoder, self.apply_kinetic_decoder)
        if self.apply_kinetic_func:
            self.main_func = KineticWrapper(self.main_func, True, False)
        print(self)
        if emb_loss_type == "mse":
            self.emb_loss_func = nn.MSELoss(reduction='none')
        elif emb_loss_type == "l1":
            self.emb_loss_func = nn.L1Loss(reduction='none')
        elif emb_loss_type == "l3":
            self.emb_loss_func = lambda x,y : (x-y).abs() ** 3
        elif emb_loss_type == "l4":
            self.emb_loss_func = lambda x,y : (x-y).abs() ** 4
            
        else:
            self.emb_loss_func = RMSELoss(reduction='none')
        self.data_loss_func = nn.MSELoss()
    def forward(self, x, emb_mode = False, return_emb_loss = True):    
        initial, all_data, from_logsig, to_logsig = x
        all_data_length = from_logsig.size(1) + 1
        if emb_mode:
            self.encoder_decoder.reset(from_logsig, self.time_length)
            e0 = self.initial_encoder(initial, self.encoder_decoder.logsig_getter[0], emb = None)
            s0 = self.initial_decoder(initial, self.encoder_decoder.logsig_getter[0], e0)
            inputs = [e0, s0]
            if self.apply_kinetic_encoder:
                inputs.append(torch.zeros(e0.size(0)).to(e0))
                inputs.append(torch.zeros(e0.size(0)).to(e0))
            if self.apply_kinetic_decoder:
                inputs.append(torch.zeros(s0.size(0)).to(s0))
                inputs.append(torch.zeros(s0.size(0)).to(s0))
            inputs = tuple(inputs)
            # import pdb ; pdb.set_trace()
            to_logsig, time_seq, self.now_idx = get_term_logsig_timeseq(self.term, self.now_idx, to_logsig.transpose(0,1))
            emb_recon = rdeint(to_logsig, inputs, self.encoder_decoder, self.rtol, self.atol, self.emb_solver, self.emb_adjoint, 
                                   True, self.max_eval, self.time_length, self.step_size, time_seq)
            dec_output = self.final_decoder_layer(emb_recon[1])
            pred_logsig = (dec_output[1:] - dec_output[:-1])
            pred_data = self.final_decoder_data(dec_output[:])
            
            emblogsig_loss = self.emb_loss_func(to_logsig, pred_logsig).mean()
            # import pdb ; pdb.set_trace()
            embdata_loss = self.data_loss_func(all_data.transpose(0,1)[:all_data_length], pred_data).mean()
            embdata = emb_recon[0]
            emblogsig_abs = (embdata[1:] - embdata[:-1]).abs()
            embdata_reg   =  (embdata ** 2).mean()
            emblogsig_abs_mean = emblogsig_abs.mean()
            emblogsig_abs_std  = emblogsig_abs.std()
            emblogsig_abs_mean_reg = (emblogsig_abs_mean - self.emblogsig_mean) ** 2
            emblogsig_abs_std_reg  = (emblogsig_abs_std - self.emblogsig_std) ** 2
            tensorboard_log = {"emblogsig_loss": emblogsig_loss, "embdata_loss" : embdata_loss, "embdata_reg": embdata_reg, 
                               "emblogsig_abs_mean" : emblogsig_abs_mean, "emblogsig_abs_std" : emblogsig_abs_std,
                               "emblogsig_abs_mean_reg":emblogsig_abs_mean_reg, "emblogsig_abs_std_reg":emblogsig_abs_std_reg}
            enc_quad, enc_jac = None, None
            if self.apply_kinetic_encoder:
                enc_quad, enc_jac = emb_recon[3][-1].mean(), emb_recon[2][-1].mean()
                tensorboard_log.update({"enc_quad":enc_quad, "enc_jac":enc_jac})
            
            dec_quad, dec_jac = None, None
            if self.apply_kinetic_decoder:
                dec_quad, dec_jac = emb_recon[-1][-1].mean(), emb_recon[-2][-1].mean()
                tensorboard_log.update({"dec_quad":dec_quad, "dec_jac":dec_jac})
            if return_emb_loss:
                return ((emblogsig_loss, embdata_loss, embdata_reg, emblogsig_abs_mean_reg, emblogsig_abs_std_reg, enc_quad, enc_jac, dec_quad, dec_jac), tensorboard_log)    
            else:
                return from_logsig.transpose(0,1), to_logsig, emb_recon[0][1:] - emb_recon[0][:-1], emb_recon[1][1:] - emb_recon[1][:-1]

            
            
        else:
            self.main_func.reset(from_logsig, self.time_length)
            e0 = self.initial_encoder(initial, self.main_func.logsig_getter[0], emb = None)
            h0 = self.initial_function(initial)
            inputs = (h0, e0)
            if self.apply_kinetic_func:
                inputs = (h0, e0, torch.zeros(h0.size(0)).to(h0), torch.zeros(h0.size(0)).to(h0))
            last_out = rdeint(to_logsig, inputs, self.main_func, self.rtol, self.atol, self.solver, self.adjoint, self.return_sequences, self.max_eval, self.time_length, self.step_size)
            
            out = last_out[0][-1]
            
            outputs = self.final_linear(out)
            tensorboard_log = {}
            if self.apply_kinetic_func:
                func_quad, func_jac = last_out[3][-1].mean(), last_out[2][-1].mean()
                tensorboard_log.update({"func_quad":func_quad, "func_jac":func_jac})
                # import pdb ; pdb.set_trace()
                return ((outputs, func_quad, func_jac), tensorboard_log)
            else:
                return ((outputs,), tensorboard_log)

    def cal_num_param(self, param):
        model_parameters = filter(lambda p: p.requires_grad, param)
        params = sum([np.prod(p.size()) for p in model_parameters])
        return params

    def get_param(self, conds = None, not_conds = None):
        lst = []
        # k_lst = []
        if conds is not None:
            for k,v in self.named_parameters():
                if any([cond in k.replace("encoder_decoder","") for cond in conds]):
                    lst.append(v)
                    # k_lst.append(k)
        else:
            for k,v in self.named_parameters():
                # print(k, [cond not in k for cond in not_conds])
                if all([cond not in k.replace("encoder_decoder","") for cond in not_conds]):
                    lst.append(v)
                    # k_lst.append(k)
        # print(k_lst)
        return lst
    def get_decoder_param_num(self):
        return self.cal_num_param(self.get_param(conds=["decoder"]))
        
    def get_rest_param_num(self):
        return self.cal_num_param(self.get_param(not_conds=["decoder"]))
    
    def get_ae_param(self):
        return self.get_param(conds=["encoder", "decoder"])
        
    def get_r_param(self):
        return self.get_param(not_conds=["encoder", "decoder"])


def get_EmbRDEFunc(initial_dim, input_dim, from_logsig_dim, decoutput_dim, emb_dim, enc_param, dec_param, func_param):
    encoder = BaseFunc1(emb_dim, emb_dim, from_logsig_dim, **enc_param)
    decoder = BaseFunc1(decoutput_dim, decoutput_dim, emb_dim, **dec_param)
    main_func = BaseFunc1(input_dim, input_dim, emb_dim, **func_param)
    return Wrapper(encoder, main_func,False), Wrapper(encoder,decoder,True)

class Wrapper(nn.Module):
    def __init__(self, encoder, m2, m2_dec):
        super().__init__()
        self.encoder = encoder
        if m2_dec:
            self.decoder = m2
        else:
            self.main_func = m2
        self.m2_dec = m2_dec
        self.logsig_getter = None
    def reset(self, logsig, time_length):
        self.logsig_getter = _GetLogsignature(logsig, time_length)
        # self.fe_num = 0
    def forward(self, t, x):
        # print(t)
        # self.fe_num += 1
        # print(self.fe_num)
        if not self.m2_dec:
            h0, e0 = x[0], x[1]
            # self.add_loss(e0, s0,t)
            det_dt = torch.bmm(self.encoder(e0), self.logsig_getter[t].unsqueeze(2))
            dht_dt = torch.bmm(self.main_func(h0), det_dt)
            return (dht_dt.squeeze(2), det_dt.squeeze(2))
        else:
            e0, s0 = x[0], x[1]
            # import pdb ; pdb.set_trace()
            det_dt = torch.bmm(self.encoder(e0), self.logsig_getter[t].unsqueeze(2))
            dst_dt = torch.bmm(self.decoder(s0), det_dt)
            return (det_dt.squeeze(2), dst_dt.squeeze(2))


class _GetLogsignature:
    def __init__(self, logsig, time_length):
        self.time_mul_factor = 1
        if time_length is not None:
            self.time_mul_factor = (logsig.size(1) + 1) / time_length
        self.knots = range(logsig.size(1))
        self.logsig = logsig

    def __getitem__(self, t):
        t = t * self.time_mul_factor
        index = bisect.bisect(self.knots, t) - 1
        # print(index)
        return self.logsig[:, index]

class _Initial(nn.Module):
    def __init__(self, data_dim, logsig_dim, emb_dim, target_dim, layers_dim, layer_type):
        """
        layer_type : {fc,no}_{x,logsig,emb} or "learnable"
        """
        super().__init__()
        if layer_type == "learnable":
            self.param =  nn.Parameter(torch.ones(target_dim) * 0.1)
            self.learnable = True
        else:
            self.learnable = False
            layer_t, self.input_t = layer_type.split("_")
            if self.input_t == "logsig":
                input_dim = logsig_dim
            elif self.input_t == "emb":
                input_dim = emb_dim
            elif self.input_t == "x":
                input_dim = data_dim
            if layer_t == "fc":
                self.layer = _Initial_part(input_dim, target_dim, layers_dim)
            else:
                self.layer = lambda x: x            
            
    def forward(self, x, logsig = None, emb = None):
        if self.learnable:
            output = self.param.repeat(len(x),1)
        else:
            if self.input_t == "logsig":
                output = self.layer(logsig)
            elif self.input_t == "emb":
                output = self.layer(emb)
            else:
                output = self.layer(x)     
        return output
class _Initial_part(nn.Module):
    def __init__(self, input_dim, target_dim, layers_dim, dropout = None):
        super().__init__()
        now_dim = input_dim
        layer = []
        first = True
        for factor in layers_dim:
            if first:
                first = False
            else:
                if dropout is not None:
                    layer.append(nn.Dropout(dropout))
                layer.append(nn.ReLU())
            next_dim = factor if type(factor) is int else max(int(factor * target_dim),10)
            layer.append(nn.Linear(now_dim,next_dim))
            now_dim = next_dim
        if not first:
            layer.append(nn.ReLU())
            if dropout is not None:
                    layer.append(nn.Dropout(dropout))
        layer.append(nn.Linear(now_dim, target_dim))
        self.layer = nn.Sequential(*layer)

    def forward(self, h):
        return self.layer(h)

class BaseFunc1(nn.Module):
    def __init__(self, start_dim, last_dim1, last_dim2, layers_dim, tanh_last, mid_batch_norm, last_batch_norm):
        super().__init__()
        self.last_dim1, self.last_dim2 = last_dim1, last_dim2
        now_dim = start_dim
        target_dim = last_dim1 * last_dim2
        layer = []
        first = True
        for factor in layers_dim:
            if first:
                first = False
            else:
                if mid_batch_norm:
                    layer.append(nn.BatchNorm1d(now_dim))
                layer.append(nn.ReLU())
            next_dim = factor if type(factor) is int else max(int(factor * target_dim),10)
            layer.append(nn.Linear(now_dim,next_dim))
            now_dim = next_dim
        if tanh_last:
            if last_batch_norm == 2:
                layer.append(nn.Tanh())
                layer.append(nn.Linear(now_dim, now_dim))
                layer.append(nn.Linear(now_dim, target_dim))
            elif last_batch_norm == 1:
                layer.append(nn.Tanh())
                layer.append(nn.Linear(now_dim, target_dim))
            else:
                layer.append(nn.ReLU())
                layer.append(nn.Linear(now_dim, target_dim))
                layer.append(nn.Tanh())
        else:
            if last_batch_norm == 2:
                layer.append(nn.ReLU())
                layer.append(nn.Linear(now_dim, now_dim))
                # layer.append(nn.ReLU())
                layer.append(nn.Linear(now_dim, target_dim))
            elif last_batch_norm == 1:
                layer.append(nn.ReLU())
                layer.append(nn.Linear(now_dim, target_dim))
            else:
                layer.append(nn.Linear(now_dim, target_dim))

        self.layer = nn.Sequential(*(layer))

    def forward(self, h):
        # print(h.shape)
        return self.layer(h).view(-1, self.last_dim1, self.last_dim2)
        
        

if __name__ == '__main__':
    # EmbRDE(10, 20, 15, 5, hidden_hidden_dim=90, num_layers=3)
    time = range(10)
    a = bisect.bisect(time, 1) - 1
    print(a)


