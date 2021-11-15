from pickle import NEWFALSE
from train import Main_Module
import os

def main():
    main_module = Main_Module(ds_folder, ds_name, model_type, from_depth, to_depth, step, batch_size,  
                func_inputdim, emb_dim_factor, ienc_param, ifunc_param, idec_param, finalfunc_param, finaldec_param, decoutput_dim, apply_final_dec,
                enc_param, dec_param, func_param, finaldecdata_param,
                term, time_length, adjoint, solver, emb_adjoint, emb_solver, rtol, atol, max_eval, step_size,
                test_name, result_folder, random_num, GPU_NUM, epochs, pre_loss_stan, AE_lr, AE_l2, R_lr, R_l2,
                emb_loss_type, emblogsig_mean, emblogsig_std, apply_kinetic_encoder, apply_kinetic_decoder, apply_kinetic_func, is_recon_folddata,
                emblogsig_loss_coef, embdata_loss_coef, embdata_reg_coef, emblogsig_abs_mean_reg_coef, emblogsig_abs_std_reg_coef, 
                enc_quad_coef, enc_jac_coef, dec_quad_coef, dec_jac_coef, func_quad_coef, func_jac_coef, decay_stan, decay_factor)
    main_module.fit()

ds_folder, ds_name, model_type, from_depth, to_depth, step, batch_size = "UEA",'EigenWorms',"embnrde", 1, 2, 128, 512
func_inputdim, emb_dim_factor = 32, None
ienc_param, ifunc_param, idec_param = {"layers_dim":[32], "layer_type" : "fc_x"}, {"layers_dim":[32], "layer_type" : "fc_x"}, {"layers_dim":[32], "layer_type" : "fc_emb"}
finalfunc_param, finaldec_param, finaldecdata_param = {"layers_dim":[]}, {"layers_dim":[]}, {"layers_dim":[]}
decoutput_dim, apply_final_dec = None, False 
enc_param = {"layers_dim":[64,64,64], "tanh_last":True, "mid_batch_norm":False, "last_batch_norm":1}
dec_param = {"layers_dim":[64,64,64], "tanh_last":True, "mid_batch_norm":False, "last_batch_norm":1}
func_param= {"layers_dim":[64,64,64], "tanh_last":True, "mid_batch_norm":False, "last_batch_norm":1}
term, time_length, adjoint, solver, emb_adjoint, emb_solver, rtol, atol, max_eval, step_size = 1, None, False, "euler", False, "euler", 1, 1, 5000, 1
test_name, result_folder, random_num, GPU_NUM, epochs = "result", "result", 0, 2, 400
pre_loss_stan, AE_lr, AE_l2, R_lr, R_l2, decay_factor, decay_stan = 2000, 1e-3, 1e-6, 1e-3, 1e-6, 0.1, 0.1
emb_loss_type, emblogsig_mean, emblogsig_std, apply_kinetic_encoder, apply_kinetic_decoder, apply_kinetic_func, is_recon_folddata  = "mse", 0, 0, False, False, False, False
emblogsig_loss_coef, embdata_loss_coef, embdata_reg_coef, emblogsig_abs_mean_reg_coef = 1.0, 0, 0, 0
emblogsig_abs_std_reg_coef, enc_quad_coef, enc_jac_coef, dec_quad_coef, dec_jac_coef, func_quad_coef, func_jac_coef = 0, 0, 0, 0, 0, 0, 0



import argparse
parser = argparse.ArgumentParser('embnrde')

parser.add_argument("--ds_folder", type=str, default="TSR")
parser.add_argument("--ds_name", type=str, default='BIDMC32HR')
parser.add_argument("--D1", type=int, default=1)
parser.add_argument("--D2", type=int, default=2)
parser.add_argument("--P", type=int, default=4)
parser.add_argument("--r", type=int, default=222)
parser.add_argument("--gpu", type=int, default=0)

args = parser.parse_args()



ds_folder      = args.ds_folder
ds_name        = args.ds_name
from_depth     = args.D1
to_depth       = args.D2
step           = args.P
random_num = args.r
GPU_NUM    = args.gpu

from hyperparams import get_hyperparam
args = get_hyperparam(ds_folder, ds_name,from_depth, to_depth, step)

func_inputdim  = args.func_inputdim
finalfunc_param["layers_dim"] = [args.h_out] * args.N_out
enc_param["layers_dim"]   = [args.hf] * args.Nf
dec_param["layers_dim"]   = [args.ho] * args.No
func_param ["layers_dim"] = [args.hg] * args.Ng
pre_loss_stan = args.iter_ae
epochs = args.iter_task
embdata_reg_coef = args.c_e
AE_l2 = args.c_ae
R_l2 = args.c_task

test_name = "result"

if "BIDMC" in ds_name:
    decay_stan = float("inf")

main()

