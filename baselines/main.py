from train import Main_Module

def main():
    main_module = Main_Module(ds_folder, ds_name, model_type, depth, step, batch_size,  
                                hidden_dim, hidden_hidden_dim, num_layers, logsig_emb_decay, logsig_emb_layers, logsig_emb_hidden,
                                adjoint, solver, test_name, result_folder, random_num, GPU_NUM)
    main_module.fit()

ds_folder, ds_name, model_type, depth, step, batch_size = "TSR", "BIDMC32HR","ancde", 1, 512, 512 # "embnrde-simple"
hidden_dim, hidden_hidden_dim, num_layers = 32, 64, 3
logsig_emb_decay, logsig_emb_layers, logsig_emb_hidden = 0.3, 3, 64  # only for embnrde-simple
adjoint, solver, test_name, result_folder, random_num, GPU_NUM = 1, "rk4", "result", "result", 0, 0


import argparse
parser = argparse.ArgumentParser('embnrde')

parser.add_argument("--model", type=str, default="nrde")
parser.add_argument("--ds_folder", type=str, default="TSR")
parser.add_argument("--ds_name", type=str, default='BIDMC32HR')
parser.add_argument("--D", type=int, default=2)
parser.add_argument("--P", type=int, default=4)
parser.add_argument("--r", type=int, default=222)
parser.add_argument("--gpu", type=int, default=0)


args = parser.parse_args()

model_type     = args.model
ds_folder      = args.ds_folder
ds_name        = args.ds_name
depth          = args.D
step           = args.P
random_num = args.r
GPU_NUM    = args.gpu

from hyperparams import get_hyperparam
args = get_hyperparam(model_type,ds_folder, ds_name, depth, step)

hidden_dim = args.h # hidden path size
hidden_hidden_dim = args.hh # attention size for ANCDE, hidden size for others
num_layers = args.l # number of hidden layers
logsig_emb_decay, logsig_emb_layers, logsig_emb_hidden = args.logsig_emb_decay, args.logsig_emb_layers, args.logsig_emb_hidden # params of DE-NRDE

test_name = "result"

if model_type == "odernn":
    model_type = "odernn_folded"
elif model_type == "denrde":
    model_type = "embnrde-simple"

main()    


