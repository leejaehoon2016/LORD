import json, sys ; sys.path.append("..")

import torch
from util.data_main_util import get_Data
from util.train_util import fix_random_seed, mkdir, set_loss
from util.evaluate_score import cal_score
from emb_func import EmbRDE
from tensorboardX import SummaryWriter
import torch.optim as optim
from util.data_config import task_type_dict
from util.base import Base_Module
from util.dataset import make_new_dataset


class Main_Module(Base_Module):
    def __init__(self, ds_folder, ds_name, model_type, from_depth, to_depth, step, batch_size,  
                 func_inputdim, emb_dim_factor, ienc_param, ifunc_param, idec_param, finalfunc_param, finaldec_param, decoutput_dim, apply_final_dec,
                 enc_param, dec_param, func_param, finaldecdata_param,
                 term, time_length, adjoint, solver, emb_adjoint, emb_solver, rtol, atol, max_eval, step_size,
                 test_name, result_folder, random_num, GPU_NUM, epochs, pre_loss_stan, AE_lr, AE_l2, R_lr, R_l2,
                 emb_loss_type, emblogsig_mean, emblogsig_std, apply_kinetic_encoder, apply_kinetic_decoder, apply_kinetic_func, is_recon_folddata,
                 emblogsig_loss_coef, embdata_loss_coef, embdata_reg_coef, emblogsig_abs_mean_reg_coef, emblogsig_abs_std_reg_coef, 
                 enc_quad_coef, enc_jac_coef, dec_quad_coef, dec_jac_coef, func_quad_coef, func_jac_coef, decay_stan, decay_factor, is_train=True):

        self.ds_folder, self.ds_name, self.model_type, self.from_depth, self.to_depth, self.step, self.batch_size = \
            ds_folder, ds_name, model_type, from_depth, to_depth, step, batch_size
        self.func_inputdim, self.emb_dim_factor, self.ienc_param, self.ifunc_param, self.idec_param, self.finalfunc_param, self.finaldec_param, self.decoutput_dim, self.apply_final_dec = \
            func_inputdim, emb_dim_factor, ienc_param, ifunc_param, idec_param, finalfunc_param, finaldec_param, decoutput_dim, apply_final_dec
        self.enc_param, self.dec_param, self.func_param, self.finaldecdata_param, = \
            enc_param, dec_param, func_param, finaldecdata_param,
        self.term, self.time_length, self.adjoint, self.solver, self.emb_adjoint, self.emb_solver, self.rtol, self.atol, self.max_eval, self.step_size = \
            term, time_length, adjoint, solver, emb_adjoint, emb_solver, rtol, atol, max_eval, step_size
        self.test_name, self.result_folder = test_name, result_folder
        self.epochs, self.pre_loss_stan, self.AE_lr, self.AE_l2, self.R_lr, self.R_l2 = \
            epochs, pre_loss_stan, AE_lr, AE_l2, R_lr, R_l2
        self.emb_loss_type, self.emblogsig_mean, self.emblogsig_std, self.apply_kinetic_encoder, self.apply_kinetic_decoder, self.apply_kinetic_func, self.is_recon_folddata = \
            emb_loss_type, emblogsig_mean, emblogsig_std, apply_kinetic_encoder, apply_kinetic_decoder, apply_kinetic_func, is_recon_folddata
        self.random_num, self.GPU_NUM = random_num, GPU_NUM
        self.emblogsig_loss_coef, self.embdata_loss_coef, self.embdata_reg_coef, self.emblogsig_abs_mean_reg_coef, self.emblogsig_abs_std_reg_coef, self.enc_quad_coef, self.enc_jac_coef, self.dec_quad_coef, self.dec_jac_coef, self.func_quad_coef, self.func_jac_coef = \
            emblogsig_loss_coef, embdata_loss_coef, embdata_reg_coef, emblogsig_abs_mean_reg_coef, emblogsig_abs_std_reg_coef, enc_quad_coef, enc_jac_coef, dec_quad_coef, dec_jac_coef, func_quad_coef, func_jac_coef
        self.decay_stan, self.decay_factor = decay_stan, decay_factor
        self.foler_name = f"{ds_folder}_{ds_name}_{from_depth}_{to_depth}_{step}"
        if is_train:
            mkdir(result_folder, self.foler_name)
        fix_random_seed(random_num)
        self.args = dict(vars(self))
        # if is_train:
        #     with open(f"{self.result_folder}/param/{self.foler_name}/{self.test_name}.json" , 'w', encoding='utf-8') as f:
        #         json.dump(self.args, f, indent="\t")

        self.device = torch.device(f'cuda:{self.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
        torch.cuda.set_device(self.device)
        
        dl, _, _, _, _, _, _ = get_Data(ds_folder, ds_name, "odernn", depth = 1, step=step, batch_size=batch_size)

        full_batch_size = len(dl.dataset)
        train_dl_todepth, val_dl_todepth, test_dl_todepth, to_logsig_dim, initial_dim, output_dim, return_sequences = get_Data(ds_folder, ds_name, model_type, depth = to_depth, step=step, batch_size=full_batch_size)
        train_dl_fromdepth, val_dl_fromdepth, test_dl_fromdepth, from_logsig_dim, _, _, _ = get_Data(ds_folder, ds_name, model_type,  depth = from_depth, step = step, batch_size = full_batch_size)
        if not is_recon_folddata:
            train_dl_alldata, val_dl_alldata, test_dl_alldata, _, _, _, _ = get_Data(ds_folder, ds_name, "odernn", depth = 1, step = step, batch_size = full_batch_size)
        else:
            train_dl_alldata, val_dl_alldata, test_dl_alldata, _, _, _, _ = get_Data(ds_folder, ds_name, "odernn_folded", depth = 1, step = step, batch_size = full_batch_size)
        
        self.train_dl, self.mean_y, self.std_y = make_new_dataset(train_dl_todepth, train_dl_fromdepth, train_dl_alldata, batch_size, ds_folder)
        self.val_dl, _ , _  = make_new_dataset(val_dl_todepth, val_dl_fromdepth, val_dl_alldata, batch_size, ds_folder, self.mean_y, self.std_y)
        self.test_dl, _ , _ = make_new_dataset(test_dl_todepth, test_dl_fromdepth, test_dl_alldata, batch_size, ds_folder, self.mean_y, self.std_y)
        if task_type_dict[(self.ds_folder,self.ds_name)] == "bi":
            output_dim = 1
        self.model = EmbRDE(initial_dim, func_inputdim, from_logsig_dim, to_logsig_dim,  emb_dim_factor, output_dim, decoutput_dim, apply_final_dec,
                            ienc_param, idec_param, ifunc_param, finalfunc_param, finaldec_param, finaldecdata_param, enc_param, dec_param, func_param,
                            term, time_length, return_sequences, adjoint, solver, emb_adjoint, emb_solver, rtol, atol, max_eval, step_size, emb_loss_type,
                            emblogsig_mean, emblogsig_std, apply_kinetic_encoder, apply_kinetic_decoder, apply_kinetic_func, is_recon_folddata, step).to(self.device)
        self.initial_dim = initial_dim

        dec_param = self.model.get_decoder_param_num()
        rest_param = self.model.get_rest_param_num()
        self.param_num_info = f"{self.test_name} dec/rest #params: {dec_param}/{rest_param} = {dec_param/(rest_param + dec_param)}"
        print(self.param_num_info)
        if is_train:
            self.writer = SummaryWriter(f"{result_folder}/tensorboard/{self.foler_name}/{test_name}")
    def fit(self):
        optimizerAE = optim.Adam(self.model.get_ae_param(), lr=self.AE_lr, weight_decay=self.AE_l2, betas=(0.5, 0.9))
        now_epoch, iteration = -1, -1
        emb_reg_set = [self.emblogsig_loss_coef, self.embdata_loss_coef, self.embdata_reg_coef, self.emblogsig_abs_mean_reg_coef, self.emblogsig_abs_std_reg_coef, 
                       self.enc_quad_coef, self.enc_jac_coef, self.dec_quad_coef, self.dec_jac_coef]
        for now_epoch in range(self.pre_loss_stan):
            now_epoch += 1
            for x, y in self.train_dl:
                # import pdb ; pdb.set_trace()
                x,y = tuple(i.to(self.device) for i in x), y.to(self.device)
                iteration += 1
                result, tensorboard_log = self.model(x, True)
                for k,v in tensorboard_log.items():
                    self.writer.add_scalar(f'loss/{k}', v, iteration)
                emb_loss = 0
                for coef, val in zip(emb_reg_set, result):
                    if val is not None and coef is not None and coef != 0:
                        emb_loss = emb_loss + coef * val
                
                optimizerAE.zero_grad()
                emb_loss.backward()
                optimizerAE.step()

        print("emb_end")
        optimizerR  = optim.Adam(self.model.get_r_param(), lr=self.R_lr, weight_decay=self.R_l2, betas=(0.5, 0.9))
        schedularR = optim.lr_scheduler.StepLR(optimizer=optimizerR, step_size=1, gamma = self.decay_factor)
        before_loss = None
        loss_func = set_loss(task_type_dict[(self.ds_folder,self.ds_name)])
        self.score_info_dic = {}
        self.model_dic = {"name": "EmbRDE", "arg" : self.args, "model" : {}}
        task_reg_set = [self.func_quad_coef, self.func_jac_coef]
        
        iteration = -1
        for epoch in range(self.epochs):
            train_score= {}
            for x, y in self.train_dl:
                # import pdb ; pdb.set_trace()
                x,y = tuple(i.to(self.device) for i in x), y.to(self.device)
                iteration += 1
                result, tensorboard_log =  self.model(x, False)
                y_hat = result[0]
                task_loss = loss_func(y_hat, y)
                self.writer.add_scalar('train/loss_task', task_loss, iteration)
                for k,v in tensorboard_log.items():
                    self.writer.add_scalar(f'loss/{k}', v, iteration)
                
                for coef, val in zip(task_reg_set, result[1:]):
                    if coef is not None and coef != 0:
                        task_loss = task_loss + coef * val
                optimizerR.zero_grad()
                task_loss.backward()
                optimizerR.step()
                task_loss = task_loss.detach().item()
                if before_loss is not None and task_loss - before_loss > self.decay_stan:
                    schedularR.step()
                before_loss = task_loss
               
                key, val = cal_score(y_hat, y, task_type_dict[(self.ds_folder,self.ds_name)])
                if len(train_score) == 0:
                    train_score = {k : [v] for k,v in zip(key,val)}
                else:
                    for k,v in zip(key,val):
                        train_score[k].append(v)
                for i,(k,v) in enumerate(train_score.items()):
                    self.writer.add_scalar(f'train/{i}_{k}', sum(v)/len(v), iteration)
            self.val_test(loss_func,epoch)
    
            


               
               
               
               

