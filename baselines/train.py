import json, sys, os ; sys.path.append("..")
from base_models.get_model import get_model
import torch
from util.data_main_util import get_Data
from util.train_util import fix_random_seed, save_score_model, mkdir, set_loss
from util.evaluate_score import cal_score
# from base_models.nrde import NeuralRDE
from tensorboardX import SummaryWriter
import torch.optim as optim
import torch.nn as nn
from util.data_config import task_type_dict
from util.base import Base_Module
from util.dataset import make_base_dataset


class Main_Module(Base_Module):
    def __init__(self, ds_folder, ds_name, model_type, depth, step, batch_size,  
                 hidden_dim, hidden_hidden_dim, num_layers, logsig_emb_decay, logsig_emb_layers, logsig_emb_hidden,
                 adjoint, solver, test_name, result_folder, random_num, GPU_NUM):

        self.ds_folder, self.ds_name, self.model_type, self.depth, self.step, self.batch_size = ds_folder, ds_name, model_type, depth, step, batch_size
        self.hidden_dim, self.hidden_hidden_dim, self.num_layers = hidden_dim, hidden_hidden_dim, num_layers
        self.logsig_emb_decay, self.logsig_emb_layers, self.logsig_emb_hidden = logsig_emb_decay, logsig_emb_layers, logsig_emb_hidden
        self.adjoint, self.solver, self.test_name, self.result_folder= adjoint, solver, test_name, result_folder
        self.random_num, self.GPU_NUM = random_num, GPU_NUM
        if model_type == "odernn_folded":
            self.foler_name = f"odernn_{ds_folder}_{ds_name}_{step}"
        elif model_type == "embnrde-simple":
            self.foler_name = f"denrde_{depth}_{ds_folder}_{ds_name}_{step}"
        elif "nrde" in model_type:
            self.foler_name = f"{model_type}_{depth}_{ds_folder}_{ds_name}_{step}"
        else:
            self.foler_name = f"{model_type}_{ds_folder}_{ds_name}_{step}"
        mkdir(result_folder, self.foler_name)
        fix_random_seed(random_num)
        self.args = dict(vars(self))

        self.device = torch.device(f'cuda:{self.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
        torch.cuda.set_device(self.device)

        self.train_dl, self.val_dl, self.test_dl, logsig_dim, initial_dim, output_dim, return_sequences = get_Data(ds_folder, ds_name, model_type, depth, step, batch_size)
        self.train_dl, self.val_dl, self.test_dl, logsig_dim, initial_dim, output_dim, return_sequences = get_Data(ds_folder, ds_name, model_type, depth, step, len(self.train_dl.dataset))
        self.train_dl, self.mean_y, self.std_y = make_base_dataset(self.train_dl, batch_size, ds_folder)
        self.val_dl,_,_   = make_base_dataset(self.val_dl,  batch_size, ds_folder, self.mean_y, self.std_y)
        self.test_dl,_,_  = make_base_dataset(self.test_dl, batch_size, ds_folder, self.mean_y, self.std_y)
        if task_type_dict[(self.ds_folder,self.ds_name)] == "bi":
            output_dim = 1

        self.model = get_model(model_type,initial_dim, logsig_dim, hidden_dim, output_dim, hidden_hidden_dim,
                               num_layers, return_sequences, adjoint, solver, logsig_emb_decay, logsig_emb_layers, logsig_emb_hidden).to(self.device)
        self.param_num_info = f"{self.test_name} total #params: {self.model.get_total_param_num()}"
        self.writer = SummaryWriter(f"{result_folder}/tensorboard/{self.foler_name}/{test_name}")
        print(self.model)
        print(self.param_num_info)
        
    def fit(self, epochs = 1000):
        loss_func = set_loss(task_type_dict[(self.ds_folder,self.ds_name)])
        ps =[]
        batch_size = self.train_dl.batch_sampler.batch_size if hasattr(self.train_dl, 'batch_sampler') else self.strain_dl.batch_size
        if self.model_type in ["ancde"]:
            lr = 1e-3
            epochs = 500
        else:
            lr = 0.01 * 32 / batch_size
        for name, param in self.model.named_parameters():
            if "nrde" in self.model_type or "odernn" in self.model_type:
                lr_ = lr if not name.startswith('final_linear') else lr * 10
            else:
                lr_ = 1e-3
            ps.append({"params": param, "lr": lr_})
        if "nrde" in self.model_type or "odernn" in self.model_type:
            optimizerR = optim.Adam(ps)
        else:
            optimizerR = optim.Adam(ps, weight_decay=1e-5)
        schedular = optim.lr_scheduler.ReduceLROnPlateau(optimizerR, patience=15, threshold=1e-6, min_lr=1e-7)
        self.moniter_loss = "acc" if task_type_dict[(self.ds_folder,self.ds_name)] == "multi" else "loss"
        self.moniter_time = 60
        self.no_improve_time = 0
        before_val = -float("inf")

        self.score_info_dic = {}
        self.model_dic = {"name": "EmbRDE", "arg" : self.args, "model" : {}}
        best_train_loss = float("inf")

        iteration = 0
        for epoch in range(epochs):
            no_improve = True
            train_score = {}
            for x,y in self.train_dl:
                
                if type(x) is not torch.Tensor:
                    x,y = tuple(i.to(self.device) for i in x), y.to(self.device)
                else:
                    x, y = x.to(self.device), y.to(self.device)
                
                iteration += 1
                y_hat = self.model(x)
                task_loss = loss_func(y_hat, y)

                if task_loss < best_train_loss:
                    best_train_loss = task_loss
                    no_improve = False

                self.writer.add_scalar('train/loss_task', task_loss, iteration)
                optimizerR.zero_grad()
                task_loss.backward()
                optimizerR.step()
                
                
                
                key, val = cal_score(y_hat, y, task_type_dict[(self.ds_folder,self.ds_name)])
                if len(train_score) == 0:
                    train_score = {k : [v] for k,v in zip(key,val)}
                else:
                    for k,v in zip(key,val):
                        train_score[k].append(v)
            for i,(k,v) in enumerate(train_score.items()):
                self.writer.add_scalar(f'train/{i}_{k}', sum(v)/len(v), epoch)
            
            

            val_loss = self.val_test(loss_func,epoch)
            if self.model_type in ["nrde", "embnrde-simple", "gru", "odernn"]:
                if before_val < self.score_info_dic[self.moniter_loss][0]:
                    self.no_improve_time = 0
                    before_val = self.score_info_dic[self.moniter_loss][0]
                else:
                    self.no_improve_time += 1
                
                if self.moniter_time <= self.no_improve_time:
                    break
                schedular.step(val_loss)
            elif self.model_type in ["ancde"]:
                if no_improve:
                    self.no_improve_time += 1
                else:
                    self.no_improve_time = 0
                if self.moniter_time <= self.no_improve_time:
                    break