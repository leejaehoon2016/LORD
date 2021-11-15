import torch, json
import torch.nn as nn
from .evaluate_score import cal_score
from .train_util import save_score_model
from .data_config import task_type_dict
class Base_Module:
    def val_test(self, loss_func, epoch):
        # val
        self.model.eval()
        val_hat_y  = []
        val_true_y = []
        for x,y in self.val_dl:
            if type(x) is not torch.Tensor:
                x,y = tuple(i.to(self.device) for i in x), y.to(self.device)
            else:
                x, y = x.to(self.device), y.to(self.device)
            y_hat = self.model(x)
            if type(y_hat) is tuple:
                y_hat = y_hat[0][0]
            val_true_y.append(y)
            val_hat_y.append(y_hat)
        val_hat, val_true = torch.cat(val_hat_y,dim=0), torch.cat(val_true_y,dim=0)
        key, v_val = cal_score(val_hat, val_true, task_type_dict[(self.ds_folder,self.ds_name)])
        for i,(k,v) in enumerate(zip(key, v_val)):
            self.writer.add_scalar(f'val/{i}_{k}', v, epoch)
        val_loss = loss_func(val_hat, val_true)
        self.writer.add_scalar(f'val/loss', val_loss, epoch)       
        
        # test
        test_hat_y  = []
        test_true_y = []
        for x,y in self.test_dl:
            if type(x) is not torch.Tensor:
                x,y = tuple(i.to(self.device) for i in x), y.to(self.device)
            else:
                x, y = x.to(self.device), y.to(self.device)
            y_hat = self.model(x)
            if type(y_hat) is tuple:
                y_hat = y_hat[0][0]
            test_true_y.append(y)
            test_hat_y.append(y_hat)
        test_hat, test_true = torch.cat(test_hat_y,dim=0), torch.cat(test_true_y,dim=0)
        key, t_val = cal_score(test_hat, test_true, task_type_dict[(self.ds_folder,self.ds_name)])
        for i,(k,v) in enumerate(zip(key, t_val)):
            self.writer.add_scalar(f'test/{i}_{k}', v, epoch)
        test_loss = loss_func(test_hat, test_true)
        self.writer.add_scalar(f'test/loss', test_loss, epoch)                 
        
        change = save_score_model(self.model, self.score_info_dic, self.model_dic, key + ["loss"] , v_val + [float(-val_loss.item())],
                                    t_val + [float(-test_loss.item())], epoch)
        if change:
            self.save_score_model_dic()
        self.model.train()      
        return val_loss    
    def save_score_model_dic(self):
        # torch.save(self.model_dic, f"{self.result_folder}/save_model/{self.foler_name}/{self.test_name}.pth")
        tmp_dic = self.score_info_dic.copy()
        tmp_dic["num_param"] = self.param_num_info
        with open(f"{self.result_folder}/score/{self.foler_name}/{self.test_name}.json" , 'w', encoding='utf-8') as f:
            json.dump(tmp_dic, f, indent="\t")

