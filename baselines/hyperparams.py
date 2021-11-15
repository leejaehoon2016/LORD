def get_hyperparam(model, folder, name, d, p):
    if "nrde" in model:
        return Args(eval(f"{model}_{d}_{folder}_{name}_{p}"))
    else:
        return Args(eval(f"{model}_{folder}_{name}_{p}"))


class Args:
    def __init__(self,arg_dic):
        dic = {"h":0, "hh":0, "l":0,"logsig_emb_decay":0, "logsig_emb_layers":0, "logsig_emb_hidden":0}
        dic.update(arg_dic)
        for k,v in dic.items():
            setattr(self,k,v)

odernn_TSR_BIDMC32HR_1  = odernn_TSR_BIDMC32HR_8   = odernn_TSR_BIDMC32HR_128   = odernn_TSR_BIDMC32HR_512   = {"h":32}
odernn_TSR_BIDMC32SpO2_1 = odernn_TSR_BIDMC32SpO2_8 = odernn_TSR_BIDMC32SpO2_128 = odernn_TSR_BIDMC32SpO2_512 = {"h":32}

ncde_TSR_BIDMC32HR_1 = ncde_TSR_BIDMC32HR_8     = ncde_TSR_BIDMC32HR_128     = ncde_TSR_BIDMC32HR_512     = {"h":64,"hh":128,"l":2}
ncde_TSR_BIDMC32SpO2_1 = ncde_TSR_BIDMC32SpO2_8   = ncde_TSR_BIDMC32SpO2_128   = ncde_TSR_BIDMC32SpO2_512   = {"h":64,"hh":192,"l":3}

ancde_TSR_BIDMC32HR_1    = {"h":128,"hh":64,"l":2}
ancde_TSR_BIDMC32HR_8    = {"h":128,"hh":64,"l":2}
ancde_TSR_BIDMC32HR_128  = {"h":128,"hh":64,"l":2}
ancde_TSR_BIDMC32HR_512  = {"h":128,"hh":64,"l":2}

ancde_TSR_BIDMC32SpO2_1     = {"h":128,"hh":64,"l":2}
ancde_TSR_BIDMC32SpO2_8     = {"h":128,"hh":64,"l":2}
ancde_TSR_BIDMC32SpO2_128   = {"h":128,"hh":64,"l":2}
ancde_TSR_BIDMC32SpO2_512   = {"h":128,"hh":64,"l":2}


nrde_2_TSR_BIDMC32HR_8     = nrde_2_TSR_BIDMC32HR_128     = nrde_2_TSR_BIDMC32HR_512     = {"h":64,"hh":128,"l":2}
nrde_2_TSR_BIDMC32SpO2_8   = nrde_2_TSR_BIDMC32SpO2_128   = nrde_2_TSR_BIDMC32SpO2_512   = {"h":64,"hh":192,"l":3}
nrde_3_TSR_BIDMC32HR_8     = nrde_3_TSR_BIDMC32HR_128     = nrde_3_TSR_BIDMC32HR_512     = {"h":64,"hh":128,"l":2}
nrde_3_TSR_BIDMC32SpO2_8   = nrde_3_TSR_BIDMC32SpO2_128   = nrde_3_TSR_BIDMC32SpO2_512   = {"h":64,"hh":192,"l":3}

denrde_2_TSR_BIDMC32HR_8   = {**nrde_2_TSR_BIDMC32HR_8,   "logsig_emb_decay":0.7, "logsig_emb_layers":2, "logsig_emb_hidden":64}
denrde_2_TSR_BIDMC32HR_128 = {**nrde_2_TSR_BIDMC32HR_128, "logsig_emb_decay":0.7, "logsig_emb_layers":1, "logsig_emb_hidden":128}
denrde_2_TSR_BIDMC32HR_512 = {**nrde_2_TSR_BIDMC32HR_512, "logsig_emb_decay":0.7, "logsig_emb_layers":1, "logsig_emb_hidden":64}
denrde_3_TSR_BIDMC32HR_8   = {**nrde_3_TSR_BIDMC32HR_8,   "logsig_emb_decay":0.7, "logsig_emb_layers":2, "logsig_emb_hidden":128}
denrde_3_TSR_BIDMC32HR_128 = {**nrde_3_TSR_BIDMC32HR_128, "logsig_emb_decay":0.7, "logsig_emb_layers":1, "logsig_emb_hidden":128}
denrde_3_TSR_BIDMC32HR_512 = {**nrde_3_TSR_BIDMC32HR_512, "logsig_emb_decay":0.7, "logsig_emb_layers":1, "logsig_emb_hidden":128}

denrde_2_TSR_BIDMC32SpO2_8   = {**nrde_2_TSR_BIDMC32SpO2_8,   "logsig_emb_decay":0.5, "logsig_emb_layers":1, "logsig_emb_hidden":128}
denrde_2_TSR_BIDMC32SpO2_128 = {**nrde_2_TSR_BIDMC32SpO2_128, "logsig_emb_decay":0.7, "logsig_emb_layers":1, "logsig_emb_hidden":64}
denrde_2_TSR_BIDMC32SpO2_512 = {**nrde_2_TSR_BIDMC32SpO2_512, "logsig_emb_decay":0.7, "logsig_emb_layers":1, "logsig_emb_hidden":64}
denrde_3_TSR_BIDMC32SpO2_8   = {**nrde_3_TSR_BIDMC32SpO2_8,   "logsig_emb_decay":0.5, "logsig_emb_layers":1, "logsig_emb_hidden":128}
denrde_3_TSR_BIDMC32SpO2_128 = {**nrde_3_TSR_BIDMC32SpO2_128, "logsig_emb_decay":0.7, "logsig_emb_layers":1, "logsig_emb_hidden":64}
denrde_3_TSR_BIDMC32SpO2_512 = {**nrde_3_TSR_BIDMC32SpO2_512, "logsig_emb_decay":0.7, "logsig_emb_layers":1, "logsig_emb_hidden":64}