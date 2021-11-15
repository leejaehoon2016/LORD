
def get_hyperparam(folder, name, d1, d2, p):
    return Args(eval(f"{folder}_{name}_{d1}_{d2}_{p}"))

class Args:
    def __init__(self,dic):
        for k,v in dic.items():
            setattr(self,k,v)



TSR_BIDMC32HR_1_2_8   = {"func_inputdim": 32, "h_out": 64,  "N_out": 0, "hf": 64, "Nf": 3, "ho": 64, "No": 3, "hg": 64, "Ng": 3, "iter_ae": 400,  "iter_task": 2000, "c_e": 0, "c_ae": 1e-5, "c_task": 1e-5}
TSR_BIDMC32HR_1_2_128 = {"func_inputdim": 64, "h_out": 32,  "N_out": 3, "hf": 32, "Nf": 3, "ho": 32, "No": 3, "hg": 64, "Ng": 3, "iter_ae": 1000, "iter_task": 500,  "c_e": 0, "c_ae": 1e-5, "c_task": 1e-5}
TSR_BIDMC32HR_1_2_512 = {"func_inputdim": 64, "h_out": 32,  "N_out": 3, "hf": 32, "Nf": 3, "ho": 32, "No": 3, "hg": 64, "Ng": 3, "iter_ae": 1000, "iter_task": 500,  "c_e": 0, "c_ae": 1e-5, "c_task": 1e-5}
TSR_BIDMC32HR_1_3_8   = {"func_inputdim": 32, "h_out": 64,  "N_out": 0, "hf": 64, "Nf": 3, "ho": 64, "No": 3, "hg": 64, "Ng": 3, "iter_ae": 400,  "iter_task": 2000, "c_e": 0, "c_ae": 1e-5, "c_task": 1e-5}
TSR_BIDMC32HR_1_3_128 = {"func_inputdim": 64, "h_out": 32,  "N_out": 2, "hf": 128,"Nf": 3, "ho": 128,"No": 3, "hg": 64, "Ng": 3, "iter_ae": 1000, "iter_task": 500,  "c_e": 0, "c_ae": 1e-5, "c_task": 1e-5}
TSR_BIDMC32HR_1_3_512 = {"func_inputdim": 64, "h_out": 128, "N_out": 2, "hf": 32, "Nf": 3, "ho": 32, "No": 3, "hg": 64, "Ng": 3, "iter_ae": 1000, "iter_task": 500,  "c_e": 0, "c_ae": 1e-5, "c_task": 1e-5}
TSR_BIDMC32HR_2_3_8   = {"func_inputdim": 32, "h_out": 64,  "N_out": 0, "hf": 64, "Nf": 3, "ho": 64, "No": 3, "hg": 64, "Ng": 3, "iter_ae": 400,  "iter_task": 2000, "c_e": 0, "c_ae": 1e-5, "c_task": 1e-5}
TSR_BIDMC32HR_2_3_128 = {"func_inputdim": 64, "h_out": 64,  "N_out": 2, "hf": 32, "Nf": 3, "ho": 32, "No": 3, "hg": 64, "Ng": 3, "iter_ae": 1000, "iter_task": 500,  "c_e": 0, "c_ae": 1e-5, "c_task": 1e-5}
TSR_BIDMC32HR_2_3_512 = {"func_inputdim": 64, "h_out": 32,  "N_out": 2, "hf": 64, "Nf": 3, "ho": 64, "No": 3, "hg": 64, "Ng": 3, "iter_ae": 1000, "iter_task": 500,  "c_e": 0, "c_ae": 1e-5, "c_task": 1e-5}


TSR_BIDMC32SpO2_1_2_8   = {"func_inputdim": 32, "h_out": 64,  "N_out": 0, "hf": 64, "Nf": 3, "ho": 64, "No": 3, "hg": 64, "Ng": 3, "iter_ae": 400,  "iter_task": 2000, "c_e": 0, "c_ae": 1e-5, "c_task": 1e-5}
TSR_BIDMC32SpO2_1_2_128 = {"func_inputdim": 64, "h_out": 32,  "N_out": 3, "hf": 32, "Nf": 3, "ho": 32, "No": 3, "hg": 64, "Ng": 3, "iter_ae": 1000, "iter_task": 500,  "c_e": 0, "c_ae": 1e-5, "c_task": 1e-5}
TSR_BIDMC32SpO2_1_2_512 = {"func_inputdim": 64, "h_out": 128, "N_out": 2, "hf": 64, "Nf": 3, "ho": 64, "No": 3, "hg": 64, "Ng": 3, "iter_ae": 1000, "iter_task": 500,  "c_e": 0, "c_ae": 1e-5, "c_task": 1e-5}
TSR_BIDMC32SpO2_1_3_8   = {"func_inputdim": 32, "h_out": 64,  "N_out": 0, "hf": 64, "Nf": 3, "ho": 64, "No": 3, "hg": 64, "Ng": 3, "iter_ae": 400,  "iter_task": 2000, "c_e": 0, "c_ae": 1e-5, "c_task": 1e-5}
TSR_BIDMC32SpO2_1_3_128 = {"func_inputdim": 32, "h_out": 64,  "N_out": 0, "hf": 64, "Nf": 3, "ho": 64, "No": 3, "hg": 64, "Ng": 3, "iter_ae": 1000, "iter_task": 500,  "c_e": 0, "c_ae": 1e-5, "c_task": 1e-5}
TSR_BIDMC32SpO2_1_3_512 = {"func_inputdim": 64, "h_out": 64,  "N_out": 2, "hf": 32, "Nf": 3, "ho": 32, "No": 3, "hg": 64, "Ng": 3, "iter_ae": 1000, "iter_task": 500,  "c_e": 0, "c_ae": 1e-5, "c_task": 1e-5}
TSR_BIDMC32SpO2_2_3_8   = {"func_inputdim": 32, "h_out": 64,  "N_out": 0, "hf": 64, "Nf": 3, "ho": 64, "No": 3, "hg": 64, "Ng": 3, "iter_ae": 400,  "iter_task": 2000, "c_e": 0, "c_ae": 1e-5, "c_task": 1e-5}
TSR_BIDMC32SpO2_2_3_128 = {"func_inputdim": 32, "h_out": 128, "N_out": 2, "hf": 64, "Nf": 3, "ho": 64, "No": 3, "hg": 64, "Ng": 3, "iter_ae": 1000, "iter_task": 500,  "c_e": 0, "c_ae": 1e-5, "c_task": 1e-5}
TSR_BIDMC32SpO2_2_3_512 = {"func_inputdim": 32, "h_out": 64,  "N_out": 2, "hf": 128,"Nf": 3, "ho": 128,"No": 3, "hg": 64, "Ng": 3, "iter_ae": 1000, "iter_task": 500,  "c_e": 0, "c_ae": 1e-5, "c_task": 1e-5}
