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

odernn_UEA_EigenWorms_4   = odernn_UEA_EigenWorms_32   = odernn_UEA_EigenWorms_128   = {"h":128}
odernn_UEA_CounterMovementJump_64  = {"h":64}
odernn_UEA_CounterMovementJump_128 = {"h":256}
odernn_UEA_CounterMovementJump_512 = {"h":64}
odernn_UEA_SelfRegulationSCP2_32   = {"h":128}
odernn_UEA_SelfRegulationSCP2_64   = {"h":64}
odernn_UEA_SelfRegulationSCP2_256  = {"h":64}
odernn_TSR_BIDMC32HR_8   = odernn_TSR_BIDMC32HR_128   = odernn_TSR_BIDMC32HR_512   = {"h":32}
odernn_TSR_BIDMC32RR_8   = odernn_TSR_BIDMC32RR_128   = odernn_TSR_BIDMC32RR_512   = {"h":192}
odernn_TSR_BIDMC32SpO2_8 = odernn_TSR_BIDMC32SpO2_128 = odernn_TSR_BIDMC32SpO2_512 = {"h":32}


ncde_UEA_EigenWorms_4   = ncde_UEA_EigenWorms_32   = ncde_UEA_EigenWorms_128   = {"h":32,"hh":64,"l":3}
ncde_UEA_CounterMovementJump_64  = {"h":64, "hh":128,"l":3}
ncde_UEA_CounterMovementJump_128 = {"h":64, "hh":64, "l":3}
ncde_UEA_CounterMovementJump_512 = {"h":128,"hh":64, "l":3}
ncde_UEA_SelfRegulationSCP2_32   = {"h":64, "hh":64, "l":2}
ncde_UEA_SelfRegulationSCP2_64   = {"h":64, "hh":256,"l":3}
ncde_UEA_SelfRegulationSCP2_256  = {"h":256,"hh":128,"l":3}
ncde_TSR_BIDMC32HR_8     = ncde_TSR_BIDMC32HR_128     = ncde_TSR_BIDMC32HR_512     = {"h":64,"hh":128,"l":2}
ncde_TSR_BIDMC32RR_8     = ncde_TSR_BIDMC32RR_128     = ncde_TSR_BIDMC32RR_512     = {"h":64,"hh":192,"l":3}
ncde_TSR_BIDMC32SpO2_8   = ncde_TSR_BIDMC32SpO2_128   = ncde_TSR_BIDMC32SpO2_512   = {"h":64,"hh":192,"l":3}


ancde_UEA_EigenWorms_4   = {"h":128,"hh":64,"l":3}
ancde_UEA_EigenWorms_32  = {"h":128,"hh":64,"l":3}
ancde_UEA_EigenWorms_128 = {"h":64, "hh":32,"l":2}
ancde_UEA_CounterMovementJump_64  = {"h":128,"hh":128,"l":2}
ancde_UEA_CounterMovementJump_128 = {"h":256,"hh":128, "l":3}
ancde_UEA_CounterMovementJump_512 = {"h":64, "hh":256, "l":2}
ancde_UEA_SelfRegulationSCP2_32   = {"h":256,"hh":128, "l":3}
ancde_UEA_SelfRegulationSCP2_64   = {"h":64, "hh":128,"l":2}
ancde_UEA_SelfRegulationSCP2_256  = {"h":128,"hh":128,"l":2}
ancde_TSR_BIDMC32HR_8   = ancde_TSR_BIDMC32HR_128   = ancde_TSR_BIDMC32HR_512 = {"h":128,"hh":64,"l":2}
ancde_TSR_BIDMC32RR_8   = ancde_TSR_BIDMC32RR_128 = {"h":128,"hh":64,"l":2} ; ancde_TSR_BIDMC32RR_512   = {"h":128,"hh":64,"l":3}
ancde_TSR_BIDMC32SpO2_8 = ancde_TSR_BIDMC32SpO2_128 = ancde_TSR_BIDMC32SpO2_512 = {"h":128,"hh":64,"l":2}


nrde_2_UEA_EigenWorms_4   = nrde_2_UEA_EigenWorms_32   = nrde_2_UEA_EigenWorms_128   = {"h":32,"hh":64,"l":3}
nrde_2_UEA_CounterMovementJump_64  = {"h":64, "hh":128,"l":2}
nrde_2_UEA_CounterMovementJump_128 = {"h":256,"hh":64, "l":2}
nrde_2_UEA_CounterMovementJump_512 = {"h":64, "hh":64, "l":2}
nrde_2_UEA_SelfRegulationSCP2_32   = {"h":64, "hh":128,"l":2}
nrde_2_UEA_SelfRegulationSCP2_64   = {"h":256,"hh":64, "l":2}
nrde_2_UEA_SelfRegulationSCP2_256  = {"h":256,"hh":64, "l":2}
nrde_2_TSR_BIDMC32HR_8     = nrde_2_TSR_BIDMC32HR_128     = nrde_2_TSR_BIDMC32HR_512     = {"h":64,"hh":128,"l":2}
nrde_2_TSR_BIDMC32RR_8     = nrde_2_TSR_BIDMC32RR_128     = nrde_2_TSR_BIDMC32RR_512     = {"h":64,"hh":192,"l":3}
nrde_2_TSR_BIDMC32SpO2_8   = nrde_2_TSR_BIDMC32SpO2_128   = nrde_2_TSR_BIDMC32SpO2_512   = {"h":64,"hh":192,"l":3}


nrde_3_UEA_EigenWorms_4   = nrde_3_UEA_EigenWorms_32   = nrde_3_UEA_EigenWorms_128   = {"h":32,"hh":64,"l":3}
nrde_3_UEA_CounterMovementJump_64  = {"h":128,"hh":128,"l":3}
nrde_3_UEA_CounterMovementJump_128 = {"h":256,"hh":64, "l":3}
nrde_3_UEA_CounterMovementJump_512 = {"h":256,"hh":256,"l":3}
nrde_3_UEA_SelfRegulationSCP2_32   = {"h":128,"hh":128,"l":2}
nrde_3_UEA_SelfRegulationSCP2_64   = {"h":128,"hh":64, "l":2}
nrde_3_UEA_SelfRegulationSCP2_256  = {"h":64, "hh":256, "l":3}
nrde_3_TSR_BIDMC32HR_8     = nrde_3_TSR_BIDMC32HR_128     = nrde_3_TSR_BIDMC32HR_512     = {"h":64,"hh":128,"l":2}
nrde_3_TSR_BIDMC32RR_8     = nrde_3_TSR_BIDMC32RR_128     = nrde_3_TSR_BIDMC32RR_512     = {"h":64,"hh":192,"l":3}
nrde_3_TSR_BIDMC32SpO2_8   = nrde_3_TSR_BIDMC32SpO2_128   = nrde_3_TSR_BIDMC32SpO2_512   = {"h":64,"hh":192,"l":3}


denrde_2_UEA_EigenWorms_4            = {**nrde_2_UEA_EigenWorms_4,           "logsig_emb_decay":0.5, "logsig_emb_layers":1, "logsig_emb_hidden":128}
denrde_2_UEA_EigenWorms_32           = {**nrde_2_UEA_EigenWorms_32,          "logsig_emb_decay":0.5, "logsig_emb_layers":1, "logsig_emb_hidden":128}
denrde_2_UEA_EigenWorms_128          = {**nrde_2_UEA_EigenWorms_128,         "logsig_emb_decay":0.7, "logsig_emb_layers":1, "logsig_emb_hidden":64}
denrde_2_UEA_CounterMovementJump_64  = {**nrde_2_UEA_CounterMovementJump_64, "logsig_emb_decay":0.3, "logsig_emb_layers":1, "logsig_emb_hidden":128}
denrde_2_UEA_CounterMovementJump_128 = {**nrde_2_UEA_CounterMovementJump_128,"logsig_emb_decay":0.3, "logsig_emb_layers":2, "logsig_emb_hidden":128}
denrde_2_UEA_CounterMovementJump_512 = {**nrde_2_UEA_CounterMovementJump_512,"logsig_emb_decay":0.7, "logsig_emb_layers":1, "logsig_emb_hidden":64}
denrde_2_UEA_SelfRegulationSCP2_32   = {**nrde_2_UEA_SelfRegulationSCP2_32,  "logsig_emb_decay":0.3, "logsig_emb_layers":1, "logsig_emb_hidden":64}
denrde_2_UEA_SelfRegulationSCP2_64   = {**nrde_2_UEA_SelfRegulationSCP2_64,  "logsig_emb_decay":0.3, "logsig_emb_layers":2, "logsig_emb_hidden":64}
denrde_2_UEA_SelfRegulationSCP2_256  = {**nrde_2_UEA_SelfRegulationSCP2_256, "logsig_emb_decay":0.3, "logsig_emb_layers":2, "logsig_emb_hidden":64}
denrde_2_TSR_BIDMC32HR_8     = {**nrde_2_TSR_BIDMC32HR_8,   "logsig_emb_decay":0.7, "logsig_emb_layers":2, "logsig_emb_hidden":64}
denrde_2_TSR_BIDMC32HR_128   = {**nrde_2_TSR_BIDMC32HR_128, "logsig_emb_decay":0.7, "logsig_emb_layers":1, "logsig_emb_hidden":128}
denrde_2_TSR_BIDMC32HR_512   = {**nrde_2_TSR_BIDMC32HR_512, "logsig_emb_decay":0.7, "logsig_emb_layers":1, "logsig_emb_hidden":64}
denrde_2_TSR_BIDMC32RR_8     = {**nrde_2_TSR_BIDMC32RR_8,   "logsig_emb_decay":0.5, "logsig_emb_layers":1, "logsig_emb_hidden":128}
denrde_2_TSR_BIDMC32RR_128   = {**nrde_2_TSR_BIDMC32RR_128, "logsig_emb_decay":0.7, "logsig_emb_layers":1, "logsig_emb_hidden":128}
denrde_2_TSR_BIDMC32RR_512   = {**nrde_2_TSR_BIDMC32RR_512, "logsig_emb_decay":0.7, "logsig_emb_layers":1, "logsig_emb_hidden":64}
denrde_2_TSR_BIDMC32SpO2_8   = {**nrde_2_TSR_BIDMC32SpO2_8,   "logsig_emb_decay":0.5, "logsig_emb_layers":1, "logsig_emb_hidden":128}
denrde_2_TSR_BIDMC32SpO2_128 = {**nrde_2_TSR_BIDMC32SpO2_128, "logsig_emb_decay":0.7, "logsig_emb_layers":1, "logsig_emb_hidden":64}
denrde_2_TSR_BIDMC32SpO2_512 = {**nrde_2_TSR_BIDMC32SpO2_512, "logsig_emb_decay":0.7, "logsig_emb_layers":1, "logsig_emb_hidden":64}


denrde_3_UEA_EigenWorms_4            = {**nrde_3_UEA_EigenWorms_4,           "logsig_emb_decay":0.5, "logsig_emb_layers":1, "logsig_emb_hidden":128}
denrde_3_UEA_EigenWorms_32           = {**nrde_3_UEA_EigenWorms_32,          "logsig_emb_decay":0.5, "logsig_emb_layers":1, "logsig_emb_hidden":128}
denrde_3_UEA_EigenWorms_128          = {**nrde_3_UEA_EigenWorms_128,         "logsig_emb_decay":0.7, "logsig_emb_layers":1, "logsig_emb_hidden":128}
denrde_3_UEA_CounterMovementJump_64  = {**nrde_3_UEA_CounterMovementJump_64, "logsig_emb_decay":0.3, "logsig_emb_layers":2, "logsig_emb_hidden":128}
denrde_3_UEA_CounterMovementJump_128 = {**nrde_3_UEA_CounterMovementJump_128,"logsig_emb_decay":0.5, "logsig_emb_layers":2, "logsig_emb_hidden":64}
denrde_3_UEA_CounterMovementJump_512 = {**nrde_3_UEA_CounterMovementJump_512,"logsig_emb_decay":0.3, "logsig_emb_layers":1, "logsig_emb_hidden":128}
denrde_3_UEA_SelfRegulationSCP2_32   = {**nrde_3_UEA_SelfRegulationSCP2_32,  "logsig_emb_decay":0.3, "logsig_emb_layers":2, "logsig_emb_hidden":128}
denrde_3_UEA_SelfRegulationSCP2_64   = {**nrde_3_UEA_SelfRegulationSCP2_64,  "logsig_emb_decay":0.3, "logsig_emb_layers":2, "logsig_emb_hidden":64}
denrde_3_UEA_SelfRegulationSCP2_256  = {**nrde_3_UEA_SelfRegulationSCP2_256, "logsig_emb_decay":0.3, "logsig_emb_layers":2, "logsig_emb_hidden":128}
denrde_3_TSR_BIDMC32HR_8     = {**nrde_3_TSR_BIDMC32HR_8,   "logsig_emb_decay":0.7, "logsig_emb_layers":2, "logsig_emb_hidden":128}
denrde_3_TSR_BIDMC32HR_128   = {**nrde_3_TSR_BIDMC32HR_128, "logsig_emb_decay":0.7, "logsig_emb_layers":1, "logsig_emb_hidden":128}
denrde_3_TSR_BIDMC32HR_512   = {**nrde_3_TSR_BIDMC32HR_512, "logsig_emb_decay":0.7, "logsig_emb_layers":1, "logsig_emb_hidden":128}
denrde_3_TSR_BIDMC32RR_8     = {**nrde_3_TSR_BIDMC32RR_8,   "logsig_emb_decay":0.5, "logsig_emb_layers":1, "logsig_emb_hidden":128}
denrde_3_TSR_BIDMC32RR_128   = {**nrde_3_TSR_BIDMC32RR_128, "logsig_emb_decay":0.7, "logsig_emb_layers":1, "logsig_emb_hidden":128}
denrde_3_TSR_BIDMC32RR_512   = {**nrde_3_TSR_BIDMC32RR_512, "logsig_emb_decay":0.7, "logsig_emb_layers":1, "logsig_emb_hidden":64}
denrde_3_TSR_BIDMC32SpO2_8   = {**nrde_3_TSR_BIDMC32SpO2_8,   "logsig_emb_decay":0.5, "logsig_emb_layers":1, "logsig_emb_hidden":128}
denrde_3_TSR_BIDMC32SpO2_128 = {**nrde_3_TSR_BIDMC32SpO2_128, "logsig_emb_decay":0.7, "logsig_emb_layers":1, "logsig_emb_hidden":64}
denrde_3_TSR_BIDMC32SpO2_512 = {**nrde_3_TSR_BIDMC32SpO2_512, "logsig_emb_decay":0.7, "logsig_emb_layers":1, "logsig_emb_hidden":64}