from .odernn import ODE_RNN
from .nrde import NeuralRDE
# from .gru import GRU
from .nrde_simple_emb import NeuralRDE_SimpleEmb
from .ancde import ANCDE

def get_model(model_type, initial_dim, input_dim, hidden_dim, output_dim, hidden_hidden_dim,
                num_layers, return_sequences, adjoint, solver, logsig_emb_decay, logsig_emb_layers, logsig_emb_hidden):
    

    model_type = model_type.split("_")[0]
    if model_type == "nrde" or model_type == "ncde":
        model = NeuralRDE(
                        initial_dim, input_dim, hidden_dim, output_dim, hidden_hidden_dim=hidden_hidden_dim,
                        num_layers=num_layers, return_sequences=return_sequences, adjoint=adjoint, solver=solver)
    elif model_type == "embnrde-simple":
        model = NeuralRDE_SimpleEmb(
                        initial_dim, input_dim, hidden_dim, output_dim, hidden_hidden_dim=hidden_hidden_dim,
                        num_layers=num_layers, return_sequences=return_sequences, adjoint=adjoint, solver=solver, 
                        logsig_emb_decay = logsig_emb_decay, logsig_emb_layers= logsig_emb_layers, logsig_emb_hidden= logsig_emb_hidden,)
    elif model_type == "odernn":
        model = ODE_RNN(input_dim, hidden_dim, output_dim, gru=False, return_sequences=return_sequences)
    
    elif model_type == "ancde":
        model = ANCDE(initial_dim, hidden_dim, output_dim, num_layers, attention_channel=hidden_hidden_dim, slope_check = False,soft = True,timewise=False, initial=False,return_sequences=return_sequences)
    else:
        assert 0, f"{model_type} no implement"
    return model

    