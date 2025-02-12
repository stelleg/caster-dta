import torch
import torch.nn as nn
from torch_geometric.utils import degree as pyg_degree

def _select_activation(activation):
        """
        Select the activation function from a string 
        (or return the activation function if it is already a module)
        """
        activation = activation.lower()

        if(isinstance(activation, nn.Module)):
            return activation

        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'leaky_relu':
            return nn.LeakyReLU(0.01)
        elif activation == 'tanh':
            return nn.Tanh()
        elif activation == 'sigmoid':
            return nn.Sigmoid()
        elif activation == 'gelu':
            return nn.GELU()
        elif activation == 'elu':
            return nn.ELU()
        elif activation == 'selu':
            return nn.SELU()
        elif activation == 'swish' or activation == 'silu':
            return nn.SiLU()
        elif activation == 'none':
            return nn.Identity()
        else:
            raise ValueError(f"Activation function {activation} not recognized as a string. You can pass the module directly as an argument instead of a string.")
        

def _calc_PNA_degrees(in_ds, for_type='molecule'):
    """
    Calculate the histogram of degrees of the graph for the PNA model
    """
    if(for_type == 'molecule'):
        ind = 1
    elif(for_type == 'protein'):
        ind = 0

    max_degree = -1

    for data in in_ds:
        data = data[ind]
        d = pyg_degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        max_degree = max(max_degree, int(d.max()))

    deg = torch.zeros(max_degree + 1, dtype=torch.long)
    for data in in_ds:
        data = data[ind]
        d = pyg_degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel())

    return deg