import torch_geometric as pyg
import torch

import ipdb

def construct_graph(node_features, edge_features, ntypes, etypes):
    """
    Construct a PyG homogeneous graph from node and edge features
    Note - we aren't constructing an explicit HeteroData object here
    as we will store the node and edge type independently in the PyG data object
    and it can be converted to a heterogeneous graph using to_heterogeneous
    """
    
    # Node and edge types as long tensors
    node_types = torch.tensor(ntypes, dtype=torch.long)
    edge_types = torch.tensor(etypes, dtype=torch.long).flatten()


    # Node feature tensor
    # Note that if this is a tuple, we will torch.tensor each of the elements of the tuple
    if(isinstance(node_features, tuple)):
        node_features = tuple(torch.tensor(feat, dtype=torch.float32) for feat in node_features)
        n_residues = node_features[0].shape[0]
    else:
        node_features = torch.tensor(node_features, dtype=torch.float32)
        n_residues = node_features.shape[0]

    # Make the edge index 
    edge_index = torch.tensor([[i, j] for i in range(n_residues) for j in range(n_residues)], dtype=torch.long).t().contiguous()

    # Edge attributes based on edge features in COO format
    # basically flatten the edge features array along the first two dimensions
    # Note that we will do this for every element of a tuple if it is one
    if(isinstance(edge_features, tuple)):
        edge_attr = tuple(torch.tensor(feat, dtype=torch.float32).flatten(0,1) for feat in edge_features)
        edge_nan_checker = edge_attr[0]
    else:
        edge_attr = torch.tensor(edge_features, dtype=torch.float32).flatten(0,1)
        edge_nan_checker = edge_attr


    # Filter out index and edge features that are nan
    # (this will handle molecule graphs too which are not actually fully connected)
    keep_indices = (~torch.isnan(edge_nan_checker).all(axis=-1))
    edge_index = edge_index[:,keep_indices]

    if(isinstance(edge_features, tuple)):
        edge_attr = tuple(feat[keep_indices,:] for feat in edge_attr)
    else:
        edge_attr = edge_attr[keep_indices,:]
    
    edge_types = edge_types[keep_indices]


    # Make the PyG data object
    # NOTE: we could include a pos attribute and use PointTransformerConv, but we won't do this
    # for now, for the sake of simplicity and for more consistency with the original PocketMiner code
    pyg_data = pyg.data.Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
    pyg_data.node_type = node_types
    pyg_data.edge_type = edge_types

    return pyg_data
