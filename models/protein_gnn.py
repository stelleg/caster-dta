import torch_geometric as pyg
import torch.nn as nn
import torch

from functools import partial
import warnings

import models.gvp_layers as gvp
from models.model_utils import _select_activation

import ipdb


class SelectableProteinModelWrapper(nn.Module):
    """
    A wrapper that can switch between the different types of protein GNNs
    based on user input and arguments (homogeneous vs heterogeneous), (scalar vs vector)
    One can bypass this entirely by passing in a specific model to this class
    or by just skipping this class altogether and using a specific model directly
    """
    def __init__(self,
                 in_channels,
                 edge_dim,
                 base_conv, # 'simplegvp', 'pocketminer', 'gatv2', 'heat'
                 **kwargs):
        
        super(SelectableProteinModelWrapper, self).__init__()

        passed_args = locals().copy()
        exclude_args = ['self', 'passed_args', '__class__', 'kwargs', 
                        'base_conv', 'force_model']
        passed_args = {k: v for k, v in passed_args.items() if k not in exclude_args}

        if(type(in_channels) is not type(edge_dim)):
            raise ValueError("in_channels and edge_dim must be the same type - \
                             either both are ints to represent scalars or both are tuples to represent (scalar, vector)")

        self.base_conv = base_conv
        self.is_scalar_data = isinstance(in_channels, int)
        vector_models = ['lbamodel', 'pocketminer', 'cpdmodel']

        if self.is_scalar_data and (self.base_conv in vector_models):
            raise ValueError(f"Cannot use a vector model {self.base_conv} with scalar input data {self.in_channels} (either define the input as (n, 0) or include vector data)")
        elif not self.is_scalar_data and (self.base_conv not in vector_models):
            raise ValueError(f"Cannot use a scalar model {self.base_conv} with vector input data {self.in_channels} (either define the input as n or exclude vector data)")


        model_selector = {
            'lbamodel': VectorProteinGNN_LBAModel,
            'pocketminer': VectorProteinGNN_PocketMiner, 
            'cpdmodel': VectorProteinGNN_CPDModel,
            'gatv2': HomoScalarProteinGNN_GATv2,
            'heat': HeteroScalarProteinGNN_HEAT,
        }

        # Select the appropriate model based on the input
        model_selection = self.base_conv


        # If a model is forced, use that model 
        # We rely on the user to know that they can't pass vector data
        # to a scalar model and vice versa
        self.gnn_model = model_selector[model_selection](**passed_args, **kwargs)


    # Pass-through forward to the base model's forward
    # TODO: allow for managing vector -> scalar conversion, if the model is scalar?
    # low-priority since we're not using scalar data for our analysis
    def forward(self, x, edge_index, ntypes, etypes, eattr=None, batch=None):
        return self.gnn_model(x, edge_index, ntypes, etypes, eattr=eattr, batch=batch)
    

    # Pass-through attribute access to the base model's attributes
    # Note, __getattr__ is only called if the requested attribute is not found in the object's __dict__
    # so this passes through everything that isn't found in the SelectableProteinModel object
    # (but also note nn.Module is special and modifies the __dict__ as well as
    # overloads __getattr__ itself, so we need to call that version with super)
    def __getattr__(self, name):
        try:
            return super(SelectableProteinModelWrapper, self).__getattr__(name)
        except AttributeError:
            return getattr(super(SelectableProteinModelWrapper, self).__getattr__('gnn_model'), name)



class BaseProteinGNN(nn.Module):
    """
    A GNN that creates protein embeddings for downstream use
    where each protein is represented as a single graph
    The **kwargs in models below are the ones that are explicitly defined in this __init__

    """
    def __init__(self, 
                 in_channels, 
                 edge_dim, 
                 num_ntypes, 
                 num_etypes, 
                 ntype_emb_dim,
                 etype_emb_dim, 
                 num_convs=1,
                 hidden_channels=None,
                 out_channels=8, 
                 dropout_rate=0.2,
                 activation='relu'):
        """
        Initialize the GNN model
        """

        super(BaseProteinGNN, self).__init__()
        self.in_channels = in_channels
        self.edge_dim = edge_dim
        self.num_ntypes = num_ntypes
        self.num_etypes = num_etypes
        self.ntype_emb_dim = ntype_emb_dim
        self.etype_emb_dim = etype_emb_dim

        self.num_convs = num_convs
        self.hidden_channels = hidden_channels if hidden_channels is not None else out_channels
        self.out_channels = out_channels
        self.dropout_rate = dropout_rate

        # Embed using nn.Embedding or one-hot encode based on the provided dimensions
        if(self.ntype_emb_dim is not None):
            self.ntype_embedding = nn.Embedding(self.num_ntypes, self.ntype_emb_dim)
        else:
            self.ntype_embedding = partial(nn.functional.one_hot, num_classes=self.num_ntypes)
            self.ntype_emb_dim = self.num_ntypes
        
        if(self.etype_emb_dim is not None):
            self.etype_embedding = nn.Embedding(self.num_etypes, self.etype_emb_dim)
        else:
            self.etype_embedding = partial(nn.functional.one_hot, num_classes=self.num_etypes)
            self.etype_emb_dim = self.num_etypes

        self.activation = _select_activation(activation)
        self.dropout = nn.Dropout(self.dropout_rate)

    
    def _embed_types_and_cat(self, x, eattr, ntypes, etypes):
        """
        Embed the node and edge types, then concatenate them
        to their respective features
        """
        # Embed the node types and concatenate to the node features
        ntype_embs = self.ntype_embedding(ntypes)
        x = torch.cat([ntype_embs, x], dim=-1)

        # Embed the edge types and concatenate to the edge features
        etype_embs = self.etype_embedding(etypes)
        eattr = torch.cat([etype_embs, eattr], dim=-1)

        return x, eattr


class HomoScalarProteinGNN_GATv2(BaseProteinGNN):
    """
    The homogeneous, scalar form of the GNN model for proteins
    Assumes all features are scalars and does homogeneous message passing
    with GATv2Conv layers (embedding the node/edge types)
    For info on **kwargs, see BaseProteinGNN
    """

    def __init__(self, 
                 aggr='sum',
                 concat=False,
                 heads=2,
                 conv_dropout=0.0,
                 conv_neg_slope=0.2,
                 **kwargs):
        """
        Initialize the GNN model
        """

        super(HomoScalarProteinGNN_GATv2, self).__init__(**kwargs)
        self.aggr = aggr
        self.concat = concat
        self.heads = heads
        self.conv_dropout = conv_dropout
        self.conv_neg_slope = conv_neg_slope

        # Define graph convolutional layers
        conv_list = []

        conv_layer = partial(pyg.nn.GATv2Conv, 
                            edge_dim=self.edge_dim + self.etype_emb_dim,
                            heads=self.heads,
                            dropout=self.conv_dropout,
                            negative_slope=self.conv_neg_slope,
                            concat=self.concat,
                            aggr=self.aggr)
        
        node_channel_dims = [self.in_channels + self.ntype_emb_dim] + [self.hidden_channels] * (self.num_convs-1) + [self.out_channels]

        for i in torch.arange(self.num_convs):
            conv_list.append(
                conv_layer(in_channels=node_channel_dims[i], 
                           out_channels=node_channel_dims[i+1])
            )

        self.conv_list = nn.ModuleList(conv_list)


    def forward(self, x, edge_index, ntypes, etypes, eattr=None, batch=None):

        # Embed node and edge types and concatenate to x and eattr, respectively
        x, eattr = self._embed_types_and_cat(x, eattr, ntypes, etypes)

        # Perform the convolutions
        for conv in self.conv_list[:-1]:
            x = conv(x, edge_index, edge_attr=eattr)
            x = self.activation(x)
            x = self.dropout(x)

        x = self.conv_list[-1](x, edge_index, edge_attr=eattr)
        x = self.activation(x)

        return x
    


class HeteroScalarProteinGNN_HEAT(BaseProteinGNN):
    """
    The heterogeneous, scalar form of the GNN model for proteins
    Assumes all features are scalars and does heterogeneous message passing
    with HEATConv layers (embedding the node/edge types)
    For info on **kwargs, see BaseProteinGNN
    """

    def __init__(self, 
                 eattr_emb_dim,
                 aggr='sum',
                 concat=True,
                 heads=2,
                 conv_dropout=0.0,
                 conv_neg_slope=0.2,
                 **kwargs):
        """
        Initialize the GNN model
        """

        super(HeteroScalarProteinGNN_HEAT, self).__init__(**kwargs)
        self.eattr_emb_dim = eattr_emb_dim
        self.aggr = aggr
        self.concat = concat
        self.heads = heads
        self.conv_dropout = conv_dropout
        self.conv_neg_slope = conv_neg_slope

        # Define graph convolutional layers
        conv_list = []

        conv_layer = partial(pyg.nn.HEATConv, 
                            num_node_types=self.num_ntypes, 
                            num_edge_types=self.num_etypes,
                            edge_type_emb_dim=self.etype_emb_dim,
                            edge_dim=self.edge_dim,
                            edge_attr_emb_dim=self.eattr_emb_dim,
                            heads=self.heads,
                            dropout=self.conv_dropout,
                            negative_slope=self.conv_neg_slope,
                            concat=self.concat,
                            aggr=self.aggr)
        
        node_channel_dims = [self.in_channels] + [self.hidden_channels] * (self.num_convs-1) + [self.out_channels]

        for i in torch.arange(self.num_convs):
            conv_list.append(
                conv_layer(in_channels=node_channel_dims[i], 
                           out_channels=node_channel_dims[i+1])
            )

        self.conv_list = nn.ModuleList(conv_list)


    def forward(self, x, edge_index, ntypes, etypes, eattr=None, batch=None):

        for conv in self.conv_list[:-1]:
            x = conv(x, edge_index, ntypes, etypes, edge_attr=eattr)
            x = self.activation(x)
            x = self.dropout(x)

        x = self.conv_list[-1](x, edge_index, ntypes, etypes, edge_attr=eattr)
        x = self.activation(x)

        return x



class VectorProteinGNN_LBAModel(BaseProteinGNN):
    """
    A GVP that creates protein embeddings for downstream use
    where each protein is represented as a single graph with scalar and vector features
    This uses geometric vector perceptron (specifically the GVP-GNN layers from gvp_model)
    Inspired by the LBA model from the original GVP repo
    """

    def __init__(self, 
                 edge_hidden_channels,
                 aggr="mean",
                 **kwargs):
        """
        Initialize the GNN model
        """

        super(VectorProteinGNN_LBAModel, self).__init__(**kwargs)
        self.edge_hidden_channels = edge_hidden_channels

        # Make sure the GVP-GNN model outputs a scalar and a vector
        # If hidden_channels is an int, we assume it's a scalar output with no vectors
        # Same with out_channels
        # For the other arguments, we don't do explicit checking since we assume the user should know how to use them
        if isinstance(self.hidden_channels, int):
            # warnings.warn(f"The SimpleGVP model requires a scalar and vector hidden dimension (or explicitly defining vector dimension as 0). Setting hidden_channels (given as {self.hidden_channels}) to (hidden_channels, 0)")
            self.hidden_channels = (self.hidden_channels, 0)

        if isinstance(self.out_channels, int):
            # warnings.warn(f"The SimpleGVP model requires a scalar and vector output dimension (or explicitly defining vector dimension as 0). Setting out_channels (given as {self.out_channels}) to (out_channels, 0)")
            self.out_channels = (self.out_channels, 0)


        # Define some input processing layers using base GVPs (not convs)
        node_channels_with_embed = (self.in_channels[0] + self.ntype_emb_dim, self.in_channels[1])
        edge_channels_with_embed = (self.edge_dim[0] + self.etype_emb_dim, self.edge_dim[1])

        self.gvp_node = nn.Sequential(
            gvp.GVP(node_channels_with_embed, self.hidden_channels, 
                    activations=(None, None), vector_gate=True),
            gvp.LayerNorm(self.hidden_channels),
        )

        self.gvp_edge = nn.Sequential(
            gvp.GVP(edge_channels_with_embed, self.edge_hidden_channels, 
                    activations=(None, None), vector_gate=True),
            gvp.LayerNorm(self.edge_hidden_channels),
        )


        # Define graph convolutional layers
        conv_list = []

        self.gvp_relu = nn.ReLU()

        for _ in range(self.num_convs):
            conv_list.append(
                gvp.GVPConvLayer(self.hidden_channels, self.edge_hidden_channels, 
                                 drop_rate=self.dropout_rate,
                                 activations=(self.gvp_relu, None),
                                 vector_gate=True,
                                 aggr=aggr)
            )
        
        self.conv_list = nn.ModuleList(conv_list)

        # GVP to produce a value for each residue
        self.gvp_norm_before_scalar = gvp.LayerNorm(self.hidden_channels)
        self.gvp_to_scalar = gvp.GVP(self.hidden_channels, self.out_channels, 
                                     activations=(self.gvp_relu, None),
                                     vector_gate=True)


    def forward(self, x, edge_index, ntypes, etypes, eattr=None, batch=None):
        # (NOTE TO SELF: GVP is indeed rotation invariant for scalar features - we tested)

        x_s, x_v = x
        eattr_s, eattr_v = eattr if eattr is not None else (torch.empty([edge_index.shape[1], 0]), torch.empty([edge_index.shape[1], 0, 3]))

        # Embed node and edge types and concatenate to the scalar parts of x and eattr, respectively
        x_s, eattr_s = self._embed_types_and_cat(x_s, eattr_s, ntypes, etypes)

        # Combine the scalar and vector parts of the input features back together
        x = (x_s, x_v)
        eattr = (eattr_s, eattr_v)

        # Process the input features with the GVPs (and LayerNorms)
        x = self.gvp_node(x)
        eattr = self.gvp_edge(eattr)

        # Perform the convolutions
        for conv in self.conv_list:
            x = conv(x, edge_index, edge_attr=eattr)

        # Produce residue-level embeddings
        # will return only a scalar for each residue at the same
        # dimensionality as the hidden scalar dimension
        x = self.gvp_norm_before_scalar(x)
        x = self.gvp_to_scalar(x)

        return x
    


class VectorProteinGNN_PocketMiner(BaseProteinGNN):
    """
    A GVP that creates protein embeddings for downstream use
    where each protein is represented as a single graph with scalar and vector features
    This uses geometric vector perceptron (specifically the GVP-GNN layers from gvp_model)
    Inspired by the PocketMiner model that uses GVPs to predict binding pockets
    """

    def __init__(self, 
                 edge_hidden_channels,
                 initial_node_project_channels,
                 initial_edge_project_channels,
                 **kwargs):
        """
        Initialize the GNN model
        """

        super(VectorProteinGNN_PocketMiner, self).__init__(**kwargs)
        self.initial_node_project_channels = initial_node_project_channels
        self.initial_edge_project_channels = initial_edge_project_channels
        self.edge_hidden_channels = edge_hidden_channels

        # Make sure the GVP-GNN model outputs a scalar and a vector
        # If hidden_channels is an int, we assume it's a scalar output with no vectors
        # Same with out_channels
        # For the other arguments, we don't do explicit checking since we assume the user should know how to use them
        if isinstance(self.hidden_channels, int):
            # warnings.warn(f"The GVPProteinGNN model requires a scalar and vector hidden dimension (or explicitly defining vector dimension as 0). Setting hidden_channels (given as {self.hidden_channels}) to (hidden_channels, 0)")
            self.hidden_channels = (self.hidden_channels, 0)

        if isinstance(self.out_channels, int):
            # warnings.warn(f"The GVPProteinGNN model requires a scalar and vector output dimension (or explicitly defining vector dimension as 0). Setting out_channels (given as {self.out_channels}) to (out_channels, 0)")
            self.out_channels = (self.out_channels, 0)

        # Initial structural feature projection layers
        # NOTE: These are present in the original GVP paper and the Tensorflow code
        # but for some reason are absent in the GVP PyTorch implementation of MQAModel
        # Also note we use the full GVP layernorm, not just a scalar-wise layernorm
        # TODO: replace the GVP layernorm with layernorms for only the scalar parts
        if(self.initial_node_project_channels is None):
            self.gvp_node_structural_proj = nn.Identity()
            self.initial_node_project_channels = self.in_channels
        else:
            self.gvp_node_structural_proj = nn.Sequential(
                gvp.GVP(self.in_channels, self.initial_node_project_channels, activations=(None, None)),
                gvp.LayerNorm(self.initial_node_project_channels)
            )

        if(self.initial_edge_project_channels is None):
            self.gvp_edge_structural_proj = nn.Identity()
            self.initial_edge_project_channels = self.edge_dim
        else:
            self.gvp_edge_structural_proj = nn.Sequential(
                gvp.GVP(self.edge_dim, self.initial_edge_project_channels, activations=(None, None)),
                gvp.LayerNorm(self.initial_edge_project_channels)
            )


        # Define some input processing layers using base GVPs (not convs)
        node_channels_with_embed = (self.initial_node_project_channels[0] + self.ntype_emb_dim, self.initial_node_project_channels[1])
        edge_channels_with_embed = (self.initial_edge_project_channels[0] + self.etype_emb_dim, self.initial_edge_project_channels[1])

        self.gvp_node = nn.Sequential(
            gvp.LayerNorm(node_channels_with_embed),
            gvp.GVP(node_channels_with_embed, self.hidden_channels, activations=(None, None)),
        )

        self.gvp_edge = nn.Sequential(
            gvp.LayerNorm(edge_channels_with_embed),
            gvp.GVP(edge_channels_with_embed, self.edge_hidden_channels, activations=(None, None)),
        )


        # Define graph convolutional layers
        conv_list = []

        for _ in range(self.num_convs):
            conv_list.append(
                gvp.GVPConvLayer(self.hidden_channels, self.edge_hidden_channels, 
                                 drop_rate=self.dropout_rate,
                                 activations=(None, None))
            )
        
        self.conv_list = nn.ModuleList(conv_list)

        # GVP to produce a value for each residue
        self.gvp_norm_before_scalar = gvp.LayerNorm(self.hidden_channels)
        self.gvp_to_scalar = gvp.GVP(self.hidden_channels, self.out_channels, 
                                     activations=(None, None))


    def forward(self, x, edge_index, ntypes, etypes, eattr=None, batch=None):
        # (NOTE TO SELF: GVP is indeed rotation invariant for scalar features - we tested)

        x_s, x_v = x
        eattr_s, eattr_v = eattr if eattr is not None else (torch.empty([edge_index.shape[1], 0]), torch.empty([edge_index.shape[1], 0, 3]))

        # Project the structural features (node and edge) to the initial hidden dimensions
        x_s, x_v = self.gvp_node_structural_proj((x_s, x_v))
        eattr_s, eattr_v = self.gvp_edge_structural_proj((eattr_s, eattr_v))

        # Embed node and edge types and concatenate to the scalar parts of x and eattr, respectively
        x_s, eattr_s = self._embed_types_and_cat(x_s, eattr_s, ntypes, etypes)

        # Combine the scalar and vector parts of the input features back together
        x = (x_s, x_v)
        eattr = (eattr_s, eattr_v)

        # Process the input features with the GVPs (and LayerNorms)
        x = self.gvp_node(x)
        eattr = self.gvp_edge(eattr)

        # Perform the convolutions
        for conv in self.conv_list:
            x = conv(x, edge_index, edge_attr=eattr)

        # Produce residue-level embeddings
        # will return only scalars for each residue at the same
        # dimensionality as the hidden scalar dimension
        x = self.gvp_norm_before_scalar(x)
        x = self.gvp_to_scalar(x)

        return x



class VectorProteinGNN_CPDModel(BaseProteinGNN):
    """
    A GVP that creates protein embeddings for downstream use
    where each protein is represented as a single graph with scalar and vector features
    This uses geometric vector perceptron (specifically the GVP-GNN layers from gvp_model)
    Inspired by the CPD model from the original GVP repo
    """

    def __init__(self, 
                 edge_hidden_channels,
                 **kwargs):
        """
        Initialize the GNN model
        """

        super(VectorProteinGNN_CPDModel, self).__init__(**kwargs)
        self.edge_hidden_channels = edge_hidden_channels

        # Make sure the GVP-GNN model outputs a scalar and a vector
        # If hidden_channels is an int, we assume it's a scalar output with no vectors
        # Same with out_channels
        # For the other arguments, we don't do explicit checking since we assume the user should know how to use them
        if isinstance(self.hidden_channels, int):
            # warnings.warn(f"The SimpleGVP model requires a scalar and vector hidden dimension (or explicitly defining vector dimension as 0). Setting hidden_channels (given as {self.hidden_channels}) to (hidden_channels, 0)")
            self.hidden_channels = (self.hidden_channels, 0)

        if isinstance(self.out_channels, int):
            # warnings.warn(f"The SimpleGVP model requires a scalar and vector output dimension (or explicitly defining vector dimension as 0). Setting out_channels (given as {self.out_channels}) to (out_channels, 0)")
            self.out_channels = (self.out_channels, 0)

        
        edge_in_dim_with_embed = (self.edge_dim[0] + self.etype_emb_dim, self.edge_dim[1])


        self.W_v = nn.Sequential(
            gvp.GVP(self.in_channels, self.hidden_channels, activations=(None, None)),
            gvp.LayerNorm(self.hidden_channels)
        )
        self.W_e = nn.Sequential(
            gvp.GVP(edge_in_dim_with_embed, self.edge_hidden_channels, activations=(None, None)),
            gvp.LayerNorm(self.edge_hidden_channels)
        )
        
        self.encoder_layers = nn.ModuleList(
                gvp.GVPConvLayer(self.hidden_channels, self.edge_hidden_channels, drop_rate=self.dropout_rate) 
            for _ in range(self.num_convs))
        
        # Use embedding from super class for this
        edge_h_dim = (self.edge_hidden_channels[0] + self.ntype_emb_dim, self.edge_hidden_channels[1])
      
        self.decoder_layers = nn.ModuleList(
                gvp.GVPConvLayer(self.hidden_channels, edge_h_dim, 
                             drop_rate=self.dropout_rate, autoregressive=True) 
            for _ in range(self.num_convs))
        
        self.W_out = gvp.GVP(self.hidden_channels, self.out_channels, activations=(None, None))



    def forward(self, x, edge_index, ntypes, etypes, eattr=None, batch=None):
        # (NOTE TO SELF: GVP is indeed rotation invariant for scalar features - we tested)

        eattr_s, eattr_v = eattr if eattr is not None else (torch.empty([edge_index.shape[1], 0]), torch.empty([edge_index.shape[1], 0, 3]))

        # Add edge type embeddings to the edge features 
        # (NOT doing this yet for node type embeddings)
        # Those are added later for "autoregressive" decoding
        etype_embs = self.etype_embedding(etypes)
        eattr_s = torch.cat([etype_embs, eattr_s], dim=-1)
        eattr = (eattr_s, eattr_v)

        x = self.W_v(x)
        eattr = self.W_e(eattr)
        
        for layer in self.encoder_layers:
            x = layer(x, edge_index, eattr)
        
        encoder_embeddings = x
        
        # Add node type embeddings to the edge attributes for autoregressive decoding
        h_S = self.ntype_embedding(ntypes)
        h_S = h_S[edge_index[0]]
        h_S[edge_index[0] >= edge_index[1]] = 0

        eattr = (torch.cat([eattr[0], h_S], dim=-1), eattr[1])
        
        for layer in self.decoder_layers:
            x = layer(x, edge_index, eattr, autoregressive_x = encoder_embeddings)
        
        x = self.W_out(x)

        return x