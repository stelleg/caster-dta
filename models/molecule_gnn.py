import torch_geometric as pyg
import torch.nn as nn
import torch

from functools import partial
import warnings

from models.model_utils import _select_activation

import ipdb


class SelectableMoleculeModelWrapper(nn.Module):
    """
    A GNN that can switch between the different types of molecule GNNs
    based on user input and arguments (homogeneous vs heterogeneous only)
    One can bypass this entirely by passing in a specific model to this class
    or by just skipping this class altogether and using a specific model directly
    """
    def __init__(self,
                 base_conv, # 'gatv2', 'gine', 'attentivefp', 'heat' (last is heterogeneous)
                 force_model=None,
                 **kwargs):
        
        super(SelectableMoleculeModelWrapper, self).__init__()

        passed_args = locals().copy()
        exclude_args = ['self', 'passed_args', '__class__', 'kwargs', 
                        'force_homogeneous', 'force_model', 'base_conv']
        passed_args = {k: v for k, v in passed_args.items() if k not in exclude_args}

        self.base_conv = base_conv.lower()

        model_selector = {
            'gatv2': HomoMoleculeGNN_GAT,
            'gine': HomoMoleculeGNN_GINE,
            'gin': HomoMoleculeGNN_GIN,
            'gps': HomoMoleculeGNN_GPS,
            'pna': HomoMoleculeGNN_PNA,
            'attentivefp': HomoMoleculeGNN_AttentiveFP,
            'heat': HeteroMoleculeGNN_HEAT,
        }

        # Select the appropriate model based on the input
        model_selection = self.base_conv


        # If a model is forced, use that model 
        # We rely on the user to know that they can't pass vector data
        # to a scalar model and vice versa
        if(force_model is not None):
            self.gnn_model = force_model(**passed_args, **kwargs)
        else:
            self.gnn_model = model_selector[model_selection](**passed_args, **kwargs)


    # Pass-through forward to the base model's forward
    def forward(self, x, edge_index, ntypes, etypes, eattr=None, batch=None):
        return self.gnn_model(x, edge_index, ntypes, etypes, eattr=eattr, batch=batch)
    
    # Pass-through attribute access to the base model's attributes
    # Note, __getattr__ is only called if the requested attribute is not found in the object's __dict__
    # so this passes through everything that isn't found in the SelectableProteinModel object
    # (but also note nn.Module is special and modifies the __dict__ as well as
    # overloads __getattr__ itself, so we need to call that version with super)
    def __getattr__(self, name):
        try:
            return super(SelectableMoleculeModelWrapper, self).__getattr__(name)
        except AttributeError:
            return getattr(super(SelectableMoleculeModelWrapper, self).__getattr__('gnn_model'), name)


class BaseMoleculeGNN(nn.Module):
    """
    A GNN that creates molecule embeddings for downstream use
    where each molecule is represented as a single graph
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
                 activation='relu',
                 aggr='sum'):
        """
        Initialize the GNN model
        """

        super(BaseMoleculeGNN, self).__init__()
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
        self.aggr = aggr

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


class HomoMoleculeGNN_GAT(BaseMoleculeGNN):
    """
    A GNN that creates molecule embeddings for downstream use
    where each molecule is represented as a single graph
    using GATv2Conv layers (and a homogeneous graph representation with concatenated embeddings)
    """

    def __init__(self,
                 concat=False,
                 heads=2,
                 conv_dropout=0.0,
                 conv_neg_slope=0.2,
                 **kwargs):
        """
        Initialize the GNN model
        """

        super(HomoMoleculeGNN_GAT, self).__init__(**kwargs)
        self.concat = concat
        self.heads = heads
        self.conv_dropout = conv_dropout
        self.conv_neg_slope = conv_neg_slope

        if self.num_convs == 1:
            warnings.warn(f"The HomoMoleculeGNN_GAT model will not use the hidden_channels parameter for a single convolution")

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
    

class HomoMoleculeGNN_GINE(BaseMoleculeGNN):
    """
    A GNN that creates molecule embeddings for downstream use
    where each molecule is represented as a single graph
    using GINEConv layers (and a homogeneous graph representation with concatenated embeddings)
    Note that GINEConv needs an MLP that projects the node embeddings, so that is included in the conv list
    as per the standard from the GIN in PyTorch Geometric
    """

    def __init__(self,
                 act_first=False,
                 gin_norm=None,
                 gin_norm_kwargs=None,
                 gin_trainable_eps=True,
                 **kwargs):
        """
        Initialize the GNN model
        """

        super(HomoMoleculeGNN_GINE, self).__init__(**kwargs)
        self.act_first = act_first
        self.gin_norm = gin_norm
        self.gin_norm_kwargs = gin_norm_kwargs
        self.gin_trainable_eps = gin_trainable_eps

        if self.num_convs == 1:
            warnings.warn(f"The HomoMoleculeGNN_GINE model will not use the hidden_channels parameter for a single convolution")

        # Define graph convolutional layers
        conv_list = []

        conv_layer = partial(self._init_conv, 
                            edge_dim=self.edge_dim + self.etype_emb_dim,
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
    

    def _init_conv(self, in_channels, out_channels, edge_dim, **kwargs):
        # Use the built-in PyG MLP as inspired by 
        # https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/basic_gnn.html#GIN
        mlp = pyg.nn.MLP([in_channels, out_channels, out_channels],
                         act=self.activation,
                         act_first=self.act_first,
                         norm=self.gin_norm,
                         norm_kwargs=self.gin_norm_kwargs)

        return pyg.nn.GINEConv(mlp, train_eps=self.gin_trainable_eps, edge_dim=edge_dim, **kwargs)


class HomoMoleculeGNN_GIN(BaseMoleculeGNN):
    """
    A GNN that creates molecule embeddings for downstream use
    where each molecule is represented as a single graph
    using GINConv layers (and a homogeneous graph representation with concatenated embeddings)
    Note that GINConv needs an MLP that projects the node embeddings, so that is included in the conv list
    as per the standard from the GIN in PyTorch Geometric
    NOTE: GIN does not accept edge attributes (unlike GINE), so edge attributes passed will be ignored
    """

    def __init__(self,
                 act_first=False,
                 gin_norm=None,
                 gin_norm_kwargs=None,
                 gin_trainable_eps=True,
                 **kwargs):
        """
        Initialize the GNN model
        """

        super(HomoMoleculeGNN_GIN, self).__init__(**kwargs)
        self.act_first = act_first
        self.gin_norm = gin_norm
        self.gin_norm_kwargs = gin_norm_kwargs
        self.gin_trainable_eps = gin_trainable_eps

        if self.num_convs == 1:
            warnings.warn(f"The HomoMoleculeGNN_GINE model will not use the hidden_channels parameter for a single convolution")

        # Define graph convolutional layers
        conv_list = []

        conv_layer = partial(self._init_conv, 
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
            x = conv(x, edge_index)
            x = self.activation(x)
            x = self.dropout(x)

        x = self.conv_list[-1](x, edge_index)
        x = self.activation(x)

        return x
    

    def _init_conv(self, in_channels, out_channels, **kwargs):
        # Use the built-in PyG MLP as inspired by 
        # https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/basic_gnn.html#GIN
        mlp = pyg.nn.MLP([in_channels, out_channels, out_channels],
                         act=self.activation,
                         act_first=self.act_first,
                         norm=self.gin_norm,
                         norm_kwargs=self.gin_norm_kwargs)

        return pyg.nn.GINConv(mlp, train_eps=self.gin_trainable_eps, **kwargs)


class HomoMoleculeGNN_AttentiveFP(BaseMoleculeGNN):
    """
    A GNN that creates molecule embeddings for downstream use
    where each molecule is represented as a single graph
    using the AttentiveFP approach (and a homogeneous graph representation with concatenated embeddings)
    Largely inspired by https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/attentive_fp.html#AttentiveFP
    (but mostly rewritten, though we do use the GATEConv ourselves,
    since we want atom-level embeddings, not molecule-level)
    """

    def __init__(self,
                 **kwargs):
        """
        Initialize the GNN model
        """

        super(HomoMoleculeGNN_AttentiveFP, self).__init__(**kwargs)

        # Projection of node features (always)
        self.lin1 = nn.Linear(self.in_channels + self.ntype_emb_dim, self.hidden_channels)

        # Define graph convolutional layers
        conv_list = []
        gru_list = []
        
        conv_layer = partial(pyg.nn.GATConv, 
                            dropout=self.dropout_rate,
                            add_self_loops=False, negative_slope=0.01)
        
        node_channel_dims = [self.hidden_channels] * (self.num_convs+1)

        for i in torch.arange(self.num_convs):
            layer_in_dim = node_channel_dims[i]
            layer_out_dim = node_channel_dims[i+1]

            # GATEConv is always first, and only one GATEConv total
            # (so we don't use the partial trick here for it)
            if(i == 0):
                conv_list.append(pyg.nn.models.attentive_fp.GATEConv(
                    layer_in_dim, layer_out_dim, 
                    self.edge_dim + self.etype_emb_dim, 
                    self.dropout_rate)
                )
            else:
                conv_list.append(
                    conv_layer(in_channels=layer_in_dim, 
                            out_channels=layer_out_dim)
                )

            # One GRU per layer
            gru_list.append(nn.GRUCell(layer_out_dim, layer_in_dim))

        self.conv_list = nn.ModuleList(conv_list)
        self.gru_list = nn.ModuleList(gru_list)

        # Last linear layer to project to output
        self.lin2 = nn.Linear(self.hidden_channels, self.out_channels)

        # ELU activation used after the graph convolutions
        self.conv_elu = nn.ELU()

        # Basic ReLU used after the GRU passes
        self.gru_relu = nn.ReLU()

        # Leaky ReLU used after the initial linear layer 
        # (we use our own in self.activation for the final output)
        self.leaky_relu = nn.LeakyReLU(0.01)


    def forward(self, x, edge_index, ntypes, etypes, eattr=None, batch=None):

        # Embed node and edge types and concatenate to x and eattr, respectively
        x, eattr = self._embed_types_and_cat(x, eattr, ntypes, etypes)

        x = self.leaky_relu(self.lin1(x))

        # Initial GATEConv convolution and GRU pass
        x_h = self.conv_list[0](x, edge_index, edge_attr=eattr)
        x_h = self.conv_elu(x_h)
        x_h = self.dropout(x_h)
        x = self.gru_relu(self.gru_list[0](x_h, x))

        # Perform the remaining convolutions (all GATConv now)
        # Note that AttentiveFP uses ELU activations
        for conv, gru in zip(self.conv_list[1:], self.gru_list[1:]):
            x_h = conv(x, edge_index)
            x_h = self.conv_elu(x_h)
            x_h = self.dropout(x_h)
            x = self.gru_relu(gru(x_h, x))

        # Below is custom on top of the AttentiveFP atom embeddings
        # to project to the final output dimensionality
        # TODO: Consider making this a convolution instead of a linear layer
        x = self.lin2(x)
        x = self.activation(x)

        return x
    

class HomoMoleculeGNN_GPS(BaseMoleculeGNN):
    """
    A GNN that creates molecule embeddings for downstream use
    where each molecule is represented as a single graph
    using the GPSConv approach (and a homogeneous graph representation with concatenated embeddings)
    """

    def __init__(self, 
                 pe_dim=8, 
                 attn_type='multihead', 
                 attn_kwargs={'dropout': 0.5},
                 **kwargs):
        
        super(HomoMoleculeGNN_GPS, self).__init__(**kwargs)

        self.pe_dim = pe_dim
        self.pe_norm = nn.BatchNorm1d(20)
        self.pe_lin = nn.Linear(20, pe_dim)

        node_channel_dims = [self.in_channels + self.ntype_emb_dim + self.pe_dim] + [self.hidden_channels] * (self.num_convs-1) + [self.out_channels]

        self.convs = nn.ModuleList()
        for i in range(self.num_convs):
            ffnn = nn.Sequential(
                nn.Linear(node_channel_dims[i], node_channel_dims[i+1]),
                nn.ReLU(),
                nn.Linear(node_channel_dims[i+1], node_channel_dims[i+1]),
            )
            conv = pyg.nn.GPSConv(node_channel_dims[i+1], 
                                  pyg.nn.GINEConv(ffnn, edge_dim=self.edge_dim + self.etype_emb_dim), 
                                    heads=4, attn_type=attn_type, attn_kwargs=attn_kwargs)
            self.convs.append(conv)


    def forward(self, x, edge_index, ntypes, etypes, eattr=None, batch=None):

        x, eattr = self._embed_types_and_cat(x, eattr, ntypes, etypes)
        pe = self._calc_pos_encs(x, edge_index, eattr, batch)

        pe = self.pe_norm(pe)
        pe = self.pe_lin(pe)

        x = torch.cat([x, pe], dim=-1)

        for conv in self.convs:
            x = conv(x, edge_index, batch, edge_attr=eattr)
        
        return x
    

    def _calc_pos_encs(self, x, edge_index, eattr, batch):
        """
        Calculate positional encodings for the batch
        Note, this is slow and should probably be done in advance rather than
        in the forward pass (but for testing, we'll do it here)
        """
        n_walks = 20

        row, col = edge_index
        N = x.shape[0]

        value = torch.ones(row.shape[0], device=row.device)
        value = pyg.utils.scatter(value, row, dim_size=N, reduce='sum').clamp(min=1)[row]
        value = 1.0 / value

        adj = pyg.utils.to_torch_csr_tensor(edge_index, value, size=(N, N))

        out = adj
        pe_list = [pyg.utils.get_self_loop_attr(*pyg.utils.to_edge_index(out), num_nodes=N)]
        for _ in range(n_walks - 1):
            out = out @ adj
            pe_list.append(pyg.utils.get_self_loop_attr(*pyg.utils.to_edge_index(out), N))
        pe = torch.stack(pe_list, dim=-1)

        return pe


class HomoMoleculeGNN_PNA(BaseMoleculeGNN):
    """
    A GNN that creates molecule embeddings for downstream use
    where each molecule is represented as a single graph
    using PNAConv layers (and a homogeneous graph representation with concatenated embeddings)
    with most of the code inspired by https://github.com/pyg-team/pytorch_geometric/blob/master/examples/pna.py
    """

    def __init__(self,
                 degree_hist,
                 aggregators=['mean', 'min', 'max', 'std'],
                 scalers=['identity', 'amplification', 'attenuation'],
                 towers=4,
                 **kwargs):
        """
        Initialize the GNN model
        """

        super(HomoMoleculeGNN_PNA, self).__init__(**kwargs)
        self.degree_hist = degree_hist
        self.aggregators = aggregators
        self.scalers = scalers
        self.towers = towers

        if self.num_convs == 1:
            warnings.warn(f"The HomoMoleculeGNN_PNA model will not use the hidden_channels parameter for a single convolution")

        # Define graph convolutional layers
        conv_list = []

        conv_layer = partial(pyg.nn.PNAConv, 
                            edge_dim=self.edge_dim + self.etype_emb_dim,
                            aggregators=self.aggregators,
                            scalers=self.scalers,
                            deg=self.degree_hist,
                            towers=self.towers)
        
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


class HeteroMoleculeGNN_HEAT(BaseMoleculeGNN):
    """
    A GNN that creates molecule embeddings for downstream use
    where each molecule is represented as a single graph
    using HEATConv layers (and a homogeneous graph representation with passed node and edge types)
    """

    def __init__(self, 
                 eattr_emb_dim,
                 concat=True,
                 heads=2,
                 conv_dropout=0.0,
                 conv_neg_slope=0.2,
                 **kwargs):
        """
        Initialize the GNN model
        """

        super(HeteroMoleculeGNN_HEAT, self).__init__(**kwargs)
        self.eattr_emb_dim = eattr_emb_dim
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