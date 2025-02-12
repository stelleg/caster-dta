import torch_geometric as pyg
import torch.nn as nn
import torch

from models.protein_gnn import SelectableProteinModelWrapper as ProteinGNN
from models.molecule_gnn import SelectableMoleculeModelWrapper as MoleculeGNN
from models.model_utils import _select_activation

from functools import partial

import ipdb

# TODO: Rename this to whatever we want to call this in the end

class JointGNN(nn.Module):
    """
    A model that learns embeddings for both proteins and molecules using GNNs
    and then performs cross-attention of residue-atom pairs 
    to get protein-molecule pair embeddings that is then used
    to compute an affinity score for the protein-molecule binding (output)
    """

    def __init__(self, 
                 protein_gnn_kwargs, 
                 molecule_gnn_kwargs,
                 residue_lin_depth,
                 atom_lin_depth,
                 n_attention_heads,
                 attention_dropout,
                 protein_lin_depth, 
                 molecule_lin_depth,
                 pairwise_embedding_dim,
                 out_lin_depth,
                 out_lin_factor=0.5,
                 out_lin_norm_type=None,
                 activation='relu',
                 dropout=0.0,
                 element_pooling='mean',
                 include_residual_stream=True,
                 residual_dim_ff_scale=2,
                 num_cross_attn_layers=1,
                 include_post_pool_layernorm=False):
        """
        Initialize the GNN model
        """
        super(JointGNN, self).__init__()

        self.pairwise_embedding_dim = pairwise_embedding_dim
        self.n_attention_heads = n_attention_heads
        self.attention_dropout = attention_dropout
        self.element_pooling = element_pooling
        self.num_cross_attn_layers = num_cross_attn_layers
        self.include_residual_stream = include_residual_stream
        self.residual_dim_ff_scale = residual_dim_ff_scale
        self.include_post_pool_layernorm = include_post_pool_layernorm
        self.out_lin_factor = out_lin_factor
        self.out_lin_norm_type = out_lin_norm_type

        # Static layers that are applied throughout
        self.activation = _select_activation(activation)
        self.dropout = nn.Dropout(dropout)


        # Individual GNNs for proteins and molecules
        self.protein_gnn = ProteinGNN(**protein_gnn_kwargs)
        self.molecule_gnn = MoleculeGNN(**molecule_gnn_kwargs)

        # Get the output dimensions of the GNNs (assuming the GNNs return scalars)
        # Note that we probably want to make sure the 
        protein_gnn_out = self.protein_gnn.out_channels
        molecule_gnn_out = self.molecule_gnn.out_channels

        # Note that GVP will return a tuple for the output dimension - we'll just take the first element if so
        # and assume that it is returning a tensor when called down below
        protein_gnn_out = protein_gnn_out[0] if isinstance(protein_gnn_out, tuple) else protein_gnn_out
        molecule_gnn_out = molecule_gnn_out[0] if isinstance(molecule_gnn_out, tuple) else molecule_gnn_out


        # Perform some linear layers on the residue and atom embeddings after the GNN and before the cross-attention
        self.residue_lins, self.residue_norms, residue_out_dim = self._make_lins_from_depth(residue_lin_depth, 
                                                                                            protein_gnn_out)
        self.atom_lins, self.atom_norms, atom_out_dim = self._make_lins_from_depth(atom_lin_depth, 
                                                                                   molecule_gnn_out)


        if self.num_cross_attn_layers > 0:
            # Perform cross-attention from proteins to molecules and vice-versa
            # the query will be either the protein or molecule embeddings
            # and the key/value will be the other one of the two options
            cross_attn_kwargs = {
                'embed_dim_1': residue_out_dim,
                'embed_dim_2': atom_out_dim,
                'n_attention_heads': n_attention_heads,
                'attn_dropout': attention_dropout,
                'include_residual_stream': include_residual_stream,
                'dim_feedforward_scale': residual_dim_ff_scale,
                'feedforward_dropout': dropout
            }
            cross_attn_base = partial(CrossAttentionModule, **cross_attn_kwargs)

            self.cross_attn_module = StackedCrossAttentionModule(cross_attn_base, 
                                                                 num_layers=self.num_cross_attn_layers)
        
        else:
            # Just set the cross-attention module to None
            self.cross_attn_module = None
            
        
        if self.include_post_pool_layernorm:
            self.protein_post_pool_norm = nn.LayerNorm(residue_out_dim)
            self.molecule_post_pool_norm = nn.LayerNorm(atom_out_dim)


        # Perform some linear layers on each of the protein and molecule embeddings after the
        # cross-attention block with (or without) the residual stream included
        self.protein_lins, self.protein_norms, protein_out_dim = self._make_lins_from_depth(protein_lin_depth, 
                                                                        residue_out_dim)
        self.molecule_lins, self.molecule_norms, molecule_out_dim = self._make_lins_from_depth(molecule_lin_depth, 
                                                                          atom_out_dim)


        # Linear embedding for the protein-molecule (pm) pair (after concatenation)
        n_concat_embeds = protein_out_dim + molecule_out_dim
        self.pm_embed_lin = nn.Linear(n_concat_embeds, self.pairwise_embedding_dim)


        # Final output layers
        # Will be a series of linear layers that halve the hidden dimension
        # successively based on the number of layers, with the final going to 1        
        self.out_fc_layers, self.out_fc_norms, pairwise_embed_out = self._make_lins_from_depth(out_lin_depth, 
                                                                            self.pairwise_embedding_dim, 
                                                                            scale_factor=self.out_lin_factor,
                                                                            include_norms=self.out_lin_norm_type)

        # Output layer (to regression target)
        self.output_layer = nn.Linear(pairwise_embed_out, 1)


    def forward_with_graphs(self,
                            protein_graph,
                            molecule_graph):
        """
        Forward pass with the graphs directly
        Will just build the data dictionaries and call the forward method
        """
        protein_graph_data, molecule_graph_data = self._graphs_to_dicts(protein_graph, molecule_graph)

        return self.forward(protein_graph_data, molecule_graph_data)
    
    @staticmethod
    def _graphs_to_dicts(protein_graph, molecule_graph):
        protein_graph_data = {
            'x': protein_graph.x,
            'edge_index': protein_graph.edge_index if protein_graph.edge_index is not None else protein_graph.adj_t,
            'ntypes': protein_graph.node_type,
            'etypes': protein_graph.edge_type,
            'eattr': protein_graph.edge_attr,
            'batch': protein_graph.batch
        }

        molecule_graph_data = {
            'x': molecule_graph.x,
            'edge_index': molecule_graph.edge_index if molecule_graph.edge_index is not None else molecule_graph.adj_t,
            'ntypes': molecule_graph.node_type,
            'etypes': molecule_graph.edge_type,
            'eattr': molecule_graph.edge_attr,
            'batch': molecule_graph.batch
        }

        return protein_graph_data, molecule_graph_data

    def forward(self, 
                protein_graph_data = {},
                molecule_graph_data = {}):
        # Both of the graph data above are dicts that should contain
        # `x`, `edge_index`, `ntypes`, `etypes`, `eattr`
        # can also contain `batch`` if applicable
        protein_batch_info = protein_graph_data.get('batch', None)
        molecule_batch_info = molecule_graph_data.get('batch', None)

        # Get the GNN embeddings for the protein and molecule nodes
        # (residues and atoms, respectively)
        residue_embed = self.protein_gnn(**protein_graph_data) # [B, R, D]
        atom_embed = self.molecule_gnn(**molecule_graph_data) # [B, A, D]
        

        # Apply linear layers to the residue-level and atom-level embeddings
        for lin_layer, norm_layer in zip(self.residue_lins, self.residue_norms):
            residue_embed = lin_layer(residue_embed)
            residue_embed = norm_layer(residue_embed)
            residue_embed = self.activation(residue_embed)
            residue_embed = self.dropout(residue_embed)
        
        for lin_layer, norm_layer in zip(self.atom_lins, self.atom_norms):
            atom_embed = lin_layer(atom_embed)
            atom_embed = norm_layer(atom_embed)
            atom_embed = self.activation(atom_embed)
            atom_embed = self.dropout(atom_embed)

        
        # Use to_dense_batch to get the batched embeddings across the graphs
        # B = batch dimension, 
        # R = number of residues (max in batch), 
        # A = number of atoms (max in batch), 
        # D = embedding dimension
        residue_embed, residue_mask = pyg.utils.to_dense_batch(residue_embed, batch=protein_batch_info) # [B, R, D]
        atom_embed, atom_mask = pyg.utils.to_dense_batch(atom_embed, batch=molecule_batch_info) # [B, A, D]


        # By default, we'll do cross-attention and global mean pool
        # to get protein and molecule embeddings
        # This will return the embeddings for the residues and proteins
        if self.num_cross_attn_layers > 0:
            residue_embed, atom_embed, attn_weights = self.cross_attn_module(residue_embed, atom_embed, 
                                                                             residue_mask, atom_mask)
        else:
            attn_weights = None
        

        # Perform element pooling to create protein/molecule embeddings
        if self.element_pooling == 'mean':
            # Take the mean across tokens as a representation of the protein/molecule
            # We need to take the masked mean, not just the mean
            # as there are masked elements across the batches that will affect the mean
            protein_embed = (residue_embed * residue_mask.unsqueeze(-1)).sum(dim=1) / residue_mask.sum(dim=1, keepdim=True)
            molecule_embed = (atom_embed * atom_mask.unsqueeze(-1)).sum(dim=1) / atom_mask.sum(dim=1, keepdim=True)
        
        elif self.element_pooling == 'max':
            # We'll add a large negative value to all masked elements so they aren't selected
            # This is a bit of a hack, but it should work to ignore masked elements entirely
            # (unless the values are all that large, but in that case something is likely wrong)
            residue_offset = ~residue_mask.unsqueeze(-1) * 1.0e10
            atom_offset = ~atom_mask.unsqueeze(-1) * 1.0e10
            # Take the max across tokens as a representation of the protein/molecule
            protein_embed = (residue_embed-residue_offset).max(dim=1).values
            molecule_embed = (atom_embed-atom_offset).max(dim=1).values

        elif self.element_pooling == 'sum':
            # Take the sum across tokens as a representation of the protein/molecule
            protein_embed = (residue_embed * residue_mask.unsqueeze(-1)).sum(dim=1)
            molecule_embed = (atom_embed * atom_mask.unsqueeze(-1)).sum(dim=1)

        
        if self.include_post_pool_layernorm:
            protein_embed = self.protein_post_pool_norm(protein_embed)
            molecule_embed = self.molecule_post_pool_norm(molecule_embed)


        protein_embed = self.activation(protein_embed)
        protein_embed = self.dropout(protein_embed)

        molecule_embed = self.activation(molecule_embed)
        molecule_embed = self.dropout(molecule_embed)


        # Apply the linear layers to the protein and molecule embeddings
        for lin_layer, norm_layer in zip(self.protein_lins, self.protein_norms):
            protein_embed = lin_layer(protein_embed)
            protein_embed = norm_layer(protein_embed)
            protein_embed = self.activation(protein_embed)
            protein_embed = self.dropout(protein_embed)

        for lin_layer, norm_layer in zip(self.molecule_lins, self.molecule_norms):
            molecule_embed = lin_layer(molecule_embed)
            molecule_embed = norm_layer(molecule_embed)
            molecule_embed = self.activation(molecule_embed)
            molecule_embed = self.dropout(molecule_embed)


        # Concatenate the protein and molecule embeddings
        # and apply a single embedding layer
        x = torch.cat([protein_embed, molecule_embed], dim=-1)
        x = self.pm_embed_lin(x)
        x = self.activation(x)
        x = self.dropout(x)


        # Apply the final output layers
        for out_fc_layer, out_fc_norm in zip(self.out_fc_layers, self.out_fc_norms):   
            x = out_fc_layer(x)
            x = out_fc_norm(x)
            x = self.activation(x)
            x = self.dropout(x)

        # Apply the final layer to get the output/prediction
        x = self.output_layer(x)

        return x, attn_weights    

    @staticmethod
    def _make_lins_from_depth(depth, in_dim, scale_factor=2, include_norms=None):
        """
        Create a series of linear layers based on the depth and input dimension
        We will successively multiply the dimensionality of the layers here by scale_factor
        (truncating if it is not an integer)
        """
        lins = []
        norms = []
        curr_in_dim = in_dim
        curr_out_dim = in_dim # in case depth=0

        if include_norms == "layer":
            base_norm_layer = nn.LayerNorm
        elif include_norms == "batch":
            from functools import partial
            base_norm_layer = partial(pyg.nn.BatchNorm, allow_single_element=True)
        else:
            base_norm_layer = nn.Identity

        for _ in range(depth):
            curr_out_dim = int(curr_in_dim * scale_factor)
            lins.append(nn.Linear(curr_in_dim, curr_out_dim))
            norms.append(base_norm_layer(curr_out_dim))
            curr_in_dim = curr_out_dim
        
        # Return the list of linear layers and the final output dimension
        return nn.ModuleList(lins), nn.ModuleList(norms), curr_out_dim



class CrossAttentionModule(nn.Module):
    """
    Module that performs cross-attention between two sets of embeddings
    """

    def __init__(self, embed_dim_1, embed_dim_2, 
                 n_attention_heads, attn_dropout,
                 include_residual_stream=True,
                 dim_feedforward_scale=2, feedforward_dropout=0.2):
        """
        Initialize the cross-attention module
        """
        super(CrossAttentionModule, self).__init__()
        self.include_residual_stream = include_residual_stream

        self.preattn_norm1 = nn.LayerNorm(embed_dim_1)
        self.preattn_norm2 = nn.LayerNorm(embed_dim_2)

        self.embed1_to_2 = nn.MultiheadAttention(embed_dim=embed_dim_1, 
                                                 kdim=embed_dim_2, vdim=embed_dim_2,
                                                 num_heads=n_attention_heads, dropout=attn_dropout,
                                                 batch_first=True)
        
        self.embed2_to_1 = nn.MultiheadAttention(embed_dim=embed_dim_2, 
                                                 kdim=embed_dim_1, vdim=embed_dim_1,
                                                 num_heads=n_attention_heads, dropout=attn_dropout,
                                                 batch_first=True)

        self.ff_norm1 = nn.LayerNorm(embed_dim_1)
        self.ff_norm2 = nn.LayerNorm(embed_dim_2)

        self.ff_dropout = nn.Dropout(feedforward_dropout)

        if(include_residual_stream):
            self.ff1 = nn.Sequential(
                nn.Linear(embed_dim_1, embed_dim_1*dim_feedforward_scale),
                nn.ReLU(),
                nn.Dropout(feedforward_dropout),
                nn.Linear(embed_dim_1*dim_feedforward_scale, embed_dim_1)
            )

            self.ff2 = nn.Sequential(
                nn.Linear(embed_dim_2, embed_dim_2*dim_feedforward_scale),
                nn.ReLU(),
                nn.Dropout(feedforward_dropout),
                nn.Linear(embed_dim_2*dim_feedforward_scale, embed_dim_2)
            )
    
    def forward(self, embed_1, embed_2, mask1, mask2, return_weights=True):
        """
        Forward pass for the cross-attention module
        mask1 and mask2 are the masks that indicate which elements are REAL
        (the opposite of these masks will be used as the key_padding_mask)
        """
        
        x1_norm = self.preattn_norm1(embed_1)
        x2_norm = self.preattn_norm2(embed_2)

        x1_attn, attn_weights1 = self.embed1_to_2(x1_norm, x2_norm, x2_norm, key_padding_mask=~mask2)
        x2_attn, attn_weights2 = self.embed2_to_1(x2_norm, x1_norm, x1_norm, key_padding_mask=~mask1)

        if self.include_residual_stream:
            # Embed 1 residual stream
            embed_1 = embed_1 + self.ff_dropout(x1_attn)

            x1 = self.ff_norm1(embed_1)
            x1 = self.ff1(x1)

            embed_1 = embed_1 + self.ff_dropout(x1)

            # Embed 2 residual stream
            embed_2 = embed_2 + self.ff_dropout(x2_attn)

            x2 = self.ff_norm2(embed_2)
            x2 = self.ff2(x2)

            embed_2 = embed_2 + self.ff_dropout(x2)

        else:
            # No residual update; use the attention output directly
            embed_1 = x1_attn
            embed_2 = x2_attn

        if return_weights:
            attn_weights = (attn_weights1, attn_weights2)
            return embed_1, embed_2, attn_weights
        else:
            return embed_1, embed_2


class StackedCrossAttentionModule(nn.Module):
    """
    Module that stacks multiple cross attention modules on top of each
    other to perform multi-layer cross-attention (similar to TransformerEncoder)
    Expects as input a base cross-attention module that just needs to be called
    (basically a partial-ed initialized module)
    """

    def __init__(self, cross_attn_base, num_layers):
        """
        Initialize the cross-attention module
        """
        super(StackedCrossAttentionModule, self).__init__()

        cross_attn_layers = nn.ModuleList()

        for _ in range(num_layers):
            cross_attn_layers.append(cross_attn_base())

        self.cross_attn_layers = cross_attn_layers
        

    
    def forward(self, embed_1, embed_2, mask1, mask2, return_weights=True):
        """
        Forward pass for the stacked cross-attention module
        """
        attn_weights = []

        # Pass the embeddings through each of the cross-attention layers
        # with each one being updated by the previous one
        for cross_attn_layer in self.cross_attn_layers:
            embed_1, embed_2, attn_weight = cross_attn_layer(embed_1, embed_2, mask1, mask2, return_weights=True)

            if return_weights:
                attn_weights.append(attn_weight)
                    
        if return_weights:
            return embed_1, embed_2, attn_weights
        else:
            return embed_1, embed_2

