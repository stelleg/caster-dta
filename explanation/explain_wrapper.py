import torch
import torch.nn as nn
import torch_geometric as pyg

import copy

import ipdb

class DTAModelExplainer():
    """
    Wrapper class for DTA model explanations
    Note that one explainer is trained on one side (protein or ligand) at a time
    One can select to just explain protein or molecule, but default is explaining both
    """

    def __init__(self, model, explain_side='both',
                 n_epochs=100, lr=0.01):
        self.wrapped_model = ExplainWrapperDTAModel(model)

        if explain_side == 'both':
            self.explain_side = ['molecule', 'protein']
        else:
            self.explain_side = [explain_side]

        self.explainers = {}

        if 'protein' in self.explain_side:
            prot_gnn_explainer = pyg.explain.GNNExplainer(epochs=n_epochs, lr=lr)

            prot_explainer = pyg.explain.Explainer(
                model=self.wrapped_model,
                algorithm=prot_gnn_explainer,
                explanation_type='model',
                node_mask_type='object',
                edge_mask_type=None,
                model_config=dict(
                    mode='regression',
                    task_level='graph'
                )
            )

            self.explainers['protein'] = prot_explainer
        
        if 'molecule' in self.explain_side:
            mol_gnn_explainer = pyg.explain.GNNExplainer(epochs=n_epochs, lr=lr)

            mol_explainer = pyg.explain.Explainer(
                model=self.wrapped_model,
                algorithm=mol_gnn_explainer,
                explanation_type='model',
                node_mask_type='object',
                edge_mask_type=None,
                model_config=dict(
                    mode='regression',
                    task_level='graph'
                )
            )

            self.explainers['molecule'] = mol_explainer
    
    def _reset_explain_flags(self):
        """
        Reset all explain flags on the convolutional layers
        by setting them to None
        """
        for module in self.wrapped_model.modules():
            if isinstance(module, pyg.nn.MessagePassing):
                module.explain = None

        return


    def _set_explain_flags(self, explain_side='protein'):
        """
        Sets flags on the convolutional layers where we allow for explanations
        and disables it elsewhere. Can only explain one side at a time.
        Will set the explain flag to False for the side we do not want to explain
        (the explainer will set it to True when appropriate if it is not explicitly False)
        """

        if explain_side == 'protein':
            keep_prefix = 'protein_gnn'
        elif explain_side == 'molecule':
            keep_prefix = 'molecule_gnn'

        for mod_name, module in self.wrapped_model.named_modules():
            if isinstance(module, pyg.nn.MessagePassing):
                if keep_prefix in mod_name:
                    continue
                else:
                    module.explain = False
        
        return
    

    def explain_model(self,
                      protein_graph_data,
                      molecule_graph_data,
                      explain_index=None):
        """
        Explain the model based on the graphs provided
        depending on which side this explainer is set to explain
        """

        out_explanations = {}
        orig_protein_graph_data = copy.deepcopy(protein_graph_data)
        orig_molecule_graph_data = copy.deepcopy(molecule_graph_data)

        for side in self.explain_side:
            self._reset_explain_flags()
            self._set_explain_flags(side)
            self.wrapped_model.zero_grad(set_to_none=True)
            self.wrapped_model.eval()

            protein_graph_data = copy.deepcopy(orig_protein_graph_data)
            molecule_graph_data = copy.deepcopy(orig_molecule_graph_data)

            if side == 'protein':
                explain_x, explain_edge_index = protein_graph_data['x'], protein_graph_data['edge_index']
            elif side == 'molecule':
                explain_x, explain_edge_index = molecule_graph_data['x'], molecule_graph_data['edge_index']

            # Check if x tuple (or list), if so, merge and pass info needed for splitting downstream
            # Remember that vectors are of shape num_nodes, num_vecs, 3
            if isinstance(explain_x, (tuple, list)):
                num_vecs = explain_x[1].shape[-2]
                explain_x = _merge_x(*explain_x)
            else:
                num_vecs = None

            # Reassign detached versions of the graph data to avoid issues
            # where the inputs are attached to the computation graph
            explain_x = explain_x.clone().detach().requires_grad_()
            explain_edge_index = explain_edge_index.clone().detach()

            model_kwargs = {
                'protein_graph_data': protein_graph_data,
                'molecule_graph_data': molecule_graph_data,
                'explain_side': side,
                'num_vecs': num_vecs
            }

            explanation = self.explainers[side](explain_x, explain_edge_index,
                                                index=explain_index,
                                                **model_kwargs)
            
            out_explanations[side] = explanation


        self._reset_explain_flags()

        return out_explanations


class ExplainWrapperDTAModel(nn.Module):
    """
    Wrapper class for the DTA model
    which takes in a trained model and wraps it with a
    new forward function for the explainer to work with
    """

    def __init__(self, model):
        super(ExplainWrapperDTAModel, self).__init__()
        self.model = model

    def forward(self, x, edge_index, 
                protein_graph_data={},
                molecule_graph_data={},
                explain_side='protein',
                num_vecs=None):
        """
        Takes in the expected passed values of the GNNExplainer
        version of forward and reconfigures it to the expected one
        for the model to work
        """
        # x and edge_index are passed in as the first two arguments
        # and will need to be reassigned to their respective graph data
        # dictionaries to pass into the joint model

        # If the input data was vector data originally (for x), then we need to resplit it
        # back into its vector forms

        if(num_vecs is not None):
            x = _resplit_x(x, num_vecs)

        if explain_side == 'protein':
            protein_graph_data['x'] = x
            protein_graph_data['edge_index'] = edge_index
        elif explain_side == 'molecule':
            molecule_graph_data['x'] = x
            molecule_graph_data['edge_index'] = edge_index

        out, _ = self.model(protein_graph_data, molecule_graph_data)

        return out

    
def _resplit_x(x, nv):
    """
    Resplit merged data into separate scalar, vector data
    Basically identical to GVP's _split function
    Returns the split scalar, vector tuple
    """
    v = torch.reshape(x[..., -3*nv:], x.shape[:-1] + (nv, 3))
    s = x[..., :-3*nv]
    return s, v

def _merge_x(s, v):
    """
    Merge separate vectors into a single tensor
    Returns the merged tensor
    """
    v = torch.reshape(v, v.shape[:-2] + (3*v.shape[-2],))
    return torch.cat([s, v], -1)