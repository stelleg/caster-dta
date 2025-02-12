import torch
import torch_geometric as pyg

import pandas as pd
import torch.multiprocessing as multiprocessing

# This is needed to avoid a deadlock in the DataLoader
# that occurs for some unknown reasons ("received 0 items of ancdata")
multiprocessing.set_sharing_strategy('file_system')
N_PROC = None

from functools import partial

from utils import pdb_utils, smiles_utils, create_graphs

from tqdm import tqdm

import ipdb

# below for debugging only, sets Pool to (1) as well
# import multiprocessing.dummy as multiprocessing
# N_PROC = 1


class ProteinMoleculeDataset(torch.utils.data.Dataset):
    """
    A dataset that stores protein-molecule pairs and their affinities.
    Will store the data in such a way that the graphs are only stored for
    unique proteins and unique molecules and then indexing will call into
    those to return the pairs.
    """
    def __init__(self, paired_dataframe, sparse_edges=False,
                 protein_dist_units='nanometers',
                 protein_edge_thresh=1.0, protein_thresh_type='dist',
                 protein_keep_selfloops=False,
                 protein_vector_features=True,
                 protein_include_esm2=False,
                 protein_include_residue_posenc=False,
                 protein_include_aa_props=True,
                 molecule_full_atomtype=False,
                 molecule_onehot_ordinal_feats=False,
                 molecule_include_selfloops=False,
                 scale_output=None):
        """
        This dataset is defined by a dataframe with columns:
        - protein_id: the protein ID
        - protein_sequence: the protein sequence
        - protein_file: a path to the protein PDB file
        - molecule_id: the molecule ID
        - molecule_smiles: the molecule SMILES string
        - affinity_score: the affinity score (target value)
        Take in the dataframe as defined above and use it to create
        the structure of this dataset with unique protein-molecule pairs.
        """

        # Reset index in case, since self.pair_indices relies on this
        paired_dataframe = paired_dataframe.reset_index(drop=True)

        if(isinstance(scale_output, str)):
            self.scale_output = [scale_output]
        else:
            self.scale_output = scale_output

        # WARN: In pyg 2.5.0, HEATConv does not work properly with sparsed edges
        # as it will for some reason modify the edge_type tensor in place
        # (so for now, we will only use dense edges - this may be fixed in the future)
        self.sparse_edges = sparse_edges
        self.protein_dist_units = protein_dist_units
        self.protein_edge_thresh = protein_edge_thresh
        self.protein_thresh_type = protein_thresh_type
        self.protein_keep_selfloops = protein_keep_selfloops
        self.protein_vector_features = protein_vector_features
        self.protein_include_esm2 = protein_include_esm2
        self.protein_include_residue_posenc = protein_include_residue_posenc
        self.protein_include_aa_props = protein_include_aa_props
        self.molecule_full_atomtype = molecule_full_atomtype
        self.molecule_onehot_ordinal_feats = molecule_onehot_ordinal_feats
        self.molecule_include_selfloops = molecule_include_selfloops

        # Load the molecule data into graphs (done first because much faster)
        molecule_df = paired_dataframe[['molecule_id', 'molecule_smiles']].drop_duplicates()
        self.molecule_data = self._load_molecules(molecule_df, self.sparse_edges,
                                                  self.molecule_full_atomtype,
                                                  self.molecule_onehot_ordinal_feats,
                                                  self.molecule_include_selfloops)

        # Load the protein data into graphs (done second because very slow)
        protein_df = paired_dataframe[['protein_id', 'protein_sequence', 'protein_file']].drop_duplicates()
        self.protein_data = self._load_proteins(protein_df, self.sparse_edges, 
                                                self.protein_dist_units,
                                                self.protein_edge_thresh, self.protein_thresh_type,
                                                self.protein_keep_selfloops,
                                                self.protein_vector_features,
                                                self.protein_include_esm2, 
                                                self.protein_include_residue_posenc,
                                                self.protein_include_aa_props)

        # Load the affinity/target data as a tensor
        # and store the indices as a mapping to the protein/molecule pairs
        self.affinity_data = torch.tensor(paired_dataframe['affinity_score'].values, dtype=torch.float32)
        self.pair_indices = paired_dataframe[['protein_id', 'molecule_id']].apply(tuple, axis=1).to_dict()

        # Store a stratification variable
        # to make making a train/test/validation split on different proteins/molecules easier
        # basically have two indices, one for the protein and one for the molecule
        self.idx_protein_strat = torch.tensor(paired_dataframe['protein_id'].astype('category').cat.codes.values, dtype=torch.long)
        self.idx_molecule_strat = torch.tensor(paired_dataframe['molecule_id'].astype('category').cat.codes.values, dtype=torch.long)


        if(self.scale_output is not None):
            self._init_scale_target()

        # Note: you can use tensor.bincount() to get counts for each protein or molecule downstream

        # Store metadata about the features in the dataset
        self.metadata_dict = self._get_feature_metadata()


    def __len__(self):
        return len(self.affinity_data)


    def __getitem__(self, idx):
        protein_id, molecule_id = self.pair_indices[idx]
        return self.protein_data[protein_id], self.molecule_data[molecule_id], self.affinity_data[idx]
    

    def __str__(self):
        out_str = f"ProteinMoleculeDataset:\n"
        out_str += f"\t{len(self)} protein-molecule pairs and targets\n"
        out_str += f"\t{len(self.protein_data)} unique proteins\n"
        out_str += f"\t{len(self.molecule_data)} unique molecules\n"
        out_str += f"\tProtein distance units: {self.protein_dist_units}\n"
        out_str += f"\tProtein edge threshold (type): {self.protein_edge_thresh} ({self.protein_thresh_type})\n"
        out_str += f"\tProtein edge keep self-loops: {self.protein_keep_selfloops}\n"
        out_str += f"\tMolecule edge include self-loops: {self.molecule_include_selfloops}\n"
        out_str += f"\tFeature metadata:\n"
        out_str += f"\t\tProtein node features: {self.metadata_dict['protein_node_features']}\n"
        out_str += f"\t\tProtein edge features: {self.metadata_dict['protein_edge_features']}\n"
        out_str += f"\t\tProtein node types: {self.metadata_dict['protein_node_types']}\n"
        out_str += f"\t\tProtein edge types: {self.metadata_dict['protein_edge_types']}\n"
        out_str += f"\t\tMolecule node features: {self.metadata_dict['molecule_node_features']}\n"
        out_str += f"\t\tMolecule edge features: {self.metadata_dict['molecule_edge_features']}\n"
        out_str += f"\t\tMolecule node types: {self.metadata_dict['molecule_node_types']}\n"
        out_str += f"\t\tMolecule edge types: {self.metadata_dict['molecule_edge_types']}\n"
        out_str += f"\tAffinity/target metadata:\n"
        out_str += f"\t\trescaling: {self.scale_output}\n"
        out_str += f"\t\tmin: {self.affinity_data.min()} ; max: {self.affinity_data.max()}\n"
        out_str += f"\t\tmean: {self.affinity_data.mean()} ; std: {self.affinity_data.std()}\n"

        return out_str


    def _init_scale_target(self):
        """
        Perform scaling and store the scaling parameters if needed
        Target/output transformations
        Multiple can be applied if passing a list
        (will be applied in the order given)
        """

        for scale_type in self.scale_output:
            self._perform_scale_type(scale_type)


    def _perform_scale_type(self, scale_type):
        """
        Perform scaling on the given values
        """
        if scale_type == 'standardize':
            # Standardize the affinity scores
            self.scale_mean_factor = torch.mean(self.affinity_data)
            self.scale_std_factor = torch.std(self.affinity_data)
            self.affinity_data = (self.affinity_data - self.scale_mean_factor) / torch.std(self.affinity_data)

        if scale_type == 'minmax':
            # Scale the affinity scores to be between -1 and 1
            self.scale_min_factor = torch.min(self.affinity_data)
            self.scale_max_factor = torch.max(self.affinity_data)
            # Between 0 and 1
            self.affinity_data = (self.affinity_data - self.scale_min_factor) / (self.scale_max_factor - self.scale_min_factor)
            # Now between -1 and 1
            self.affinity_data = self.affinity_data * 2 - 1

        if scale_type == 'log':
            self.affinity_data = torch.log1p(self.affinity_data)


    def unscale_target(self, values):
        """
        Unscale the given values using the scales computed in the original dataset
        Do so in reverse of the scaling done in the __init__ method to return the original values
        """

        for scale_type in self.scale_output[::-1]:
            values = self._perform_unscale_type(values, scale_type)            

        return values
    

    def _perform_unscale_type(self, values, scale_type):
        """
        Perform UNSCALING on the given values
        (note, we do not change the affinity values inplace here)
        """
        if scale_type == 'standardize':
            values = (values * self.scale_std_factor) + self.scale_mean_factor

        if scale_type == 'minmax':
            values = (values + 1) * 0.5
            values = values * (self.scale_max_factor - self.scale_min_factor) + self.scale_min_factor

        if scale_type == 'log':
            values = torch.expm1(values)

        return values
    

    def _report_scale_data(self):
        """
        Report the scaling data for the dataset
        as a dictionary
        """
        scale_data_dict = {}
        scale_data_dict['scale_output'] = self.scale_output

        for scale_type in self.scale_output:
            scale_type_dict = {}

            if scale_type == 'standardize':
                scale_type_dict['scale_mean_factor'] = self.scale_mean_factor.item()
                scale_type_dict['scale_std_factor'] = self.scale_std_factor.item()

            if scale_type == 'minmax':
                scale_type_dict['scale_min_factor'] = self.scale_min_factor.item()
                scale_type_dict['scale_max_factor'] = self.scale_max_factor.item()

            if scale_type == 'log':
                pass

            scale_data_dict[scale_type] = scale_type_dict

        return scale_data_dict
    

    def _load_scale_data_from_dict(self, scale_data_dict):
        """
        Load the scaling data from a dictionary
        """
        self.scale_output = scale_data_dict['scale_output']

        for scale_type in self.scale_output:
            scale_type_dict = scale_data_dict[scale_type]

            if scale_type == 'standardize':
                self.scale_mean_factor = torch.tensor(scale_type_dict['scale_mean_factor'])
                self.scale_std_factor = torch.tensor(scale_type_dict['scale_std_factor'])

            if scale_type == 'minmax':
                self.scale_min_factor = torch.tensor(scale_type_dict['scale_min_factor'])
                self.scale_max_factor = torch.tensor(scale_type_dict['scale_max_factor'])

            if scale_type == 'log':
                pass

        return


    def _get_feature_metadata(self):
        """
        Get metadata about the features contained in the dataset
        Specifically:
        the number of node features for proteins and molecules,
        the number of edge features for proteins and molecules,
        and the number of node types and edge types for proteins and molecules
        """
        metadata_dict = {}
        example_pgraph = next(iter(self.protein_data.values()))
        example_mgraph = next(iter(self.molecule_data.values()))
        
        metadata_dict['protein_node_features'] = _shape_from_feature(example_pgraph.x)
        metadata_dict['protein_edge_features'] = _shape_from_feature(example_pgraph.edge_attr)
        
        metadata_dict['molecule_node_features'] = _shape_from_feature(example_mgraph.x)
        metadata_dict['molecule_edge_features'] = _shape_from_feature(example_mgraph.edge_attr)

        metadata_dict['protein_node_types'] = self._get_num_types(graph_type='protein', entity_type='node')
        metadata_dict['protein_edge_types'] = self._get_num_types(graph_type='protein', entity_type='edge')

        metadata_dict['molecule_node_types'] = self._get_num_types(graph_type='molecule', entity_type='node')
        metadata_dict['molecule_edge_types'] = self._get_num_types(graph_type='molecule', entity_type='edge')

        return metadata_dict
    

    def _get_num_types(self, graph_type='protein', entity_type='node'):
        """
        Get the number of unique node types in all the graphs in the dataset
        (This is the maximum node type in the graphs + 1)
        """
        key_val = entity_type + '_type'

        max_type_val = 0

        if(graph_type == 'protein'):
            base_data = self.protein_data
        elif(graph_type == 'molecule'):
            base_data = self.molecule_data

        for g_data in base_data.values():
            max_type_val = max(max_type_val, g_data[key_val].max().item())

        return max_type_val+1


    @staticmethod
    def _load_proteins(protein_df, sparse_edges, 
                       dist_units,
                       edge_thresh, 
                       thresh_type,
                       keep_self_loops,
                       protein_vector_features,
                       protein_include_esm2,
                       protein_include_residue_posenc,
                       protein_include_aa_props):
        """
        Load the proteins into a graph representation
        """

        # Load the protein data into graphs and store as a dictionary
        # of IDs to graphs
        protein_dict = {}
        _protein_file_to_graph_withedge = partial(_protein_file_to_graph, 
                                                  dist_units=dist_units,
                                                  edge_thresh=edge_thresh, 
                                                  thresh_type=thresh_type,
                                                  keep_self_loops=keep_self_loops,
                                                  vectorize_features=protein_vector_features,
                                                  add_esm2_embeds=protein_include_esm2, 
                                                  add_residue_posenc=protein_include_residue_posenc,
                                                  include_aa_props=protein_include_aa_props)

        with multiprocessing.Pool(N_PROC) as pool:
            protein_list = list(tqdm(pool.imap(_protein_file_to_graph_withedge, protein_df['protein_file'], chunksize=1), total=protein_df.shape[0]))
        
        for protein_id, protein_seq, protein_g in zip(protein_df['protein_id'], 
                                                      protein_df['protein_sequence'], protein_list):
            protein_g.protein_sequence = protein_seq
            protein_g.protein_id = protein_id
            protein_dict[protein_id] = protein_g

        # Sort the protein dictionary by the protein ID (just for consistency)
        protein_dict = {k: protein_dict[k] for k in sorted(protein_dict.keys())}

        # Sparse transform if needed
        if(sparse_edges):
            ToSparseTransform = pyg.transforms.ToSparseTensor(attr='edge_attr')
            protein_dict = {k: ToSparseTransform(v) for k, v in protein_dict.items()}

        return protein_dict
    
    
    @staticmethod
    def _load_molecules(molecule_df, sparse_edges,
                        molecule_full_atomtype,
                        molecule_onehot_ordinal_feats,
                        molecule_include_selfloops):
        """
        Load the molecules into a graph representation
        """

        # Load the molecule data into graphs and store as a dictionary
        # of IDs to graphs
        molecule_dict = {}

        _molecule_smiles_to_graph_withtype = partial(_molecule_smiles_to_graph, 
                                                     molecule_full_atomtype=molecule_full_atomtype, 
                                                     molecule_onehot_ordinal_feats=molecule_onehot_ordinal_feats,
                                                     molecule_include_selfloops=molecule_include_selfloops) 

        with multiprocessing.Pool(N_PROC) as pool:
            molecule_list = list(tqdm(pool.imap(_molecule_smiles_to_graph_withtype, molecule_df['molecule_smiles'], chunksize=1), total=molecule_df.shape[0]))
        
        for molecule_id, molecule_smiles, molecule_g in zip(molecule_df['molecule_id'], 
                                                            molecule_df['molecule_smiles'],
                                                            molecule_list):
            molecule_g.molecule_smiles = molecule_smiles
            molecule_g.molecule_id = molecule_id
            molecule_dict[molecule_id] = molecule_g

        # Sort the molecule dictionary by the molecule ID (just for consistency)
        molecule_dict = {k: molecule_dict[k] for k in sorted(molecule_dict.keys())}
        
        # Sparse transform if needed
        if(sparse_edges):
            ToSparseTransform = pyg.transforms.ToSparseTensor(attr='edge_attr')
            molecule_dict = {k: ToSparseTransform(v) for k, v in molecule_dict.items()}

        return molecule_dict


class PMD_DataLoader(torch.utils.data.DataLoader):
    """
    DataLoader for the ProteinMoleculeDataset
    This is just a basic PyTorch DataLoader wrapper around the dataset
    with a custom collate function based on PyG's collate_fn
    """
    def __init__(self, dataset, num_workers=0, 
                 shuffle=False, max_num=12000000, 
                 count_elem='edge', graph_type='protein', 
                 include_nodepair=True,
                 skip_too_big=False, max_batch_size=None,
                 **kwargs):
        super().__init__(dataset, 
                         num_workers=num_workers, 
                         collate_fn=PMDCollator(dataset),
                         batch_sampler=PMD_BatchSampler(dataset, max_num, count_elem, 
                                                        graph_type, shuffle, 
                                                        include_nodepair, skip_too_big,
                                                        max_batch_size), 
                         **kwargs)


class PMD_BatchSampler(torch.utils.data.Sampler):
    """
    Batch sampler for the ProteinMoleculeDataset
    This will sample batches based on the protein or molecule
    based on the number of edges in the graphs of these objects
    Highly inspired by the PyG implementation:
    https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/loader/dynamic_batch_sampler.html
    """
    def __init__(
        self,
        dataset,
        max_num,
        count_elem='edge',
        graph_type='both',
        include_nodepair=True,
        shuffle=True,
        skip_too_big=False,
        max_bsize=None,
    ):
        if max_num <= 0:
            raise ValueError(f"`max_num` should be a positive integer value "
                             f"(got {max_num})")
        if count_elem not in ['node', 'edge']:
            raise ValueError(f"`max_count` choice should be either "
                             f"'node' or 'edge' (got '{count_elem}')")
        if graph_type not in ['protein', 'molecule', 'both']:
            raise ValueError(f"`graph_type` choice should be one of "
                             f"'protein', 'molecule', or 'both' (got '{graph_type}')")

        self.dataset = dataset
        self.max_num = max_num
        self.count_elem = count_elem
        self.graph_type = graph_type
        self.include_nodepair = include_nodepair
        self.shuffle = shuffle
        self.skip_too_big = skip_too_big
        self.max_bsize = max_bsize

    def __iter__(self):
        if self.shuffle:
            indices = torch.randperm(len(self.dataset)).tolist()
        else:
            indices = range(len(self.dataset))

        samples = []
        current_num = 0
        num_steps = 0
        num_processed = 0
        max_npair = 0

        while (num_processed < len(self.dataset)):

            for i in indices[num_processed:]:
                p_data, m_data, _ = self.dataset[i]
                p_num = p_data.num_nodes if self.count_elem == 'node' else p_data.num_edges
                m_num = m_data.num_nodes if self.count_elem == 'node' else m_data.num_edges
                
                if self.graph_type == 'protein':
                    num = p_num
                elif self.graph_type == 'molecule':
                    num = m_num
                else:
                    num = p_num + m_num


                # If requested, add the initialized size of the residue-pair attention matrix
                # to the number of elements considered for the maximum "size" of the batch
                # Note that this is a rough estimate and may not be exact, 
                # and that all samples (even if smaller) when added will add to the nodepair size 
                # as the batch increases even if the maximum "sequence" length doesn't
                if self.include_nodepair:
                    prev_max_npair = max_npair
                    max_npair = max(max_npair, p_data.num_nodes * m_data.num_nodes)

                    # Difference between the current maximum nodepair initialized size and the previous
                    # since the previous would have been added already to the total
                    num += ( max_npair * (len(samples)+1) ) - ( prev_max_npair * len(samples) )
                    

                if current_num + num > self.max_num:
                    if current_num == 0:
                        if self.skip_too_big:
                            continue
                    else:  # Mini-batch filled:
                        break


                samples.append(i)
                num_processed += 1
                current_num += num

                if (self.max_bsize is not None) and (len(samples) >= self.max_bsize):
                    break
            

            yield samples
            samples = []
            num_steps += 1
            current_num = 0
            max_npair = 0


class PMDCollator:
    """
    Collator class for the ProteinMoleculeDataset
    On call, takes in a batch of data and returns three things:
    - A batched protein PyG data object
    - A batched molecule PyG data object
    - A tensor of affinity scores
    """
    
    def __init__(self, dataset):
        self.dataset = dataset

    def __call__(self, batch):
        """
        Takes in a batch (which is a list of tuples of protein, molecule, affinity)
        """
        batch_T = [list(x) for x in zip(*batch)]
        protein_batch = pyg.data.Batch.from_data_list(batch_T[0])
        molecule_batch = pyg.data.Batch.from_data_list(batch_T[1])
        affinity_batch = torch.stack(batch_T[2])

        return protein_batch, molecule_batch, affinity_batch
    

def _protein_file_to_graph(protein_file, 
                           dist_units,
                           edge_thresh, thresh_type, 
                           keep_self_loops,
                           vectorize_features,
                           add_esm2_embeds,
                           add_residue_posenc,
                           include_aa_props):
    """
    Load a single protein file into a protein graph
    """
    node_features, edge_features, ntypes, etypes = pdb_utils.process_pdb(protein_file, 
                                                                         dist_units,
                                                                         edge_thresh, thresh_type, 
                                                                         keep_self_loops,
                                                                         vectorize_features, 
                                                                         add_esm2_embeds, 
                                                                         add_residue_posenc,
                                                                         include_aa_props)
    protein_g = create_graphs.construct_graph(node_features, edge_features, ntypes, etypes)

    return protein_g


def _molecule_smiles_to_graph(molecule_smiles, molecule_full_atomtype,
                              molecule_onehot_ordinal_feats,
                              molecule_include_selfloops):
    """
    Load a single molecule SMILES into a molecule graph
    """
    node_features, edge_features, ntypes, etypes = smiles_utils.process_smiles(molecule_smiles, 
                                                                               molecule_full_atomtype,
                                                                               molecule_onehot_ordinal_feats,
                                                                               molecule_include_selfloops)
    mol_g = create_graphs.construct_graph(node_features, edge_features, ntypes, etypes)

    return mol_g


def _shape_from_feature(fdata):
    """
    Get the shape of a set of features
    (accounting for the fact that it may be a tensor or a tuple of tensors)
    In the case of a tensor, return the number of features
    In the case of a tuple, return a tuple of the number of features in each tuple
    """

    if(isinstance(fdata, torch.Tensor)):
        return fdata.shape[1]
    
    elif(isinstance(fdata, tuple)):
        return tuple(x.shape[1] for x in fdata)

    else:
        raise ValueError(f"Expected a tensor or tuple of tensors, got {type(fdata)}.")