import torch_geometric.utils.smiles as pyg_smiles
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from utils.create_smiles_features import compute_atom_node_features, compute_bond_edge_features
import utils.smiles_definitions as smiles_maps

import ipdb


def process_smiles(smiles_str, molecule_full_atomtype, 
                   one_hot_ordinal_feats, 
                   molecule_include_selfloops,
                   include_gasteiger_charges=True):
    """
    Load a SMILES string and get graph information for it.
    PyG does have a built-in function to do this, but it isn't 
    extraordinarily well-documented and we need some additional info,
    so we'll implement this ourselves.

    Heavily inspired by DGraphDTA's `smile_to_graph` in
    https://github.com/595693085/DGraphDTA/blob/master/data_process.py
    """

    # # Disable annoying logging that isn't useful
    # rdk.RDLogger.DisableLog('rdApp.*')
    
    rdk_mol = Chem.MolFromSmiles(smiles_str)
    
    if(include_gasteiger_charges):
        AllChem.ComputeGasteigerCharges(rdk_mol)

    # Get the node features (atom information)
    node_features = compute_atom_node_features(rdk_mol, one_hot_ordinal_feats, include_gasteiger_charges)

    # Get the node types
    if molecule_full_atomtype:
        map_dict = smiles_maps.ALL_ATOMICNUM_TO_NTYPE
    else:
        map_dict = smiles_maps.SELECT_ATOMICNUM_TO_NTYPE
    
    node_types = [map_dict[atom.GetAtomicNum()] for atom in rdk_mol.GetAtoms()]

    # Get the edge features (bond information)
    edge_features = compute_bond_edge_features(rdk_mol, include_selfloops=molecule_include_selfloops)
    edge_types = np.empty_like(edge_features[:,:,-1])
    edge_types.fill(np.nan)

    # Self-loops if desired will be bond type 0 and all other bond types will be offset by 1
    if molecule_include_selfloops:
        offset = 1
        for i in range(rdk_mol.GetNumAtoms()):
            edge_types[i,i] = 0
    else:
        offset = 0

    for bond in rdk_mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_types[start, end] = smiles_maps.SMILES_BOND_MAP[str(bond.GetBondType())] + offset
        edge_types[end, start] = smiles_maps.SMILES_BOND_MAP[str(bond.GetBondType())] + offset

    return node_features, edge_features, node_types, edge_types