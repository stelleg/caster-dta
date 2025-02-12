import numpy as np
import utils.smiles_definitions as smiles_maps

import ipdb

def compute_atom_node_features(rdk_mol, one_hot_ordinal_feats, include_gasteiger_charges=True):
    """
    Get the node features (atom information)
    Includes: chirality, hybridization,
    number of hydrogens, number of radical electrons
    total valence, number of implicit hydrogens
    degree, formal charge, 
    if is in ring, if is aromatic
    and Gasteiger charge if requested by the user
    """

    # Note that the atoms are in order of their index
    # so the position in the list matches their index in the molecule
    # for matching with the edge list
    node_features = []

    for atom in rdk_mol.GetAtoms():
        atom_feats = []
        # One hot encoded features
        atom_feats.extend(smiles_maps.SMILES_CHIRALITY_MAP[str(atom.GetChiralTag())])
        atom_feats.extend(smiles_maps.SMILES_HYBRID_MAP[str(atom.GetHybridization())])
        atom_feats.extend(smiles_maps.SMILES_H_MAP[atom.GetTotalNumHs()])
        atom_feats.extend(smiles_maps.SMILES_DEGREE_MAP[atom.GetDegree()])
        atom_feats.extend(smiles_maps.SMILES_VALENCE_MAP[atom.GetImplicitValence()])

        if(one_hot_ordinal_feats):
            atom_feats.extend(smiles_maps.SMILES_CHARGE_MAP[atom.GetFormalCharge()])
            atom_feats.extend(smiles_maps.SMILES_RADICAL_MAP[atom.GetNumRadicalElectrons()])
        else:
            # Features that we might include at some point as categorical (as above), 
            # but for now are included as ordinal due to their high cardinality
            atom_feats.append(atom.GetFormalCharge())
            atom_feats.append(atom.GetNumRadicalElectrons())

        # Binary features (already one-hot encoded)
        atom_feats.append(int(atom.IsInRing()))
        atom_feats.append(int(atom.GetIsAromatic()))

        if(include_gasteiger_charges):
            g_charge = atom.GetDoubleProp("_GasteigerCharge")

            # Replace NaNs with 0
            if np.isnan(g_charge):
                g_charge = 0.0

            # Replace infinities with 0 (not sure why these exist, but they do)
            if ~np.isfinite(g_charge):
                g_charge = 0.0

            atom_feats.append(g_charge)

        # The atom features should be a flattened list
        # due to use of extend/append as appropriate
        # 40 features per atom (+1 if Gasteiger)

        node_features.append(atom_feats)

    # 40 node features per atom
    # (+1 if Gasteiger charges are included)
    node_features = np.asarray(node_features, dtype=np.float32)

    # assert node_features.shape[-1] == smiles_maps.NUM_ATOM_FEATURES, f"Expected {smiles_maps.NUM_ATOM_FEATURES} features, got {node_features.shape[-1]}.\n \
    # Change the NUM_ATOM_FEATURES constant in smiles_definitions.py if the current number of features is correct."

    return node_features


def compute_bond_edge_features(rdk_mol, include_selfloops=False):
    """
    Get the edge features (bond information)
    Includes: stereo (one-hot of 7), is conjugated, is in ring
    """

    n_atoms = rdk_mol.GetNumAtoms()
    edge_features = np.empty((n_atoms, n_atoms, 9))
    edge_features.fill(np.nan)

    for bond in rdk_mol.GetBonds():
        bond_feats = []

        # One hot encoded features
        bond_feats.extend(smiles_maps.SMILES_STEREO_MAP[str(bond.GetStereo())])
        
        # Binary features (already one-hot encoded)
        bond_feats.append(int(bond.GetIsConjugated()))
        bond_feats.append(int(bond.IsInRing()))
        # bond_feats.append(int(bond.GetIsAromatic())) # redundant with bond type

        bond_feats = np.asarray(bond_feats, dtype=np.float32)

        edge_features[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()] = bond_feats
        edge_features[bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()] = bond_feats

    # Add self loops if desired
    # where all edge features are 0
    if include_selfloops:
        for i in range(n_atoms):
            edge_features[i, i] = np.zeros(9)

    # 9 edge features per bond

    # assert edge_features.shape[-1] == smiles_maps.NUM_BOND_FEATURES, f"Expected {smiles_maps.NUM_BOND_FEATURES} features, got {edge_features.shape[-1]}.\n \
    # Change the NUM_BOND_FEATURES constant in smiles_definitions.py if the current number of features is correct."

    return edge_features