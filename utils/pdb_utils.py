import mdtraj as md
import numpy as np
import warnings

import ipdb

import utils.protein_definitions as protein_maps
from utils.create_protein_features import compute_residue_node_features, compute_residue_edge_features


def process_pdb(pdb_file, dist_units,
                edge_thresh, thresh_type,
                keep_self_loops, 
                vectorize_features, add_esm2_embeds, 
                add_residue_posenc, include_aa_props):
    """
    Load a PDB file using MDTraj and then process it to get:
    residue (node) features and edge features for a single protein
    Heavily inspired by `process_strucs` from here: https://github.com/Mickdub/gvp/blob/pocket_pred/src/validate_performance_on_xtals.py
    (PocketMiner codebase)
    """
    
    warnings.filterwarnings("ignore", message="Unlikely unit cell vectors")

    md_pdb = md.load(pdb_file)

    # We will ultimately compute the same features as PocketMiner
    # but store it in a different way (using PyG data structures)
    # and we will precompute these features while processing the data
    # so that we can use them in the training loop without needing to recompute every time

    # Selects atoms that are part of the protein, 
    # and are nitrogen, carbon, oxygen atoms (and alpha carbon atoms)
    idx = md_pdb.top.select("is_protein and (name N or name CA or name C or name O)")
    md_pdb_sub = md_pdb.atom_slice(idx)

    # Check if the protein has more than one frame - if so, only use the first frame
    # (this often results from multiple conformers being deposited, but generally conformers
    # are very similar to one another - so a consistent approach works best here)
    # Note that we could also use the average of all conformers/frames, but those coordinates
    # definitively do not represent a single conformer, so it's not clear what the best approach is
    if md_pdb_sub.n_frames > 1:
        md_pdb_sub = md_pdb_sub[0]

    # Drop first or last residue if they don't have all 4 atoms in order (N, CA, C, O)
    # (this is generally acceptable for most proteins and may happen since the terminal residues
    # tend to be poorly resolved sometimes)
    n_atoms = md_pdb_sub.n_atoms
    atoms = [atom.name for atom in md_pdb_sub.top.atoms]
    atomstr = ''.join(atoms)
    expected_atomstr = 'NCACO'

    first_ind = atomstr.find(expected_atomstr)
    last_ind = atomstr[::-1].find(expected_atomstr[::-1])

    md_pdb_sub = md_pdb_sub.atom_slice(range(first_ind, n_atoms-last_ind))


    # Get the number of residues in the protein
    # and get coords that correspond to each residue
    # (this should be a 3D array with shape (n_residues, n_atoms_per_residue, 3))
    # and n_atoms_per_residue should be 4 for all residues
    # (N, CA, C, O) is the order for every single residue, basically
    atom_coords = md_pdb_sub.xyz
    n_residues = md_pdb_sub.top.n_residues

    res_coords = atom_coords.reshape(n_residues, 4, 3)

    # Default distance units are nanometers in MDTraj
    # if user requests, convert to Angstroms
    if dist_units == 'angstroms':
        res_coords = res_coords * 10
    elif dist_units == 'nanometers':
        pass
    else:
        raise ValueError(f"Distance units {dist_units} not recognized. Must be 'angstroms' or 'nanometers'")

    # Get the actual residue identities for each atom as an index
    # (these will be either node types or a particular feature)
    res_names = [res.name for res in md_pdb_sub.top.residues]
    res_1letter_names = [protein_maps.PROTEIN_3LETTER_1LETTER_MAP[res] for res in res_names]
    res_idents = [protein_maps.PROTEIN_1LETTER_INT_MAP[res] for res in res_1letter_names]


    # Compute the node (residue) features
    # Node types are the residue identities
    node_features = compute_residue_node_features(res_coords, res_idents, 
                                                  vectorize_features, add_esm2_embeds, 
                                                  add_residue_posenc, include_aa_props)
    node_types = np.asarray(res_idents, dtype=np.int32)

    # Compute the edge features
    # Edge types are all 0s for now (may change in the future)
    edge_features = compute_residue_edge_features(res_coords, res_idents, 
                                                  edge_thresh, thresh_type, keep_self_loops,
                                                  vectorize_features)
    edge_types = np.zeros([n_residues, n_residues], dtype=np.int32)

    return node_features, edge_features, node_types, edge_types