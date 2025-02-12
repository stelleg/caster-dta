import numpy as np
import torch
import scipy.spatial.distance as sp_dist

import utils.protein_definitions as pd_maps

import ipdb


# TODO: Consider one-hot encoding some of these features

def compute_residue_node_features(res_coords, res_idents, 
                                  vectorize_features, add_esm2_embeds, 
                                  add_residue_posenc, include_aa_props):
    """
    Take in residue coordinates and indices for a single protein to produce 
    a set of residue node features for every residue
    Heavily inspired by `StructuralFeatures` from here: https://github.com/Mickdub/gvp/blob/pocket_pred/src/models.py
    (PocketMiner codebase)
    """
    # res_coords [n_residues, 4, 3]
    # res_idents [n_residues]

    # TODO: consider splitting each of these into separate functions
    # for clarity and ease of testing

    ## Dihedral angles (phi, psi, omega)

    # Each residue has 4 atoms (N, CA, C, O) 
    # and we will compute the dihedral angles
    # we want phi, psi, omega angles

    # We only need the first three coordinates for each residue (N, CA, C)
    dihedral_coords = res_coords[:,:3,:]
    dihedral_coords = dihedral_coords.reshape(-1, 3) # Now N, CA, C, N, CA, C, ...
    
    shifted_dist = dihedral_coords[1:,:] - dihedral_coords[:-1,:]
    shifted_dist = normalize_vecs(shifted_dist, axis=-1)

    # Backbone relative coordinates
    u0 = shifted_dist[2:,:]
    u1 = shifted_dist[1:-1,:]
    u2 = shifted_dist[:-2,:]

    # Backbone normals (cross products)
    n1 = normalize_vecs(np.cross(u1, u0, axis=-1), axis=-1)
    n2 = normalize_vecs(np.cross(u2, u1, axis=-1), axis=-1)

    # Angles between normals (note, do we need an epsilon here for clipping?)
    # Produces a vector that is in the order of phi, psi, omega for each residue
    # (with the first phi and last psi and omega not included)
    cosAngles = np.clip(np.sum(n1 * n2, axis=-1), -1.0, 1.0)
    allAngles = np.arccos(cosAngles) * np.sign(np.sum(n1 * u2, axis=-1))

    # We need to pad some values for the first and last residues
    # Specifically, the first phi, the last psi, and the last omega
    # will be set to 0 as a base feature
    allAngles = np.pad(allAngles, [1,2], 'constant', constant_values=0.0)

    # We will split this back into residue-level features
    # of phi, psi, omega for each residue
    allAngles = allAngles.reshape(-1, 3)

    # PocketMiner got the sin and cos of these angles as features
    # so we will do the same, now having 6 features per residue
    residueAngleFeats = np.concatenate([np.cos(allAngles), np.sin(allAngles)], axis=-1)


    ## Orientations 
    # Note, in PocketMiner they have forward and backward separately, but these are identical
    # and just -1 * themselves, so we just need to compute one of them
    # (we do need both as we offset by 1 for forward/backward at opposite residue ends)
    alpha_coords = res_coords[:, 1, :]
    forward = normalize_vecs(alpha_coords[1:,:] - alpha_coords[:-1,:], axis=-1)
    backward = -1 * forward
    residueForwardFeats = np.pad(forward, [(0, 1), (0, 0)], 'constant', constant_values=0.0)
    residueBackwardFeats = np.pad(backward, [(1, 0), (0, 0)], 'constant', constant_values=0.0)


    ## Sidechains vectors
    # We will compute the sidechain vectors for each residue
    # we will use the alpha_coords from above as the origin
    # (not entirely sure what is happening here, but this is what PocketMiner does)
    N_coords, C_coords = res_coords[:,0,:], res_coords[:,2,:]
    N_coords = normalize_vecs(N_coords - alpha_coords, axis=-1)
    C_coords = normalize_vecs(C_coords - alpha_coords, axis=-1)
    bisector = normalize_vecs(N_coords + C_coords, axis=-1)
    perp = normalize_vecs(np.cross(C_coords, N_coords, axis=-1), axis=-1)

    # Compute the sidechain vector (?) features
    # three values per residue for cartesian coordinates
    residueSidechainFeats = -bisector * np.sqrt(1/3) - perp * np.sqrt(2/3)


    ## Residue-level features such as hydrophobicity, charge, etc.
    # A lot of these features are inspired by DGraphDTA
    res_letter_idents = [pd_maps.PROTEIN_INT_1LETTER_MAP[res] for res in res_idents]

    include_dicts = [pd_maps.AA_WEIGHTS, 
                     pd_maps.AA_PKAS, pd_maps.AA_PKBS, pd_maps.AA_PKCS,
                     pd_maps.AA_PKIS, pd_maps.AA_HYDROPHOB,
                     pd_maps.AA_ALIPHATIC, pd_maps.AA_AROMATIC,
                     pd_maps.AA_ACIDIC, pd_maps.AA_BASIC,
                     pd_maps.AA_POLAR_NEUTRAL] 

    # List comprehension for all of these residues and their features
    if include_aa_props:
        residueAAFeats = [[d[res] for d in include_dicts] for res in res_letter_idents]
        residueAAFeats = np.asarray(residueAAFeats, dtype=np.float32)
    else:
        residueAAFeats = np.empty((len(res_idents), 0), dtype=np.float32)


    # Each residue has an index from the beginning of the protein
    # We will do this in a directed sense so that directionality is handled distinctly
    # (from the destination index - the source index)
    n_residues = len(res_idents)
    res_inds = np.arange(n_residues)


    # Posititionally encode this using a cosine and sine function
    # we use n_embeds sin and n_embeds cos embeddings, with frequencies defined by the embedding number
    # much like PositionalEncodings from PocketMiner 
    # (and the original Attention is All You Need paper)
    if add_residue_posenc:
        residuePosEncFeats = calc_pos_encoding(res_inds, n_embeds=8)
    else:
        residuePosEncFeats = np.empty((n_residues, 0), dtype=np.float32)


    # Include ESM2 embeddings if requested
    # (We use the smallest model for speed and memory reasons)
    # but this produces 320 features per residue (a lot)
    if(add_esm2_embeds):
        torch.hub.set_dir('./ext-packages/esm2')
        esm_model, esm_alpha = torch.hub.load('facebookresearch/esm', 'esm2_t6_8M_UR50D', verbose=False)
        batch_converter = esm_alpha.get_batch_converter()
        esm_model.eval()

        # Get the sequence from the residue identities
        # and parse into a format that ESM2 can use
        # TODO: handle unknown residues (X) - maybe by replacing with <mask> token?
        res_seq = ''.join(res_letter_idents)
        res_data = [('protein1', res_seq)]
        _, _, res_tokens = batch_converter(res_data)

        # Get the ESM2 embeddings
        # Use CPU for this since we are only doing one protein
        # and we process multiple proteins at the same time
        with torch.no_grad():
            with torch.autocast('cpu'):
                esm_out = esm_model(res_tokens, repr_layers=[6], return_contacts=False)
                esm_embeds = esm_out["representations"][6]

            # First and last tokens are special tokens and we don't want them
            residueESMEmbeds = esm_embeds[:, 1:-1, :].cpu().numpy().squeeze(axis=0)
    else:
        residueESMEmbeds = np.empty((n_residues, 0), dtype=np.float32)


    ## Concatenate all features based on whether we want a (scalar, Vector) or all-scalar representation
    if vectorize_features:
        residueScalarFeatures = np.concatenate(
            [residueAngleFeats, 
             residueAAFeats,
             residuePosEncFeats,
             residueESMEmbeds],
             axis=-1)
        
        residueVectorFeatures = np.stack(
            [residueForwardFeats,
             residueBackwardFeats, 
             residueSidechainFeats], 
             axis=1)
        
        residueAllFeatures = (residueScalarFeatures, residueVectorFeatures)

        tuple_shape = (residueAllFeatures[0].shape[-1], 
                       residueAllFeatures[1].shape[-2])

        # assert tuple_shape == pd_maps.NUM_RESIDUE_FEATURES, f"Expected {pd_maps.NUM_RESIDUE_FEATURES} features, got {tuple_shape}.\n \
        # Change the NUM_RESIDUE_FEATURES constant in protein_definitions.py if the current number of features is correct."

    else:
        residueAllFeatures = np.concatenate(
            [residueAngleFeats, 
             residueAAFeats,
             residuePosEncFeats,
             residueESMEmbeds,
             residueForwardFeats,
             residueBackwardFeats, 
             residueSidechainFeats], 
             axis=-1)

        # assert residueAllFeatures.shape[-1] == pd_maps.NUM_RESIDUE_FEATURES, f"Expected {pd_maps.NUM_RESIDUE_FEATURES} features, got {residueAllFeatures.shape[-1]}.\n \
        # Change the NUM_RESIDUE_FEATURES constant in protein_definitions.py if the current number of features is correct."

    return residueAllFeatures


def compute_residue_edge_features(res_coords, res_idents, 
                                  edge_thresh, thresh_type, keep_self_loops,
                                  vectorize_features):
    """
    Take in residue coordinates and indices for a single protein to produce 
    a set of edge features for every possible residue-residue edge
    Heavily inspired by `StructuralFeatures` from here: https://github.com/Mickdub/gvp/blob/pocket_pred/src/models.py
    (PocketMiner codebase)
    """
    # res_coords [n_residues, 4, 3]
    # res_idents [n_residues]


    # We will ultimately compute the same features as PocketMiner
    # but store it in a different way (using PyG data structures)
    # and we will precompute these features while processing the data
    # so that we can use them in the training loop without needing to recompute every time

    ## Edge distances

    # Get the geometric distance between each residue's alpha carbons
    # the alpha carbon is always the second set of coordinates for each residue
    # normalized by the number of residues
    # Note: we don't use THIS as a feature, just the RBFs of it
    alpha_coords = res_coords[:, 1, :]

    edgeDistFeats = sp_dist.squareform(sp_dist.pdist(alpha_coords))
    edgeDistFeats = np.expand_dims(edgeDistFeats, axis=-1)

    # We include a Gaussian radial basis function of this distance as a feature
    # Based on the PocketMiner implementation (and the original GVP paper)
    # where the distance between the means represents sqrt(2) * sigma in the RBF equation
    D_min, D_max, D_count = 0., 20., 16
    D_step = (D_max - D_min) / D_count
    D_mu = np.linspace(D_min, D_max, D_count)
    D_mu = np.reshape(D_mu, [1, 1, -1])
    edgeRBFFeats = np.exp(-np.square((edgeDistFeats - D_mu) / (D_step)))


    ## Edge directions

    # Get the direction vectors between each residue's alpha carbons in a pairwise manner
    # (this is the direction of the edge between each pair of residues)
    edgeDirFeats = normalize_vecs(alpha_coords[:,np.newaxis] - alpha_coords[np.newaxis,:], axis=-1)
    

    ## Edge positional encodings of residue distances (based on sequence)
    
    # Each residue has an index from the beginning of the protein
    # We will do this in a directed sense so that directionality is handled distinctly
    # (from the destination index - the source index)
    n_residues = len(res_idents)
    res_inds = np.arange(n_residues)
    res_ind_diffs = res_inds[np.newaxis, :] - res_inds[:, np.newaxis]


    # Posititionally encode residue differences using a cosine and sine function
    edgePosEncFeats = calc_pos_encoding(res_ind_diffs, n_embeds=16)

    # # Scale the positional encodings to be between 0 and 1 (add 1, multiply by 0.5)
    # # We do this since it makes (?) the convolutions more stable by making edge attributes consistently
    # # between 0 and 1
    # edgePosEncFeats = (edgePosEncFeats + 1) * 0.5


    ## Concatenate all the edge features
    # row is source residue, column is destination residue
    # Note that we will be pulling out vectors from this later
    edgeAllFeatures = np.concatenate(
        [edgeRBFFeats,  
         edgePosEncFeats, 
         edgeDirFeats],
         axis=-1)

    
    # Remove the diagonal/self-loop edges by setting their respective values to NaN
    # (NaN means they will not be included later)
    # (we also need to redefine edgeDistFeats in case it is used to select the topk below)
    if not keep_self_loops:
        inds = np.arange(n_residues)
        edgeAllFeatures[inds, inds, :] = np.nan
        edgeDistFeats[inds, inds, :] = np.nan

    # Sparsifying the edge features to only include the 
    # features for the top % proportional based on 
    # distance for each residue pair
    # or use a set number of edges to keep for each residue
    # (PocketMiner uses 30 k-nearest neighbors)

    if(edge_thresh is not None):

        # Distance-based approach
        if(thresh_type == 'dist'):
            # Use a distance-based thresholding approach
            # where we keep the residues within a certain distance

            edgeDistMask = (edgeDistFeats <= edge_thresh).astype(np.float32)
            edgeDistMask[edgeDistMask == 0] = np.nan

            # NaN out the features for edges that are not within the threshold
            edgeMaskedFeatures = edgeAllFeatures * edgeDistMask

        # Number of edges-based approaches
        else:
            if(thresh_type == 'prop'):
                # Get the number of edges to keep for each residue
                # as a proportion of the total number of residues
                n_edges = int(np.ceil(edge_thresh * n_residues))
            elif(thresh_type == 'num'):
                # Defined number of edges to keep (k-nearest neighbors)
                # as a static number for each residue
                n_edges = int(edge_thresh)
        
        
            # Get the indices of the top n_edges closest residues
            # for each residue (note we unsqueezed this earlier so we 
            # need to resqueeze it here to make the calculations easy)
            edge_indices = np.argsort(edgeDistFeats.squeeze(), axis=-1)[:, :n_edges]

            # Create a mask to zero out all but the top n_edges
            # for each residue
            edgeMaskedFeatures = np.empty_like(edgeAllFeatures)
            edgeMaskedFeatures.fill(np.nan)

            for i in range(n_residues):
                residue_nearest = edge_indices[i]
                edgeMaskedFeatures[i, residue_nearest, :] = edgeAllFeatures[i, residue_nearest, :]
    
    else:
        # If we are keeping all edges, just return the original features
        edgeMaskedFeatures = edgeAllFeatures

    # Split out the vector features from the scalar features (if indicated)
    # In this case, it will be just the direction vectors
    if vectorize_features:
        edgeScalarFeatures = edgeMaskedFeatures[:,:,:-3]
        
        edgeVectorFeatures = edgeMaskedFeatures[:,:,-3:]
        edgeVectorFeatures = np.expand_dims(edgeVectorFeatures, axis=-2)
        
        edgeMaskedFeatures = (edgeScalarFeatures, edgeVectorFeatures)

        tuple_shape = (edgeMaskedFeatures[0].shape[-1], 
                       edgeMaskedFeatures[1].shape[-2])

        # assert tuple_shape == pd_maps.NUM_EDGE_FEATURES, f"Expected {pd_maps.NUM_EDGE_FEATURES} features, got {tuple_shape}.\n \
        # Change the NUM_EDGE_FEATURES constant in protein_definitions.py if the current number of features is correct."

    else:
        # No change, everything is already concatenated for a scalar form

        # assert edgeMaskedFeatures.shape[-1] == pd_maps.NUM_EDGE_FEATURES, f"Expected {pd_maps.NUM_EDGE_FEATURES} features, got {edgeMaskedFeatures.shape[-1]}.\n \
        # Change the NUM_EDGE_FEATURES constant in protein_definitions.py if the current number of features is correct."

        pass

    return edgeMaskedFeatures


def normalize_vecs(in_vec, axis=None):
    """
    Normalize an array of vectors along a given axis
    """
    norms = np.linalg.norm(in_vec, axis=axis, keepdims=True)
    return np.divide(in_vec, norms, out=np.zeros_like(in_vec), where=norms!=0)


def calc_pos_encoding(indices, n_embeds=16):
    """
    Posititionally encode indices (or any values) 
    using a cosine and sine function.
    We use n_embeds/2 sin and n_embeds/2 cos embeddings, 
    with frequencies defined by the embedding number
    much like PositionalEncodings from PocketMiner 
    (and the original Attention is All You Need paper)
    """

    per_sincos = n_embeds // 2

    # log-transformed version of the
    # canonical equation used in transformer-based models
    enc_freqs = np.exp(2 * np.arange(per_sincos) * -(np.log(10000.0) / per_sincos))
    enc_expand = tuple(np.arange(len(indices.shape)))
    indices_freqs = np.expand_dims(indices, -1) * np.expand_dims(enc_freqs, enc_expand)

    return np.concatenate([np.cos(indices_freqs), np.sin(indices_freqs)], axis=-1)