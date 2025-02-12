import pandas as pd
import numpy as np
import os, sys

from inference.visualization_utils import draw_mol_with_attn, draw_protein_with_attn

from rdkit import Chem

import ipdb

# Directory for results (output)
output_dir = "./ad_results/1107base_2protconv_2molconv_withprotselfloop_withmolselfloop_bdb_9/visualizations"
os.makedirs(output_dir, exist_ok=True)

# Location for results dataframe (from test_protein_binders.py)
df_pickle = "./ad_results/1107base_2protconv_2molconv_withprotselfloop_withmolselfloop_bdb_9/ad_raw_results.pkl"
results_df = pd.read_pickle(df_pickle)

# Score type to use
# attention for values from attention matrix; explanation for values from GNNExplainer
base_score = 'attention' if len(sys.argv) == 1 else sys.argv[1]

# Whether to overwrite existing protein visualization files
# (Molecule visualizations are cheap, so they are always overwritten)
overwrite_existing = False

# Whether to prune out some compounds that are overly identified as binders
# likely due to the model having some issues in out-of-distribution regions of molecular space
prune_suffixes = None

# Whether to only get putative binders per protein 
# (basically the top X percentile molecules per protein)
# NOTE: Davis, Metz, BindingDB are pKd values, so larger is stronger affinity
# KIBA is different and is a score, so smaller is stronger affinity
# which one indicates with "affinity_direction" (which direction of affinity is preferable)
top_percentile_cutoff = 0.995
stronger_affinity_direction = "high" # "high" or "low"


## END USER INPUTS ##

output_dir = os.path.join(output_dir, base_score)
os.makedirs(output_dir, exist_ok=True)

print(f"Using score: {base_score} and outputting to {output_dir}")


if prune_suffixes is not None:
    print(f"Pruning out molecules with suffixes: {prune_suffixes}...")

    for suffix in prune_suffixes:
        has_suffix = results_df['molecule_id'].str.lower().str.endswith(suffix)
        print(f"\tPruning out {has_suffix.sum()} molecules with suffix: {suffix}...")
        results_df = results_df[~has_suffix]


    print(f"\tPruned out molecules with suffixes: {prune_suffixes}!")


keep_drugs = ['rivastigmine', 'donepezil', 'galantamine',
              'oseltamivir', 'zanamivir', 'peramivir']


if top_percentile_cutoff is not None:
    print(f"Filtering for top {top_percentile_cutoff*100}% percentile molecules per protein...")

    if top_percentile_cutoff > 1:
        raise ValueError("Top percentile cutoff must be between 0 and 1!")

    # Get the top X percentile molecules per protein
    results_df['percentile'] = results_df.groupby('protein_id')['affinity_score'].rank(pct=True)

    if stronger_affinity_direction == "low":
        results_df['percentile'] = 1 - results_df['percentile'] # flip for lower score meaning higher affinity

    cutoff_drugs = results_df['percentile'] >= top_percentile_cutoff
    drug_keep_bool = results_df['molecule_id'].isin(keep_drugs)

    results_df = results_df[cutoff_drugs | drug_keep_bool]


    print(f"\tFiltered for top {top_percentile_cutoff*100}% molecules per protein!")


print(f"Making visualizations for {len(results_df)} protein-molecule pairs...")



## Use pymol to visualize the attention scores on the proteins and molecules
# Loop through the dataframe for each protein-molecule pair

for idx, row in results_df.iterrows():
    prot_id = row['protein_id']
    prot_folder = os.path.join(output_dir, prot_id)
    os.makedirs(prot_folder, exist_ok=True)

    mol_id = row['molecule_id']
    prot_mol_folder = os.path.join(prot_folder, mol_id)
    os.makedirs(prot_mol_folder, exist_ok=True)

    curr_out_folder = prot_mol_folder

    base_id = f"{prot_id}--{mol_id}"

    print(f"Running visualization for {base_id}...")

    prot_file = row['protein_file']
    prot_attn = row[f'protein_{base_score}']


    max_prot_attn = max(prot_attn)

    attn_id = mol_id

    unscaled_prot_id = f"{prot_id}_unscaled"
    unscaled_prot_ret_file = os.path.join(curr_out_folder, f"prot_{mol_id}_{unscaled_prot_id}.png")

    # Draw the protein with attention (if the file doesn't already exist)
    if not os.path.exists(unscaled_prot_ret_file) or overwrite_existing:
        draw_protein_with_attn(prot_file, prot_attn, unscaled_prot_id, mol_id, 
                               max_attn=max_prot_attn, 
                               curr_out_folder=curr_out_folder)
        print("\tSaved unscaled-attention protein visualization!")
    else:
        print(f"\tUnscaled-attention protein visualization {unscaled_prot_ret_file} already exists. Skipping...")

    
    # Draw the protein with scaled attention where we scale by the length of the protein
    # and log-transform to get a ratio of how much more attention is paid to a residue
    # compared to a uniform baseline
    scaled_prot_id = f"{prot_id}_scaled"
    scaled_prot_ret_file = os.path.join(curr_out_folder, f"prot_{mol_id}_{scaled_prot_id}.png")

    # NOTE: we do not care if a residue was paid less attention to here (just more)
    # but we will visualize that regardless, just for the sake of completeness
    scaled_prot_attn = prot_attn * len(prot_attn)
    scaled_prot_attn = np.log10(scaled_prot_attn)
    scaled_max_attn = np.abs(scaled_prot_attn).max()
    scaled_min_attn = -scaled_max_attn

    # Draw the protein with attention (if the file doesn't already exist)
    if not os.path.exists(scaled_prot_ret_file) or overwrite_existing:
        draw_protein_with_attn(prot_file, scaled_prot_attn, scaled_prot_id, mol_id, 
                               max_attn=scaled_max_attn, min_attn=scaled_min_attn,
                               curr_out_folder=curr_out_folder)
        print("\tSaved scaled-attention protein visualization!")
    else:
        print(f"\tScaled-attention protein visualization {scaled_prot_ret_file} already exists. Skipping...")


    # Draw the protein with the maximal attention (not scaled by length of protein)
    maximal_prot_id = f"{prot_id}_maximal"
    maximal_prot_ret_file = os.path.join(curr_out_folder, f"prot_{mol_id}_{maximal_prot_id}.png")

    maximal_prot_attn = row[f'max_protein_{base_score}']
    max_maximal_prot_attn = max(maximal_prot_attn)

    # Draw the protein with attention (if the file doesn't already exist)
    if not os.path.exists(maximal_prot_ret_file) or overwrite_existing:
        draw_protein_with_attn(prot_file, maximal_prot_attn, maximal_prot_id, mol_id, 
                               max_attn=max_maximal_prot_attn, 
                               curr_out_folder=curr_out_folder)
        print("\tSaved maximal-attention protein visualization!")

    
    # Load RDKit molecule and draw with attention
    smiles = row['molecule_smiles']
    mol = Chem.MolFromSmiles(smiles)

    # Highlights based on attention scores of atoms
    mol_attn = row[f'molecule_{base_score}']
    scaled_mol_attn = 0.25 * mol_attn / mol_attn.max() # scale to 0-0.25 for alpha values
    maximal_mol_attn = row[f'max_molecule_{base_score}']
    scaled_maximal_mol_attn = 0.25 * maximal_mol_attn / maximal_mol_attn.max() # scale to 0-0.25 for alpha values


    # Draw atoms with attention scores as highlights (alpha values)
    # and save the images separately
    draw_mol_with_attn(mol, mol_attn, mol_id, "unscaled", curr_out_folder)
    print("\tSaved unscaled-attention molecule visualization!")
    draw_mol_with_attn(mol, scaled_mol_attn, mol_id, "scaled", curr_out_folder)
    print("\tSaved scaled-attention molecule visualization!")
    draw_mol_with_attn(mol, maximal_mol_attn, mol_id, "maximal", curr_out_folder)
    print("\tSaved maximal-attention molecule visualization!")
    draw_mol_with_attn(mol, scaled_maximal_mol_attn, mol_id, "scaled_maximal", curr_out_folder)
    print("\tSaved scaled-maximal-attention molecule visualization!")
