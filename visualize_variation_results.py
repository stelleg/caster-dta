import pandas as pd
import numpy as np
import os, sys

from inference.visualization_utils import draw_mol_with_attn, draw_protein_difference

from rdkit import Chem

import ipdb

# Directory for results (output)
output_dir = "./pgx_results/1107base_2protconv_2molconv_withprotselfloop_withmolselfloop_bdb_9/visualizations"
os.makedirs(output_dir, exist_ok=True)

# Location for results dataframe (from test_dta_variation.py)
df_pickle = "./pgx_results/1107base_2protconv_2molconv_withprotselfloop_withmolselfloop_bdb_9/pgx_delta_results.pkl"
results_df = pd.read_pickle(df_pickle)

# Score type to use
# attention for values from attention matrix; explanation for values from GNNExplainer
base_score = 'attention' if len(sys.argv) == 1 else sys.argv[1]

# Whether to do rows where the drugs match the input only or not
do_match_only = True

# Number of top residues to draw labels for in the difference overlay
top_residues = 10

# Whether to overwrite existing protein visualization files
# (Molecule visualizations are cheap, so they are always overwritten)
overwrite_existing = True

## END USER INPUTS ##

output_dir = os.path.join(output_dir, base_score)
os.makedirs(output_dir, exist_ok=True)

print(f"Using score: {base_score} and outputting to {output_dir}")


if do_match_only:
    results_df = results_df[results_df['match_drug']]



## Use pymol to visualize the attention scores on the proteins and molecules
# Loop through the dataframe for each ref/alt-molecule pairs

for idx, row in results_df.iterrows():
    variant_id = row['variant_id']
    variant_folder = os.path.join(output_dir, variant_id)
    os.makedirs(variant_folder, exist_ok=True)

    refalt_pair_name = f"ref_{row['ref_id']}__alt_{row['alt_id']}"
    variant_refalt_folder = os.path.join(variant_folder, refalt_pair_name)
    os.makedirs(variant_refalt_folder, exist_ok=True)

    molecule_id = row['molecule_id']
    variant_refalt_molecule_folder = os.path.join(variant_refalt_folder, f"{molecule_id}")
    os.makedirs(variant_refalt_molecule_folder, exist_ok=True)

    curr_out_folder = variant_refalt_molecule_folder

    base_id = f"{variant_id}--{refalt_pair_name}--{molecule_id}"

    print(f"Running visualization for {base_id}...")

    ref_pdb = row['ref_file']
    alt_pdb = row['alt_file']

    ref_attn = row[f'ref_prot_{base_score}']
    alt_attn = row[f'alt_prot_{base_score}']

    ref_id = f"ref_prot_{variant_id}--{refalt_pair_name}--{molecule_id}"
    alt_id = f"alt_prot_{variant_id}--{refalt_pair_name}--{molecule_id}"
    diff_id = f"diff_prot_{variant_id}--{refalt_pair_name}--{molecule_id}"

    max_attn = max(max(ref_attn), max(alt_attn))

    ret_file = os.path.join(curr_out_folder, f"grid_{variant_id}_{molecule_id}.png")

    # Draw the protein differences (if the file doesn't already exist)
    if not os.path.exists(ret_file) or overwrite_existing:
        draw_protein_difference(ref_pdb, alt_pdb, ref_attn, alt_attn,
                                base_id, curr_out_folder,
                                scale_attn_by_length=True,
                                logscale_attn=True,
                                n_top_labels=top_residues)
        print("\tSaved protein (grid) visualization!")
    else:
        print(f"\tProtein grid visualization {ret_file} already exists. Skipping...")

    
    # Load RDKit molecule and draw each
    # ref - alt, then 
    # diff - labeled diff
    smiles = row['molecule_smiles']
    mol = Chem.MolFromSmiles(smiles)

    # Highlights based on attention scores of atoms
    ref_mol_attn = row[f'ref_mol_{base_score}']
    alt_mol_attn = row[f'alt_mol_{base_score}']
    diff_mol_attn = np.abs(row[f'delta_mol_{base_score}'])
    top_diff_mol_attn_inds = diff_mol_attn.argsort()[-10:]
    top_diff_mol_attn = np.zeros_like(diff_mol_attn)
    top_diff_mol_attn[top_diff_mol_attn_inds] = diff_mol_attn[top_diff_mol_attn_inds]
    top_diff_mol_attn = 0.25 * top_diff_mol_attn / top_diff_mol_attn.max() # scale to 0-0.25 for alpha values

    # Draw atoms with attention scores as highlights (alpha values)
    # and save the images separately
    draw_mol_with_attn(mol, ref_mol_attn, molecule_id, "ref", curr_out_folder)
    print("\tSaved molecule with ref attn.", end="\r")
    draw_mol_with_attn(mol, alt_mol_attn, molecule_id, "alt", curr_out_folder)
    print("\tSaved molecule with alt attn.", end="\r")
    draw_mol_with_attn(mol, diff_mol_attn, molecule_id, "diff", curr_out_folder)
    print("\tSaved molecule with diff attn.", end="\r")
    draw_mol_with_attn(mol, top_diff_mol_attn, molecule_id, "top_diff", curr_out_folder)
    print("\tSaved molecule with top differences highlighted.", end="\r")

    print("\tSaved molecule visualizations!                          ")

    # ipdb.set_trace()