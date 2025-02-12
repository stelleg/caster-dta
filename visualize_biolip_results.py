import pandas as pd
import numpy as np
import scipy.stats as stats
import os, sys

import matplotlib.pyplot as plt

from inference.visualization_utils import draw_mol_with_attn, draw_protein_with_attn

from rdkit import Chem

import ipdb

# Directory for results (output)
output_dir = "./biolip_results/1107base_2protconv_2molconv_withprotselfloop_withmolselfloop_bdb_9/visualizations"
os.makedirs(output_dir, exist_ok=True)

# Location for results dataframe (from test_biolip_binding.py)
df_pickle = "./biolip_results/1107base_2protconv_2molconv_withprotselfloop_withmolselfloop_bdb_9/biolip_raw_results.pkl"
results_df = pd.read_pickle(df_pickle)

# Score type to use
# attention for values from attention matrix; explanation for values from GNNExplainer
base_score = 'attention' if len(sys.argv) == 1 else sys.argv[1]

# Whether to overwrite existing protein visualization files
# (Molecule visualizations are cheap, so they are always overwritten)
overwrite_existing = False

# Number of bins in histograms
num_bins = 50

# Top cutoff for plotting (most difference in attention)
# Done for each of scaled and unscaled (may have overlap, so not always 2 * top_cutoff results)
top_cutoff = 20


## END USER INPUTS ##

output_dir = os.path.join(output_dir, base_score)
os.makedirs(output_dir, exist_ok=True)

print(f"Using score: {base_score} and outputting to {output_dir}")


# Process binding residue column to be a list of integers
# and make a new column with the binding residue identities
# (defined by BioLIP)
results_df['binding_residues'] = results_df['binding_residues'].str.split(' ')
results_df['binding_residue_pos'] = results_df['binding_residues'].apply(lambda x: [int(y[1:]) for y in x if y != ''])
results_df['binding_residue_ident'] = results_df['binding_residues'].apply(lambda x: [y[0] for y in x if y != ''])

def get_idx_residues(row):
    prot_attn = row[f'protein_{base_score}']
    binding_residues = row['binding_residue_pos']
    binding_residue_score = [prot_attn[i-1] for i in binding_residues]
    return np.array(binding_residue_score)

def get_nonidx_residues(row):
    prot_attn = row[f'protein_{base_score}']
    binding_residues = row['binding_residues']
    nonbinding_residue_score = [prot_attn[i-1] for i in range(len(prot_attn)) if i not in binding_residues]
    return np.array(nonbinding_residue_score)

results_df[f'binding_residue_{base_score}'] = results_df.apply(get_idx_residues, axis=1)
results_df[f'nonbinding_residue_{base_score}'] = results_df.apply(get_nonidx_residues, axis=1)

results_df[f'binding_residue_{base_score}_scaled'] = (results_df[f'binding_residue_{base_score}'] * results_df['protein_sequence'].str.len()).apply(lambda x: np.log10(x))
results_df[f'nonbinding_residue_{base_score}_scaled'] = (results_df[f'nonbinding_residue_{base_score}'] * results_df['protein_sequence'].str.len()).apply(lambda x: np.log10(x))


# Get the mean attention for each protein-molecule pair of the binding and nonbinding residues
results_df[f'mean_binding_residue_{base_score}'] = results_df[f'binding_residue_{base_score}'].apply(np.mean)
results_df[f'mean_nonbinding_residue_{base_score}'] = results_df[f'nonbinding_residue_{base_score}'].apply(np.mean)

results_df[f'mean_binding_residue_{base_score}_scaled'] = results_df[f'binding_residue_{base_score}_scaled'].apply(np.mean)
results_df[f'mean_nonbinding_residue_{base_score}_scaled'] = results_df[f'nonbinding_residue_{base_score}_scaled'].apply(np.mean)


# Compute paired t-test for the binding and nonbinding residues (mean attention over binding and nonbinding, compared)
t_stat, p_val = stats.ttest_rel(results_df[f'mean_binding_residue_{base_score}'], 
                                results_df[f'mean_nonbinding_residue_{base_score}'])

print(f"Paired t-test for binding and nonbinding residues: t-statistic = {t_stat}, p-value = {p_val}")

# Histogram of the difference in attention between binding and nonbinding residues
results_df[f'diff_binding_nonbinding_residue_{base_score}'] = results_df[f'mean_binding_residue_{base_score}'] - results_df[f'mean_nonbinding_residue_{base_score}']
bdata = results_df[f'diff_binding_nonbinding_residue_{base_score}']

absmax = np.abs(bdata.max())

ax = bdata.hist(bins=np.linspace(-absmax, absmax, num_bins))
ax.grid(False)
ax.grid(axis='x', linestyle='--', alpha=0.5)
plt.axvline(bdata.mean(), color='k', linestyle='dashed', linewidth=1.2)
plt.text(bdata.mean(), ax.get_ylim()[1], f"  Mean: {bdata.mean():.4f}", verticalalignment='top')

plt.text(0.7, 0.9, f"Paired t-test\nt-statistic = {t_stat:.4f}\np-value = {p_val:.3e}", 
         ha='left', va='top', transform=ax.transAxes)

plt.xlabel("Difference")
plt.ylabel("Frequency")
plt.title("Difference in Average Attention (Binding - Nonbinding)")
plt.savefig(os.path.join(output_dir, f"diff_binding_nonbinding_residue_{base_score}_hist.png"),
            dpi=600)

plt.close()

# Compute paired t-test for the binding and nonbinding residues (scaled mean attention over binding and nonbinding, compared)
t_stat, p_val = stats.ttest_rel(results_df[f'mean_binding_residue_{base_score}_scaled'],
                                results_df[f'mean_nonbinding_residue_{base_score}_scaled'])

print(f"Paired t-test for binding and nonbinding residues (scaled): t-statistic = {t_stat}, p-value = {p_val}")

# Histogram of the difference in log-scaled attention between binding and nonbinding residues
results_df[f'diff_binding_nonbinding_residue_{base_score}_scaled'] = results_df[f'mean_binding_residue_{base_score}_scaled'] - results_df[f'mean_nonbinding_residue_{base_score}_scaled']
bdata = results_df[f'diff_binding_nonbinding_residue_{base_score}_scaled']

absmax = np.abs(bdata.max())

ax = bdata.hist(bins=np.linspace(-absmax, absmax, num_bins))
ax.grid(False)
ax.grid(axis='x', linestyle='--', alpha=0.5)
plt.axvline(bdata.mean(), color='k', linestyle='dashed', linewidth=1.2)
plt.text(bdata.mean(), ax.get_ylim()[1], f"  Mean: {bdata.mean():.4f}", verticalalignment='top')

plt.text(0.7, 0.9, f"Paired t-test\nt-statistic = {t_stat:.4f}\np-value = {p_val:.3e}", 
         ha='left', va='top', transform=ax.transAxes)

plt.xlabel("Difference")
plt.ylabel("Frequency")
plt.title("Difference in Average Log-Scaled Attention (Binding - Nonbinding)")
plt.savefig(os.path.join(output_dir, f"diff_binding_nonbinding_residue_{base_score}_scaled_hist.png"),
            dpi=600)

plt.close()


# Scatterplot of the differences in attention between binding and nonbinding residues 
# based on length of the original protein
# (for both scaled and unscaled)
plt.scatter(results_df['protein_sequence'].str.len(), results_df[f'diff_binding_nonbinding_residue_{base_score}'], alpha=0.5, s=5)
plt.xlabel("Protein Length")
plt.ylabel("Difference in Attention (Binding - Nonbinding)")
plt.title("Difference in Attention vs. Protein Length")
plt.savefig(os.path.join(output_dir, f"diff_binding_nonbinding_residue_{base_score}_vs_protein_length.png"),
            dpi=600)
plt.close()

plt.scatter(results_df['protein_sequence'].str.len(), results_df[f'diff_binding_nonbinding_residue_{base_score}_scaled'], alpha=0.5, s=5)
plt.xlabel("Protein Length")
plt.ylabel("Difference in Log-Scaled Attention (Binding - Nonbinding)")
plt.title("Difference in Log-Scaled Attention vs. Protein Length")
plt.savefig(os.path.join(output_dir, f"diff_binding_nonbinding_residue_{base_score}_scaled_vs_protein_length.png"),
            dpi=600)
plt.close()


## Verify that the residues attention is being pulled from are the right ones
# def get_seq_residue_idx(row):
#     prot_seq = row['protein_sequence']
#     binding_residues = row['binding_residue_pos']
#     binding_residue_seq = [prot_seq[i-1] for i in binding_residues]
#     return binding_residue_seq
# results_df['binding_residue_seq'] = results_df.apply(get_seq_residue_idx, axis=1)



# Get the ones with the largest difference between binding and nonbinding residues
# to plot the attention scores for 
# (both normal scale and log-scaled, separately)
top_inds_unscaled = results_df[f'diff_binding_nonbinding_residue_{base_score}'].argsort()[-top_cutoff:].values
top_inds_scaled = results_df[f'diff_binding_nonbinding_residue_{base_score}_scaled'].argsort()[-top_cutoff:].values

top_inds = np.union1d(top_inds_unscaled, top_inds_scaled)
results_df_top = results_df.iloc[top_inds]


print(f"Making visualizations for {len(results_df_top)} protein-molecule pairs...")
output_dir = os.path.join(output_dir, f"plotted_{base_score}s")
os.makedirs(output_dir, exist_ok=True)



## Use pymol to visualize the attention scores on the proteins and molecules
# Loop through the dataframe for each protein-molecule pair

for idx, row in results_df_top.iterrows():
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



    # Draw the protein with attention of 1 at binding residues and 0 elsewhere
    # as a visualization of the binding residues
    binding_residues = np.array(row['binding_residue_pos']) - 1
    binding_residue_score = np.zeros_like(prot_attn)
    binding_residue_score[binding_residues] = 10.0

    binding_residue_id = f"{prot_id}_binding_residues"
    binding_residue_ret_file = os.path.join(curr_out_folder, f"prot_{mol_id}_{binding_residue_id}.png")

    # Draw the protein with attention (if the file doesn't already exist)
    if not os.path.exists(binding_residue_ret_file) or overwrite_existing:
        draw_protein_with_attn(prot_file, binding_residue_score, binding_residue_id, mol_id, 
                               max_attn=1.0, 
                               curr_out_folder=curr_out_folder)
        print("\tSaved binding-residue protein visualization!")
    else:
        print(f"\tBinding-residue protein visualization {binding_residue_ret_file} already exists. Skipping...")



    # Create a linear plot of residue number vs. attention score, with binding residues highlighted
    plt.scatter(np.arange(len(prot_attn)), prot_attn, label="Non-binding")
    plt.scatter(binding_residues, prot_attn[binding_residues], c='r', label="Binding")
    plt.xlabel("Residue Number")
    plt.ylabel("Attention Score")
    plt.title("Protein Attention Scores")
    plt.legend()

    # Horizontal line at the mean attention score overall and the mean of binding only
    plt.axhline(prot_attn.mean(), linestyle='dashed', linewidth=1.2)
    plt.axhline(prot_attn[binding_residues].mean(), color='r', linestyle='dashed', linewidth=1.2)

    plt.savefig(os.path.join(curr_out_folder, f"prot_{prot_id}_{mol_id}_{base_score}_scores.png"),
                dpi=600)
    
    plt.close()

    print("\tSaved protein unscaled-attention scores plot!")

    # Same as above but for scaled attention
    scaled_prot_attn = prot_attn * len(prot_attn)
    scaled_prot_attn = np.log10(scaled_prot_attn)
    plt.scatter(np.arange(len(scaled_prot_attn)), scaled_prot_attn, label="Non-binding")
    plt.scatter(binding_residues, scaled_prot_attn[binding_residues], c='r', label="Binding")
    plt.xlabel("Residue Number")
    plt.ylabel("Log-Scaled Attention Score")
    plt.title("Protein Log-Scaled Attention Scores")
    plt.legend()

    # Horizontal line at the mean attention score overall and the mean of binding only
    plt.axhline(scaled_prot_attn.mean(), linestyle='dashed', linewidth=1.2)
    plt.axhline(scaled_prot_attn[binding_residues].mean(), color='r', linestyle='dashed', linewidth=1.2)

    plt.savefig(os.path.join(curr_out_folder, f"prot_{prot_id}_{mol_id}_scaled_{base_score}_scores.png"),
                dpi=600)
    
    plt.close()

    print("\tSaved protein scaled-attention scores plot!")


    
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


ipdb.set_trace()