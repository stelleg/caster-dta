import os
import numpy as np
import pandas as pd
import scipy
import json, pickle, hashlib
from collections import OrderedDict
from dataset import process_data

from rdkit import Chem

import ipdb

def load_dataset(dataset_name, 
                 skip_pdb_dl=True,
                 allow_complexed_pdb=False,
                 create_comp=False,
                 reverse_comp_fold_order=False,
                 verbose_pdb_dl = False,
                 verbose_comp_fold = False,
                 do_mostcommon_filter=False,
                 do_proteinseqdupe_filter=False):
    """
    Function to load the requested dataset. Supports Plinder, Davis, and KIBA currently.
    Loads it into the same form that is provided by the Davis/KIBA datasets
    where one file is a set of proteins and sequences, one file is a set of ligands and SMILES, and one file is
    a matrix of affinity scores between the proteins and ligands in order.
    This is then passed to process_data to get the data in the form needed for the model.
    """
    dataset_name = dataset_name.lower()

    pdb_dir_name = 'pdb_files'
    pdb_dir_name += '_createcomp' if create_comp else '_nocreatecomp'
    pdb_dir_name += '_complex' if allow_complexed_pdb else '_nocomplex'

    print(f"Processing {dataset_name} data now...")


    if dataset_name == 'plinder':
        from dataset import get_plinder

        data_path = f'./data/plinder_data/{dataset_name}' # './data/plinder_data/plinder'
        os.makedirs(os.path.join(data_path, pdb_dir_name), exist_ok=True)

        plinder_df = get_plinder.parse_plinder(data_path, save_to_csv=True, force_reparse=False, 
                                               need_structures=True, dedupe_systems=True)
        
        unique_prots = plinder_df['protein_id'].sort_values().unique()
        unique_ligs = plinder_df['molecule_id'].sort_values().unique()
        affinity = np.empty((len(unique_ligs), len(unique_prots)))
        affinity[:] = np.nan

        # For each protein ID, use the most common sequence provided for it (first one listed if multiple)
        # (this is a bit of a hack, but it's the best we can do with the data)
        proteins = {}
        for prot in unique_prots:
            proteins[prot] = plinder_df[plinder_df['protein_id'] == prot]['protein_sequence'].mode().values[0]

        # For each ligand ID, use the most common SMILES provided for it (first one listed if multiple)
        ligands = {}
        for lig in unique_ligs:
            ligands[lig] = plinder_df[plinder_df['molecule_id'] == lig]['molecule_smiles'].mode().values[0]

        # Fill in the affinity matrix with the values provided
        for _, row in plinder_df.iterrows():
            lig_idx = np.where(unique_ligs == row['molecule_id'])[0][0]
            prot_idx = np.where(unique_prots == row['protein_id'])[0][0]

            affinity[lig_idx, prot_idx] = row['affinity_score']


    elif dataset_name == 'davis' or dataset_name == 'kiba':
        data_path = f'./data/deepdta_data/{dataset_name}' # './data/deepdta_data/kiba' or './data/deepdta_data/davis'

        proteins = json.load(open(os.path.join(data_path, 'proteins.txt')), object_pairs_hook=OrderedDict)
        ligands = json.load(open(os.path.join(data_path, 'ligands_iso.txt')), object_pairs_hook=OrderedDict)
        affinity = pickle.load(open(os.path.join(data_path, 'Y'), 'rb'), encoding='latin1')


    elif dataset_name == 'metz':
        data_path = f'./data/other_data/{dataset_name}' # './data/other_data/metz'

        interaction_df = pd.read_csv(os.path.join(data_path, 'Metz_interaction.csv'))
        protinfo_df = pd.read_csv(os.path.join(data_path, 'prot_info.csv'))

        drug_cols = ['PUBCHEM_SID', 'Canonical_Smiles']
        prot_cols = protinfo_df['name'].tolist()

        cols_to_keep = drug_cols + prot_cols
        interaction_df = interaction_df.dropna(subset=['PUBCHEM_SID', 'Canonical_Smiles'])
        interaction_df = interaction_df[cols_to_keep]

        interaction_df['PUBCHEM_SID'] = interaction_df['PUBCHEM_SID'].astype(int).astype(str)

        drug_info = interaction_df[drug_cols].set_index('PUBCHEM_SID').to_dict(into=OrderedDict)
        ligands = drug_info['Canonical_Smiles']

        prot_info = protinfo_df[['name', 'sequence']].set_index('name').to_dict(into=OrderedDict)
        proteins = prot_info['sequence']

        # Ensure order of rows is the same as the ligands dictionary
        assert interaction_df['PUBCHEM_SID'].tolist() == list(ligands.keys())

        interaction_df = interaction_df.drop(columns=drug_cols)

        # Ensure order of columns is the same as the proteins dictionary
        assert interaction_df.columns.tolist() == list(proteins.keys())

        # Set values to be the affinity scores (with NaNs for missing values and <X being set to NaN)
        interaction_df = interaction_df.replace('<', np.nan, regex=True)
        interaction_df = interaction_df.astype(float)

        # This results in a matrix where the rows are ligands and the columns are proteins
        # 36136 interactions for 1470 ligands and 170 proteins
        affinity = interaction_df.values

    
    elif 'bindingdb' in dataset_name:
        # For bindingdb, we allow for multiple tasks to be parsed, though we
        # only evaluate on Kd for the paper
        # specifically, we allow for kd, ki, ic50, and ec50 to be passed
        task_name = dataset_name.split('_')[-1]
        dataset_name = 'bindingdb'

        if(task_name == 'bindingdb'):
            task_name = 'kd'
            print("\tNo task specified for bindingdb dataset; defaulting to Kd")

        data_path = f'./data/other_data/{dataset_name}' # './data/other_data/bindingdb'

        # We only want the column defined by the task requested by the user (defined by the columns below)
        task_to_col = {
            'kd': 'Kd (nM)',
            'ki': 'Ki (nM)',
            'ic50': 'IC50 (nM)',
            'ec50': 'EC50 (nM)',
        }

        if task_name not in task_to_col:
            raise ValueError(f"Task {task_name} not recognized for BindingDB dataset")
        else:
            affinity_col = task_to_col[task_name]

        bdb_file = 'BindingDB_All.tsv'

        # If the parsed file for this task already exists, load it to avoid reprocessing
        bdb_parsed_file = os.path.join(data_path, f'{bdb_file}_parsed_{task_name}.pkl')
        if os.path.exists(bdb_parsed_file):
            bdb_parsed = pd.read_pickle(bdb_parsed_file)
        else:
            # Parse the BindingDB file            
            # Some lines have a ton of extra columns, so we'll skip them (relatively small number of rows)
            bdb_parsed = pd.read_csv(os.path.join(data_path, bdb_file), sep='\t', on_bad_lines='skip')
            
            bdb_cols_to_keep = ['BindingDB Ligand Name', 'Ligand SMILES', 
                                'Target Name', 'BindingDB Target Chain Sequence',
                                affinity_col]
            
            bdb_parsed = bdb_parsed[bdb_cols_to_keep]

            # Set any affinity values that can't be converted to numeric ones to NaN
            bdb_parsed[affinity_col] = pd.to_numeric(bdb_parsed[affinity_col], errors='coerce')
            
            # Drop any rows with missing values for the affinity data
            bdb_parsed = bdb_parsed.dropna(subset=[affinity_col])

            # Convert the affinity data to log-scale
            bdb_parsed[affinity_col] = -np.log10(bdb_parsed[affinity_col] / 1e9)

            # Drop any non-finite values (usually a value of 0 in the original data)
            bdb_parsed = bdb_parsed[np.isfinite(bdb_parsed[affinity_col])]

            # Make sure all all ligand SMILES create valid RDKit molecules
            # (some are invalid and will cause issues with the model, so we prune these out)
            def validate_smiles(smiles):
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is None:
                        return False
                    return True
                except:
                    return False
                
            val_smiles = bdb_parsed['Ligand SMILES'].apply(validate_smiles)
            bdb_parsed = bdb_parsed[val_smiles]

            # Start handling dupes
            # Start by dropping dupe protein sequences + ligand SMILES pairs and keeping the first
            # (This will incidentally keep the first ligand name and target name as the primary as well)
            bdb_parsed_nodupes = bdb_parsed.drop_duplicates(subset=['Ligand SMILES', 'BindingDB Target Chain Sequence'], keep='first')

            # Get the means of the affinity values for each unique protein-ligand pair
            bdb_parsed_aff_meaned = bdb_parsed.groupby(['Ligand SMILES', 'BindingDB Target Chain Sequence'])[affinity_col].mean().reset_index()
            
            # Merge the meaned affinity values back into the original dataframe
            bdb_parsed_nodupes = bdb_parsed_nodupes.drop(columns=[affinity_col])
            bdb_parsed = bdb_parsed_nodupes.merge(bdb_parsed_aff_meaned, on=['Ligand SMILES', 'BindingDB Target Chain Sequence'], how='left')

            # Sort by ligand name and target name, then reset index
            bdb_parsed = bdb_parsed.sort_values(by=['BindingDB Ligand Name', 'Target Name']).reset_index(drop=True)

            # Add a hash of the ligand smiles to each ligand name (since some names are duplicated with different smiles...)
            bdb_parsed['BindingDB Ligand Name'] = bdb_parsed['BindingDB Ligand Name'] + '__' + bdb_parsed['Ligand SMILES'].apply(lambda x: hashlib.sha1(x.encode("utf-8")).hexdigest())

            # Add a hash of the protein sequence to each protein name (since some names are duplicated with different sequences...)
            bdb_parsed['Target Name'] = bdb_parsed['Target Name'] + '__' + bdb_parsed['BindingDB Target Chain Sequence'].apply(lambda x: hashlib.sha1(x.encode("utf-8")).hexdigest())

            # Save the parsed data to avoid reprocessing in the future
            bdb_parsed.to_pickle(bdb_parsed_file)


        # Initial processing steps for model-specific issues
        # Prune any rows with protein sequences of length < 25
        # (can't search for these in PDB or reliably fold them with AF2 - sequence-based models also suffer)
        bdb_parsed = bdb_parsed[bdb_parsed['BindingDB Target Chain Sequence'].str.len() >= 25]

        # Prune any rows with protein sequences that are >3000 (previously 5000)
        # (AF2 is generally not good at handling these, and they are unlikely to have a good PDB match)
        bdb_parsed = bdb_parsed[bdb_parsed['BindingDB Target Chain Sequence'].str.len() <= 3000]

        # Make sure all sequences are uppercase
        bdb_parsed['BindingDB Target Chain Sequence'] = bdb_parsed['BindingDB Target Chain Sequence'].str.upper()

        # Get unique ligands and proteins
        lig_info = bdb_parsed[['BindingDB Ligand Name', 'Ligand SMILES']].drop_duplicates()
        prot_info = bdb_parsed[['Target Name', 'BindingDB Target Chain Sequence']].drop_duplicates()

        ligands = lig_info.set_index('BindingDB Ligand Name')['Ligand SMILES'].to_dict(into=OrderedDict)
        proteins = prot_info.set_index('Target Name')['BindingDB Target Chain Sequence'].to_dict(into=OrderedDict)

        # Make the interaction  by pivoting the dataframe
        interaction_df = bdb_parsed.drop(columns=['Ligand SMILES', 'BindingDB Target Chain Sequence'])
        interaction_df = interaction_df.pivot(index='BindingDB Ligand Name', columns='Target Name', values=affinity_col)

        # Change the order of the index to match the ligand dictionary
        interaction_df = interaction_df.reindex(index=ligands.keys())

        # Change the order of the columns to match the protein dictionary
        interaction_df = interaction_df[proteins.keys()]

        # Assert that the order of the columns and rows is the same as the dictionaries
        assert interaction_df.columns.tolist() == list(proteins.keys())
        assert interaction_df.index.tolist() == list(ligands.keys())

        # Set the affinity matrix to the values in the dataframe
        # For Kd, this results in a matrix where the rows are ligands and the columns are proteins
        # 47695 interactions for 26409 ligands and 2192 proteins (with 5000 protein sequence maximum)
        # 47569 interactions for 26356 ligands and 2183 proteins (with 3000 protein sequence maximum)
        affinity = interaction_df.values


    else:
        raise ValueError(f"Dataset name {dataset_name} not recognized")


    data_df = process_data.process_data(proteins, ligands, affinity,
                                        data_path, 
                                        pdb_dir_name=pdb_dir_name,
                                        skip_pdb_dl=skip_pdb_dl,
                                        allow_complexed_pdb=allow_complexed_pdb, 
                                        create_comp=create_comp,
                                        reverse_comp_fold_order=reverse_comp_fold_order,
                                        verbose_pdb_dl=verbose_pdb_dl,
                                        verbose_comp_fold=verbose_comp_fold)
    
    # Add the splits column back in if we used PLINDER
    # since it predefines the splits for us
    if dataset_name == 'plinder':
        plinder_df_sub = plinder_df[['protein_id', 'molecule_id', 'affinity_score', 'split']]
        data_df = data_df.merge(plinder_df_sub, on=['protein_id', 'molecule_id', 'affinity_score'], how='left')


    # If any values comprise a huge portion of the dataset
    # downsample them to prevent overfitting
    # (for example, in Davis, 75% of the observations are a filler value of 5.0)
    if(do_mostcommon_filter):
        most_common = data_df['affinity_score'].value_counts()
        most_common = most_common / most_common.sum()
        most_common_thresh = 0.1
        if(most_common.max() > most_common_thresh):
            over_represented = most_common[most_common > most_common_thresh]
            print(f"Removing most common values to prevent overfitting")
            print(f"\t{over_represented.index.tolist()} are overrepresented")
            data_df = data_df[~data_df['affinity_score'].isin(over_represented.index)]
            data_df = data_df.reset_index(drop=True)

    # If any sequences are duplicates across proteins, can use only the sequences from
    # the protein that has the most unique values for protein-molecule affinity and drop the rest
    # This helps mitigate overfitting and ensures that the model is learning from the most diverse set of data
    if(do_proteinseqdupe_filter):
        # Get unique protein IDs and sequences of those protein IDs
        prot_seqs = data_df.groupby('protein_id')['protein_sequence'].first()
        prot_seqs = prot_seqs.reset_index()

        # Find nonunique protein sequences
        dupe_seqs = prot_seqs['protein_sequence'].value_counts()
        dupe_seqs = dupe_seqs[dupe_seqs > 1]

        if(len(dupe_seqs) > 0):
            print(f"Removing proteins with identical sequences to prevent overfitting")
            print(f"{len(dupe_seqs)} sequences are duplicated across proteins")

        # For each sequence, find the protein IDs that have that sequence
        for seq in dupe_seqs.index:
            dupe_prots = prot_seqs[prot_seqs['protein_sequence'] == seq]['protein_id']

            # Find the protein IDs that have the most unique values for protein-molecule affinity
            prot_affinity_counts = data_df[data_df['protein_id'].isin(dupe_prots)].groupby('protein_id')['affinity_score'].nunique()
            best_prot = prot_affinity_counts.idxmax()

            # Drop the other protein IDs
            drop_prots = dupe_prots[dupe_prots != best_prot]
            print(f"\tKeeping {best_prot}; dropping protein IDs: {drop_prots.tolist()}")

            data_df = data_df[~data_df['protein_id'].isin(drop_prots)]


    return data_df, data_path, pdb_dir_name
