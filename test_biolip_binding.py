import os
import pandas as pd
import torch
import hashlib

from inference.inference_utils import load_model_from_checkpoint, create_dataset_with_checkpoint_params
from inference.evaluation import run_model_on_dataset
from inference.download_utils import get_af2_from_uniprot_accession

from utils import protein_definitions as prot_defs, smiles_definitions as mol_defs

import rdkit

import ipdb

if __name__ == "__main__":
    # Directory for results (output)
    results_dir = "./biolip_results/1107base_2protconv_2molconv_withprotselfloop_withmolselfloop_bdb_9"
    os.makedirs(results_dir, exist_ok=True)

    # Base data directory for cache, PDB, and other data
    base_data_dir = "./data/biolip_data/"
    os.makedirs(base_data_dir, exist_ok=True)

    # Folder from where we define the model and dataset using various files saved during training
    # (will consider creating a torchscripted model for the joint GNN at some point instead)
    model_folder = 'pretrained_model_downstream'

    # BioLIP dataset files
    # https://zhanggroup.org/BioLiP/download/readme.txt
    biolip_file = './data/biolip_data/BioLiP_nr.txt.gz'
    biolip_ligand_file = './data/biolip_data/ligand.tsv.gz'



    ## END USER INPUTS ##


    # BioLIP dataframes, load from files
    biolip_df = pd.read_csv(biolip_file, sep='\t', compression='gzip', header=None)
    biolip_ligand_df = pd.read_csv(biolip_ligand_file, sep='\t', compression='gzip',
                                   on_bad_lines='skip')
    


    # Filter ligand DF to only get SMILES, CCD, and name
    # Drop rows with missing SMILES, and get only first SMILES for each ligand
    biolip_ligand_df['SMILES'] = biolip_ligand_df['SMILES'].str.split('; ')
    biolip_ligand_df = biolip_ligand_df.dropna(subset=['SMILES'])
    # biolip_ligand_smiles_num = biolip_ligand_df['SMILES'].apply(lambda x: len(x) if isinstance(x, list) else 1)
    biolip_ligand_df['SMILES'] = biolip_ligand_df['SMILES'].apply(lambda x: x[0] if isinstance(x, list) else x)

    biolip_ligand_df = biolip_ligand_df[['#CCD', 'SMILES', 'name']]
    biolip_ligand_df = biolip_ligand_df.rename({
        '#CCD': 'id',
        'SMILES': 'smiles'
    }, axis=1)

    # Get drugs and SMILES to test from dataframe
    # (using only drugs that survive an RDKit check)
    # Consider getting rid of molecules with multiple fragments like 
    # [Tc+].COC(C)(C)C[N+]#[C-].COC(C)(C)C[N+]#[C-]
    # (which is implemented, but ignored for now)

    def validate_smiles(smiles, allow_fragments=False):
        try:
            mol = rdkit.Chem.MolFromSmiles(smiles)
            if mol is not None:
                if allow_fragments:
                    return True
                else:
                    n_frags = rdkit.Chem.GetMolFrags(mol)
                    if len(n_frags) == 1:
                        return True
                
        except Exception as e:
            print(f"Error with SMILES {smiles}: {e}")
        
        return False

    print("Removing drugs with invalid SMILES...")

    orig_len = len(biolip_ligand_df)
    # Remove empty SMILES
    biolip_ligand_df['smiles'] = biolip_ligand_df['smiles'].replace('', pd.NA)
    biolip_ligand_df = biolip_ligand_df.dropna(subset=['smiles'])

    # Remove drugs with invalid SMILES
    biolip_ligand_df['valid_smiles'] = biolip_ligand_df['smiles'].apply(validate_smiles)
    biolip_ligand_df = biolip_ligand_df[biolip_ligand_df['valid_smiles']]
    
    print(f"{len(biolip_ligand_df)}/{orig_len} drugs with valid SMILES.")

    drug_df = biolip_ligand_df.copy()


    # Get protein data from BioLIP dataframe
    # Get only relevant columns from BioLIP dataframe
    biolip_df = biolip_df[[4, 8, 17, 20]]
    biolip_df = biolip_df.rename({
        4: '#CCD',
        8: 'binding_residues',
        17: 'protein_id',
        20: 'seq',
    }, axis=1)

    # Drop proteins without a uniprot ID
    biolip_df = biolip_df.dropna(subset=['protein_id'])
    biolip_df['protein_id'] = biolip_df['protein_id'].str.split(',').apply(lambda x: x[0] if isinstance(x, list) else x)

    # Append a hash of the protein sequence to the protein ID
    # seqhashes = biolip_df['seq'].apply(lambda x: hashlib.md5(x.encode()).hexdigest())
    # biolip_df['protein_id'] = biolip_df['protein_id'] + "__" + seqhashes

    # Select only proteins with less than 1500 residues
    # (to avoid very large proteins that the AF2 database splits into fragments)
    # biolip_seqlen = biolip_df['seq'].apply(len)
    # biolip_df = biolip_df[(biolip_seqlen <= 1500)]

    prot_df = biolip_df.copy()


    # Data/cache directory for AD other data (non-PDB files)
    # like dataset files, results, etc.
    other_data_dir = os.path.join(base_data_dir, "other_data")
    os.makedirs(other_data_dir, exist_ok=True)



    # Load the model and set to evaluation mode
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model_from_checkpoint(model_folder, 
                                       best_model_type='val', 
                                       device=device)
    model.eval()

    print("Model loaded successfully!")


    # Sanity check the model node types and edge types
    # and compare to the dataset node types and edge types
    # (If the model was trained on data with no unknown residues or unknown atoms,
    # then any proteins or drugs with such things need to be removed)
    # (This is because the model will not know how to handle these unknowns)
    n_prot_ntypes = model.protein_gnn.num_ntypes
    n_mol_ntypes = model.molecule_gnn.num_ntypes
    n_prot_etypes = model.protein_gnn.num_etypes
    n_mol_etypes = model.molecule_gnn.num_etypes

    print(f"Model has parameters for {n_prot_ntypes} protein node types and {n_mol_ntypes} molecule node types.")

    # Check if the model has unknown node types
    unk_res_val = prot_defs.PROTEIN_1LETTER_INT_MAP['X']
    if n_prot_ntypes == unk_res_val:
        print("Model was not trained with unknown protein node types. Need to remove proteins with unknown residues.")
        no_unk_res = prot_df['seq'].apply(lambda x: unk_res_val not in [prot_defs.PROTEIN_1LETTER_INT_MAP.get(y, prot_defs.PROTEIN_1LETTER_INT_MAP['X']) for y in x])
        print(f"Removing {(~no_unk_res).sum()} proteins with unknown residues:")
        print(prot_df[~no_unk_res])
        prot_df = prot_df[no_unk_res]
        

    unk_atom_val = mol_defs.SELECT_ATOMICNUM_TO_NTYPE['X']
    if n_mol_ntypes == unk_atom_val:
        print("Model was not trained with unknown molecule node types. Need to remove drugs with unknown atoms.")

        def check_for_unk_atom(smiles):
            mol = rdkit.Chem.MolFromSmiles(smiles)
            for atom in mol.GetAtoms():
                if atom.GetAtomicNum() not in mol_defs.SELECT_ATOMICNUM_TO_NTYPE:
                    return False
            return True
    
        no_unk_atom = drug_df['smiles'].apply(check_for_unk_atom)
        print(f"Removing {(~no_unk_atom).sum()} drugs with unknown atoms:")
        print(drug_df[~no_unk_atom])

        drug_df = drug_df[no_unk_atom]


    # Check if the model has unknown edge types
    # (only bonds are considered here, since residues are always the same edge type)
    unk_bond_val = mol_defs.SMILES_BOND_MAP['X'] + 1 # need to offset by 1 here for self-loops as well
    if n_mol_etypes == unk_bond_val:
        print("Model was not trained with unknown molecule edge types. Need to remove drugs with unknown bonds.")

        def check_for_unk_bond(smiles):
            mol = rdkit.Chem.MolFromSmiles(smiles)
            for bond in mol.GetBonds():
                if str(bond.GetBondType()) not in mol_defs.SMILES_BOND_MAP:
                    return False
            return True

        no_unk_bond = drug_df['smiles'].apply(check_for_unk_bond)
        print(f"Removing {(~no_unk_bond).sum()} drugs with unknown bonds:")
        print(drug_df[~no_unk_bond])
        drug_df = drug_df[no_unk_bond]


    # Merge the BioLIP ligand dataframe with the BioLIP dataframe
    # and this is what is passed later
    prot_df = prot_df.rename({
        'seq': 'protein_sequence',
        '#CCD': 'molecule_id'
    }, axis=1)

    drug_df = drug_df.rename({
        "id": "molecule_id",
        "smiles": "molecule_smiles",
        "name": "molecule_name"
    }, axis=1).drop(columns=['valid_smiles'])
    
    combined_df = prot_df.merge(drug_df, on='molecule_id', how='inner')
    combined_df['affinity_score'] = 0.0



    # Drop any rows with missing values of any kind of the remaining columns
    combined_df = combined_df.dropna()


    # Find the proteins' structure files
    # (will NOT do our own computational folding here, just use the PDB files, 
    # though PDBank does include AF2 prefolded structures which may be picked up)

    # Define some base directories for the data
    # Data directory for PDB files
    data_dir = os.path.join(base_data_dir, "pdb_files")
    os.makedirs(data_dir, exist_ok=True)

    # Get only the first protein for each ID (since we're not doing any folding)
    prot_df_base = combined_df[['protein_id', 'protein_sequence']].drop_duplicates(subset=['protein_id'])
    prot_df_base.rename({
        'protein_id': 'id',
        'protein_sequence': 'seq'
    }, axis=1, inplace=True)

    prot_out_files = [os.path.join(data_dir, f"{x}.pdb") for x in prot_df_base['id']]
    prot_df_base['file'] = prot_out_files
    prot_df_base = prot_df_base.dropna().sort_values(by=['id', 'seq']).reset_index(drop=True)

    # Get the PDB files for the proteins (from AF2 or local download, only using local download for now)
    prot_df_success = get_af2_from_uniprot_accession(prot_df_base, do_api=False)

    # Merge to get the files into the combined dataframe (keeping only proteins with files)
    # Merge on both protein ID and sequence (since we may have multiple sequences for the same ID)
    combined_df = combined_df.merge(prot_df_success, 
                                    left_on=['protein_id', 'protein_sequence'], 
                                    right_on=['id', 'seq'], 
                                    how='inner')
    
    combined_df = combined_df.drop(columns=['id', 'seq'])
    combined_df = combined_df.rename({
        'file': 'protein_file'
    }, axis=1)

    combined_df = combined_df.sort_values(by=['protein_id', 'molecule_id']).reset_index(drop=True)

    # Append a hash of the protein sequence to the protein ID
    seqhashes = combined_df['protein_sequence'].apply(lambda x: hashlib.md5(x.encode()).hexdigest())
    combined_df['protein_id'] = combined_df['protein_id'] + "__" + seqhashes

    # Combine cases where the same protein and molecule has different binding sites
    # aka the same protein-molecule pair
    # (for now, just drop all duplicates including the first one)
    combined_df = combined_df.drop_duplicates(subset=['protein_id', 'molecule_id'], keep=False)

    # Create the dataset object and wrap it in a dataloader
    # Also allowing for reloading if it already exists
    dataset = create_dataset_with_checkpoint_params(combined_df, model_folder, cache_dir=other_data_dir)

    # Now we can loop through the dataloader and make predictions
    # We'll save the results to a dataframe for now
    results_df = combined_df.copy()
    parsed_df = run_model_on_dataset(model, dataset, device=device,
                                     max_batch_size=16)

    results_df = results_df.drop(columns=['affinity_score'])
    results_df = results_df.merge(parsed_df, on=['protein_id', 'molecule_id'], how='left')
    

    # Save the raw results to a file
    results_file = os.path.join(results_dir, 'biolip_raw_results.pkl')
    results_df.to_pickle(results_file)

    ipdb.set_trace()