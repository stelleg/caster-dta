import os
import pandas as pd
import torch

from inference.inference_utils import load_model_from_checkpoint, create_dataset_with_checkpoint_params
from inference.evaluation import run_model_on_dataset
from inference.load_drugbank import load_drugbank

from utils import protein_definitions as prot_defs, smiles_definitions as mol_defs

import mdtraj as md
import rdkit
import warnings

import ipdb

if __name__ == "__main__":
    # Directory for results (output)
    results_dir = "./ad_results/1107base_2protconv_2molconv_withprotselfloop_withmolselfloop_bdb_9"
    os.makedirs(results_dir, exist_ok=True)

    # Base data directory for cache, PDB, and other data
    base_data_dir = "./data/ad_data/"
    os.makedirs(base_data_dir, exist_ok=True)

    # Folder from where we define the model and dataset using various files saved during training
    # (will consider creating a torchscripted model for the joint GNN at some point instead)
    model_folder = 'pretrained_model_downstream'

    # Drugbank XML or XML.zip file 
    # (only used if not preprocessed before)
    drugbank_file = './data/full database.xml'

    # Flag for only approved (and not withdrawn) drugs
    only_approved_drugs = True


    protids_to_files = {
        # https://www.rcsb.org/structure/6SZF
        'Abeta_6SZF': './data/ad_data/pdb_files/6SZF.pdb',
        # https://www.rcsb.org/structure/1iyt
        'Abeta_1IYT': './data/ad_data/pdb_files/1IYT.pdb',
        # https://www.rcsb.org/structure/1AML
        'Abeta_1AML': './data/ad_data/pdb_files/1AML.pdb',
        # https://www.rcsb.org/structure/7Q4B
        'Abeta_7Q4B': './data/ad_data/pdb_files/7Q4B.pdb',
        
        # https://www.rcsb.org/structure/1OWT
        'APP_1OWT': './data/ad_data/pdb_files/1OWT.pdb',
        # https://www.rcsb.org/structure/2fk3
        'APP_2FK3': './data/ad_data/pdb_files/2FK3.pdb',
        # https://www.rcsb.org/structure/4PWQ
        'APP_4PWQ': './data/ad_data/pdb_files/4PWQ.pdb',
        # https://www.rcsb.org/structure/3NYL
        'APP_3NYL': './data/ad_data/pdb_files/3NYL.pdb',
        # https://www.rcsb.org/structure/1mwp
        'APP_1MWP': './data/ad_data/pdb_files/1MWP.pdb',
        # https://www.rcsb.org/structure/AF_AFP05067F1
        'APP_AFP05067F1': './data/ad_data/pdb_files/AF-P05067-F1-model_v4.pdb',

        # https://www.rcsb.org/structure/1eea
        'AChE_1EEA': './data/ad_data/pdb_files/1EEA.pdb',
        # https://www.rcsb.org/structure/4pqe
        'AChE_4PQE': './data/ad_data/pdb_files/4PQE.pdb',
        # https://www.rcsb.org/structure/AF_AFP04058F1
        'AChE_AFP04058F1': './data/ad_data/pdb_files/AF-P04058-F1-model_v4.pdb',

        # https://www.rcsb.org/structure/5iqp
        'Tau_5IQP': './data/ad_data/pdb_files/5IQP.pdb',
        # https://www.rcsb.org/structure/2mz7
        'Tau_2MZ7': './data/ad_data/pdb_files/2MZ7.pdb',
        # https://www.rcsb.org/structure/AF_AFP27348F1
        'Tau_AFP27348F1': './data/ad_data/pdb_files/AF-P27348-F1-model_v4.pdb',


        # https://www.rcsb.org/structure/6VMZ
        'H5N1Hemagg_6VMZ': './data/ad_data/pdb_files/6VMZ.pdb',
        # https://www.rcsb.org/structure/4mhi
        'H5N1Hemagg_4MHI': './data/ad_data/pdb_files/4MHI.pdb',
        # https://www.rcsb.org/structure/6PD3
        'H5N1Hemagg_6PD3': './data/ad_data/pdb_files/6PD3.pdb',
        # https://www.rcsb.org/structure/6pd5
        'H5N1Hemagg_6PD5': './data/ad_data/pdb_files/6PD5.pdb',
        # https://www.rcsb.org/structure/6pd6
        'H5N1Hemagg_6PD6': './data/ad_data/pdb_files/6PD6.pdb',
        # https://www.rcsb.org/structure/2fk0
        'H5N1Hemagg_2FK0': './data/ad_data/pdb_files/2FK0.pdb',
        # https://www.rcsb.org/structure/3S11
        'H5N1Hemagg_3S11': './data/ad_data/pdb_files/3S11.pdb',
        # https://www.rcsb.org/structure/3s12
        'H5N1Hemagg_3S12': './data/ad_data/pdb_files/3S12.pdb',

        # https://www.rcsb.org/structure/7DXP
        'H5N1Nucleo_7DXP': './data/ad_data/pdb_files/7DXP.pdb',
        # https://www.rcsb.org/structure/2q06
        'H5N1Nucleo_2Q06': './data/ad_data/pdb_files/2Q06.pdb',

        # https://www.rcsb.org/structure/2hty
        'H5N1Neuram_2HTY': './data/ad_data/pdb_files/2HTY.pdb',

        # https://www.rcsb.org/structure/1O8A
        'HTNACE_1O8A': './data/ad_data/pdb_files/1O8A.pdb',

    }

    # Using DrugBank 5.1.12 https://go.drugbank.com/releases/5-1-12
    # Load DrugBank dataframe (premade/from cache if available)
    drugbank_df_file = os.path.join(base_data_dir, 'drugbank.pkl')
    if not os.path.exists(drugbank_df_file):
        print(f"Loading DrugBank data from XML at {drugbank_file}...")
        drugbank_df = load_drugbank(drugbank_file)
        drugbank_df.to_pickle(drugbank_df_file)
        print(f"DrugBank data loaded and saved to cache at {drugbank_df_file}!")
    else:
        print(f"Loading DrugBank data from cache at {drugbank_df_file}...")
        drugbank_df = pd.read_pickle(drugbank_df_file)
        print(f"DrugBank data loaded successfully!")


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

    orig_len = len(drugbank_df)
    # Remove empty SMILES
    drugbank_df['smiles'] = drugbank_df['smiles'].replace('', pd.NA)
    drugbank_df = drugbank_df.dropna(subset=['smiles'])

    # Remove drugs with invalid SMILES
    drugbank_df['valid_smiles'] = drugbank_df['smiles'].apply(validate_smiles)
    drugbank_df = drugbank_df[drugbank_df['valid_smiles']]
    
    print(f"{len(drugbank_df)}/{orig_len} drugs with valid SMILES.")


    # Keep only approved drugs if requested (not withdrawn)
    if only_approved_drugs:
        orig_len = len(drugbank_df)
        
        def approved_drug_check(groups):
            if groups is None:
                return False
            return 'approved' in groups and 'withdrawn' not in groups
        
        drugbank_df = drugbank_df[drugbank_df['groups'].apply(approved_drug_check)]
        print(f"{len(drugbank_df)}/{orig_len} drugs are approved (not withdrawn).")


    # Create dict of drug name to SMILES
    test_drugs = drugbank_df[['name', 'smiles']].set_index('name').to_dict()['smiles']


    # Drugs not in DrugBank (too new or just not captured)
    # or drugs where we want to update the SMILES
    addl_drugs = {
        "Valiltramiprosate": r"CC(C)[C@H](N)C(=O)NCCCS(O)(=O)=O",
    }

    test_drugs.update(addl_drugs)

    # Special drugs to look at (AD-related?)
    # from within the dataframe
    ad_prior_drugs = [
        "Valiltramiprosate",
        "Tacrine",
        "Donepezil",
        "Galantamine",
        "Rivastigmine",
        "Memantine",
        "Epigallocatechin gallate",
    ]
    

    ## END USER INPUTS ##


    # Data/cache directory for AD other data (non-PDB files)
    # like dataset files, results, etc.
    other_data_dir = os.path.join(base_data_dir, "other_data")
    os.makedirs(other_data_dir, exist_ok=True)


    # Convert inputs to dictionaries if not already dicts
    # where the key is the (base)name of the file and the value is the file path
    if not isinstance(protids_to_files, dict):
        protids_to_files = {x: os.path.basename(x) for x in protids_to_files}


    # Get the protein sequences for each of the PDB files presented here
    protid_seq_map = {}
    warnings.filterwarnings("ignore", message="Unlikely unit cell vectors")

    for prot_id, prot_file in protids_to_files.items():
        md_obj = md.load(prot_file)

        md_top_protinds = md_obj.top.select("protein")
        md_obj = md_obj.atom_slice(md_top_protinds)

        res_seq = []

        for residue in md_obj.top.residues:
            res1letter = residue.code
            res_seq.append(res1letter)

        protid_seq_map[prot_id] = ''.join(res_seq)
            
    prot_ids = list(protid_seq_map.keys())
    prot_seqs = list(protid_seq_map.values())
    prot_out_files = [protids_to_files[prot_id] for prot_id in prot_ids]
    prot_df = pd.DataFrame({"id": prot_ids, "seq": prot_seqs, "file": prot_out_files})

    drug_df = pd.DataFrame(test_drugs.items(), columns=['id', 'smiles'])


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


    # We'll create a dataset object for the entire dataset and then use the dataloader
    # First make a dataframe that has all the combinations of proteins and molecules
    prot_df = prot_df.rename({
        "id": "protein_id",
        "seq": "protein_sequence",
        "file": "protein_file"
    }, axis=1)

    drug_df = drug_df.rename({
        "id": "molecule_id",
        "smiles": "molecule_smiles"
    }, axis=1)

    combined_df = prot_df.merge(drug_df, how='cross')
    combined_df['affinity_score'] = 0.0

    combined_df = combined_df.sort_values(by=['protein_id', 'molecule_id']).reset_index(drop=True)

    # Create the dataset object and wrap it in a dataloader
    # Also allowing for reloading if it already exists
    dataset = create_dataset_with_checkpoint_params(combined_df, model_folder, cache_dir=other_data_dir)


    # Now we can loop through the dataloader and make predictions
    # We'll save the results to a dataframe for now
    results_df = combined_df.copy()
    parsed_df = run_model_on_dataset(model, dataset, device=device,
                                     max_batch_size=16,
                                     do_gnn_explainer=False)

    results_df = results_df.drop(columns=['affinity_score'])
    results_df = results_df.merge(parsed_df, on=['protein_id', 'molecule_id'], how='left')

    # Save the raw results to a file
    results_file = os.path.join(results_dir, 'ad_raw_results.pkl')
    results_df.to_pickle(results_file)


    # Find the top k drugs for each protein
    # Note that for KIBA, lower values are better
    # and for Davis, higher values are better
    k_drugs = 10
    topk_affinity = results_df.sort_values('affinity_score').groupby('protein_id').head(k_drugs)

    ipdb.set_trace()