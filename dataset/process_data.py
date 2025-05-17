import os, glob, shutil
import re
import pandas as pd
import numpy as np

import requests
import subprocess, shlex

import hashlib

import torch

import rcsbsearchapi as rcsb 
import Bio.PDB

import ipdb

def process_data(proteins, ligands, affinity=None,
                 data_path='./data/deepdta_data/davis', 
                 known_pdb_ids=None,
                 pdb_dir_name='pdb_files',
                 overwrite_csv=True, 
                 skip_pdb_dl=False, overwrite_pdb=False, 
                 allow_complexed_pdb=False,
                 create_comp=False,
                 reverse_comp_fold_order=False,
                 verbose_pdb_dl=False,
                 verbose_comp_fold=False):
    """
    Loads the original GraphDTA data for DAVIS and KIBA
    and processes it into a CSV file
    with columns of the following format:
    - protein_id
    - protein_sequence
    - protein_file
    - molecule_id
    - molecule_smiles
    - affinity_score
    """

    if(affinity is None):
        affinity = np.full((len(ligands), len(proteins)), -9999.0)

    # Define a directory for the PDB files
    pdb_dir = os.path.join(data_path, pdb_dir_name)    
            

    # TODO: check if the fully processed CSV already exists, if so, load it and skip this step
    csv_fpath = os.path.join(data_path, 'processed_data_full.csv')
    if os.path.exists(csv_fpath) and not overwrite_csv:
        print(f"Processed data CSV file {csv_fpath} already exists, loading it...")
        full_data = pd.read_csv(csv_fpath)

    else:
        # Process the data into a list of dictionaries
        full_data = []
        for prot_i, (prot_id, prot_seq) in enumerate(proteins.items()):
            for mol_i, (mol_id, mol_smiles) in enumerate(ligands.items()):
                affinity_score = affinity[mol_i][prot_i]

                if(pd.isna(affinity_score)):
                    continue

                cleaned_prot_id = re.sub(r'[^0-9a-zA-Z\-]', '_', prot_id)
                prot_fname = os.path.join(pdb_dir, f'{cleaned_prot_id}.pdb')

                full_data.append({
                    'protein_id': prot_id,
                    'protein_sequence': prot_seq,
                    'protein_file': prot_fname,
                    'molecule_id': mol_id,
                    'molecule_smiles': mol_smiles,
                    'affinity_score': affinity_score
                })

        # Write the processed data to a CSV file
        full_data = pd.DataFrame(full_data)
        full_data.to_csv(csv_fpath, index=False)


    # Make the PDB directory in case it doesn't exist
    os.makedirs(pdb_dir, exist_ok=True)

    prot_cols = ['protein_id', 'protein_sequence', 'protein_file']
    unique_prot_data = full_data.drop_duplicates(subset=prot_cols)[prot_cols]

    if not skip_pdb_dl:
        print(f"Downloading files from PDBank for {len(unique_prot_data)} proteins. Verbosity: {verbose_pdb_dl}")

        # Download the PDB files for the proteins
        # map_fpath = os.path.join(data_path, 'uniprot_mapping_file.dat.gz')
        all_success_dls = download_pdb_files(unique_prot_data['protein_id'], 
                                            unique_prot_data['protein_sequence'], 
                                            unique_prot_data['protein_file'],
                                            known_pdb_ids=known_pdb_ids,
                                            overwrite=overwrite_pdb,
                                            allow_complexed_pdb=allow_complexed_pdb,
                                            verbose_pdb_dl=verbose_pdb_dl)
    
    
    # Now check and see which proteins are actually missing after downloading
    dl_prot_list = glob.glob(os.path.join(pdb_dir, '*.pdb'))

    not_dl_prot_bool = ~np.in1d(unique_prot_data['protein_file'], dl_prot_list)
    not_dl_prot_ids = unique_prot_data[not_dl_prot_bool]['protein_id'].values


    # If we are creating protein structures, then we need to do that here
    if create_comp and len(not_dl_prot_ids) > 0:
        missing_prot_data = unique_prot_data[unique_prot_data['protein_id'].isin(not_dl_prot_ids)]

        print(f"Folding proteins using computational model (AF2) for {len(missing_prot_data)} proteins. Verbosity: {verbose_comp_fold}")

        # Reverse the order of the proteins to fold (useful if running this script twice)
        if(reverse_comp_fold_order):
            missing_prot_data = missing_prot_data.iloc[::-1].reset_index(drop=True)

        # Create the AlphaFold2 models for the proteins
        successful_af2 = create_comp_models(missing_prot_data['protein_id'], 
                                            missing_prot_data['protein_sequence'], 
                                            missing_prot_data['protein_file'],
                                            overwrite=overwrite_pdb,
                                            verbose_comp_fold=verbose_comp_fold)

        # TODO: make sure to remove any proteins that were successfully made with AF2
        # from `not_dl_prot_ids` so that we don't remove them unnecessarily from the below
        dl_prot_list = glob.glob(os.path.join(pdb_dir, '*.pdb'))

        not_dl_prot_bool = ~np.in1d(unique_prot_data['protein_file'], dl_prot_list)
        not_dl_prot_ids = unique_prot_data[not_dl_prot_bool]['protein_id'].values


    if(len(not_dl_prot_ids) > 0):
        print(f"Proteins that are still missing PDB files: {not_dl_prot_ids}")
    

    # Now, we need to remove the proteins that are missing PDB files from the processed data
    # And store the processed data with the PDB files
    final_csv_fpath = os.path.join(pdb_dir, 'processed_data.csv')
    processed_data = full_data[~full_data['protein_id'].isin(not_dl_prot_ids)].reset_index(drop=True)
    processed_data.to_csv(final_csv_fpath, index=False)

    return processed_data


def download_pdb_files(prot_ids, prot_seqs, out_paths, known_pdb_ids=None,
                       overwrite=True, allow_complexed_pdb=False,
                       verbose_pdb_dl=False):
    """
    Download PDB files given a list of IDs
    Will use experimental structures if available (from PDB database)
    otherwise will use AlphaFold2 folds if available (from PDB AlphaFold2 database)
    """
    # Silence print statements if not verbose
    if not verbose_pdb_dl:
        print = lambda *args, **kwargs: None
    else:
        import builtins as __builtin__
        print = __builtin__.print

    all_success_dls = []

    # if we know the pdb_ids we want, just use those
    if known_pdb_ids is not None:
        for prot_id, pdb_id, out_path in zip(prot_ids, known_pdb_ids, out_paths):
            search_record_fpath = out_path.replace('.pdb', '_search_record.txt')

            if os.path.exists(out_path) and not overwrite:
                print(f"PDB file {out_path} already exists, skipping download...")
                all_success_dls.append(prot_id)
                continue

            if os.path.exists(search_record_fpath) and not overwrite:
                print(f"Search record file {search_record_fpath} already exists (PDB file does not). Skipping search and download...")
                continue

            _select_and_download_pdb([pdb_id + "_1"], out_path)

            with open(search_record_fpath, 'w') as f:
                f.write(f"Search options: {pdb_id}_1\n")
                f.write(f"Date: {pd.Timestamp.now()}")
        
    else:
        for prot_id, prot_seq, out_path in zip(prot_ids, prot_seqs, out_paths):
            prot_ver = None

            search_record_fpath = out_path.replace('.pdb', '_search_record.txt')

            if os.path.exists(out_path) and not overwrite:
                print(f"PDB file {out_path} already exists, skipping download...")
                all_success_dls.append(prot_id)
                continue

            if os.path.exists(search_record_fpath) and not overwrite:
                print(f"Search record file {search_record_fpath} already exists (PDB file does not). Skipping search and download...")
                continue

            # Download the PDB file
            print(f"Downloading PDB file for protein {prot_id}. ", end="", flush=True)

            # Perform experimental query using RCSB API and check for acceptable results     
            exp_res = get_rcsb_res(prot_seq, query_type="experimental", allow_complex=allow_complexed_pdb)
            acceptable_results = check_pdb_result(exp_res, res_type='experimental')

            # If no experimental perfect matches, get computational structure instead
            # This will almost certainly be an AlphaFold2-folded protein
            if(len(acceptable_results) == 0):
                print(f"No experimental... ", end="", flush=True)

                # Perform computational query using RCSB API and check for acceptable results
                comp_res = get_rcsb_res(prot_seq, query_type="computational", allow_complex=allow_complexed_pdb)
                acceptable_results = check_pdb_result(comp_res, res_type='computational')

                if(len(acceptable_results) == 0):
                    print(f"No computational - skipping.")
                else:
                    print(f"Computational found.")
                    prot_ver = 'computational'

            else:
                print(f"Experimental found.")
                prot_ver = 'experimental'


            if(len(acceptable_results) > 0):
                chosen_pdb_accession = _select_and_download_pdb(acceptable_results, out_path, prot_ver)
            else:
                chosen_pdb_accession = None

            if(chosen_pdb_accession is not None):
                all_success_dls.append(prot_id)

            print(f"\tOptions: {acceptable_results}")
            print(f"\tChosen: {chosen_pdb_accession}")
            # print(f"\tSeq: {prot_seq}")

            # Save the record that we searched for this protein in the past
            # so we can skip doing so in the future if needed
            # (this will be a simple text file with the options and chosen above
            # as well as the current date and time)
            with open(search_record_fpath, 'w') as f:
                f.write(f"Search options: {acceptable_results}\n")
                f.write(f"Chosen: {chosen_pdb_accession}\n")
                f.write(f"Sequence: {prot_seq}\n")
                f.write(f"Date: {pd.Timestamp.now()}")

        
    return all_success_dls

    
def _select_and_download_pdb(pdb_list, out_path, result_ver='experimental', also_save_accession=True):
    """
    Selects the best PDB file option from a list of possible results
    and then downloads it to the specified output path
    Optionally also include a README of the file accession for each protein
    """

    # If only one option found, then just use that as the "best option"
    if(len(pdb_list) == 1):
        pdb_base_identifier = pdb_list[0]
    else:
        # If more than one option found, then we need to select the best one
        # This will involve checking the sequence identity and length
        # of the structure to the query sequence
        # Likely a different function here for experimental and computational
        
        if(result_ver == 'computational'):
            pdb_base_identifier = _select_computational_pdb(pdb_list)
        elif(result_ver == 'experimental'):
            pdb_base_identifier = _select_experimental_pdb(pdb_list)


    # If computational, we need to look up the AlphaFold2 identifier/source
    # and then download from the AlphaFold database instead
    if(result_ver == 'computational'):
        entry_id, _ = pdb_base_identifier.rsplit('_', 1)
        
        data_api_url = f'https://data.rcsb.org/rest/v1/core/entry/{entry_id}'
        data_api_resp = requests.get(data_api_url)
        data_api_resp.raise_for_status()
        data_api_resp_json = data_api_resp.json()

        af_download = data_api_resp_json['rcsb_comp_model_provenance']['source_url']
        # Need to get the PDB download (not the CIF one, which many of these are)
        af_download = af_download.replace('.cif.gz', '.pdb').replace('.cif', '.pdb')

        # Download the PDB file
        pdb_resp = requests.get(af_download)
        pdb_resp.raise_for_status()

        with open(out_path, 'wb') as f:
            _ = f.write(pdb_resp.content)
    
    elif(result_ver == 'experimental'):
        entry_id, _ = pdb_base_identifier.rsplit('_', 1)

        try:
            # Download the PDB file (if it exists)
            download_url = f'https://files.rcsb.org/download/{entry_id}.pdb'
            pdb_resp = requests.get(download_url)
            pdb_resp.raise_for_status()

            with open(out_path, 'wb') as f:
                _ = f.write(pdb_resp.content)

        except requests.exceptions.HTTPError:
            # Download the mmCIF file instead 
            # and then convert using Biopython
            download_url = f'https://files.rcsb.org/download/{entry_id}.cif'
            pdb_resp = requests.get(download_url)
            pdb_resp.raise_for_status()

            tmp_cif_fpath = out_path.replace('.pdb', '.cif')

            with open(tmp_cif_fpath, 'wb') as f:
                _ = f.write(pdb_resp.content)

            # Convert the CIF file to a PDB file
            cif_parser = Bio.PDB.MMCIFParser()
            structure = cif_parser.get_structure(entry_id, tmp_cif_fpath)

            # There shouldn't be more than 1 chain, but just in case
            # we will assign a SINGLE-CHARACTER chain ID to each chain in the structure
            allow_chain_ids = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            chain_i = 0

            for model in structure:
                for chain in model:
                    chain.id = allow_chain_ids[chain_i]
                    chain_i += 1

            pdb_io = Bio.PDB.PDBIO()
            pdb_io.set_structure(structure)

            pdb_io.save(out_path)

    if(also_save_accession):
        accession_fpath = out_path.replace('.pdb', '_accession.txt')
        with open(accession_fpath, 'w') as f:
            _ = f.write(f"Downloaded from PDB with accession: {pdb_base_identifier}")

    return pdb_base_identifier


def _select_computational_pdb(pdb_list):
    """
    Selects the best computational PDB file option from a list of possible results
    and then downloads it to the specified output path
    """

    best_plddt_score = -float('inf')
    pdb_base_identifier = None

    # For a computational structure, we want to select the one that meets:
    # - Must be an AlphaFold2 structure (faster to check here than to add a query filter)
    # - pLDDT score (based on AlphaFold2), lower is better
    # (add more if needed)

    for pdb_id in pdb_list:
        entry_id, _ = pdb_id.rsplit('_', 1)

        data_api_url = f'https://data.rcsb.org/rest/v1/core/entry/{entry_id}'
        data_api_resp = requests.get(data_api_url)
        data_api_resp.raise_for_status()
        data_api_resp_json = data_api_resp.json()

        source_db = data_api_resp_json['rcsb_comp_model_provenance']['source_db']
        if(source_db != 'AlphaFoldDB'):
            continue

        plddt_score = data_api_resp_json['rcsb_ma_qa_metric_global'][0]['ma_qa_metric_global'][0]['value']
        
        if(plddt_score > best_plddt_score):
            best_plddt_score = plddt_score
            pdb_base_identifier = pdb_id

    return pdb_base_identifier


def _select_experimental_pdb(pdb_list):
    """
    Selects the best experimental PDB file option from a list of possible results
    and then downloads it to the specified output path
    """

    # For an experimental structure, we want to select the one that meets
    # some of the criteria from: https://www.rcsb.org/docs/general-help/assessing-the-quality-of-3d-structures
    # - Modelled residue count (higher is better)
    # - Resolution (lower is better)
    # (add more if needed)
    
    # TODO: consider porting over to PDBe API for more detailed information
    # https://www.ebi.ac.uk/pdbe/api/doc/pdb.html

    pdb_base_identifier = None
    best_modeled_res_count = -float('inf')
    best_reso_score = float('inf')

    for pdb_id in pdb_list:
        entry_id, _ = pdb_id.rsplit('_', 1)

        data_api_url = f'https://data.rcsb.org/rest/v1/core/entry/{entry_id}'
        data_api_resp = requests.get(data_api_url)
        data_api_resp.raise_for_status()
        data_api_resp_json = data_api_resp.json()

        modeled_score = data_api_resp_json['rcsb_entry_info']['deposited_modeled_polymer_monomer_count']
        try:
            resolution_score = np.max(data_api_resp_json['rcsb_entry_info']['resolution_combined'])
        except:
            resolution_score = float('inf')

        # This is not actually a good way of doing this
        # Should probably use numpy and find a nondominated solution or something
        # but for now, this will work (if we add more criteria, this will need to be updated)
        if(modeled_score > best_modeled_res_count):
            best_modeled_res_count = modeled_score
            best_reso_score = resolution_score
            pdb_base_identifier = pdb_id
        elif(modeled_score == best_modeled_res_count):
            if(resolution_score < best_reso_score):
                best_modeled_res_count = modeled_score
                best_reso_score = resolution_score
                pdb_base_identifier = pdb_id

    return pdb_base_identifier



def check_pdb_result(rcsb_res_list, res_type='experimental'):
    """
    Check if the RCSB search result for PDB files 
    that meets the criteria for a "good" structure
    """

    # For both experimental and computational, check the following:
    # 100% sequence identity, 
    # matching lengths (to avoid other ligands/binding things in the structure),
    # ... (add more criteria here as needed)

    good_results = []

    for res in rcsb_res_list:
        seq_id = res['identifier']
        print("checking ", seq_id)

        # Figure out which one is the sequence service type
        res_sequence = None
        for resp_service in res['services']:
            if resp_service['service_type'] == 'sequence':
                res_sequence = resp_service['nodes']

        # Check if the sequence identity is 100%
        seq_id_check = res_sequence[0]['match_context'][0]['sequence_identity'] == 1.0

        # Check if score is 1 
        score_check = res['score'] == 1.0

        # Check if the sequence alignment is perfect (query_length == subject_length)
        length_match_check = res_sequence[0]['match_context'][0]['query_length'] == res_sequence[0]['match_context'][0]['subject_length']

        # Add computational specific checks here (if any)
        if(res_type == 'computational'):
            pass

        # Add experimental specific checks here (if any)
        if(res_type == 'experimental'):
            pass

        # If both checks pass, then add to the list of good results
        if(seq_id_check and length_match_check and score_check):
            good_results.append(seq_id)
    

    #if there are no good results relax the requirement:
    return good_results


def get_rcsb_res(prot_seq, query_type="experimental", allow_complex=False):
    # RCSB attribute query for getting human proteins
    # with an option to get experimental or computational structures

    # Sequence query for the protein sequence (with a 90% sequence identity threshold
    # and an evidence threshold of 1.0) - lenient because we check the results later anyways
    prot_seq_query = rcsb.search.SequenceQuery(prot_seq, 1.0, 0.9)

    # Get entries where at least 80% of the sequence is modeled
    seq_len = len(prot_seq)
    max_unmodeled = int(np.floor(0.2 * seq_len))

    unmodeled_attrib = rcsb.search.AttributeQuery(attribute="rcsb_assembly_info.unmodeled_polymer_monomer_count",
                                                  operator="less_or_equal",
                                                  value=max_unmodeled)


    # Get human proteins (either common name or taxonomy lineage)
    # The latter tends to be more used by computational fold depositions
    human_attrib = rcsb.search.AttributeQuery(attribute="rcsb_entity_source_organism.common_name",
                                              operator="contains_words",
                                              value="human")
    
    homosapiens_attrib = rcsb.search.AttributeQuery(attribute="rcsb_entity_source_organism.taxonomy_lineage.name",
                                                    operator="exact_match",
                                                    value="Homo sapiens")

    # Get proteins that are not bound/complexed with a ligand (small molecule)
    ligand_attrib = rcsb.search.AttributeQuery(attribute="rcsb_entry_info.deposited_nonpolymer_entity_instance_count",
                                                operator="equals",
                                                value=0)
    
    # Proteins that are not bound to another PROTEIN
    protein_sole_attrib = rcsb.search.AttributeQuery(attribute="rcsb_entry_info.deposited_polymer_entity_instance_count",
                                                     operator="equals",
                                                     value=1)
    
    # Proteins that are not multimers of a single chain
    protein_monomer = rcsb.search.AttributeQuery(attribute="rcsb_assembly_info.polymer_entity_instance_count",
                                                 operator="equals",
                                                 value=1)
    
    # Proteins that only have protein polymers (other types of polymers are not useful)
    protein_only_attrib = rcsb.search.AttributeQuery(attribute="rcsb_entry_info.selected_polymer_entity_types",
                                                     operator="exact_match",
                                                     value="Protein (only)")

    
    # Get proteins only made by AlphaFold2 (unused)
    # alphafold_attrib = rcsb.search.AttributeQuery(attribute="rcsb_comp_model_provenance.source_db",
    #                                                 operator="exact_match",
    #                                                 value="AlphaFoldDB")
    

    base_query = prot_seq_query & (human_attrib | homosapiens_attrib)

    if(query_type == 'experimental'):
        # Experimental has a ligand attribute that we want to prune if needed
        # (though we will need to check the full structure separately, too)
        # Note that we always want a single protein chain for the experimental structure
        # and so we will use the ligand_attrib_2 for this - folding changes too much in polypeptides
        full_query = base_query & protein_sole_attrib & protein_monomer & unmodeled_attrib

        # Allow a structure where is binding to a ligand if specified
        if(not allow_complex):
            full_query = prot_seq_query & ligand_attrib & protein_only_attrib

        res = full_query("polymer_entity", return_content_type=["experimental"], results_verbosity='verbose')

    else:
        # Computational results from AlphaFold2 are always ligand-less (so we can drop that for speed)
        # They also always have all the residues modeled (so we can drop that for speed)
        # full_query = base_query
        res = prot_seq_query("polymer_entity", return_content_type=["computational"], results_verbosity='verbose')

    print("query : ", list(res))
    return res 


def create_comp_models(prot_ids, prot_seqs, out_paths, overwrite=True, 
                       model_types=['af2'],
                       also_save_accession=True,
                       verbose_comp_fold=False):
    """
    Create computationally-folded models for the given proteins
    using the specified model type(s) (AlphaFold2, ESM, etc.)
    We will use AlphaFold2 for all proteins not already available
    ESMFold is priority if allowed, only for residues <= 400 in length
    (default ESMFold is disabled since ESM can be inaccurate for some proteins)
    """
    # Silence print statements if not verbose
    if not verbose_comp_fold:
        print = lambda *args, **kwargs: None
    else:
        import builtins as __builtin__
        print = __builtin__.print

    successful_ids = []
    models_used = []

    for prot_id, prot_seq, out_path in zip(prot_ids, prot_seqs, out_paths):
        if os.path.exists(out_path) and not overwrite:
            print(f"Protein model {out_path} already exists, skipping model creation...", flush=True)
            successful_ids.append(prot_id)
            models_used.append("NA")
            continue

        # Create the computational model
        print(f"Creating computational fold for {prot_id} (length {len(prot_seq)}). ", end="", flush=True)

        if('esm' in model_types):
            if(len(prot_seq) <= 400):
                # Use the ESMFold API for the protein
                # (if the sequence is less than 400 residues)
                print(f"Using ESMFold API... ", end="", flush=True)
                esm_api = "https://api.esmatlas.com/foldSequence/v1/pdb/"
                esm_headers = {'Content-Type': 'application/x-www-form-urlencoded'}
                esm_data = prot_seq
                esm_resp = requests.post(esm_api, headers=esm_headers, data=esm_data, verify=False)
                esm_resp.raise_for_status()

                with open(out_path, 'wb') as f:
                    _ = f.write(esm_resp.content)

                if also_save_accession:
                    accession_fpath = out_path.replace('.pdb', '_accession.txt')
                    with open(accession_fpath, 'w') as f:
                        _ = f.write(f"ESMFold_API with sequence of {len(prot_seq)} residues: \n{prot_seq}")

                print("Completed.")
                successful_ids.append(prot_id)
                models_used.append("ESMFold_API")

            else:
                # Use a local ESMFold model for the protein (slower)
                print(f"Using local ESMFoldv1 model... ", end="", flush=True)

                esm_success_flag = _run_esmfoldv1(prot_seq, out_path)

                if(esm_success_flag):
                    print("Completed.")
                    successful_ids.append(prot_id)
                    models_used.append("ESMFold_v1_Local")

                    if also_save_accession:
                        accession_fpath = out_path.replace('.pdb', '_accession.txt')
                        with open(accession_fpath, 'w') as f:
                            _ = f.write(f"ESMFold_v1 with sequence of {len(prot_seq)} residues: \n{prot_seq}")


        
        elif('af2' in model_types):
            # Create an AlphaFold2 model for the protein
            # this will use a set-up Docker container for ColabFold to run it locally
            print(f"Using AlphaFold2... ", end="", flush=True)
            
            # Run the AlphaFold2 model creation
            # (this will use the ColabFold container)
            af2_success_flag = _run_af2(prot_seq, out_path, is_retry=False)

            if not af2_success_flag:
                print("Failed. Retry... ", end="", flush=True)
                af2_success_flag = _run_af2(prot_seq, out_path, is_retry=True)

            if(af2_success_flag):
                print("Completed.", flush=True)
                successful_ids.append(prot_id)
                models_used.append("AlphaFold2")

                if also_save_accession:
                    accession_fpath = out_path.replace('.pdb', '_accession.txt')
                    with open(accession_fpath, 'w') as f:
                        _ = f.write(f"AlphaFold2 with sequence of {len(prot_seq)} residues: \n{prot_seq}")
                
            else:
                print("Failed. Skipping...", flush=True)


    return successful_ids, models_used


def _run_af2(prot_seq, out_path, is_retry=False):
    """
    Run the AlphaFold2 model creation for a given protein sequence
    stored inside of the FASTA file at `temp_fa_fpath`
    using the ColabFold Docker/Singularity container
    """

    def _search_for_result(in_dir):
        """
        Search for the best result in the given directory
        (either relaxed or unrelaxed, with relaxed as priority)
        """

        af2_results = glob.glob(os.path.join(in_dir, f'*_relaxed_rank_001*.pdb'))
        
        if(len(af2_results) == 0):
            af2_results = glob.glob(os.path.join(in_dir, f'*rank_001*.pdb'))

        if(len(af2_results) > 0):
            return sorted(af2_results)[0]
        else:
            return None


    colabfold_dir = './ext-packages/colabfold'

    # Get a unique identifier for the protein sequence
    # (this allows us to skip re-running the same sequence if it appears again)
    prot_id = hashlib.sha256(prot_seq.encode('utf-8')).hexdigest()


    # Run the AlphaFold2 model creation
    # (this will use the ColabFold container)
    # (need to set up the container for this to work)
    # Storing the results in a temporary directory for the run
    # and then moving the results to the final output path
    tmp_dir = f'{colabfold_dir}/tmp/{prot_id}'
    os.makedirs(tmp_dir, exist_ok=True)


    # Check if there are already results
    # If so, copy them and skip the running of AF2
    # (We'll assume the user checked and wanted dupe sequences)
    af2_result = _search_for_result(tmp_dir)


    if(af2_result is not None):
        shutil.copy2(af2_result, out_path)
        return True
    

    # If no results found and this is a retry, then delete the temporary directory
    # and start again from scratch (possible that the previous run downloaded a corrupted MSA search result)
    if(is_retry):
        shutil.rmtree(tmp_dir)
        os.makedirs(tmp_dir, exist_ok=True)
    

    # Create a temporary .fa file for the protein sequence
    temp_fa_fpath = os.path.join(tmp_dir, f'{prot_id}_fasta.fa')
    with open(temp_fa_fpath, 'w') as f:
        _ = f.write(f">{prot_id}\n{prot_seq}")

    # TODO: create the docker equivalent
    af2_cmd = f"singularity run --nv \
        -B {colabfold_dir}/cache:/cache -B ./:/work \
        -B {tmp_dir}:/output/ \
        {colabfold_dir}/colabfold.sif \
        colabfold_batch /work/{temp_fa_fpath} /output/ \
            --num-models 5 \
            --num-recycle 3 \
            --stop-at-score 85 \
            --random-seed 9 \
            --templates \
            --amber \
            --num-relax 1 \
            --relax-max-iterations 2000"
    # --use-gpu-relax # note this causes a lot of issues and often fails
    # Consider using 5 models
    
    split_af2_cmd = shlex.split(af2_cmd)
    af2_proc = subprocess.run(split_af2_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    # Search for result in directory and return if found
    af2_result = _search_for_result(tmp_dir)

    if(af2_result is not None):
        shutil.copy2(af2_result, out_path)
        return True
    else:
        return False
    

def _run_esmfoldv1(prot_seq, out_path):
    """
    Run the ESMFoldv1 model creation for a given protein sequence
    using the torch.hub model for it
    """

    # TODO: Test this - it is currently untested
    # as we haven't properly set up ESMFoldv1 yet

    torch.hub.set_dir('./ext-packages/esmfold')
    esm_model = torch.hub.load('facebookresearch/esm', 'esmfold_v1', verbose=False)
    esm_model.eval()
    if(torch.cuda.is_available()):
        esm_model.cuda()

    try:
        with torch.no_grad():
            output = esm_model.infer_pdb(prot_seq)
        
        with open(out_path, 'wb') as f:
            _ = f.write(output)

        return True

    except:
        return False
