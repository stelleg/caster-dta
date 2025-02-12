import os, glob, shutil
import gzip
from dataset.process_data import download_pdb_files, create_comp_models
import numpy as np

import requests
import mdtraj as md
import warnings

import ipdb

# Note: this function is very similar to the process in dataset.process_data.process_data, but
# it is easier to handle this separately for the purposes of inference here
def acquire_pdbs(prot_df, pdb_data_dir,
                 verbose_print=False,
                 do_comp_folds=True,
                 require_completeness=False):
    print("Downloading PDB files for the proteins...")
    _ = download_pdb_files(prot_df['id'], prot_df['seq'], prot_df['file'], 
                           overwrite=False, verbose_pdb_dl=verbose_print)


    if do_comp_folds:
        # Make computational predictions for the proteins that weren't found or don't already have a file
        dl_prot_list = glob.glob(os.path.join(pdb_data_dir, '*.pdb'))

        not_dl_prot_bool = ~np.in1d(prot_df['file'], dl_prot_list)
        not_dl_prot_ids = prot_df[not_dl_prot_bool]['id'].values
        missing_prot_df = prot_df[prot_df['id'].isin(not_dl_prot_ids)]

        print(f"\tMissing PDB files for the following proteins:")
        print(missing_prot_df['id'].values)

        # Create the computational models for the missing proteins
        print("Making computational models for the missing proteins using AF2...")
        create_comp_models(missing_prot_df['id'], missing_prot_df['seq'], missing_prot_df['file'],
                           verbose_comp_fold=verbose_print)


    # Check again to see if any are still missing after this and report
    dl_prot_list = glob.glob(os.path.join(pdb_data_dir, '*.pdb'))
    dl_prot_list = [os.path.basename(x).replace('.pdb', '') for x in dl_prot_list]

    not_dl_prot_bool = ~np.in1d(prot_df['id'], dl_prot_list)
    not_dl_prot_ids = prot_df[not_dl_prot_bool]['id'].values
    missing_prot_df = prot_df[prot_df['id'].isin(not_dl_prot_ids)]


    print(f"\tStill missing PDB files for the following proteins:")
    print(missing_prot_df['id'].values)

    if len(missing_prot_df) > 0:
        if require_completeness:
            raise ValueError("Some proteins are still missing (see above), and require_completeness is set to True - exiting")
        else:
            # Subset protein DF to only include the proteins that were successfully downloaded
            print("Subsetting protein DF to only include the proteins that were successfully downloaded...")
            prot_df = prot_df[prot_df['id'].isin(dl_prot_list)]

    return prot_df


# Function modified from this notebook: https://colab.research.google.com/github/paulynamagana/AFDB_notebooks/blob/main/AFDB_API.ipynb
# Will allow one to try to get files from the AlphaFold2 API or a locally-downloaded set of predictions 
# (the latter is here: https://alphafold.ebi.ac.uk/download)
def get_af2_from_uniprot_accession(df, local_predownload="./data/predownloaded_AF_preds",
                                   do_api=True):
    successful_rows = []

    # Your API endpoint and other parameters
    api_endpoint = "https://alphafold.ebi.ac.uk/api/prediction/"

    for idx, row in df.iterrows():
        accession = row['id']
        sequence = row['seq']
        out_file = row['file']

        split_dir, split_base = os.path.split(out_file)
        tmp_outfile = os.path.join(split_dir, f"tmp_{split_base}")
        failed_file = out_file.replace('.pdb', '_failed.txt')

        success_flag = True

        if os.path.exists(out_file):
            print(f"{accession}: PDB already exists. Skipping.")
            successful_rows.append(idx)
            continue

        if os.path.exists(failed_file):
            print(f"{accession}: Previously attempted to find and failed due to sequence mismatch or lack of result.")
            continue

        predl_search = os.path.join(local_predownload, f"AF-{accession}-F1-model_v4.pdb.gz")

        ## Get original folded file (not subset)
        if os.path.exists(predl_search):
            print(f"{accession}: Found local predownloaded file. Copying... ", end="")
            with gzip.open(predl_search, 'rb') as f_in:
                with open(tmp_outfile, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
                
        elif do_api:
            print(f"{accession}: No local predownloaded file found. Querying API... ", end="")
            url = f"{api_endpoint}{accession}"
            # Make the API request
            response = requests.get(url, timeout=10)

            # Check if the request was successful (status code 200)
            if response.status_code != 200:
                print(f"{accession}: Request failed with status code {response.status_code}. ", end="")
                success_flag = False
            else:
                # Process the API response as needed
                result = response.json()
                if len(result) == 0:
                    print(f"{accession}: No prediction available. ", end="")
                    success_flag = False
                else:
                    result = result[0]
                    # result_seq = result['uniprotSequence']
                    dl_url = result['pdbUrl']

                    # Download the PDB file to the appropriate location
                    response = requests.get(dl_url, timeout=10)

                    with open(tmp_outfile, 'wb') as f:
                        f.write(response.content)
        else:
            print(f"{accession}: No local predownloaded file found and API querying is disabled. Skipping (will not save flag file)... ")
            success_flag = False
            continue

        ## If got original file, try to subset it to match the sequence needed (only starting position important)
        if success_flag:
            # Load the content using mdtraj and get sequence
            # use warnings to ignore the warnings about unlikely unit cell vectors
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                traj = md.load(tmp_outfile)
            topo = traj.top
            md_seq = topo.to_fasta()[0]

            # Get the offset of the sequence 
            # (assuming that the required sequence is fully contained in the PDB)
            offset = md_seq.find(sequence)

            # If the sequence is not found, print and delete the downloaded file
            # and move on to the next protein
            if(offset == -1):
                print(f"{accession}: Required sequence not found in the PDB (may have a mutation). ", end="")
                success_flag = False            
            else:
                # Select only the part of the PDB that corresponds to the sequence we want
                # since the AF2 file might have a longer sequence
                sub_inds = topo.select(f"protein and resid>={offset}")
                sub_traj = traj.atom_slice(sub_inds)

                sub_fasta = sub_traj.top.to_fasta()[0]

                if sub_fasta != sequence:
                    if sequence not in sub_fasta:
                        print(f"{accession}: Sequence mismatch between PDB and AF2 prediction. ", end="")
                        success_flag = False
                    else:
                        print(f"{accession}: AF2 prediction has extra residues. Will save with them and continue. ", end="")
        
        if success_flag:
            # Save the subsetted PDB file (and delete the temporary file)
            sub_traj.save(out_file)
            print(f"{accession}: PDB saved successfully")
            successful_rows.append(idx)
        else:
            print("Failed to download and acquire sequence as requested from AF2 API or from local copy.")
            # Save a record of the failed download to skip in future
            with open(failed_file, 'w') as f:
                f.write("Failed to download and acquire sequence as requested from AF2 API or from local copy")

        if os.path.isfile(tmp_outfile):    
            os.remove(tmp_outfile)

    success_df = df.iloc[successful_rows]

    return success_df
