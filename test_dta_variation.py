import os
import pandas as pd
import numpy as np
import torch
import hashlib
from functools import partial

from inference.variant_to_protein import variant_to_protein_seq
from inference.inference_utils import load_model_from_checkpoint, create_dataset_with_checkpoint_params
from inference.download_utils import acquire_pdbs
from inference.evaluation import run_model_on_dataset

import ipdb

if __name__ == "__main__":
    # Directory for results (output)
    results_dir = "./pgx_results/1107base_2protconv_2molconv_withprotselfloop_withmolselfloop_bdb_9"
    os.makedirs(results_dir, exist_ok=True)

    # Base data directory for cache, PDB, and other data
    base_data_dir = "./data/pgx_data/"
    os.makedirs(base_data_dir, exist_ok=True)

    # Folder from where we define the model and dataset using various files saved during training
    # (will consider creating a torchscripted model for the joint GNN at some point instead)
    model_folder = 'pretrained_model_downstream'


    # List of genetic variants to test (should be coding)
    # pulled from PharmGKB variant annotations (various)
    # Especially the VIPs = https://www.pharmgkb.org/vips
    test_variants = {
        "SLCO1B1_var1_atorvastatin-simvastatin-pravastatin-rosuvastatin": "rs4149056", 
        "SLCO1B1_var2_atorvastatin-rosuvastatin-methotrexate-pravastatin": "rs2306283",
        "TPMT_var1_mercaptopurine": "rs1800462",
        "VKORC1_var_warfarin": "rs61742245",


        ## Other examples below, though not necessarily VIPs or have too many sequences pulled in
        ## to compare properly with the analysis below (ABCG2 for example has over 20 possibilities)

        # "ABCG2_var_rosuvastatin-methotrexate-sunitinib": "rs2231142",
        # "ADRB1_var_metoprolol": "rs1801253",
        # "CYP2C19_var_clopidogrel": "rs4986893", 
        # "CYP2B6_var_efavirenz": "rs3745274",
        # "CYP2C9_var1_warfarin": "rs1799853",
        # "CYP2C9_var2_warfarin": "rs1057910",
        # "NUDT15_var_mercaptopurine": "rs116855232",
        # "TPMT_var2_mercaptopurine-azathioprine": "rs1800460",
        # "TPMT_var3_mercaptopurine-azathioprine": "rs1142345",
    }


    # List of manually paired sequences to compare/test (added to the set of variants)
    # This should be a dict where the key is some name/identifier for the pair(s) and
    # the value is a dict with ref and alt protein sequences (protein IDs will be the hashes of the sequence) 
    # this is only an example for users if they want to use this feature
    test_sequence_pairs = {
        # rs67666821 (sort of, this is actually a fake termination at the frameshift location, but it should be terminating several positions later instead)
        # midazolam, risperidone, tacrolimus
        # "CYP3A4_var_midazolam-risperidone-tacrolimus": {
        #     "ref_seq": "MALIPDLAMETWLLLAVSLVLLYLYGTHSHGLFKKLGIPGPTPLPFLGNILSYHKGFCMFDMECHKKYGKVWGFYDGQQPVLAITDPDMIKTVLVKECYSVFTNRRPFGPVGFMKSAISIAEDEEWKRLRSLLSPTFTSGKLKEMVPIIAQYGDVLVRNLRREAETGKPVTLKDVFGAYSMDVITSTSFGVNIDSLNNPQDPFVENTKKLLRFDFLDPFFLSITVFPFLIPILEVLNICVFPREVTNFLRKSVKRMKESRLEDTQKHRVDFLQLMIDSQNSKETESHKALSDLELVAQSIIFIFAGYETTSSVLSFIMYELATHPDVQQKLQEEIDAVLPNKAPPTYDTVLQMEYLDMVVNETLRLFPIAMRLERVCKKDVEINGMFIPKGVVVMIPSYALHRDPKYWTEPEKFLPERFSKKNKDNIDPYIYTPFGSGPRNCIGMRFALMNMKLALIRVLQNFSFKPCKETQIPLKLSLGGLLQPEKPVVLKVESRDGTVSGA",
        #     "alt_seq": "MALIPDLAMETWLLLAVSLVLLYLYGTHSHGLFKKLGIPGPTPLPFLGNILSYHKGFCMFDMECHKKYGKVWGFYDGQQPVLAITDPDMIKTVLVKECYSVFTNRRPFGPVGFMKSAISIAEDEEWKRLRSLLSPTFTSGKLKEMVPIIAQYGDVLVRNLRREAETGKPVTLKDVFGAYSMDVITSTSFGVNIDSLNNPQDPFVENTKKLLRFDFLDPFFLSITVFPFLIPILEVLNICVFPREVTNFLRKSVKRMKESRLEDTQKHRVDFLQLMIDSQNSKETESHKALSDLELVAQSIIFIFAGYETTSSVLSFIMYELATHPDVQQKLQEEIDAVLPNKAPPTYDTVLQMEYLDMVVNETLRLFPIAMRLERVCKKDVEINGMFIPKGVVVMIPSYALHRDPKYWTEPEKFLPERFSKKNKDNIDPYIYTPFGSGPRNCIGMRFALMNMKLALIRVLQNFSFKPCKETQIPLKLSLGGLLQPEKT"
        # },

        # # KIT imatinib V654A mutation
        # # https://cancer.sanger.ac.uk/cosmic/mutation/overview?id=189370265
        # "KIT_var_imatinib": {
        #     "ref_seq": "MRGARGAWDFLCVLLLLLRVQTGSSQPSVSPGEPSPPSIHPGKSDLIVRVGDEIRLLCTDPGFVKWTFEILDETNENKQNEWITEKAEATNTGKYTCTNKHGLSNSIYVFVRDPAKLFLVDRSLYGKEDNDTLVRCPLTDPEVTNYSLKGCQGKPLPKDLRFIPDPKAGIMIKSVKRAYHRLCLHCSVDQEGKSVLSEKFILKVRPAFKAVPVVSVSKASYLLREGEEFTVTCTIKDVSSSVYSTWKRENSQTKLQEKYNSWHHGDFNYERQATLTISSARVNDSGVFMCYANNTFGSANVTTTLEVVDKGFINIFPMINTTVFVNDGENVDLIVEYEAFPKPEHQQWIYMNRTFTDKWEDYPKSENESNIRYVSELHLTRLKGTEGGTYTFLVSNSDVNAAIAFNVYVNTKPEILTYDRLVNGMLQCVAAGFPEPTIDWYFCPGTEQRCSASVLPVDVQTLNSSGPPFGKLVVQSSIDSSAFKHNGTVECKAYNDVGKTSAYFNFAFKGNNKEQIHPHTLFTPLLIGFVIVAGMMCIIVMILTYKYLQKPMYEVQWKVVEEINGNNYVYIDPTQLPYDHKWEFPRNRLSFGKTLGAGAFGKVVEATAYGLIKSDAAMTVAVKMLKPSAHLTEREALMSELKVLSYLGNHMNIVNLLGACTIGGPTLVITEYCCYGDLLNFLRRKRDSFICSKQEDHAEAALYKNLLHSKESSCSDSTNEYMDMKPGVSYVVPTKADKRRSVRIGSYIERDVTPAIMEDDELALDLEDLLSFSYQVAKGMAFLASKNCIHRDLAARNILLTHGRITKICDFGLARDIKNDSNYVVKGNARLPVKWMAPESIFNCVYTFESDVWSYGIFLWELFSLGSSPYPGMPVDSKFYKMIKEGFRMLSPEHAPAEMYDIMKTCWDADPLKRPTFKQIVQLIEKQISESTNHIYSNLANCSPNRQKPVVDHSVRINSVGSTASSSQPLLVHDDV",
        #     "alt_seq": "MRGARGAWDFLCVLLLLLRVQTGSSQPSVSPGEPSPPSIHPGKSDLIVRVGDEIRLLCTDPGFVKWTFEILDETNENKQNEWITEKAEATNTGKYTCTNKHGLSNSIYVFVRDPAKLFLVDRSLYGKEDNDTLVRCPLTDPEVTNYSLKGCQGKPLPKDLRFIPDPKAGIMIKSVKRAYHRLCLHCSVDQEGKSVLSEKFILKVRPAFKAVPVVSVSKASYLLREGEEFTVTCTIKDVSSSVYSTWKRENSQTKLQEKYNSWHHGDFNYERQATLTISSARVNDSGVFMCYANNTFGSANVTTTLEVVDKGFINIFPMINTTVFVNDGENVDLIVEYEAFPKPEHQQWIYMNRTFTDKWEDYPKSENESNIRYVSELHLTRLKGTEGGTYTFLVSNSDVNAAIAFNVYVNTKPEILTYDRLVNGMLQCVAAGFPEPTIDWYFCPGTEQRCSASVLPVDVQTLNSSGPPFGKLVVQSSIDSSAFKHNGTVECKAYNDVGKTSAYFNFAFKGNNKEQIHPHTLFTPLLIGFVIVAGMMCIIVMILTYKYLQKPMYEVQWKVVEEINGNNYVYIDPTQLPYDHKWEFPRNRLSFGKTLGAGAFGKVVEATAYGLIKSDAAMTVAVKMLKPSAHLTEREALMSELKVLSYLGNHMNIANLLGACTIGGPTLVITEYCCYGDLLNFLRRKRDSFICSKQEDHAEAALYKNLLHSKESSCSDSTNEYMDMKPGVSYVVPTKADKRRSVRIGSYIERDVTPAIMEDDELALDLEDLLSFSYQVAKGMAFLASKNCIHRDLAARNILLTHGRITKICDFGLARDIKNDSNYVVKGNARLPVKWMAPESIFNCVYTFESDVWSYGIFLWELFSLGSSPYPGMPVDSKFYKMIKEGFRMLSPEHAPAEMYDIMKTCWDADPLKRPTFKQIVQLIEKQISESTNHIYSNLANCSPNRQKPVVDHSVRINSVGSTASSSQPLLVHDDV"
        # }
        
    }


    # Get some drug SMILES to test here
    # A lot are from here: https://www.cureffi.org/wp-content/uploads/2013/10/drugs.txt
    # https://www.cureffi.org/2013/10/04/list-of-fda-approved-drugs-and-cns-drugs-with-smiles/
    test_drugs = {
        "Rosuvastatin": r"O[C@@H](C[C@H](CC(=O)O)O)/C=C/c1c(nc(nc1c1ccc(cc1)F)N(S(=O)(=O)C)C)C(C)C",
        "Methotrexate": r"OC(=O)CC[C@@H](C(=O)O)NC(=O)c1ccc(cc1)N(Cc1cnc2c(n1)c(N)nc(n2)N)C",
        "Sunitinib": r"CCN(CCNC(=O)c1c(C)[nH]c(c1C)/C=C/1\C(=O)Nc2c1cc(F)cc2)CC",
        "Metoprolol": r"COCCc1ccc(cc1)OCC(CNC(C)C)O",
        "Clopidogrel": r"COC(=O)[C@H](c1ccccc1Cl)N1CCc2c(C1)ccs2",
        "Efavirenz": r"FC([C@@]1(C#CC2CC2)OC(=O)Nc2c1cc(Cl)cc2)(F)F",
        "Warfarin": r"CC(=O)C[C@@H](C1=CC=CC=C1)C2=C(C3=CC=CC=C3OC2=O)O",
        "Mercaptopurine": r"Sc1ncnc2c1nc[nH]2",
        "Atorvastatin": r"O[C@@H](C[C@H](CC(=O)O)O)CCn1c(C(C)C)c(c(c1c1ccc(cc1)F)c1ccccc1)C(=O)Nc1ccccc1",
        "Pravastatin": r"CC[C@@H](C(=O)O[C@H]1C[C@H](O)C=C2[C@H]1[C@@H](CC[C@H](C[C@H](CC(=O)O)O)O)[C@H](C=C2)C)C",
        "Azathioprine": r"Cn1cnc(c1Sc1ncnc2c1nc[nH]2)N(=O)=O",
        "Midazolam": r"Clc1ccc2c(c1)C(=NCc1n2c(C)nc1)c1ccccc1F",
        "Risperidone": r"Fc1ccc2c(c1)onc2C1CCN(CC1)CCc1c(C)nc2n(c1=O)CCCC2",
        "Tacrolimus": r"C=CC[C@@H]1/C=C(\C)/C[C@H](C)C[C@H](OC)[C@H]2O[C@](O)([C@@H](C[C@@H]2OC)C)C(=O)C(=O)N2[C@H](C(=O)O[C@@H]([C@@H]([C@H](CC1=O)O)C)/C(=C/[C@@H]1CC[C@H]([C@@H](C1)OC)O)/C)CCCC2",

    }
    

    ## END USER INPUTS ##


    # Define some base directories for the data
    # Data directory for PGx PDB files
    data_dir = os.path.join(base_data_dir, "pdb_files")
    os.makedirs(data_dir, exist_ok=True)

    # Data/cache directory for PGx other data (non-PDB files)
    # like dataset files, results, etc.
    other_data_dir = os.path.join(base_data_dir, "other_data")
    os.makedirs(other_data_dir, exist_ok=True)


    # Convert inputs to dictionaries if not already dicts
    # where the key is the name of the variant/drug and the value is the variant/drug SMILES itself
    if not isinstance(test_variants, dict):
        test_variants = {x: x for x in test_variants}

    if not isinstance(test_drugs, dict):
        test_drugs = {x: x for x in test_drugs}

    # Convert variants to get different protein sequences
    # 1) the original protein sequence (before variation) - note there may be multiple
    # 2) the variant protein sequence (after variation) - note there may be multiple
    # So that we can compare the two to each other based on the model outputs
    # Will pull existing results from the other data directory if they exist instead of requerying
    print("Getting protein sequences for the variants...")
    var_protid_map, protid_seq_map = variant_to_protein_seq(test_variants, other_data_dir,
                                                            force_requery=False)
    print("\tProtein sequences obtained!")


    # Add manually indicated sequences to the protein sequence map
    # and to the variant to protein ID map
    for k, v in test_sequence_pairs.items():
        ref_id = hashlib.sha256(v['ref_seq'].encode()).hexdigest()
        alt_id = hashlib.sha256(v['alt_seq'].encode()).hexdigest()

        protid_seq_map[ref_id] = v['ref_seq']
        protid_seq_map[alt_id] = v['alt_seq']

        var_protid_map[k] = {
            "ref_id": [ref_id],
            "alt_id": [alt_id]
        }


    # Download the PDB files for the proteins if available on PDB
    prot_ids = list(protid_seq_map.keys())
    prot_seqs = list(protid_seq_map.values())
    prot_out_files = [os.path.join(data_dir, f"{x}.pdb") for x in protid_seq_map.keys()]
    prot_df_base = pd.DataFrame({"id": prot_ids, "seq": prot_seqs, "file": prot_out_files})

    prot_df = acquire_pdbs(prot_df_base, data_dir, require_completeness=False)

    # Now, we take the successful PDB files and create a dataset for the combinations of proteins and molecules
    # We'll create a dataset object for the entire dataset and then use the dataloader

    # First make a dataframe that has all the combinations of proteins and molecules
    drug_df = pd.DataFrame(test_drugs.items(), columns=['id', 'smiles'])

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

    # Load the model and set to evaluation mode
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model_from_checkpoint(model_folder, 
                                       best_model_type='val', 
                                       device=device)
    model.eval()

    print("Model loaded successfully!")


    # Now we can loop through the dataloader and make predictions
    # We'll save the results to a dataframe for now
    results_df = combined_df.copy()
    parsed_df = run_model_on_dataset(model, dataset, device=device,
                                     max_batch_size=8)

    results_df = results_df.drop(columns=['affinity_score'])
    results_df = results_df.merge(parsed_df, on=['protein_id', 'molecule_id'], how='left')

    # Save the raw results to a file
    results_file = os.path.join(results_dir, 'pgx_raw_results.pkl')
    results_df.to_pickle(results_file)


    # Parse to get results for each variant-drug pair
    # where one column is the mean affinity for all the ref proteins of the variant
    # and one column is the mean affinity for all the alt proteins of the variant
    # Note: attention scores NOT included in this matrix
    all_test_variants = {k: 'manual' for k in test_sequence_pairs.keys()}
    all_test_variants.update(test_variants)

    var_col = pd.DataFrame(all_test_variants.items(), columns=["variant_id", "variant_code"])
    drug_col = pd.DataFrame(test_drugs.items(), columns=['molecule_id', 'molecule_smiles'])

    variant_drug_delta_df = var_col.merge(drug_col, how='cross')


    # For each variant-drug pair, get the results for the reference and alternate proteins
    # and for each ref-alt pair, compute various statistics (delta, differences in attention, etc.)

    # Flag to allow for "smarter" matching of ref-alt pairs
    # by comparing the names of the ref/alt IDs to see if the alt is a mutation of the ref
    # for all non-manual entries
    attempt_smarter_matching = True

    def process_data_match_variant_refalt_pairs(row, attn_type='attention'):
        variant_code = row['variant_code']

        var_id, mol_id = row['variant_id'], row['molecule_id']

        ref_prot_ids = var_protid_map[var_id]['ref_id']
        alt_prot_ids = var_protid_map[var_id]['alt_id']

        ref_prot_data = results_df[results_df['protein_id'].isin(ref_prot_ids) &
                                        (results_df['molecule_id'] == mol_id)]
        
        alt_prot_data = results_df[results_df['protein_id'].isin(alt_prot_ids) &
                                        (results_df['molecule_id'] == mol_id)]

        # Construct a dataframe with a row for each ref-alt pair
        # with various statistics from the results
        refalt_list = []

        for _, ref_row in ref_prot_data.iterrows():
            ref_id = ref_row['protein_id']
            ref_affinity = ref_row['affinity_score']
            ref_protattention = ref_row[f'protein_{attn_type}']
            ref_molattention = ref_row[f'molecule_{attn_type}']
            ref_len = len(ref_protattention)

            for _, alt_row in alt_prot_data.iterrows():
                alt_id = alt_row['protein_id']
                alt_affinity = alt_row['affinity_score']
                alt_protattention = alt_row[f'protein_{attn_type}']
                alt_molattention = alt_row[f'molecule_{attn_type}']
                alt_len = len(alt_protattention)

                ref_seq = protid_seq_map[ref_id]
                alt_seq = protid_seq_map[alt_id]

                # Skip if ref name is not in alt name (if smarter matching is attempted)
                # for all non-manual variants
                # (also, if the sequences are the same, we'll skip)
                if attempt_smarter_matching and variant_code != 'manual':
                    if ref_id not in alt_id:
                        continue
                    else:
                        if ref_seq == alt_seq:
                            continue


                delta_affinity = alt_affinity - ref_affinity
                delta_molattention = alt_molattention - ref_molattention 

                # Protein length can change between ref and alt; if so, 
                # we'll return nan for the delta attention scores for the protein
                # and handle this separately later
                if ref_len == alt_len:
                    delta_protattention = ref_protattention - alt_protattention
                else:
                    delta_protattention = np.nan


                refalt_list.append({
                    "variant_id": var_id,
                    "molecule_id": mol_id,
                    "variant_code": variant_code,
                    "molecule_smiles": row['molecule_smiles'],
                    "ref_id": ref_id,
                    "alt_id": alt_id,
                    "ref_affinity": ref_affinity,
                    "alt_affinity": alt_affinity,
                    "delta_affinity": delta_affinity,
                    f'ref_prot_{attn_type}': ref_protattention,
                    f'alt_prot_{attn_type}': alt_protattention,
                    f"delta_prot_{attn_type}": delta_protattention,
                    f'ref_mol_{attn_type}': ref_molattention,
                    f'alt_mol_{attn_type}': alt_molattention,
                    f"delta_mol_{attn_type}": delta_molattention,
                    "ref_file": ref_row['protein_file'],
                    "alt_file": alt_row['protein_file'],
                    "ref_len": ref_len,
                    "alt_len": alt_len,
                    "ref_seq": protid_seq_map[ref_id],
                    "alt_seq": protid_seq_map[alt_id]
                })

        return pd.DataFrame(refalt_list)

    attn_refalt_func = partial(process_data_match_variant_refalt_pairs, attn_type='attention')
    expl_refalt_func = partial(process_data_match_variant_refalt_pairs, attn_type='explanation')

    delta_results_attn = pd.concat(variant_drug_delta_df.apply(attn_refalt_func, axis=1).tolist())
    delta_results_expl = pd.concat(variant_drug_delta_df.apply(expl_refalt_func, axis=1).tolist())

    expl_cols = delta_results_expl.columns.difference(delta_results_attn.columns).tolist()
    delta_results_attn[expl_cols] = delta_results_expl[expl_cols]

    delta_results = delta_results_attn

    variant_drug_delta_df = delta_results.reset_index(drop=True)
    match_drug = variant_drug_delta_df.apply(lambda x: x['molecule_id'].lower() in x['variant_id'].lower(), axis=1)
    variant_drug_delta_df['match_drug'] = match_drug

    # Save the results to a file
    delta_results_file = os.path.join(results_dir, 'pgx_delta_results.pkl')
    variant_drug_delta_df.to_pickle(delta_results_file)

    ipdb.set_trace()