import requests
import os, json

import ipdb

def variant_to_protein_seq(var_dict, cache_dir, 
                           force_requery=False):
    """
    Convert variant IDs to protein sequences
    Specifically, the "normal" protein sequence and
    the "variant" protein sequence(s)
    Returns a dict for each input variant of protein ref and alt IDs
    and a dict of protein IDs to sequences (since there are likely to be dupes)
    """
    var_cache_dir = os.path.join(cache_dir, 'variants')
    seq_cache_dir = os.path.join(cache_dir, 'sequences')
    os.makedirs(var_cache_dir, exist_ok=True)
    os.makedirs(seq_cache_dir, exist_ok=True)

    var_ids = list(var_dict.values())
    var_ids = list(set(var_ids))
    var_ids_copy = var_ids.copy()

    all_var_dict = {}
    uniq_seq_dict = {}

    # Attempt to load all variants from the cache directory
    # if force_requery is False
    if not force_requery:
        for var_id in var_ids_copy:
            var_file = os.path.join(var_cache_dir, f"{var_id}.json")
            if os.path.exists(var_file):
                with open(var_file, 'r') as f:
                    all_var_dict[var_id] = json.load(f)

                var_ids.remove(var_id)


    # Query only if there are variants that are not in the cache directory
    # that have been mapped to HGVS
    if var_ids:
        # Use Ensembl's variant recoder to take variant IDs and get the HGVS for the protein
        data = [f'"{x}"' for x in var_ids]
        data = ', '.join(data)
        data = f'{{ "ids" : [{data}] }}'
        
        vr_server_api = "https://rest.ensembl.org/variant_recoder/homo_sapiens"
        headers={ "Content-Type" : "application/json", "Accept" : "application/json"}
        resp = requests.post(vr_server_api, headers=headers, data=data)
        
        if not resp.ok:
            resp.raise_for_status()
        
        decoded_resp = resp.json()


        # Get unique protein reference and alt HGVS IDs for each input
        # Generally, there should be just one reference and one alt, but not always
        # as some variants refer to multiple proteins or multiple alleles
        # TODO: pair the ref and alts (but this may not be possible)
        for var_id, var_resp in zip(var_ids, decoded_resp):
            curr_var_dict = {}
            alt_ids = []

            for allele_id, allele_data in var_resp.items():
                assert allele_data['input'] == var_id
                
                allele_alt_ids = [x for x in allele_data.get('hgvsp', []) if x.startswith('NP_')]
                alt_ids.extend(allele_alt_ids)

            alt_ids = list(set(alt_ids))
            ref_ids = list(set([x.split(':')[0] for x in alt_ids]))


            curr_var_dict['ref_id'] = ref_ids
            curr_var_dict['alt_id'] = alt_ids

            all_var_dict[var_id] = curr_var_dict


    # Get the protein sequences for the reference and alternate proteins
    # by using Mutalyzer
    # For reference we append :p.= to get the base protein sequence
    # and for alt we use the actual HGVS identifier
    uniq_ref_ids = list(set([x for y in all_var_dict.values() for x in y['ref_id']]))
    uniq_alt_ids = list(set([x for y in all_var_dict.values() for x in y['alt_id']]))

    uniq_ref_ids_copy = uniq_ref_ids.copy()
    uniq_alt_ids_copy = uniq_alt_ids.copy()

    # Check which sequences are already in the cache directory
    # and only requery if force_requery is True
    if not force_requery:
        for ref_id in uniq_ref_ids_copy:
            ref_file = os.path.join(seq_cache_dir, f"{ref_id}.txt")
            if os.path.exists(ref_file):
                with open(ref_file, 'r') as f:
                    uniq_seq_dict[ref_id] = f.read()

                uniq_ref_ids.remove(ref_id)

        for alt_id in uniq_alt_ids_copy:
            alt_file = os.path.join(seq_cache_dir, f"{alt_id}.txt")
            if os.path.exists(alt_file):
                with open(alt_file, 'r') as f:
                    uniq_seq_dict[alt_id] = f.read()
                
                uniq_alt_ids.remove(alt_id)


    # Use Mutalyzer to get the protein sequences
    # TODO: use hgvs package to get the sequences instead of Mutalyzer
    # since Mutalyzer doesn't always support some variants (such as frameshift)
    mut_server_api = "https://mutalyzer.nl/api/mutate/"
    headers={ "accept" : "application/json"}
    
    for ref_id in uniq_ref_ids:
        ref_url = f"{mut_server_api}{ref_id}:p.="
        ref_resp = requests.get(ref_url, headers=headers).json()
        uniq_seq_dict[ref_id] = process_seq(ref_resp['sequence']['seq'])
    
    for alt_id in uniq_alt_ids:
        alt_url = f"{mut_server_api}{alt_id}"
        alt_resp = requests.get(alt_url, headers=headers).json()
        uniq_seq_dict[alt_id] = process_seq(alt_resp['sequence']['seq'])


    # Add variants to the cache directory
    # (overwrite if necessary if force_requery is True)
    for var_id, var_data in all_var_dict.items():
        var_file = os.path.join(var_cache_dir, f"{var_id}.json")

        if not os.path.exists(var_file) or force_requery:
            with open(var_file, 'w') as f:
                json.dump(var_data, f)

    # Add sequences to the cache directory
    # (overwrite if necessary if force_requery is True)
    for prot_id, seq in uniq_seq_dict.items():
        seq_file = os.path.join(seq_cache_dir, f"{prot_id}.txt")

        if not os.path.exists(seq_file) or force_requery:
            with open(seq_file, 'w') as f:
                f.write(seq)


    # Remap the variant IDs to the original variant names
    ids_to_name = {v: k for k, v in var_dict.items()}
    all_var_dict = {ids_to_name[k]: v for k, v in all_var_dict.items()}
        
    return all_var_dict, uniq_seq_dict


def process_seq(seq):
    """
    Process the sequence to remove unneeded components 
    (for now, just removing all characters after a termination character)
    In the future, this might involve replacing characters with a standard
    set of amino acids (selenocysteine to cysteine, etc.)
    """

    term_chars = ['*']
    return seq.split(term_chars[0])[0]