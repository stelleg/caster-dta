import pandas as pd
import os

import ipdb

def parse_plinder(dataset_dir,
                  save_to_csv=True,
                  force_reparse=False,
                  need_structures=False,
                  dedupe_systems=True):
    """
    Parse the PLINDER dataset
    Note that seed here is irrelevant as the dataset is already split
    but we take it as an argument to return an error if passed
    """
    data_path = dataset_dir
    data_csv_path = os.path.join(data_path, 'processed_data_plinderdl.csv') # './data/plinder/processed_data.csv'

    # If data already processed, load and skip processing
    if os.path.exists(data_csv_path) and not force_reparse:
        print("Data already processed, loading...", flush=True)
        plinder_data = pd.read_csv(data_csv_path)
        return plinder_data

    from pandarallel import pandarallel
    import hashlib

    import plinder.core.utils.config
    from plinder.core.scores import query_index, query_links
    from plinder.core import PlinderSystem
    pandarallel.initialize(progress_bar=True)


    # Make sure PLINDER loads (or saves into) the correct directory
    os.environ['PLINDER_MOUNT'] = dataset_dir

    # Make PLINDER logger quiet
    os.environ['PLINDER_LOG_LEVEL'] = '40'

    cfg = plinder.core.get_config()
    # Report where downloading files from and saving to
    print(f"local directory: {cfg.data.plinder_dir}")
    print(f"remote data directory: {cfg.data.plinder_remote}")

    # Load the dataset by querying the Plindex
    # Columns documented at 
    # https://plinder-org.github.io/plinder/dataset.html#annotation-table-target

    # NOTE: for CASTER-DTA, we have "ligand_binding_affinity" as a column!

    print("Querying PLINDER index (+links) for systems to use...")

    # NOTE: system_pass_validation_criteria is a bit stringent
    # if we don't need structures; however, it is necessary to ensure
    # that comparisons to CASTER-DTA are fair(-ish? keep in mind that having different test/val sets already limits this)
    plindex_single_proper = query_index(
        filters=[
            ("system_num_ligand_chains", "==", 1),
            ("system_num_protein_chains", "==", 1),
            ("system_pass_validation_criteria", "==", True),
            ("ligand_is_proper", "==", True),
            ("system_has_binding_affinity", "==", True),
        ],
        splits=["train", "val", "test"],
        columns=["system_id", "ligand_id", "entry_pdb_id", 
                "ligand_smiles", "ligand_binding_affinity",],
    )

    # Columns for linked systems to identify systems with apo and pred structures
    # https://plinder-org.github.io/plinder/dataset.html#linked-systems-links

    all_scored_links = query_links(
        filters=[("reference_system_id", "in", set(plindex_single_proper.system_id))],
        columns=["reference_system_id", "id", "target_id", "receptor_file", "kind"],
    )

    all_scored_links = all_scored_links.rename(
        columns = {
        "reference_system_id": "system_id",
        "id": "linked_structure_id",
        "target_id": "target_structure_id",
        "kind": "linked_structure_type",
    })

    plindex_single_proper = plindex_single_proper.merge(all_scored_links, 
                                on='system_id', 
                                how='left', suffixes=('', '_linked'))
    
    plindex_single_proper = plindex_single_proper.sort_values(by=plindex_single_proper.columns.tolist())

    # If structures needed, drop systems without linked structures
    if need_structures:
        plindex_single_proper = plindex_single_proper.dropna(subset=['linked_structure_id'])
    
    # Deduplicate systems (these have the same protein and ligand but different linked structures)
    # Only if requested and if structures are not requested
    if dedupe_systems and not need_structures:
        plindex_single_proper = plindex_single_proper.drop_duplicates(subset=['entry_pdb_id', 'ligand_smiles', 'ligand_binding_affinity'])

    print(plindex_single_proper['split'].value_counts(dropna=False))


    # Function to get needed data from a system
    def get_data_from_system(row):
        system_id = row['system_id']
        plsys = PlinderSystem(system_id=system_id)

        # Get protein sequence of system protein, which is the same as 
        # the linked structure (hopefully?)
        seq_opts = plsys.sequences
        protein_seq = list(seq_opts.values())[0]

        # Get file path to input (linked) structure
        # (DO NOT RUN THIS PART LOCALLY UNLESS YOU WANT TO DOWNLOAD
        # AND UNPACK EVERY SINGLE LINKED STRUCTURE)
        link_type = row['linked_structure_type']
        link_name = row['linked_structure_id']

        # Get file path to input structure
        if need_structures:
            input_prot_file = plsys.get_linked_structure(link_type, link_name)
        else:
            input_prot_file = 'NOSTRUCTREQUESTED'

        # Get file path to holo structure
        holo_prot_file = plsys.system_cif

        # Ligand smiles
        # NOTE: ligand IDs for identical ligands are different
        # because they're tied to the system ID (for some reason)
        # so we'll use a hash of the smiles string instead
        smiles_opts = plsys.smiles
        first_lig = list(smiles_opts.keys())[0]
        lig_smiles = smiles_opts[first_lig]
        lig_id = hashlib.sha1(lig_smiles.encode("utf-8")).hexdigest()


        # NOTE: the evaluation function will align the proteins and ligands for the competition
        # but we do not have the luxury of doing that when computing our own RMSDs for torch to optimize
        # so we need to align the protein atoms to the holo protein atoms
        # and the ligand atoms to the holo ligand atoms
        # (noting that there may be multiple ligand alignments and we pick the best one)

        return pd.Series({
            'protein_id': link_name,
            'protein_sequence': protein_seq,
            'protein_file': input_prot_file,
            'molecule_id': lig_id,
            'molecule_smiles': lig_smiles,
            'affinity_score': row['ligand_binding_affinity'],
            'complex_id': system_id,
            'complex_file': holo_prot_file,
            'split': row['split'],
        })
    
    plinder_data = plindex_single_proper.parallel_apply(get_data_from_system, axis=1)

    need_cols = ['protein_id', 'protein_sequence', 'molecule_id', 'molecule_smiles', 'affinity_score', 'split']
    
    if need_structures:
        need_cols += ['protein_file', 'complex_file', 'complex_id']

    # Subset columns to actually needed ones
    plinder_data = plinder_data[need_cols]

    # Add a name for any proteins without IDs
    # since we need to have a unique identifier for each protein
    # (we'll hash the sequence and use that as an ID)
    plinder_data['protein_id'] = plinder_data['protein_id'].fillna(
        plinder_data['protein_sequence'].apply(lambda x: hashlib.sha1(x.encode("utf-8")).hexdigest())
    )

    # Remove all duplicate protein-molecule pairs (keeping the first one)
    # (remember that we have multiple linked structures for each system, so this happens)
    # (to evaluate we will only use the first one as this is the "fairest" comparison)
    plinder_data = plinder_data.drop_duplicates(subset=['protein_id', 'molecule_id'])
    
    # Reset the index
    plinder_data = plinder_data.reset_index(drop=True)

    if save_to_csv:
        plinder_data.to_csv(data_csv_path, index=False)

    return plinder_data