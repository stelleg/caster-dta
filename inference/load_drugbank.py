import xml.etree.ElementTree as ET
import zipfile
import pandas as pd

import ipdb

def load_drugbank(drugbank_file):
    """
    Load the Drugbank data from a provided XML file
    and converts it to a Pandas DataFrame
    """    
    # Load the XML file 
    # (if zipped as is the default download, load the first XML inside)
    if drugbank_file.endswith('.zip'):
        print("Loading Drugbank data from XML in ZIP archive...")
        base_name = None
        with zipfile.ZipFile(drugbank_file, 'r') as z:
            for fname in z.namelist():
                if fname.endswith('.xml'):
                    base_name = fname
                    break

            if base_name is None:
                raise ValueError("No XML file found in the ZIP archive")
            
            with z.open(base_name) as f:
                tree = ET.parse(f)
    else:
        print("Loading Drugbank data from XML file...")
        tree = ET.parse(drugbank_file)

    # Load the XML file
    root = tree.getroot()

    # Get the namespace
    ns = root.tag.split('}')[0] + '}'

    # Initialize a list where each item is a dictionary of the drug's properties
    drugbank_data = []

    # Loop through the drugs
    # Consider only small molecule drugs (TODO: biotech?)
    for i, drug in enumerate(root):
        # Initialize a dictionary for the drug's properties
        drug_dict = {}

        # Drug type (small molecule, biotech, etc.)
        drug_type = drug.get('type')

        if drug_type != 'small molecule':
            continue

        # Get the drugbank ID
        drug_dict['drugbank_id'] = drug.findtext(f"{ns}drugbank-id[@primary='true']")

        # Get the drug's name
        drug_dict['name'] = drug.findtext(f"{ns}name")

        # Get the drug's description
        # drug_dict['description'] = drug.findtext(f"{ns}description")

        # Get the drug's groups (things like "approved", "withdrawn", etc.)
        drug_dict['groups'] = [group.text for group in drug.findall(f"{ns}groups/{ns}group")]

        # Get the drug's SMILES string
        chem_props = drug.findall(f"{ns}calculated-properties/{ns}property")
        smiles = [prop.findtext(f'{ns}value') for prop in chem_props if prop.findtext(f"{ns}kind") == 'SMILES']

        if len(smiles) == 0:
            # some small molecules don't have SMILES strings because they're actually just short peptides
            # we'll filter on this later
            smiles = ''
        elif len(smiles) == 1:
            smiles = smiles[0]
        
        drug_dict['smiles'] = smiles

        # Get the drug's target data, including, for each (polypeptide) target:
        # parent ID, parent name, target ID, target source, names, and sequences
        # Initially a list of tuples, will be converted to one list for each property
        targets = drug.findall(f"{ns}targets/")
        target_info = get_all_target_info(targets, ns=ns)

        drug_dict['target_superid'] = [x[0] for x in target_info]
        drug_dict['target_supername'] = [x[1] for x in target_info]
        drug_dict['target_id'] = [x[2] for x in target_info]
        drug_dict['target_source'] = [x[3] for x in target_info]
        drug_dict['target_name'] = [x[4] for x in target_info]
        drug_dict['target_sequence'] = [x[5] for x in target_info]

        # Do the same thing for enzymes, with the exact same properties
        enzymes = drug.findall(f"{ns}enzymes/")
        enzyme_info = get_all_target_info(enzymes, ns=ns)

        drug_dict['enzyme_superid'] = [x[0] for x in enzyme_info]
        drug_dict['enzyme_supername'] = [x[1] for x in enzyme_info]
        drug_dict['enzyme_id'] = [x[2] for x in enzyme_info]
        drug_dict['enzyme_source'] = [x[3] for x in enzyme_info]
        drug_dict['enzyme_name'] = [x[4] for x in enzyme_info]
        drug_dict['enzyme_sequence'] = [x[5] for x in enzyme_info]

        # Get some other useful properties
        # such as whether the drug crosses the blood-brain barrier
        drugbank_data.append(drug_dict)

    # Convert the list of dictionaries to a DataFrame
    drugbank_df = pd.DataFrame(drugbank_data)

    return drugbank_df



def get_all_target_info(targets, ns='{http://www.drugbank.ca}'):
    """
    From a set of targets in the Drugbank XML, for each target get a parent name and ID,
    find each polypeptide associated with the target, 
    then for each polypeptide, extract the target's ID, source, name, and sequence
    and return for each the (parent_id, parent_name, target_id, target_source, target_name, target_sequence)
    (for now, we only consider protein targets in humans)
    Ultimately returns a list of tuples
    """

    return_list = []

    for parent_target in targets:
        target_organism = parent_target.findtext(f"{ns}organism")

        if target_organism != 'Humans':
            continue

        parent_id = parent_target.findtext(f"{ns}id")
        parent_name = parent_target.findtext(f"{ns}name")

        polypeptides = parent_target.findall(f"{ns}polypeptide")

        for polypeptide in polypeptides:
            target_id, target_source, target_name, target_sequence = get_single_target_info(polypeptide, ns=ns)

            return_list.append((parent_id, parent_name, target_id, target_source, target_name, target_sequence))

    return return_list
        

def get_single_target_info(target_element, ns='{http://www.drugbank.ca}'):
    """
    For a single polypeptide target, get the target ID, ID-source, name, and sequence
    """
    
    target_id = target_element.get('id')
    target_source = target_element.get('source')
    target_name = target_element.findtext(f"{ns}name")
    target_seq = target_element.findtext(f"{ns}amino-acid-sequence")

    # Target sequence is in FASTA format, usually, so we need to remove the header
    # and any subsequent newlines
    # basically, split by newlines, remove any lines starting with '>' (the header)
    # and then join the remaining lines together
    target_seq = ''.join([x for x in target_seq.split('\n') if not x.startswith('>')])

    return target_id, target_source, target_name, target_seq


