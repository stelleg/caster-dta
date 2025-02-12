from collections import defaultdict

# List of possible values initially obtained from
# https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/utils/smiles.html
# and updated based on any other values that are encountered

# One-hot encodings based on the ranges from https://pubs.acs.org/doi/10.1021/acs.jmedchem.9b00959
# and https://pubs.acs.org/doi/10.1021/acs.chemmater.9b01294


def map_intdict_to_onehot(in_dict, add_other=False):
    """
    Map values to one-hot vectors
    """
    num_elements = len(in_dict)
    if(add_other):
        num_elements += 1
    
    onehot_dict = {}
    for k, v in in_dict.items():
        onehot = [0] * num_elements
        onehot[v] = 1
        onehot_dict[k] = onehot

    if(add_other):
        onehot_other = ([0] * (num_elements-1)) + [1]
        onehot_dict = defaultdict(lambda: onehot_other, onehot_dict)

    return onehot_dict


## Node features
# Mapped to numeric/categorical values


# Node types defined based on atom identities
# Mapped to numeric/categorical values
# Flag to whether to keep only specific atoms in the dataset 
# or all atoms that could exist as possible node types
SELECT_ATOMICNUMS_TO_KEEP = [
    1, # 'H', # 1
    6, # 'C', # 6
    7, # 'N', # 7
    8, #'O', # 8
    9, #'F', # 9
    15, #'P', # 15
    16, #'S', # 16
    17, #'Cl', # 17
    35, #'Br', # 35
    53, #'I', # 53
]

# All other atomic symbols are mapped to the next highest value ("other") - 
# though there are none in the KIBA or DAVIS datasets as far as we can tell
SELECT_ATOMICNUM_TO_NTYPE = {j: i for i, j in enumerate(SELECT_ATOMICNUMS_TO_KEEP)}
SELECT_ATOMICNUM_TO_NTYPE = defaultdict(lambda: len(SELECT_ATOMICNUMS_TO_KEEP), SELECT_ATOMICNUM_TO_NTYPE)

ALL_ATOMICNUM_TO_NTYPE = {j: i for i, j in enumerate(range(1, 119))}
ALL_ATOMICNUM_TO_NTYPE = defaultdict(lambda: len(ALL_ATOMICNUM_TO_NTYPE), ALL_ATOMICNUM_TO_NTYPE)


# Chirality defined based on RDKit
CHIRALITIES = [
    'CHI_TETRAHEDRAL_CW', # R
    'CHI_TETRAHEDRAL_CCW', # S
]
SMILES_CHIRALITY_MAP = {j: i for i, j in enumerate(CHIRALITIES)}
SMILES_CHIRALITY_MAP = map_intdict_to_onehot(SMILES_CHIRALITY_MAP, add_other=True)

# Hybridization defined based on RDKit
HYBRIDIZATIONS = [
    'S',
    'SP',
    'SP2',
    'SP3',
    'SP3D',
    'SP3D2',
]
SMILES_HYBRID_MAP = {j: i for i, j in enumerate(HYBRIDIZATIONS)}
SMILES_HYBRID_MAP = map_intdict_to_onehot(SMILES_HYBRID_MAP, add_other=True)

# Number of hydrogens
NUM_HYDROGENS = [0, 1, 2, 3, 4]
SMILES_H_MAP = {j: i for i, j in enumerate(NUM_HYDROGENS)}
SMILES_H_MAP = map_intdict_to_onehot(SMILES_H_MAP, add_other=True)

# Formal charge
FORMAL_CHARGES = [-2, -1, 0, 1, 2]
SMILES_CHARGE_MAP = {j: i for i, j in enumerate(FORMAL_CHARGES)}
SMILES_CHARGE_MAP = map_intdict_to_onehot(SMILES_CHARGE_MAP, add_other=True)

# Number of radical electrons
NUM_RADICALS = [0, 1, 2]
SMILES_RADICAL_MAP = {j: i for i, j in enumerate(NUM_RADICALS)}
SMILES_RADICAL_MAP = map_intdict_to_onehot(SMILES_RADICAL_MAP, add_other=True)

# Degree
DEGREES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
SMILES_DEGREE_MAP = {j: i for i, j in enumerate(DEGREES)}
SMILES_DEGREE_MAP = map_intdict_to_onehot(SMILES_DEGREE_MAP, add_other=True)

# Implicit valence
VALENCES = [0, 1, 2, 3, 4, 5, 6]
SMILES_VALENCE_MAP = {j: i for i, j in enumerate(VALENCES)}
SMILES_VALENCE_MAP = map_intdict_to_onehot(SMILES_VALENCE_MAP, add_other=True)


## Edge features

# Bond types defined based on RDKit
# Mapped to numeric/categorical values
BOND_TYPES = [
    'SINGLE',
    'DOUBLE',
    'TRIPLE',
    'AROMATIC',
]
SMILES_BOND_MAP = {j: i for i, j in enumerate(BOND_TYPES)}
# Below if there are any bond types not in the list above (there are none in the KIBA or DAVIS datasets)
SMILES_BOND_MAP = defaultdict(lambda: len(BOND_TYPES), SMILES_BOND_MAP)

# Stereo configurations defined based on RDKit
# Mapped to numeric/categorical values
STEREO_CONFIGS = [
    'STEREONONE',
    'STEREOANY',
    'STEREOZ',
    'STEREOE',
    'STEREOCIS',
    'STEREOTRANS',
]
SMILES_STEREO_MAP = {j: i for i, j in enumerate(STEREO_CONFIGS)}
SMILES_STEREO_MAP = map_intdict_to_onehot(SMILES_STEREO_MAP, add_other=True)


