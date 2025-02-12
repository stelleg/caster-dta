import os, sys

import json
import torch
from models.joint_gnn import JointGNN 
from dataset.dual_dataset import ProteinMoleculeDataset, PMD_DataLoader

import pandas as pd
import numpy as np
import hashlib

def get_best_model(checkpoint_path, use_best):
    if(os.path.isdir(checkpoint_path)):
        model_paths = sorted(os.listdir(checkpoint_path))
        # Get the best model from the folder based on --use-best
        if(use_best == 'val'):
            model_paths = [x for x in model_paths if (x.startswith('bestvalmodel') or x.startswith('bestmodel'))]
        elif(use_best == 'train'):
            model_paths = [x for x in model_paths if x.startswith('besttrainmodel')]
        elif(use_best == 'final'):
            model_paths = [x for x in model_paths if x.startswith('finalmodel')]
        
        
        if(len(model_paths) == 0):
            print("No best model found in checkpoint folder - exiting", flush=True)
            sys.exit(1)

        checkpoint_path = os.path.join(checkpoint_path, model_paths[0])

        print("\tUsing best model (from above):", checkpoint_path, flush=True)
    elif(os.path.isfile(checkpoint_path)):
        print("\tPath above is a file. Will load as checkpoint directly...", flush=True)
    else:
        print("Invalid checkpoint path - exiting", flush=True)
        sys.exit(1)

    return checkpoint_path
    

def load_model_from_checkpoint(check_path, best_model_type='val', device=None,
                               allow_return_compile=False):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_param_file = get_best_model(check_path, best_model_type)
    model_kwargs = json.load(open(os.path.join(check_path, 'model_kwargs.json'), 'r'))

    model = JointGNN(model_kwargs['protein_gnn_kwargs'], model_kwargs['molecule_gnn_kwargs'], 
                    **model_kwargs['joint_gnn_kwargs'])
    
    model = model.to(device)
    model_params = torch.load(model_param_file, map_location=device)
    
    # Check if model keys have _orig_mod. in them and compile if so
    was_compiled = '_orig_mod.' in list(model_params.keys())[0]

    if was_compiled:
        if allow_return_compile:
            # Compile the model if user requests
            print("Model was detected as being compiled, recompiling...")
            model = torch.compile(model)
        else:
            # Change the keys in the state dict remove the prefix instead and leave uncompiled
            print("Model was detected as being compiled, will remove prefix from state dict keys...")
            model_params = {x.replace('_orig_mod.', ''): y for x,y in model_params.items()}
    
    model.load_state_dict(model_params)
    model.eval()

    return model
    

def create_dataset_with_checkpoint_params(dataset_df, check_path, cache_dir=None):
    # Hash the dataset to see if it is already saved; if so, load it. 
    # Otherwise, make the dataset and save it
    df_hash = hashlib.sha256(pd.util.hash_pandas_object(dataset_df, index=True).values).hexdigest()
    ds_file = os.path.join(cache_dir, f'dataset_{df_hash}.pt')

    if(os.path.exists(ds_file)):
        dataset = torch.load(ds_file)
    else:
        dataset_kwargs = json.load(open(os.path.join(check_path, 'dataset_kwargs.json'), 'r'))
        dataset = ProteinMoleculeDataset(dataset_df, **dataset_kwargs)
        torch.save(dataset, ds_file)

    # Load the rescaling parameters from the folder (since this model was trained to these)
    rescale_file = os.path.join(check_path, 'dataset_rescale_params.json')
    rescale_params = json.load(open(rescale_file, 'r'))
    dataset._load_scale_data_from_dict(rescale_params)

    return dataset