from models.joint_gnn import JointGNN
from dataset.dual_dataset import ProteinMoleculeDataset, PMD_DataLoader
from dataset.load_data import load_dataset
import torch
from torch_geometric.nn import summary as pyg_summary
import numpy as np
import json, os, sys
import shutil
import random
from utils.other_utils import seed_everything_reproducibility
from inference.inference_utils import get_best_model
import pandas as pd
import hashlib

import ipdb


# Example command to execute for a folder based on timestamp, will log the outputs and each best model here:
# env DTA_OUTFOLDER="logs/testing/"$(date +"%Y%m%d__%H_%M_%S")"" bash -c 'mkdir "${DTA_OUTFOLDER}" && python train_model.py --dataset davis --seed 9 --out-folder $DTA_OUTFOLDER 2>&1 | tee -a "${DTA_OUTFOLDER}/log.txt"'

import argparse

parser = argparse.ArgumentParser(description='Train the joint GNN model on a dataset')

parser.add_argument('--dataset', type=str, default='davis', help='The name of the dataset to train on')
parser.add_argument('--out-folder', type=str, default='output', help='The folder to save the output to')
parser.add_argument('--seed', type=int, default=9, help='The seed value to use for reproducibility')
parser.add_argument('--checkpoint', type=str, default=None, help='A checkpoint to load from before training (if any). If a folder, uses the best model found inside based on the name.')
parser.add_argument('--use-best', type=str, default='val', help='Best model to use at the end of training (or from --checkpoint if --skip-training is passed). If "val", uses the best validation model. If "train", uses the best training model. If "final", uses the final model. Default "val"')
parser.add_argument('--skip-training', action='store_true', help='Skips training and goes to evaluating on test data (useful if loading a checkpoint)')

args = parser.parse_args()

dataset_name = args.dataset
seed_val = args.seed
output_folder = args.out_folder
checkpoint_path = args.checkpoint
skip_training = args.skip_training

os.makedirs(output_folder, exist_ok=True)

print("Using dataset:", dataset_name, flush=True)
print("Using seed:", seed_val, flush=True)
print("Output folder:", output_folder, flush=True)

if(checkpoint_path is not None):
    print("Using checkpoint from:", checkpoint_path, flush=True)
    checkpoint_path = get_best_model(checkpoint_path, args.use_best)



if(skip_training):
    print("Skipping training and going to test evaluation", flush=True)

# Make a copy of this script and store in the output folder
# as a record of the training run to be able to reproduce it
# (also useful for debugging)
print("Making a copy of this script in the output folder...", flush=True)
shutil.copy(__file__, os.path.join(output_folder, 'train_model.py'))

# Save a copy of the command used to run this script in the output folder
# to be able to reproduce the run even more cleanly
print("Saving the command used to run this script in the output folder...", flush=True)
with open(os.path.join(output_folder, 'train_command.txt'), 'w') as f:
    f.write(sys.executable + ' ' + ' '.join(sys.argv))

# Set random seed for debug (and eventually some level of reproducibility)
# Note that cuda determinism is mandatory for true reproducibility if using a GPU
# as otherwise, the GPU will select different kernels for the same operation
force_cuda_determinism = False
error_on_nondeterministic = False

# Seed everything for "reproducibility" 
# (though note that this is not true reproducibility if cuda determinism is off)
seed_everything_reproducibility(seed_val, force_cuda_determinism, error_on_nondeterministic)


# Speed-based optimization, but can be nondeterministic (commented out for now)
# Note that this can be an issue with variable sized inputs
# torch.backends.cudnn.benchmark = True


# Load the requested dataset based on the name (davis, kiba, plinder, etc.)
load_dataset_kwargs = {
    'skip_pdb_dl': False,
    'allow_complexed_pdb': False,
    'create_comp': False,
    'reverse_comp_fold_order': False,
    'verbose_pdb_dl': True,
    'verbose_comp_fold': True,
    'do_mostcommon_filter': False,
    'do_proteinseqdupe_filter': False,
}

data_df, data_path, pdb_dir_name = load_dataset(dataset_name,
                                                **load_dataset_kwargs)


# Use the dataframe to create a dataset
# Would likely need to perform on each dataset separately
output_scaling = ['standardize'] #standardize, minmax, log
print("Loading dataset now...", flush=True)

# Consider higher edge threshold for protein
# based on angstrom distance (8.0?) or based on number of top edges (50?)
protein_dist_units = 'angstroms' # 'angstroms' or 'nanometers' (1 nm = 10 angstroms); note that this actually matters as it affects the edge distance RBF as well (which is 0-20 regardless of nm or angstroms)
protein_edge_thresh = 4 # if dist, this needs to be in the same unit as the distance above
protein_thresh_type = 'dist' # options are 'dist', 'num', or 'prop'
protein_keep_selfloops = True #Whether to keep self-loops in the protein graph
protein_vector_features = True #If set to true, will create protein features as a tuple (scalar, vector) for GVPs
protein_include_esm2 = False #Adds 320 scalar features from ESM2 to the protein node features
protein_include_residue_posenc = False #Includes 8 positional encoding features based on residue index (as node features).
protein_include_aa_props = True #Includes amino acid properties as node features
molecule_full_atomtype = False #Whether to one-hot encode every atomic number as a separate type (True) or to selectively encode a few (False)
molecule_onehot_ordinal_feats = False #Whether to make ordinal features (# radical electrons and formal charge) one-hot encoded or not
molecule_include_selfloops = True #Whether to include self-loops in the molecule graph (will be added as a separate edge type)

dataset_kwargs = {'sparse_edges': False,
                  'protein_dist_units': protein_dist_units,
                  'protein_edge_thresh': protein_edge_thresh,
                  'protein_thresh_type': protein_thresh_type,
                  'protein_keep_selfloops': protein_keep_selfloops,
                  'protein_vector_features': protein_vector_features,
                  'protein_include_esm2': protein_include_esm2,
                  'protein_include_residue_posenc': protein_include_residue_posenc,
                  'protein_include_aa_props': protein_include_aa_props,
                  'molecule_full_atomtype': molecule_full_atomtype,
                  'molecule_onehot_ordinal_feats': molecule_onehot_ordinal_feats,
                  'molecule_include_selfloops': molecule_include_selfloops,
                  'scale_output': output_scaling}

data_df_hash = hashlib.sha256(pd.util.hash_pandas_object(data_df, index=True).values).hexdigest()
dataset_kwargs_hash = hashlib.sha256(json.dumps(dataset_kwargs).encode()).hexdigest()

print(f"Dataframe hash: {data_df_hash}", flush=True)
print(f"Dataset kwargs hash: {dataset_kwargs_hash}", flush=True)

dataset_path = os.path.join(data_path, pdb_dir_name, f'00_datasetobj__{data_df_hash}_{dataset_kwargs_hash}.pt')

# Create the dataset if it doesn't exist
# NOTE: This is NOT robust to changes in the actual dataset or file processing code
# unless it changes the kwargs, so you need to delete and remake the dataset if processing is changed
if(not os.path.exists(dataset_path)):
    print(f"No preprocessed dataset found - creating dataset at {dataset_path}...", flush=True)
    dataset = ProteinMoleculeDataset(data_df, **dataset_kwargs)
    torch.save(dataset, dataset_path)
else:
    print(f"Preprocessed dataset found - loading from {dataset_path}...", flush=True)
    dataset = torch.load(dataset_path, weights_only=False)

print("--Full dataset details--")
print(dataset)

# Save the dataset kwargs to a JSON file in the output folder
# to allow for reloading of dataset or easy reference later
with open(os.path.join(output_folder, 'dataset_kwargs.json'), 'w') as f:
    json.dump(dataset_kwargs, f, indent=4)

# Save the rescaling parameters to a JSON file in the output folder
# to allow for reloading of dataset or easy reference later
scale_data_dict = dataset._report_scale_data()

with open(os.path.join(output_folder, 'dataset_rescale_params.json'), 'w') as f:
    json.dump(scale_data_dict, f, indent=4)

# Seed everything again before model has been made to ensure similar dataset splits
seed_everything_reproducibility(seed_val, force_cuda_determinism, error_on_nondeterministic)


# If the dataframe contains a column for predefined splits
# we will use those splits instead of random splits
if 'split' in data_df.columns:
    split_probs = data_df['split'].value_counts(normalize=True)
    print("Using predefined splits in the dataframe with the following probabilities:")
    print("Split probabilities:", split_probs)

    train_subset = data_df[data_df['split'] == 'train'].index.tolist()
    val_subset = data_df[data_df['split'] == 'val'].index.tolist()
    test_subset = data_df[data_df['split'] == 'test'].index.tolist()

    train_ds = torch.utils.data.Subset(dataset, train_subset)
    val_ds = torch.utils.data.Subset(dataset, val_subset)
    test_ds = torch.utils.data.Subset(dataset, test_subset)

else:
    # Otherwise, we will use random splits based on the seed
    speedy_run_testing = False

    if(speedy_run_testing):
        split_probs = [0.05, 0.05, 0.9]
    else:
        split_probs = [0.7, 0.15, 0.15]

    print(f"Using random splits based on the seed value {seed_val}", flush=True)
    print("Split probabilities:", split_probs)

    train_ds, val_ds, test_ds = torch.utils.data.random_split(dataset, split_probs)


# Print first (index-wise, sorted) 20 training, testing, and val indices to ensure consistency
# (if less than 20, will print all)
train_inds_sort = sorted(train_ds.indices)
val_inds_sort = sorted(val_ds.indices)
test_inds_sort = sorted(test_ds.indices)

print("First 20 training indices:", train_inds_sort[:20], flush=True)
print("First 20 validation indices:", val_inds_sort[:20], flush=True)
print("First 20 testing indices:", test_inds_sort[:20], flush=True)


# Print first (index-wise, sorted) 5 training, testing, and val pair names to ensure consistency
# (if less than 5, will print all)
n_print = 5
train_pairs = data_df.iloc[train_inds_sort]
val_pairs = data_df.iloc[val_inds_sort]
test_pairs = data_df.iloc[test_inds_sort]

print(f"First {n_print} training pairs:", train_pairs[['protein_id', 'molecule_id']].head(n_print), flush=True)
print(f"First {n_print} validation pairs:", val_pairs[['protein_id', 'molecule_id']].head(n_print), flush=True)
print(f"First {n_print} testing pairs:", test_pairs[['protein_id', 'molecule_id']].head(n_print), flush=True)

# Create a DataLoader for the dataset
# Custom loader implements dynamic batching (implement batch_sampler) 
# so that batch size depends on number of nodes/edges to prevent very large graphs
# we also have a max batch size of 32 to prevent very large batches as this can actually slow convergence
# and make it more likely for the model to overfit (the noise of smaller batches is a semi-regularizer)

# Worker reproducibility with seeding
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(seed_val)


# Different batch sizes/maximums needed for different datasets
# due to larger proteins or molecules being present
if dataset_name == 'kiba':
    bsize_max_num = 8000000
    max_bsize = 64
elif dataset_name == 'bindingdb' or dataset_name == 'belka':
    bsize_max_num = 4000000
    max_bsize = 32
else:
    bsize_max_num = 16000000
    max_bsize = 128


dloader_kwargs = {
                  'max_num': bsize_max_num, 'max_batch_size': max_bsize,
                  'shuffle': True, 'pin_memory': True, 'num_workers': 4,
                  'generator': g, 'worker_init_fn': seed_worker,
                  'persistent_workers': False
                  }

train_dl = PMD_DataLoader(train_ds, **dloader_kwargs)
val_dl = PMD_DataLoader(val_ds, **dloader_kwargs)
test_dl = PMD_DataLoader(test_ds, **dloader_kwargs)


# Number of gradient accumulations before updating weights (optimizer.step)
# The effective batch size is thus max_bsize * num_accum 
# (with variance accounting for differences in actual batch sizes)
num_batches_to_accum = 1


# Seed everything again after dataset has been made to ensure similar model initialization
seed_everything_reproducibility(seed_val, force_cuda_determinism, error_on_nondeterministic)


# Define the model
# increasing hidden is more high yield than increasing number of convolutions for increasing fit
# Note: aggr = 'sum' seems to make sense mostly, especially if the edge thresh means a different number of edges
protein_gnn_kwargs = dict(base_conv='lbamodel',
                          in_channels=dataset.metadata_dict['protein_node_features'],
                          edge_dim=dataset.metadata_dict['protein_edge_features'],
                          num_ntypes=dataset.metadata_dict['protein_node_types'],
                          num_etypes=dataset.metadata_dict['protein_edge_types'],
                          ntype_emb_dim=None, 
                          etype_emb_dim=None,
                        #   initial_node_project_channels=(16,8),
                        #   initial_edge_project_channels=(32,4),
                          num_convs=2, 
                          hidden_channels=(16,4),
                          edge_hidden_channels=(32,1),
                          out_channels=64,
                          dropout_rate=0.2,
                          activation='leaky_relu',
                          aggr='sum',
                          )

molecule_gnn_kwargs = dict(base_conv='gine',
                           in_channels=dataset.metadata_dict['molecule_node_features'],
                           edge_dim=dataset.metadata_dict['molecule_edge_features'],
                           num_ntypes=dataset.metadata_dict['molecule_node_types'],
                           num_etypes=dataset.metadata_dict['molecule_edge_types'],
                           ntype_emb_dim=None, 
                           etype_emb_dim=None,
                           num_convs=2, 
                           hidden_channels=16,
                           out_channels=64,
                           dropout_rate=0.2,
                           activation='leaky_relu',
                           aggr='sum',
                           gin_trainable_eps=True,
                        #    concat=True, 
                        #    heads=1,
                        #    conv_dropout=0.0,
                        #    conv_neg_slope=0.0,
                          )

joint_gnn_kwargs = dict(residue_lin_depth=1,
                        atom_lin_depth=1,
                        n_attention_heads=8,
                        attention_dropout=0.0,
                        protein_lin_depth=1,
                        molecule_lin_depth=1,
                        pairwise_embedding_dim=512, 
                        out_lin_depth=1,
                        out_lin_factor=0.5,
                        out_lin_norm_type=None,
                        activation='leaky_relu',
                        dropout=0.1,
                        element_pooling='mean',
                        include_residual_stream=True,
                        residual_dim_ff_scale=2,
                        num_cross_attn_layers=1,
                        include_post_pool_layernorm=False,
                        )

model = JointGNN(protein_gnn_kwargs, molecule_gnn_kwargs, 
                 **joint_gnn_kwargs)


print("--Model details--")
# print(model)
print(f"ProteinGNN: {protein_gnn_kwargs}")
print(f"MoleculeGNN: {molecule_gnn_kwargs}")
print(f"JointGNN: {joint_gnn_kwargs}")
print("")

# Save the kwargs for the model to a JSON file in the output folder
# to allow for reloading of model or easy reference later
with open(os.path.join(output_folder, 'model_kwargs.json'), 'w') as f:
    json.dump({'protein_gnn_kwargs': protein_gnn_kwargs,
               'molecule_gnn_kwargs': molecule_gnn_kwargs,
               'joint_gnn_kwargs': joint_gnn_kwargs}, f, indent=4)


# Define device and send model to device
cuda_avail = torch.cuda.is_available()
device = torch.device('cuda' if cuda_avail else 'cpu')
model = model.to(device)


# Get a dummy batch to define uninitialized parameters
# and then enumerate total model parameters
with torch.no_grad():
    protein_g, molecule_g, _ = next(iter(train_dl))
    protein_g = protein_g.to(device)
    molecule_g = molecule_g.to(device)

    _ = model.forward_with_graphs(protein_g.to(device), molecule_g.to(device))

    n_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"Number of parameters: {n_param}")

    protein_graph_data, molecule_graph_data = model._graphs_to_dicts(protein_g, molecule_g)

    model_summ = pyg_summary(model, protein_graph_data, molecule_graph_data)
    # This will print the model summary, but it's a bit unwieldy with the full model
    # to print in console, so we'll keep it but send it to a file instead in the output directory
    
    with open(os.path.join(output_folder, 'model_summary.txt'), 'w') as f:
        print(model_summ, file=f)

    del protein_g, molecule_g, protein_graph_data, molecule_graph_data


# Save model print to output separately
# as it's a bit unwieldy to print in the console
with open(os.path.join(output_folder, 'model_standardprint.txt'), 'w') as f:
    print(model, file=f)

print("")

# Define training hyperparameters
# and one param group for the final prediction layers
# and then having different weight decay parameters?
n_epochs = 2000

# Optimizer: SGD, Adam, or AdamW (or SGD without momentum)
# Note, this also selects a LR and weight decay (make these separate?)
# Consider Lion optimizer?
optimizer_select = 'adam' # 'adam', 'adamw', 'sgd', 'sgd_nomomentum'
lr = 1e-4
weight_decay = 0

# Scheduler: 'plateau', 'cosine', 'anneal_restart', 'anneal_restart_decay', 'exponential', or None
scheduler_select = 'plateau'
batch_schedulers = ['cosine', 'anneal_restart', 'anneal_restart_decay']
do_batch_schedule = True

# Number of epochs at the base learning rate before starting the scheduler
warmup_epochs = 0

# Early stopping number of epochs
n_epochs_before_stop = 200 # epochs before stopping if no improvement (in validation)

# Select loss functions (expects two inputs of predicted and target, one output)
loss_fn = torch.nn.MSELoss()
report_loss_fn = torch.nn.MSELoss()

# Other training parameters to speed things up
# and prevent divergence
clip_norm = None
do_amp = True

# Compile model (removed fullgraph mode for now, even though there are no graph breaks)
model = torch.compile(model, dynamic=True)



# Set up optimizer
# Consider changing AdamW parameters betas = (0.9,0.99), eps = 1e-6

if(optimizer_select == 'adamw'):
    # lr = 1e-3
    # weight_decay = 1e-2
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8,
                                amsgrad=False, weight_decay=weight_decay, fused=cuda_avail)

elif(optimizer_select == 'adam'):
    # lr = 1e-3
    # weight_decay = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8,
                                amsgrad=False, weight_decay=weight_decay, fused=cuda_avail)

elif(optimizer_select == 'sgd'):
    # lr = 1e-2
    # weight_decay = 1e-4
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, nesterov=True, 
                                momentum=0.9, weight_decay=weight_decay)

elif(optimizer_select == 'sgd_nomomentum'):
    # lr = 1e-2
    # weight_decay = 1e-4
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, nesterov=False, 
                                momentum=0.0, weight_decay=weight_decay)


# Set up scheduler
if(scheduler_select == 'plateau'):
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',
                                                           factor=0.8, 
                                                           patience=50)
elif(scheduler_select == 'cosine'):
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10,
                                                           eta_min=0)
elif(scheduler_select == 'anneal_restart'):
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10,
                                                                     eta_min=0)
elif(scheduler_select == 'anneal_restart_decay'):
    from models.custom_optims import CosineAnnealingWarmRestartsDecay
    scheduler = CosineAnnealingWarmRestartsDecay(optimizer, T_0=10,
                                                 eta_min=0,
                                                 decay_rate=0.95)
elif(scheduler_select == 'exponential'):
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
elif(scheduler_select is None):
    scheduler = None


# Variables
is_cuda = device.type == 'cuda'
scaler = torch.cuda.amp.GradScaler(enabled=(do_amp & is_cuda))
total_train = len(train_ds)
total_val = len(val_ds)
total_test = len(test_ds)


print("--Training parameters--")
print(f"\tDevice: {device}")
print(f"\tOptimizer: {optimizer_select}: {type(optimizer)}")
print(f"\tLearning rate: {lr}")
print(f"\tWeight decay: {weight_decay}")
print(f"\tScheduler type: {scheduler_select}: {type(scheduler)}")
print(f"\tTrain Loss Function: {loss_fn}")
print(f"\tReport Loss Function: {report_loss_fn}")
print(f"\tNumber of epochs: {n_epochs}")
print(f"\tWarmup epochs: {warmup_epochs}")
print(f"\tEarly stop epoch limit: {n_epochs_before_stop}")
print(f"\tGradient clipping norm: {clip_norm}")
print(f"\tAutomatic mixed precision: {do_amp}")
print(f"\tTotal model parameters: {n_param}")
print(f"\tTotal train / val / test samples: {total_train} / {total_val} / {total_test}")
print(f"\tProtein edge info: (Threshold: {protein_edge_thresh}, {protein_thresh_type}) ({protein_dist_units}, selfloops: {protein_keep_selfloops})")
print(f"\tMolecule feature info: (Full atomtype: {molecule_full_atomtype}, Onehot ordinal: {molecule_onehot_ordinal_feats}, Selfloops: {molecule_include_selfloops})")
print(f"\tMax batch size / batch element number: {max_bsize} / {bsize_max_num}")
print(f"\tGradient accumulation steps // effective max batch size: {num_batches_to_accum} // {max_bsize * num_batches_to_accum}")
print("")


# Argument for whether reported losses and values are scaled or unscaled
# (relative to the target scaling, not the gradient scaling)
print_unscaled_loss = True

# Set first learning rate and best losses
curr_lr = lr
best_train_loss = np.inf
best_val_loss = np.inf
n_since_best_train = -1
n_since_best_val = -1
curr_batch_accum = 0



# Load checkpoint if provided after initialization
# NOTE: This requires the model to have the same architecture as the checkpoint
# including compilation and device
if(checkpoint_path is not None):
    model.load_state_dict(torch.load(checkpoint_path))
    print(f"Model loaded from checkpoint at {checkpoint_path}", flush=True)



# One last reseed before the training loop 
# (in case we implement anything earlier that ticks the generator)
g.manual_seed(seed_val)
seed_everything_reproducibility(seed_val, force_cuda_determinism, error_on_nondeterministic)

# Training loop
if not skip_training:
    print("Starting training loop...", flush=True)
    for epoch in range(n_epochs):
        model.train()

        epoch_losses = []
        epoch_sizes = []
        n_processed = 0

        epoch_start_lr = curr_lr

        print(f'E {epoch:<5} |  LR: {epoch_start_lr:.2E}    ', end='\r', flush=True)

        for protein_g, molecule_g, target in train_dl:
            # Get number of processed elements and store
            curr_batch_size = len(target)
            n_processed += curr_batch_size

            # Zero the gradients
            optimizer.zero_grad(set_to_none=True)

            # Send the data to the device
            protein_g = protein_g.to(device, non_blocking=True)
            molecule_g = molecule_g.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            
            with torch.autocast(device.type, enabled=do_amp):
                # Get the affinity score
                # Use the forward_with_graphs call for simplicity
                affinity_score, _ = model.forward_with_graphs(protein_g, molecule_g)

                # Compute the loss
                loss = loss_fn(affinity_score.squeeze(), target.squeeze())

            # Backpropagate
            scaler.scale(loss).backward()

            curr_batch_accum += 1

            # Do optimizer step if accumulated enough (or are at end of batch)
            do_optim_step = (curr_batch_accum == num_batches_to_accum) or (n_processed == total_train)
            if (do_optim_step):
                # Unscale gradients
                scaler.unscale_(optimizer)

                # Clip gradients if requested
                if(clip_norm is not None):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)

                # Update the weights
                # (won't reunscale gradients if they were already unscaled)
                scaler.step(optimizer)
                scaler.update()

                curr_batch_accum = 0

            
            # Printing scaled/unscaled [reported] loss - (we need to use the scaled loss for training though)
            if(print_unscaled_loss):
                # Compute the unscaled loss for printing
                target_unscaled = dataset.unscale_target(target)
                affinity_score_unscaled = dataset.unscale_target(affinity_score)

                curr_loss_item = report_loss_fn(affinity_score_unscaled.squeeze(), target_unscaled.squeeze()).item()
            else:
                # Use the scaled loss for printing
                curr_loss_item = report_loss_fn(affinity_score.squeeze(), target.squeeze()).item()
            

            # Update scheduler at batch level if cosine or annealing
            if (scheduler_select in batch_schedulers) and (do_batch_schedule):
                if(epoch >= warmup_epochs):
                    scheduler.step(epoch - warmup_epochs + n_processed / total_train)
                    curr_lr = scheduler.get_last_lr()[0]
            

            # Print the loss (after special target unscaling if necessary)
            print(f'E {epoch:<5} |  LR: {curr_lr:.2E}   T: {curr_loss_item:.4f} ({n_processed}/{total_train})    ', end='\r', flush=True)
            epoch_losses.append(curr_loss_item)
            epoch_sizes.append(curr_batch_size)


        # Need to do weighted mean based on batch size
        # since the batch size is dynamic based on some limiting function
        mean_train_loss = np.average(epoch_losses, weights=epoch_sizes)
        print(f'E {epoch:<5} |  LR: {epoch_start_lr:.2E}   T: {mean_train_loss:.4f}                      ', end='\r', flush=True)

        model.eval()
        with torch.no_grad():
            epoch_val_losses = []
            epoch_val_sizes = []
            n_val_processed = 0

            for protein_g, molecule_g, target in val_dl:
                protein_g = protein_g.to(device)
                molecule_g = molecule_g.to(device)
                target = target.clone().detach().to(device) #because when unscaling, may modify inplace

                with torch.autocast(device.type, enabled=do_amp):
                    affinity_score, _ = model.forward_with_graphs(protein_g, molecule_g)

                curr_batch_size = len(target)
                n_val_processed += curr_batch_size

                if(print_unscaled_loss):
                    # Compute unscaled loss for printing
                    target_unscaled = dataset.unscale_target(target)
                    affinity_score_unscaled = dataset.unscale_target(affinity_score)

                    curr_val_loss_item = report_loss_fn(affinity_score_unscaled.squeeze(), target_unscaled.squeeze()).item()
                else:
                    # Compute the scaled loss for printing
                    val_loss = report_loss_fn(affinity_score.squeeze(), target.squeeze())
                    curr_val_loss_item = val_loss.item()


                print(f'E {epoch:<5} |  LR: {epoch_start_lr:.2E}   T: {mean_train_loss:.4f}   V: {curr_val_loss_item:.4f} ({n_val_processed}/{total_val})    ', end='\r', flush=True)
                epoch_val_losses.append(curr_val_loss_item)
                epoch_val_sizes.append(curr_batch_size)


        # Same weighted mean as for the training loss
        mean_val_loss = np.average(epoch_val_losses, weights=epoch_val_sizes)

        # Update epochs since best train and val (reset below if necessary)
        n_since_best_train += 1
        n_since_best_val += 1

        # Set new bests if necessary
        train_indicator_str = ''
        test_indicator_str = ''
        if(mean_train_loss < best_train_loss):
            best_train_loss = mean_train_loss
            train_indicator_str = f'*({n_since_best_train})  '
            n_since_best_train = 0

            # Save model state dict if train loss is better
            train_model_save_path = os.path.join(output_folder, f'besttrainmodel_{dataset_name}_train{best_train_loss:.4f}_epoch{epoch:>05d}.pt')
            torch.save(model.state_dict(), train_model_save_path)

        if(mean_val_loss < best_val_loss):
            best_val_loss = mean_val_loss
            test_indicator_str = f'**({n_since_best_val}) '
            n_since_best_val = 0

            # Save model state dict if validation loss is better
            val_model_save_path = os.path.join(output_folder, f'bestvalmodel_{dataset_name}_val{best_val_loss:.4f}_epoch{epoch:>05d}.pt')
            torch.save(model.state_dict(), val_model_save_path)

        indicator_str = f"{train_indicator_str:<8} {test_indicator_str}"

        print(f'E {epoch:<5} |  LR: {epoch_start_lr:.2E}   T: {mean_train_loss:.4f}   V: {mean_val_loss:.4f}   Best T/V: {best_train_loss:.4f} / {best_val_loss:.4f}   {indicator_str}', flush=True)

        if(n_since_best_val >= n_epochs_before_stop):
            print(f"\tEarly stopping at epoch {epoch} due to no improvement in validation loss for {n_epochs_before_stop} epochs")
            break


        # Update the learning rate based on plateau or other schedule
        if(epoch >= warmup_epochs):
            if(scheduler_select is not None):
                # Plateau needs metrics as an argument, all the others take in the current epoch
                # (note we zero-index epoch but the scheduler actually starts at 1)
                if(scheduler_select != 'plateau'):
                    scheduler.step(epoch-warmup_epochs+1)
                else:
                    scheduler.step(mean_val_loss)
                
                curr_lr = scheduler.get_last_lr()[0]

    
    # Save the final model after training completed (even if early stopped)
    finalmodel_save_path = os.path.join(output_folder, f'finalmodel_{dataset_name}_currval{mean_val_loss:.4f}_bestval{best_val_loss:.4f}_epoch{epoch:>05d}.pt')
    torch.save(model.state_dict(), finalmodel_save_path)

# Evaluate the model on the test set
# Note, we have additional metrics we want to compute here
# Such as the concordance index, RMSE, Pearson correlation, etc.
# So we will get the predictions and targets and compute these afterwards
print("\nEvaluating model on test set...", flush=True)

# Load the best requested model from the training process for evaluation
# (only if we did do additional training)
if not skip_training:
    model_load_path = get_best_model(output_folder, args.use_best)
    model.load_state_dict(torch.load(model_load_path))
    print(f"Model loaded for evaluation from best [{args.use_best}] model at {model_load_path}", flush=True)


model.eval()


with torch.no_grad():
    n_test_processed = 0

    test_preds = []
    test_targets = []

    for protein_g, molecule_g, target in test_dl:
        protein_g = protein_g.to(device)
        molecule_g = molecule_g.to(device)
        target = target.clone().detach().to(device) #because when unscaling, may modify inplace

        with torch.autocast(device.type, enabled=do_amp):
            affinity_score, _ = model.forward_with_graphs(protein_g, molecule_g)

        curr_batch_size = len(target)
        n_test_processed += curr_batch_size

        print(f'Test: ({n_test_processed}/{total_test})    ', end='\r', flush=True)
        test_preds.append(affinity_score)
        test_targets.append(target)
    
    # Concatenate the predictions and targets
    test_preds = torch.cat(test_preds, dim=0)
    test_targets = torch.cat(test_targets, dim=0)

    # Compute the unscaled loss for printing (always for test)
    test_preds_unscaled = dataset.unscale_target(test_preds).squeeze()
    test_targets_unscaled = dataset.unscale_target(test_targets).squeeze()

    test_loss = report_loss_fn(test_preds_unscaled, test_targets_unscaled).item()


    # Compute the MSE for the test set (since may be different from report loss)
    mse_loss = torch.nn.MSELoss()(test_preds_unscaled, test_targets_unscaled).item()

    # Compute RMSE
    rmse_loss = np.sqrt(mse_loss)

    # Compute MAE
    mae_loss = torch.nn.L1Loss()(test_preds_unscaled, test_targets_unscaled).item()

    # Compute Pearson correlation
    pearson_corr = torch.corrcoef(torch.stack([test_preds_unscaled, test_targets_unscaled]))[0][1].item()

    # Compute Concordance Index
    # Function from https://gitlab.com/mahnewton/daap/-/blob/main/scripts/prediction.py?ref_type=heads
    def calc_concordance_index(y_true, y_pred):
        summ = 0
        pair = 0

        for i in range(1, len(y_true)):
            for j in range(0, i):
                pair += 1
                if y_true[i] > y_true[j]:
                    summ += 1 * (y_pred[i] > y_pred[j]) + 0.5 * (y_pred[i] == y_pred[j])
                elif y_true[i] < y_true[j]:
                    summ += 1 * (y_pred[i] < y_pred[j]) + 0.5 * (y_pred[i] == y_pred[j])
                else:
                    pair -= 1

        if pair != 0:
            return summ / pair
        else:
            return 0

    # Compute the concordance index using the above function
    concordance_index = calc_concordance_index(test_targets_unscaled.cpu().numpy(), test_preds_unscaled.cpu().numpy())

    # Print the test metrics
    print(f'Test metrics                ', flush=True)
    print(f'\tReport Loss: {test_loss:.4f}', flush=True)
    print(f'\tMSE Loss: {mse_loss:.4f}', flush=True)
    print(f'\tRMSE Loss: {rmse_loss:.4f}', flush=True)
    print(f'\tMAE Loss: {mae_loss:.4f}', flush=True)
    print(f'\tPearson Correlation: {pearson_corr:.4f}', flush=True)
    print(f'\tConcordance Index: {concordance_index:.4f}', flush=True)



# Determine leakage of proteins and molecules in this dataset split
dses = {'train': train_ds, 
        'val': val_ds, 
        'test': test_ds}

ds_counts = {}

for ds_type, ds in dses.items():
    prots = [ds.dataset.pair_indices[x][0] for x in ds.indices]
    mols = [ds.dataset.pair_indices[x][1] for x in ds.indices]

    unique_prots, prot_counts = np.unique(prots, return_counts=True)
    unique_mols, mol_counts = np.unique(mols, return_counts=True)

    prot_dict = {prot: count for prot, count in zip(unique_prots, prot_counts)}
    mol_dict = {mol: count for mol, count in zip(unique_mols, mol_counts)}

    ds_counts[ds_type] = {'prot': prot_dict,
                          'mol': mol_dict}


print("Output folder: ", output_folder, flush=True)

ipdb.set_trace()
