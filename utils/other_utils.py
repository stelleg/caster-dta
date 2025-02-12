import random
import torch
import numpy as np
import os

# Seed everything for reproducibility
def seed_everything_reproducibility(seed, force_cuda_determinism=False,
                                    error_on_nondeterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Required for true reproducibility if running on GPU
    # But severely degrades performance, generally
    if(force_cuda_determinism):
        # Make sure the GPU tries to be deterministic
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Note this will deteriorate performance or increase memory usage
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"

    if(error_on_nondeterministic):
        # PyTorch function that forces true determinism and errors on non-deterministic behavior
        # Note that this can slow down the code and should be used only for debugging
        torch.use_deterministic_algorithms(True)