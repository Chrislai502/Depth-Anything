# MIT License

# Copyright (c) 2022 Intelligent Systems Lab Org

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# Import necessary modules from the zoedepth library
from zoedepth.utils.misc import count_parameters, parallelize
from zoedepth.utils.config import get_config
from zoedepth.utils.arg_utils import parse_unknown
from zoedepth.trainers.builder import get_trainer
from zoedepth.models.builder import build_model
from zoedepth.data.data_mono import MixedARTKITTINYU, DepthDataLoader

import sys
import os

current_file_path = os.path.abspath(__file__)  # Absolute path to train_mix_infer.py
root_dir = os.path.abspath(os.path.join(current_file_path, '..', '..'))  # Go two levels up
sys.path.append(root_dir)  # Add the root directory to sys.path
from depth_anything.dpt import DPT_DINOv2

# Import PyTorch modules for distributed data handling and multiprocessing
import torch.utils.data.distributed
import torch.multiprocessing as mp
import torch
import numpy as np
from pprint import pprint
import argparse

# Set environment variables for OpenGL and Weights & Biases (WandB)
os.environ["PYOPENGL_PLATFORM"] = "egl"  # Set OpenGL platform to egl for rendering
os.environ["WANDB_START_METHOD"] = "thread"  # Set WandB to start with threading

def fix_random_seed(seed: int):
    """
    Fix the random seed for reproducibility.

    Args:
        seed (int): Random seed value for deterministic behavior across runs.
    """
    import random
    import numpy
    import torch

    # Set seeds for random number generators across different libraries
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU environments

    # Set deterministic behavior for CUDA to ensure reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_ckpt(config, model, checkpoint_dir="./checkpoints", ckpt_type="best"):
    """
    Load model checkpoint based on configuration and checkpoint pattern.

    Args:
        config: Configuration object that may contain checkpoint details.
        model: The model to load weights into.
        checkpoint_dir (str): Directory where checkpoints are stored.
        ckpt_type (str): Type of checkpoint to load (e.g., 'best').

    Returns:
        model: Model with loaded weights if checkpoint exists, else original model.
    """
    import glob
    import os
    from zoedepth.models.model_io import load_wts

    # Check if the config has a specific checkpoint file
    if hasattr(config, "checkpoint"):
        checkpoint = config.checkpoint
    elif hasattr(config, "ckpt_pattern"):
        # Use a pattern to find checkpoint files matching the specified type
        pattern = config.ckpt_pattern
        matches = glob.glob(os.path.join(checkpoint_dir, f"*{pattern}*{ckpt_type}*"))
        if not matches:
            raise ValueError(f"No matches found for the pattern {pattern}")

        checkpoint = matches[0]
    else:
        return model  # Return the model unchanged if no checkpoint info is found

    # Load weights into the model and return it
    model = load_wts(model, checkpoint)
    print("Loaded weights from {0}".format(checkpoint))
    return model

def main_worker(gpu, ngpus_per_node, config, eval_config, dataset):
    """
    Main function to set up and start training on a specific GPU.

    Args:
        gpu: The GPU ID for the current process.
        ngpus_per_node: Number of GPUs per node.
        config: Configuration object for the training setup.
    """
    try:
        # Fix the random seed for reproducibility
        fix_random_seed(43)
        
        # Set the GPU ID in the configuration
        config.gpu = gpu

        # Build the model according to the config
        assert config.encoder in ['vits', 'vitb', 'vitl']
        prefix = "../"
        if config.encoder == 'vits': # 24.79 M Params
            model = DPT_DINOv2(encoder='vits', features=64, out_channels=[48, 96, 192, 384], localhub='localhub', prefix=prefix)
        elif config.encoder == 'vitb': # 97M Params
            model = DPT_DINOv2(encoder='vitb', features=128, out_channels=[96, 192, 384, 768], localhub='localhub', prefix=prefix)
        else: # 330M + Params
            model = DPT_DINOv2(encoder='vitl', features=256, out_channels=[256, 512, 1024, 1024], localhub='localhub', prefix=prefix)

        # Load model weights from checkpoint if specified in config
        model = load_ckpt(config, model)
        
        # Distribute the model across GPUs if needed
        model = parallelize(config, model)

        # Count and log the total number of parameters in the model
        total_params = f"{round(count_parameters(model)/1e6,2)}M"
        config.total_params = total_params
        print(f"Total parameters : {total_params}")

        # Create training and testing dataloaders
        if dataset == "mix":
            train_loader = MixedARTKITTINYU(config, "train").data
        else:
            train_loader = DepthDataLoader(config, "train").data
        test_loader = DepthDataLoader(eval_config, 'eval').data
        
        # Printing Trainloader and test loader information
        print(f"Trainloader length: {len(train_loader)}")
        print(f"Testloader length: {len(test_loader)}")

        # Get the trainer based on the config and initialize it with model and dataloaders
        trainer = get_trainer(config)(
            config, model, train_loader, test_loader, device=config.gpu
        )

        # Start training
        trainer.train()
    finally:
        # Ensure WandB logs are properly closed and saved
        import wandb
        wandb.finish()

if __name__ == '__main__':
    # Set the start method for multiprocessing to 'forkserver' for better compatibility
    mp.set_start_method('forkserver')

    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="depthanything")
    parser.add_argument("-d", "--dataset", type=str, default='mix') # art, kitti
    parser.add_argument("--trainer", type=str, default=None)

    # Parse known and unknown arguments
    args, unknown_args = parser.parse_known_args()
    overwrite_kwargs = parse_unknown(unknown_args)

    # Update model and trainer arguments
    overwrite_kwargs["model"] = args.model
    if args.trainer is not None:
        overwrite_kwargs["trainer"] = args.trainer

    # Load configuration based on model, mode, and dataset arguments
    train_config = get_config(args.model, "train", args.dataset, **overwrite_kwargs)
    eval_config = get_config(args.model, "eval", 'art_test', **overwrite_kwargs)

    # Initialize a shared dictionary if specified in the config for multiprocessing
    if train_config.use_shared_dict:
        shared_dict = mp.Manager().dict()
    else:
        shared_dict = None
    train_config.shared_dict = shared_dict

    # Set additional configurations
    train_config.batch_size = train_config.bs  # Set batch size from config
    train_config.mode = 'train'          # Set mode to 'train'

    # Ensure the root directory exists, create it if necessary
    if train_config.root != "." and not os.path.isdir(train_config.root):
        os.makedirs(train_config.root)

    # Try to configure distributed settings using SLURM environment variables
    try:
        node_str = os.environ['SLURM_JOB_NODELIST'].replace('[', '').replace(']', '')
        nodes = node_str.split(',')

        train_config.world_size = len(nodes)  # Total number of nodes in SLURM job
        train_config.rank = int(os.environ['SLURM_PROCID'])  # Rank of the current process
    except KeyError:
        # If SLURM is not being used, default to a single-node setup
        train_config.world_size = 1
        train_config.rank = 0
        nodes = ["127.0.0.1"]

    # Set up distributed training if enabled in config
    if train_config.distributed:
        port = np.random.randint(15000, 15025)  # Random port for distributed communication
        train_config.dist_url = f'tcp://{nodes[0]}:{port}'
        train_config.dist_backend = 'nccl'  # Use NCCL backend for GPU communication
        train_config.gpu = None

    # Determine the number of GPUs available on this node
    ngpus_per_node = torch.cuda.device_count()
    train_config.num_workers = train_config.workers  # Set number of data loading workers
    train_config.ngpus_per_node = ngpus_per_node

    # Print the final configuration for verification
    print("Config:")
    pprint(train_config)

    if train_config.distributed:
        # In distributed mode, calculate the total world size across all nodes
        train_config.world_size = ngpus_per_node * train_config.world_size

        # Spawn a separate process for each GPU on the node for distributed training
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, train_config, eval_config, args.dataset))
    else:
        # For non-distributed mode, if only one GPU, set GPU ID to 0
        if ngpus_per_node == 1:
            train_config.gpu = 0
        main_worker(train_config.gpu, ngpus_per_node, train_config, eval_config, args.dataset)
