import argparse
from zoedepth.utils.config import get_config
from zoedepth.data.data_mono import MixedARTKITTINYU
from pprint import pprint
import torch
import os

# Set environment variables for OpenGL and WandB (if needed)
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["WANDB_START_METHOD"] = "thread"

def fix_random_seed(seed: int):
    """
    Fix random seed for reproducibility

    Args:
        seed (int): random seed
    """
    import random
    import numpy
    import torch

    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

def test_data_loading(config, dataset_class, sample_ratio = None, mode="train"):
    """
    Test loading and iterating over a dataset.

    Args:
        config: Configuration object for the dataset.
        dataset_class: Dataset class to be tested (MixedNYUKITTI or MixedARTKITTINYU).
        mode (str): Mode for data loading, e.g., "train" or "online_eval".
    """
    print(f"\nTesting {dataset_class.__name__} in '{mode}' mode")
    # print("Debug check torch cuda is available: ", torch.cuda.is_available())

    # Initialize the dataset loader
    if sample_ratio:
        data_loader = dataset_class(config, mode, sample_ratio).data
    else:
        data_loader = dataset_class(config, mode).data

    # Print the config to verify settings
    # pprint(config)

    # Iterate through a few batches to test data loading
    num_batches_to_test = 2000
    print
    for i, batch in enumerate(data_loader):
        print(f"Batch {i + 1}/{num_batches_to_test}")
        # for key, value in batch.items():
        #     if torch.is_tensor(value):
        #         # print(f"  {key}: shape {value.shape}")
        #     # else:
        #     #     print(f"  {key}: {value}")
        #         pass
        
        if i + 1 >= num_batches_to_test:
            break  # Stop after testing a few batches

if __name__ == '__main__':
    # Fix random seed for reproducibility
    fix_random_seed(41)
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA is not available. Please use a GPU to run the script.")

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Test data loading for MixedARTKITTINYU datasets.")
    parser.add_argument("-m", "--model", type=str, default="zoedepth", help="Model name (default: synunet)")
    parser.add_argument("-d", "--dataset", type=str, default='mix', help="Dataset name (default: mix)")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for testing (default: 4)")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of workers for data loading (default: 2)")

    args, unknown_args = parser.parse_known_args()

    # Load configuration
    config = get_config(args.model, "train", args.dataset)
    config = get_config(args.model, "eval", args.dataset)
    config.batch_size = args.batch_size
    config.num_workers = args.num_workers
    config.mode = 'train'  # Ensure mode is set to 'train'
    config.mode = 'eval'

    # Test MixedNYUKITTI
    # test_data_loading(config, MixedNYUKITTI, mode="train")

    # Test MixedARTKITTINYU
    test_data_loading(config, MixedARTKITTINYU, {'art': 1, 'kitti': 2}, mode="train")
