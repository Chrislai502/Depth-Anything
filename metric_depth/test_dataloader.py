import argparse
from zoedepth.utils.config import get_config
from zoedepth.data.data_mono import MixedARTKITTINYU, DepthDataLoader
from pprint import pprint
import torch
import os
import matplotlib.pyplot as plt  # Added for displaying images
import torchvision.transforms as T  # For unnormalizing tensors

# Set environment variables for OpenGL and WandB (if needed)
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["WANDB_START_METHOD"] = "thread"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

def vstack_batch_images(batch_images):
    """
    Vertically stack a batch of images.

    Args:
        batch_images (torch.Tensor): A batch of images with shape (B, C, H, W).
    
    Returns:
        torch.Tensor: A single image tensor with all images vertically stacked.
    """
    # Assuming the batch is a tensor of shape (B, C, H, W)
    return torch.cat([img for img in batch_images], dim=1)  # Stack along height (dim=1)

def show_vstacked_images(batch_images):
    """
    Display a batch of images vertically stacked using matplotlib.

    Args:
        batch_images (torch.Tensor): A batch of images with shape (B, C, H, W).
    """
    # batch_images = torch.stack([unnormalize(img) for img in batch_images])  # Unnormalize all images
    batch_images = torch.stack([img for img in batch_images])  # Unnormalize all images
    
    batch_images = torch.clamp(batch_images, 0, 1)  # Clip values to [0, 1]

    # Vertically stack all images in the batch
    vstacked_image = vstack_batch_images(batch_images)

    # Convert to NumPy for displaying
    vstacked_image_np = vstacked_image.permute(1, 2, 0).cpu().numpy()

    # Display using matplotlib
    plt.imshow(vstacked_image_np)
    plt.axis('off')
    plt.show(block=False)  # Show without blocking execution
    plt.waitforbuttonpress()  # Wait for a key press
    plt.close()  # Close the window after key press

def test_data_loading(config, dataset_class, sample_ratio=None, mode="train"):
    """
    Test loading and iterating over a dataset.

    Args:
        config: Configuration object for the dataset.
        dataset_class: Dataset class to be tested (MixedNYUKITTI or MixedARTKITTINYU).
        mode (str): Mode for data loading, e.g., "train" or "online_eval".
    """
    print(f"\nTesting {dataset_class.__name__} in '{mode}' mode")

    # Initialize the dataset loader
    if sample_ratio:
        data_loader = dataset_class(config, mode, sample_ratio).data
    else:
        data_loader = DepthDataLoader(config, mode, device=device).data


    num_batches_to_test = 200
    # Iterate through batches to test data loading
    exit(0)
    for i, batch in enumerate(data_loader):
        print(f"Batch {i + 1}")
        
        if isinstance(False, bool):
            print( f"Batch {i + 1}/{num_batches_to_test} has no valid depth")
        for key, value in batch.items():
            if torch.is_tensor(value):
                print(f"  {key}: shape {value.shape}")
            # else:
        # # Display the vertically stacked images of the batch
        # if 'image' in batch:
        #     print(f"  Displaying vertically stacked images from batch {i + 1}")
        #     images = batch['image']  # Entire batch of images (B, C, H, W)
        #     show_vstacked_images(images)

        # # Log shapes of other batch data
        # for key, value in batch.items():
        #     if torch.is_tensor(value):
        #         print(f"  {key}: shape {value.shape}")

        # # Press a key to proceed to the next batch
        # print("Press a key to proceed to the next batch.")
        # # break  # Remove this break to process all batches

if __name__ == '__main__':
    # Fix random seed for reproducibility
    fix_random_seed(43)
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA is not available. Please use a GPU to run the script.")

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Test data loading for MixedARTKITTINYU datasets.")
    parser.add_argument("-m", "--model", type=str, default="zoedepth", help="Model name (default: synunet)")
    parser.add_argument("-d", "--dataset", type=str, default='mix', help="Dataset name (default: mix)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for testing (default: 4)")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for data loading (default: 2)")

    args, unknown_args = parser.parse_known_args()

    # Load configuration
    mode = "train"
    config = get_config(args.model, mode, args.dataset)
    print(f"Config in test_dataloader.py: {hex(id(config))}")
    config.batch_size = args.batch_size
    config.num_workers = args.num_workers
    config.mode = mode

    # Test MixedARTKITTINYU
    test_data_loading(config, MixedARTKITTINYU, {'art': 1, 'kitti': 2}, mode=mode)
    # test_data_loading(config, DepthDataLoader, mode=mode)
