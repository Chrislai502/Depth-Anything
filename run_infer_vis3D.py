import argparse
import cv2
import numpy as np
import os
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
import matplotlib.pyplot as plt
from PIL import Image
import yaml
import open3d as o3d

from depth_anything.dpt import DepthAnything, DPT_DINOv2
from depth_anything.util.transform import NormalizeImage, PrepareForNet

# Constants for cropping
CROP_HEIGHT = 154
CROP_WIDTH = 1008

# Calibration parameters
FX = 491.44
FY = 491.44
CX = 510.0
CY = 398.0

def load_ckpt(model, checkpoint_path="rear_stuff/DepthAnythingv1_11-Dec_12-04-50740ac911a2_latest.pt", ckpt_type="best"):
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    state_dict = ckpt.get('model', ckpt)
    model.load_state_dict(state_dict, strict=False)
    print(f"✅ Loaded model checkpoint from {checkpoint_path}")
    return model

def crop_art_style(image):
    """Apply ART dataset-style cropping to an image."""
    if isinstance(image, Image.Image):
        image = np.asarray(image, dtype=np.float32) / 255.0

    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)

    height, width, _ = image.shape
    crop_height = min(height, CROP_HEIGHT)
    crop_width = min(width, CROP_WIDTH)

    temp_pad = 0
    bottom_margin = ((height - crop_height) // 2) + temp_pad
    top_margin = height - bottom_margin - temp_pad
    left_margin = (width - crop_width) // 2
    right_margin = width - left_margin

    image = image[bottom_margin:top_margin, left_margin:right_margin, :]

    if height < CROP_HEIGHT or width < CROP_WIDTH:
        image = cv2.resize(image, (CROP_WIDTH, CROP_HEIGHT), interpolation=cv2.INTER_LINEAR)

    return image

def backproject_to_3d(depth_map, fx, fy, cx, cy):
    """
    Back-projects a 2D depth map into 3D space using given camera intrinsics.
    """
    height, width = depth_map.shape
    i, j = np.meshgrid(np.arange(width), np.arange(height), indexing='xy')
    x = (i - cx) / fx
    y = (j - cy) / fy
    x = -x
    y = -y
    X = x * depth_map
    Y = y * depth_map
    Z = depth_map
    points = np.stack((X, Y, Z), axis=-1)
    return points

def visualize_depth_map(depth_map, valid_mask):
    
    # Display the original image and depth prediction
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(raw_image)
    plt.title("Original Image")
    plt.axis("off")
    
    title_str = f"Depth Prediction (Target: {args.target_depth} m)" if args.target_depth is not None else "Depth Prediction"
    plt.subplot(1, 2, 2)
    cmap_used = "gray" if args.grayscale else "inferno"
    plt.imshow(depth, cmap=cmap_used)
    plt.title(title_str)
    plt.axis("off")
    plt.show()

def visualize_point_cloud(points, valid_mask):
    """
    Visualizes a 3D point cloud from depth information using Open3D.
    """
    points = points.reshape(-1, 3)
    valid_mask = valid_mask.flatten()
    points = points[valid_mask]
    
    depth = points[:, 2]
    depth_normalized = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
    cmap = plt.get_cmap('jet')
    colors = cmap(depth_normalized)[:, :3]
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3.0, origin=[0, 0, 0])
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Point Cloud', width=800, height=600)
    vis.add_geometry(pcd)
    vis.add_geometry(coordinate_frame)
    render_option = vis.get_render_option()
    render_option.background_color = np.asarray([0, 0, 0])
    render_option.point_size = 1.0
    vis.run()
    vis.destroy_window()

def get_first_image_from_config():
    """
    Reads config.yml to get the dataset folder and returns the first image found.
    """
    with open("config.yml", "r") as file:
        config = yaml.safe_load(file)
    
    dataset_folder = config.get("dataset")
    if not dataset_folder:
        raise ValueError("No 'dataset' folder defined in config.yml")
    
    image_paths = []
    for root, _, files in os.walk(dataset_folder):
        for file in sorted(files):  # sort for consistency
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                image_paths.append(os.path.join(root, file))
    
    if not image_paths:
        raise ValueError(f"No images found in the dataset folder: {dataset_folder}")
    
    print(f"🔍 Found {len(image_paths)} images in dataset. Using the first image: {image_paths[0]}")
    return image_paths[1]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-path', type=str, help="Path to the input image. If not provided, the first image from the config.yml dataset will be used.")
    parser.add_argument('--outdir', type=str, default='./vis_depth', help="Output directory.")
    parser.add_argument('--encoder', type=str, default='vits', choices=['vits', 'vitb', 'vitl'], help="Encoder type for the model.")
    parser.add_argument('--pred-only', dest='pred_only', action='store_true', help="Only display the prediction (skip 3D visualization).")
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help="Do not apply a colorful palette.")
    parser.add_argument('--target-depth', type=float, default=None, help="Optional target depth value for display.")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    
    global DEVICE
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # If an image path is not provided, get the first image from the config dataset folder.
    if not args.img_path:
        args.img_path = get_first_image_from_config()

    # Initialize and load the model
    depth_anything = DPT_DINOv2(encoder=args.encoder, features=64, out_channels=[48, 96, 192, 384], localhub=False)
    depth_anything = load_ckpt(depth_anything).to(DEVICE).eval()
    
    total_params = sum(param.numel() for param in depth_anything.parameters())
    print('Total parameters: {:.2f}M'.format(total_params / 1e6))
    
    transform = Compose([
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])
    
    # Read and preprocess the image
    raw_image = cv2.cvtColor(cv2.imread(args.img_path), cv2.COLOR_BGR2RGB)
    raw_image = cv2.resize(raw_image, (1006, 696), interpolation=cv2.INTER_CUBIC)
    raw_image = crop_art_style(raw_image)
    image = raw_image / 255.0
    h, w = image.shape[:2]
    image_input = transform({'image': image})['image']
    image_input = torch.from_numpy(image_input).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        depth = depth_anything(image_input)
    depth = F.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
    depth = 120 - torch.clamp(depth, min=0, max=120)
    depth = depth.cpu().numpy()
    
    # 3D visualization if not skipped
    if not args.pred_only:
        valid_mask = depth > 0
        points_3d = backproject_to_3d(depth, fx=FX, fy=FY, cx=CX, cy=CY)
        visualize_depth_map(depth, valid_mask)
        visualize_point_cloud(points_3d, valid_mask)