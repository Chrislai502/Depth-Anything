import argparse
import cv2
import numpy as np
import os
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
from tqdm import tqdm
import json
from PIL import Image
from matplotlib import pyplot as plt
import yaml

from depth_anything.dpt import DepthAnything, DPT_DINOv2
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

CROP_HEIGHT = 154
CROP_WIDTH = 1008

def load_ckpt(model, checkpoint_path="rear_stuff/DepthAnythingv1_11-Dec_12-04-50740ac911a2_latest.pt", ckpt_type="best"):
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    state_dict = ckpt.get('model', ckpt)
    model.load_state_dict(state_dict, strict=False)
    print(f"✅ Loaded model checkpoint from {checkpoint_path}")
    return model

def crop_art_style(image):
    """Applies ART dataset-style cropping to an image."""
    if isinstance(image, Image.Image):  # Convert PIL to NumPy
        image = np.asarray(image, dtype=np.float32) / 255.0

    if image.ndim == 2:  # Convert grayscale to RGB
        image = np.stack([image] * 3, axis=-1)

    height, width, _ = image.shape
    crop_height = min(height, CROP_HEIGHT)
    crop_width = min(width, CROP_WIDTH)

    # Compute cropping margins
    temp_pad = 0
    bottom_margin = ((height - crop_height) // 2)  + temp_pad
    top_margin = height - bottom_margin - temp_pad
    left_margin = (width - crop_width) // 2
    right_margin = width - left_margin

    # Crop image
    image = image[bottom_margin:top_margin, left_margin:right_margin, :]

    # Resize only if necessary
    if height < CROP_HEIGHT or width < CROP_WIDTH:
        image = cv2.resize(image, (CROP_WIDTH, CROP_HEIGHT), interpolation=cv2.INTER_LINEAR)

    return image

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-path', type=str)
    parser.add_argument('--outdir', type=str, default='./vis_depth')
    parser.add_argument('--encoder', type=str, default='vits', choices=['vits', 'vitb', 'vitl'])
    
    parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='only display the prediction')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')
    
    args = parser.parse_args()
    
    margin_width = 50
    caption_height = 60
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    depth_anything = DPT_DINOv2(encoder='vits', features=64, out_channels=[48, 96, 192, 384], localhub=False)
    depth_anything = load_ckpt(depth_anything).to(DEVICE).eval()
    # DepthAnything.from_pretrained('LiheYoung/depth_anything_{}14'.format(args.encoder)).to(DEVICE).eval()
    
    total_params = sum(param.numel() for param in depth_anything.parameters())
    print('Total parameters: {:.2f}M'.format(total_params / 1e6))
    
    transform = Compose([
        # Resize(
        #     width=1006,
        #     height=696,
        #     resize_target=False,
        #     keep_aspect_ratio=True,
        #     ensure_multiple_of=14,
        #     resize_method='lower_bound',
        #     image_interpolation_method=cv2.INTER_CUBIC,
        # ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])
    
    with open("config.yml", "r") as file:
        config = yaml.safe_load(file)
    
    image_paths = []
    for root, _, files in os.walk(config["dataset"]):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                image_paths.append(os.path.join(root, file))

    print(f"🔍 Found {len(image_paths)} images for inference.")
    
    os.makedirs(args.outdir, exist_ok=True)
    
    for img_path in tqdm(image_paths):
        
        filename = os.path.basename(img_path)
        timestamp_str = filename.split(".")[0]  # Extract timestamp from filename
        target_depth = config["target_depths"].get(int(timestamp_str), "Unknown")

        raw_image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        # resize raw image
        raw_image = cv2.resize(raw_image, (1006, 696), interpolation=cv2.INTER_CUBIC)
        raw_image = crop_art_style(raw_image)
        # image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
        image = raw_image /255.0
        
        # image = crop_art_style(image)
        h, w = image.shape[:2]
        image = transform({'image': image})['image']
        image = torch.from_numpy(image).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            depth = depth_anything(image)
        
        depth = F.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
        # depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        
        depth = 120 - torch.clamp(depth, min=0, max=120) # 
        depth = depth.cpu().numpy()
        
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(raw_image)
        plt.title("Original Image")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(depth, cmap="inferno")
        plt.title(f"Rear car is {target_depth} m away")
        plt.axis("off")

        plt.show()