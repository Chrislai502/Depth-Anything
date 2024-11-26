# This script saves the Zoe Depth Maps into a PNG format, for faster validation.

import argparse
from pprint import pprint

import torch
from zoedepth.utils.easydict import EasyDict as edict
from tqdm import tqdm

from zoedepth.data.data_mono import DepthDataLoader
from zoedepth.models.builder import build_model
from zoedepth.utils.arg_utils import parse_unknown
from zoedepth.utils.config import HOME_DIR, get_config, ALL_EVAL_DATASETS, ALL_INDOOR, ALL_OUTDOOR
from zoedepth.utils.misc import (RunningAverageDict, colors, compute_metrics,
                        count_parameters, colorize, colorize3D)

import numpy as np
import os
from PIL import Image
import torch.nn as nn

import argparse
from pprint import pprint
import torchvision.transforms as transforms

def get_depth_from_prediction(pred):
    if isinstance(pred, torch.Tensor):
        pred = pred  # pass
    elif isinstance(pred, (list, tuple)):
        pred = pred[-1]
    elif isinstance(pred, dict):
        pred = pred['metric_depth'] if 'metric_depth' in pred else pred['out']
    else:
        raise NotImplementedError(f"Unknown output type {type(pred)}")
    return pred

def main(config):
    model = build_model(config)
    
    test_loader = DepthDataLoader(config, 'online_eval').data
    model = model.cuda()
    model.eval()
    
    # Save path prefix
    depth_prefix = os.path.join(HOME_DIR, config.dataset[:3].upper(), config.track, config.bag, "groundtruth_dense")
    os.makedirs(depth_prefix, exist_ok=True)
    print(f"Saving to {depth_prefix}")
    for i, sample in tqdm(enumerate(test_loader), total=len(test_loader)):

        image = sample['image']
        image = image.cuda()
        focal = sample.get('focal', torch.Tensor(
                [715.0873]).cuda())  # This magic number (focal) is only used for evaluating BTS model
        with torch.no_grad():
            pred1 = model(image, dataset=sample['dataset'][0], focal=focal)
        pred1 = get_depth_from_prediction(pred1)
        pred1 = pred1.cpu().squeeze().squeeze().numpy().astype(dtype = np.float32)
        color_map_img = colorize3D(pred1, 0.0001, 120, cmap='magma_r')

        # For the whole batch, get the save path
        pred1 = np.round(pred1) # If not, truncated will happen
        pred1 = np.clip(pred1, 0, 255) # To prevent overflow
        pred1 *= 256.0
        pred1 = pred1.astype(np.uint16) # to png, 0 - 65535
        pred1 = Image.fromarray(pred1, mode='I;16')
        color_map_img = Image.fromarray(color_map_img)
        for i in range(len(sample['depth_path'])):
            depth_path = os.path.join(depth_prefix, sample['depth_path'][i].split('/')[-1])
            pred1.save(depth_path)
            color_path = os.path.join(depth_prefix, "../color", sample['depth_path'][i].split('/')[-1])
            color_map_img.save(color_path)
            # raise NotImplementedError
    print("Done Saving Files!")
            

def eval_model(model_name, pretrained_resource, dataset='nyu', **kwargs):

    # Load default pretrained resource defined in config if not set
    overwrite = {**kwargs, "pretrained_resource": pretrained_resource} if pretrained_resource else kwargs
    config = get_config(model_name, "eval", dataset, **overwrite)
    pprint(config)
    print(f"Evaluating {model_name} on {dataset}...")
    metrics = main(config)
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str,
                        required=True, help="Name of the model to evaluate")
    parser.add_argument("-p", "--pretrained_resource", type=str,
                        required=False, default="", help="Pretrained resource to use for fetching weights. If not set, default resource from model config is used,  Refer models.model_io.load_state_from_resource for more details.")
    parser.add_argument("-d", "--dataset", type=str, required=False,
                        default='art_test', help="Dataset to evaluate on")

    args, unknown_args = parser.parse_known_args()
    overwrite_kwargs = parse_unknown(unknown_args)

    if "ALL_INDOOR" in args.dataset:
        datasets = ALL_INDOOR
    elif "ALL_OUTDOOR" in args.dataset:
        datasets = ALL_OUTDOOR
    elif "ALL" in args.dataset:
        datasets = ALL_EVAL_DATASETS
    elif "," in args.dataset:
        datasets = args.dataset.split(",")
    else:
        datasets = [args.dataset]
    
    for dataset in datasets:
        eval_model(args.model, pretrained_resource=args.pretrained_resource,
                    dataset=dataset, **overwrite_kwargs)
