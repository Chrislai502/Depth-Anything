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

# File author: Shariq Farooq Bhat

import argparse
from pprint import pprint

import torch
from zoedepth.utils.easydict import EasyDict as edict
from tqdm import tqdm

from zoedepth.data.data_mono import DepthDataLoader
from zoedepth.models.builder import build_model
from zoedepth.utils.arg_utils import parse_unknown
from zoedepth.utils.config import change_dataset, get_config, ALL_EVAL_DATASETS, ALL_INDOOR, ALL_OUTDOOR
from zoedepth.utils.misc import (RunningAverageDict, colors, compute_metrics,
                        count_parameters, colorize, colorize3D)

import numpy as np
import os
from PIL import Image
import torch.nn as nn

import argparse
import time  # Import time for measuring durations
from pprint import pprint
import torchvision.transforms as transforms


# Placeholders for timing
data_loading_time = 0
inference_time = 0
post_processing_time = 0
image_saving_time = 0

# Global timing variables
global_timer_compute_errors = 0
global_timer_colorize = 0
global_timer_metrics_calculation = 0


def compute_errors(gt, pred):
    """Compute metrics for 'pred' compared to 'gt'

    Args:
        gt (numpy.ndarray): Ground truth values
        pred (numpy.ndarray): Predicted values

        gt.shape should be equal to pred.shape

    Returns:
        dict: Dictionary containing the following metrics:
            'a1': Delta1 accuracy: Fraction of pixels that are within a scale factor of 1.25
            'a2': Delta2 accuracy: Fraction of pixels that are within a scale factor of 1.25^2
            'a3': Delta3 accuracy: Fraction of pixels that are within a scale factor of 1.25^3
            'abs_rel': Absolute relative error
            'rmse': Root mean squared error
            'log_10': Absolute log10 error
            'sq_rel': Squared relative error
            'rmse_log': Root mean squared error on the log scale
            'silog': Scale invariant log error
    """
    
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    log_10 = (np.abs(np.log10(gt) - np.log10(pred))).mean()
    return dict(a1=a1, a2=a2, a3=a3, abs_rel=abs_rel, rmse=rmse, log_10=log_10, rmse_log=rmse_log,
                silog=silog, sq_rel=sq_rel)

def compute_errors_2d(img, i, gt, pred, valid_mask=None, save_err_img=False, path=None, max_depth_eval=80, min_depth_eval=0.1):
    """Compute metrics for 'pred' compared to 'gt'

    Args:
        gt (numpy.ndarray): Ground truth values
        pred (numpy.ndarray): Predicted values

        gt.shape should be equal to pred.shape

    Returns:
        dict: Dictionary containing the following metrics:
            'a1': Delta1 accuracy: Fraction of pixels that are within a scale factor of 1.25
            'a2': Delta2 accuracy: Fraction of pixels that are within a scale factor of 1.25^2
            'a3': Delta3 accuracy: Fraction of pixels that are within a scale factor of 1.25^3
            'abs_rel': Absolute relative error
            'rmse': Root mean squared error
            'log_10': Absolute log10 error
            'sq_rel': Squared relative error
            'rmse_log': Root mean squared error on the log scale
            'silog': Scale invariant log error
    """
    global global_timer_compute_errors, global_timer_colorize, global_timer_metrics_calculation

    start_time = time.time()


    # Mask invalid regions in gt and pred (THIS TAKES 2 SECONDS...)
    gt_filtered = np.where(valid_mask, gt, np.nan)  # Set invalid areas to NaN for visual clarity
    pred_filtered = np.where(valid_mask, pred, 0)
    global_timer_compute_errors += time.time() - start_time

    # Flatten valid values for error calculations
    gt_1d = gt[valid_mask]
    pred_1d = pred[valid_mask]


    # Calculate accuracy metrics
    metrics_start = time.time()
    thresh = np.maximum((gt_filtered / pred_filtered), (pred_filtered / gt_filtered))

    # Accuracy Metric Calculation
    a1_2d = (thresh < 1.25)
    a2_2d = (thresh < 1.25 ** 2)
    a3_2d = (thresh < 1.25 ** 3)

    a1 = a1_2d[valid_mask].mean()
    a2 = a2_2d[valid_mask].mean()
    a3 = a3_2d[valid_mask].mean()

    abs_rel_2d = np.absolute(gt - pred)
    abs_rel = (np.abs(gt_1d - pred_1d) / gt_1d).mean()

    # Rest can be done in 1d. Not interested anymore.
    sq_rel = np.mean(((gt_1d - pred_1d) ** 2) / gt_1d)

    rmse = (gt_1d - pred_1d) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt_1d) - np.log(pred_1d)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    err = (np.log(pred_1d) - np.log(gt_1d))
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    log_10 = (np.abs(np.log10(gt_1d) - np.log10(pred_1d))).mean()
    global_timer_metrics_calculation += time.time() - metrics_start

    if save_err_img:
        color_start = time.time()

        # Creating the delta accuracy image
        color_map_img = colorize3D(pred, min_depth_eval, max_depth_eval, cmap='magma_r')
        
        delta_img = color_map_img.copy()
        delta_img[~a3_2d & valid_mask] =  [54 ,0 ,0 ]# red
        delta_img[~a2_2d & valid_mask] =  [109,2 ,2 ]# red
        delta_img[~a1_2d & valid_mask] =  [163,24,24]# red

        # Creating abs_rel image
        abs_rel_img = colorize3D(abs_rel_2d, 0, 10.0, cmap='cool')
        abs_rel_img[~valid_mask] = color_map_img[~valid_mask]

        # Vertical Stack the images
        delta_img = np.vstack([img, abs_rel_img, delta_img])

        Image.fromarray(delta_img).save(os.path.join(path, f"{i}_delta_img.png"))

        global_timer_colorize += time.time() - color_start

    return dict(a1=a1, a2=a2, a3=a3, abs_rel=abs_rel, rmse=rmse, log_10=log_10, rmse_log=rmse_log,
                silog=silog, sq_rel=sq_rel)


def compute_metrics_and_save(img, i, gt, pred, interpolate=True, garg_crop=False, eigen_crop=True, dataset='nyu', min_depth_eval=0.1, max_depth_eval=10, **kwargs):
    """Compute metrics of predicted depth maps. Applies cropping and masking as necessary or specified via arguments. Refer to compute_errors for more details on metrics.
    """

    if 'config' in kwargs:
        config = kwargs['config']
        garg_crop = config.garg_crop
        eigen_crop = config.eigen_crop
        min_depth_eval = config.min_depth_eval
        max_depth_eval = config.max_depth_eval
        # print("MAX DEPTH EVAL: ", max_depth_eval)

    # If ground truth and prediction sizes do not match, and interpolation is requested, interpolate prediction
    if gt.shape[-2:] != pred.shape[-2:] and interpolate:
        pred = nn.functional.interpolate(
            pred, gt.shape[-2:], mode='bilinear', align_corners=True)

    # Prepare prediction data for evaluation: remove channel dimension, convert to numpy array, and enforce depth limits
    pred = pred.squeeze().cpu().numpy()
    pred[pred < min_depth_eval] = min_depth_eval
    pred[pred > max_depth_eval] = max_depth_eval
    pred[np.isinf(pred)] = 500
    pred[np.isnan(pred)] = min_depth_eval

    # Create a mask to ignore regions in the ground truth outside the min/max depth range
    gt_depth = gt.squeeze().cpu().numpy()
    valid_mask = np.logical_and(
        gt_depth > min_depth_eval, gt_depth < max_depth_eval)

    # Apply cropping if requested by either garg_crop or eigen_crop
    # print("GARG CROP: ", garg_crop, "EIGEN CROP: ", eigen_crop)
    if garg_crop or eigen_crop:
        gt_height, gt_width = gt_depth.shape
        eval_mask = np.zeros(valid_mask.shape)

        if garg_crop:
            eval_mask[int(0.40810811 * gt_height):int(0.99189189 * gt_height),
                      int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1

        elif eigen_crop:
            # print("-"*10, " EIGEN CROP ", "-"*10)
            if dataset == 'kitti':
                eval_mask[int(0.3324324 * gt_height):int(0.91351351 * gt_height),
                          int(0.0359477 * gt_width):int(0.96405229 * gt_width)] = 1
            else:
                # assert gt_depth.shape == (480, 640), "Error: Eigen crop is currently only valid for (480, 640) images"
                eval_mask[45:471, 41:601] = 1
        else:
            eval_mask = np.ones(valid_mask.shape)
    
    
    return compute_errors_2d(
        img, i, gt_depth, pred, 
        valid_mask, save_err_img=True, path=kwargs['path'] if 'path' in kwargs else None, 
        max_depth_eval=max_depth_eval, min_depth_eval=min_depth_eval)


@torch.no_grad()
def infer(model, images, **kwargs):
    start_time = time.time()  # Start timing inference

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

    pred1 = model(images, **kwargs)
    pred1 = get_depth_from_prediction(pred1)

    pred2 = model(torch.flip(images, [3]), **kwargs)
    pred2 = get_depth_from_prediction(pred2)
    pred2 = torch.flip(pred2, [3])

    mean_pred = 0.5 * (pred1 + pred2)

    global inference_time
    inference_time += time.time() - start_time  # Accumulate inference time

    return mean_pred


@torch.no_grad()
def evaluate(model, test_loader, config, round_vals=True, round_precision=3):
    model.eval()
    metrics = RunningAverageDict()

    global data_loading_time, inference_time, post_processing_time, image_saving_time
    counter = 0

    # Create output directory
    if config.dataset == 'art':
        path = os.path.join("./output", config.dataset, config.track, config.bag)
    else:
        path = os.path.join("./output", config.dataset)
    os.makedirs(path, exist_ok=True)

    # Dataset Root Directory

    for i, sample in tqdm(enumerate(test_loader), total=len(test_loader)):
        if 'has_valid_depth' in sample:
            if not sample['has_valid_depth']:
                continue
        image, depth = sample['image'], sample['depth']
        if "save_images" in config and config.save_images:
            image_cpu = (image.clone().detach().squeeze().numpy().transpose(1, 2, 0) * 255).astype(np.uint8) # This doesn't work
        
        image, depth = image.cuda(), depth.cuda()
        depth = depth.squeeze().unsqueeze(0).unsqueeze(0)
        focal = sample.get('focal', torch.Tensor(
            [715.0873]).cuda())  # This magic number (focal) is only used for evaluating BTS model
        pred = infer(model, image, dataset=sample['dataset'][0], focal=focal)

        if "save_images" in config and config.save_images:

            # Image saving timing
            image_save_start_time = time.time()
            curr_metrics = compute_metrics_and_save(image_cpu, i, depth, pred, path=path, config=config)
            image_saving_time += time.time() - image_save_start_time
        else:
            curr_metrics = compute_metrics(depth, pred, config=config)
        
        metrics.update(curr_metrics)
        # if counter == 10:
        #     break
        counter += 1

    # Print timing summary after evaluation
    print("\n--- Timing Summary ---")
    print(f"Data Loading Time: {data_loading_time/counter:.3f} seconds")
    print(f"Inference Time: {inference_time/counter:.3f} seconds")
    print(f"Post-Processing Time: {post_processing_time/counter:.3f} seconds")
    print(f"Image Saving Time: {image_saving_time/counter:.3f} seconds")

    print("\n--- Timing Breakdown ---")
    print(f"Average compute_errors time: {global_timer_compute_errors / counter:.3f} seconds")
    print(f"Average color mapping (colorize3D) time: {global_timer_colorize / counter:.3f} seconds")
    print(f"Average metric calculation time: {global_timer_metrics_calculation / counter:.3f} seconds")
    # print(f"Average image saving time: {global_timer_image_saving / counter:.3f} seconds")

    if round_vals:
        def r(m): return round(m, round_precision)
    else:
        def r(m): return m
    metrics = {k: r(v) for k, v in metrics.get_value().items()}
    return metrics

def main(config):
    model = build_model(config)
    
    test_loader = DepthDataLoader(config, 'online_eval').data
    model = model.cuda()
    metrics = evaluate(model, test_loader, config)
    print(f"{colors.fg.green}")
    print(metrics)
    print(f"{colors.reset}")
    metrics['#params'] = f"{round(count_parameters(model, include_all=True)/1e6, 2)}M"
    return metrics


def eval_model(model_name, pretrained_resource, dataset='nyu', **kwargs):

    # Load default pretrained resource defined in config if not set
    overwrite = {**kwargs, "pretrained_resource": pretrained_resource} if pretrained_resource else kwargs
    config = get_config(model_name, "eval", dataset, **overwrite)
    # config = change_dataset(config, dataset)  # change the dataset
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
                        default='nyu', help="Dataset to evaluate on")

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
