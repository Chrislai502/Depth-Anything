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

"""Miscellaneous utility functions."""

from scipy import ndimage

import base64
import math
import re
from io import BytesIO

import matplotlib
import matplotlib.cm
import numpy as np
import requests
import torch
import torch.distributed as dist
import torch.nn
import torch.nn as nn
import torch.utils.data.distributed
from PIL import Image
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from collections import defaultdict
import io
from PIL import Image

class RunningAverage:
    def __init__(self):
        self.avg = 0
        self.count = 0

    def append(self, value):
        self.avg = (value + self.count * self.avg) / (self.count + 1)
        self.count += 1

    def get_value(self):
        return self.avg


def denormalize(x):
    """Reverses the imagenet normalization applied to the input.

    Args:
        x (torch.Tensor - shape(N,3,H,W)): input tensor

    Returns:
        torch.Tensor - shape(N,3,H,W): Denormalized input
    """
    mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
    std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
    return x * std + mean


class RunningAverageDict:
    """A dictionary of running averages."""
    def __init__(self):
        self._dict = None

    def update(self, new_dict):
        if new_dict is None:
            return

        if self._dict is None:
            self._dict = dict()
            for key, value in new_dict.items():
                self._dict[key] = RunningAverage()

        for key, value in new_dict.items():
            if not math.isnan(value):
                self._dict[key].append(value)

    def get_value(self):
        if self._dict is None:
            return None
        return {key: value.get_value() for key, value in self._dict.items()}

    def get_n_count(self):
        if self._dict is None:
            return None
        return {key: value.count for key, value in self._dict.items()}

class RunningCountDict:
    def __init__(self):
        self._dict = None

    def update(self, new_dict):
        if new_dict is None:
            return

        if self._dict is None:
            self._dict = dict()
            for key, value in new_dict.items():
                self._dict[key] = 0

        for key, value in new_dict.items():
            self._dict[key] += int(value)
    
    def get_value(self):
        if self._dict is None:
            return None
        return self._dict

class RunningStats:
    """Tracks running statistics for a single variable."""
    def __init__(self):
        self.values = []  # Store values for statistics

    def append(self, value):
        """Add a new value to the running stats."""
        if value is not None:  # Ignore None values
            self.values.append(value)

    def mean(self):
        """Compute the mean of the values."""
        return np.mean(self.values) if self.values else None

    def stddev(self):
        """Compute the standard deviation of the values."""
        return np.std(self.values, ddof=1) if len(self.values) > 1 else None

    def median(self):
        """Compute the median of the values."""
        return np.median(self.values) if self.values else None

    def min(self):
        """Compute the minimum value."""
        return np.min(self.values) if self.values else None

    def max(self):
        """Compute the maximum value."""
        return np.max(self.values) if self.values else None

    def count(self):
        """Return the number of values."""
        return len(self.values)

    def summary(self):
        """Return a dictionary summarizing all statistics."""
        return {
            "mean": self.mean(),
            "stddev": self.stddev(),
            "median": self.median(),
            "min": self.min(),
            "max": self.max(),
            "count": self.count(),
        }


class RunningStatsDict:
    """Tracks running statistics for multiple variables stored as a dictionary."""
    def __init__(self):
        self._stats = defaultdict(RunningStats)  # Dictionary of RunningStats objects

    def update(self, new_dict):
        """Update the stats with a new dictionary of values."""
        if new_dict is None:
            return

        for key, value in new_dict.items():
            self._stats[key].append(value)

    def get_value(self):
        """Return a dictionary with all running statistics for each key."""
        return {key: stats.summary() for key, stats in self._stats.items()}

    def __str__(self):
        """Pretty-print the stats."""
        stats_str = "\n".join(f"{key}: {value.summary()}" for key, value in self._stats.items())
        return f"RunningStatsDict(\n{stats_str}\n)"
    
    
def colorize(value, vmin=None, vmax=None, cmap='gray_r', invalid_val=-99, invalid_mask=None, background_color=(128, 128, 128, 255), gamma_corrected=False, value_transform=None):
    """Converts a depth map to a color image.

    Args:
        value (torch.Tensor, numpy.ndarry): Input depth map. Shape: (H, W) or (1, H, W) or (1, 1, H, W). All singular dimensions are squeezed
        vmin (float, optional): vmin-valued entries are mapped to start color of cmap. If None, value.min() is used. Defaults to None.
        vmax (float, optional):  vmax-valued entries are mapped to end color of cmap. If None, value.max() is used. Defaults to None.
        cmap (str, optional): matplotlib colormap to use. Defaults to 'magma_r'.
        invalid_val (int, optional): Specifies value of invalid pixels that should be colored as 'background_color'. Defaults to -99.
        invalid_mask (numpy.ndarray, optional): Boolean mask for invalid regions. Defaults to None.
        background_color (tuple[int], optional): 4-tuple RGB color to give to invalid pixels. Defaults to (128, 128, 128, 255).
        gamma_corrected (bool, optional): Apply gamma correction to colored image. Defaults to False.
        value_transform (Callable, optional): Apply transform function to valid pixels before coloring. Defaults to None.

    Returns:
        numpy.ndarray, dtype - uint8: Colored depth map. Shape: (H, W, 4)
    """
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()

    value = value.squeeze()
    if invalid_mask is None:
        invalid_mask = value == invalid_val
    mask = np.logical_not(invalid_mask)

    # normalize
    vmin = np.percentile(value[mask],2) if vmin is None else vmin
    vmax = np.percentile(value[mask],85) if vmax is None else vmax
    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax
    else:
        # Avoid 0-division
        value = value * 0.

    # squeeze last dim if it exists
    # grey out the invalid values

    value[invalid_mask] = np.nan
    cmapper = matplotlib.cm.get_cmap(cmap)
    if value_transform:
        value = value_transform(value)
        # value = value / value.max()
    value = cmapper(value, bytes=True)  # (nxmx4)

    # img = value[:, :, :]
    img = value[...]
    img[invalid_mask] = background_color

    #     return img.transpose((2, 0, 1))
    if gamma_corrected:
        # gamma correction
        img = img / 255
        img = np.power(img, 2.2)
        img = img * 255
        img = img.astype(np.uint8)
    return img


def colorize3D(value, vmin=None, vmax=None, cmap='gray_r', invalid_val=-99, invalid_mask=None, background_color=(128, 128, 128), gamma_corrected=False, value_transform=None):
    """Converts a depth map to a color image with RGB channels.

    Args:
        value (torch.Tensor, numpy.ndarray): Input depth map. Shape: (H, W) or (1, H, W) or (1, 1, H, W). All singular dimensions are squeezed.
        vmin (float, optional): vmin-valued entries are mapped to the start color of cmap. If None, value.min() is used. Defaults to None.
        vmax (float, optional): vmax-valued entries are mapped to the end color of cmap. If None, value.max() is used. Defaults to None.
        cmap (str, optional): matplotlib colormap to use. Defaults to 'magma_r'.
        invalid_val (int, optional): Specifies value of invalid pixels that should be colored as 'background_color'. Defaults to -99.
        invalid_mask (numpy.ndarray, optional): Boolean mask for invalid regions. Defaults to None.
        background_color (tuple[int], optional): 3-tuple RGB color to give to invalid pixels. Defaults to (128, 128, 128).
        gamma_corrected (bool, optional): Apply gamma correction to colored image. Defaults to False.
        value_transform (Callable, optional): Apply transform function to valid pixels before coloring. Defaults to None.

    Returns:
        numpy.ndarray, dtype - uint8: Colored depth map. Shape: (H, W, 3)
    """
    # Convert to numpy if the input is a torch tensor
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()

    value = value.squeeze()  # Remove any singleton dimensions
    if invalid_mask is None:
        invalid_mask = value == invalid_val  # Create invalid mask if not provided
    mask = np.logical_not(invalid_mask)  # Invert mask to get valid pixels

    # Normalize the valid values to the range [0, 1]
    vmin = np.percentile(value[mask], 2) if vmin is None else vmin
    vmax = np.percentile(value[mask], 85) if vmax is None else vmax
    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)  # Scale between vmin and vmax
    else:
        value = value * 0.  # If vmin == vmax, set all values to 0 to avoid division by zero

    # Apply any custom transformation on valid pixels if provided
    if value_transform:
        value = value_transform(value)

    # Set invalid pixels to NaN so they will be ignored in color mapping
    value[invalid_mask] = np.nan

    # Apply colormap and convert to uint8, with shape (H, W, 4)
    cmapper = matplotlib.cm.get_cmap(cmap)
    color_mapped = cmapper(value, bytes=True)  # The result is (H, W, 4), including an alpha channel

    # Extract only the RGB channels, discarding the alpha channel
    img = color_mapped[..., :3]

    # Set the background color for invalid pixels
    img[invalid_mask] = background_color

    # Apply gamma correction if needed
    if gamma_corrected:
        img = img / 255.0  # Normalize to [0, 1] for gamma correction
        img = np.power(img, 2.2)  # Apply gamma correction
        img = (img * 255).astype(np.uint8)  # Scale back to [0, 255] and convert to uint8

    return img

def count_parameters(model, include_all=False):
    return sum(p.numel() for p in model.parameters() if p.requires_grad or include_all)


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
    # If gt and pred is empty, return empty dict with all the keys and math.nan
    if gt.size == 0 or pred.size == 0:
        return {
            "a1": np.nan,
            "a2": np.nan,
            "a3": np.nan,
            "abs_rel": np.nan,
            "rmse": np.nan,
            "log_10": np.nan,
            "sq_rel": np.nan,
            "rmse_log": np.nan,
            "silog": np.nan,
        }
    
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


def compute_errors_2d(gt, pred, valid_mask=None, save_err_img=True, max_depth_eval=80, min_depth_eval=0.1):
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

    # Mask invalid regions in gt and pred (THIS TAKES 2 SECONDS...)
    if valid_mask is not None:
        
        gt_filtered = np.where(valid_mask, gt, np.nan)  # Set invalid areas to NaN for visual clarity
        pred_filtered = np.where(valid_mask, pred, 0)

        # Flatten valid values for error calculations
        gt_1d = gt[valid_mask]
        pred_1d = pred[valid_mask]

    else:
        gt_filtered = gt
        pred_filtered = pred

        # Flatten values for error calculations
        gt_1d = gt.flatten()
        pred_1d = pred.flatten()


    # Calculate accuracy metrics
    thresh = np.maximum((gt_filtered / pred_filtered), (pred_filtered / gt_filtered))
    thresh_1d = np.maximum((gt_1d / pred_1d), (pred_1d / gt_1d))

    # Accuracy Metric Calculation
    a1_2d = (thresh < 1.25)
    a2 = (thresh_1d < 1.25 ** 2).mean()
    a3 = (thresh_1d < 1.25 ** 3).mean()
    a1 = (thresh_1d < 1.25).mean()

    abs_rel_2d = (pred - gt)
    abs_rel = (np.abs(gt_1d - pred_1d)).mean()

    # Rest can be done in 1d. Not interested anymore.
    sq_rel = np.mean(((gt_1d - pred_1d) ** 2) / gt_1d)

    rmse = (gt_1d - pred_1d) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt_1d) - np.log(pred_1d)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    err = (np.log(pred_1d) - np.log(gt_1d))
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    log_10 = (np.abs(np.log10(gt_1d) - np.log10(pred_1d))).mean()

    if save_err_img:

        # Creating the delta accuracy image
        color_map_img = colorize3D(pred, min_depth_eval, max_depth_eval, cmap='magma')
        
        delta_img = color_map_img.copy()
        if valid_mask is not None:
            delta_img[~a1_2d & valid_mask] =  [163,24,24]# red
        else:
            delta_img[~a1_2d] =  [163,24,24]

        # Creating abs_rel image
        abs_rel_img = colorize3D(abs_rel_2d, -10, 10.0, cmap='cool')
        if valid_mask is not None:
            abs_rel_img[~valid_mask] = color_map_img[~valid_mask]
        else:
            abs_rel_img = color_map_img

    return dict(a1=a1, a2=a2, a3=a3, abs_rel=abs_rel, rmse=rmse, log_10=log_10, rmse_log=rmse_log,
                silog=silog, sq_rel=sq_rel), dict(val_d1=delta_img, abs_err=abs_rel_img)

def compute_metrics_and_save(gt, pred, interpolate=True, garg_crop=False, eigen_crop=True, dataset='nyu', min_depth_eval=0.1, max_depth_eval=10, **kwargs):
    """Compute metrics of predicted depth maps. Applies cropping and masking as necessary or specified via arguments. Refer to compute_errors for more details on metrics.
    """

    if 'config' in kwargs:
        config = kwargs['config']
        garg_crop = config.garg_crop
        eigen_crop = config.eigen_crop
        min_depth_eval = min_depth_eval
        max_depth_eval = max_depth_eval
    
    # # Invert gt and pred
    # gt   = max_depth_eval - gt
    # pred = max_depth_eval - pred

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
    
    return compute_errors_2d(gt_depth, pred, valid_mask, save_err_img=True, max_depth_eval=max_depth_eval, min_depth_eval=min_depth_eval)

def compute_metrics_and_bin(gt, pred, seg_mask=None,interpolate=True, garg_crop=False, eigen_crop=True, dataset='nyu', min_depth_eval=0.1, max_depth_eval=10, **kwargs):
    """Compute metrics of predicted depth maps. Applies cropping and masking as necessary or specified via arguments. Refer to compute_errors for more details on metrics.
    """

    if 'config' in kwargs:
        config = kwargs['config']
        garg_crop = config.garg_crop
        eigen_crop = config.eigen_crop
        min_depth_eval = min_depth_eval
        max_depth_eval = max_depth_eval
    
    # Invert gt and pred
    gt   = max_depth_eval - gt
    pred = max_depth_eval - pred
    
    final_results = {}
    freq_hist = {}

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

    # Create a mask to ignore regions in the ground truth outside the min/max depth range
    if seg_mask is not None:
        seg_mask = np.logical_and(seg_mask.squeeze().squeeze().cpu().numpy(), valid_mask)
        if seg_mask.sum() <= 0:
            return final_results, freq_hist
        gt_depth = gt_depth[seg_mask]
        pred = pred[seg_mask]
    else:
        gt_depth = gt_depth[valid_mask]
        pred = pred[valid_mask]
        
    # Calculating binwise metrics
    gt_bin = (gt_depth//10).astype(int)
    maxbin = int(max_depth_eval // 10)

    # Iterate over all possible bins
    for i in range(maxbin):
        mask = (gt_bin == i) # mask for the current bin
        masked_gt = gt_depth[mask]
        masked_pred = pred[mask]
        freq_hist[i*10] = int(len(masked_gt))
        metrics = compute_errors(masked_gt, masked_pred)
        for key, val in metrics.items():
            keyname = key + '/'
            if seg_mask is not None:
                keyname = keyname + 'seg_cls/'
            keyname += str(i*10)
            final_results[keyname] = val
    return final_results, freq_hist

def compute_metrics(gt, pred, interpolate=True, garg_crop=False, eigen_crop=True, dataset='nyu', min_depth_eval=0.1, max_depth_eval=10, **kwargs):
    """Compute metrics of predicted depth maps. Applies cropping and masking as necessary or specified via arguments. Refer to compute_errors for more details on metrics.
    """
    if 'config' in kwargs:
        config = kwargs['config']
        garg_crop = config.garg_crop
        eigen_crop = config.eigen_crop
        min_depth_eval = config.min_depth_eval
        max_depth_eval = config.max_depth_eval
        # max_depth_eval = 300

    # If ground truth and prediction sizes do not match, and interpolation is requested, interpolate prediction
    if gt.shape[-2:] != pred.shape[-2:] and interpolate:
        pred = nn.functional.interpolate(
            pred, gt.shape[-2:], mode='bilinear', align_corners=True)

    # Prepare prediction data for evaluation: remove channel dimension, convert to numpy array, and enforce depth limits
    pred = pred.squeeze().cpu().numpy()
    pred[pred < min_depth_eval] = min_depth_eval
    pred[pred > max_depth_eval] = max_depth_eval
    pred[np.isinf(pred)] = max_depth_eval
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
        valid_mask = np.logical_and(valid_mask, eval_mask)
    return compute_errors(gt_depth[valid_mask], pred[valid_mask])


#################################### Model uilts ################################################


def parallelize(config, model, find_unused_parameters=True):

    if config.gpu is not None:
        torch.cuda.set_device(config.gpu)
        model = model.cuda(config.gpu)

    config.multigpu = False
    if config.distributed:
        # Use DDP
        config.multigpu = True
        config.rank = config.rank * config.ngpus_per_node + config.gpu
        dist.init_process_group(backend=config.dist_backend, init_method=config.dist_url,
                                world_size=config.world_size, rank=config.rank)
        config.batch_size = int(config.batch_size / config.ngpus_per_node)
        # config.batch_size = 8
        config.workers = int(
            (config.num_workers + config.ngpus_per_node - 1) / config.ngpus_per_node)
        print("Device", config.gpu, "Rank",  config.rank, "batch size",
              config.batch_size, "Workers", config.workers)
        torch.cuda.set_device(config.gpu)
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = model.cuda(config.gpu)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.gpu], output_device=config.gpu,
                                                          find_unused_parameters=find_unused_parameters)

    elif config.gpu is None:
        # Use DP
        config.multigpu = True
        model = model.cuda()
        model = torch.nn.DataParallel(model)

    return model


#################################################################################################


#####################################################################################################


class colors:
    '''Colors class:
    Reset all colors with colors.reset
    Two subclasses fg for foreground and bg for background.
    Use as colors.subclass.colorname.
    i.e. colors.fg.red or colors.bg.green
    Also, the generic bold, disable, underline, reverse, strikethrough,
    and invisible work with the main class
    i.e. colors.bold
    '''
    reset = '\033[0m'
    bold = '\033[01m'
    disable = '\033[02m'
    underline = '\033[04m'
    reverse = '\033[07m'
    strikethrough = '\033[09m'
    invisible = '\033[08m'

    class fg:
        black = '\033[30m'
        red = '\033[31m'
        green = '\033[32m'
        orange = '\033[33m'
        blue = '\033[34m'
        purple = '\033[35m'
        cyan = '\033[36m'
        lightgrey = '\033[37m'
        darkgrey = '\033[90m'
        lightred = '\033[91m'
        lightgreen = '\033[92m'
        yellow = '\033[93m'
        lightblue = '\033[94m'
        pink = '\033[95m'
        lightcyan = '\033[96m'

    class bg:
        black = '\033[40m'
        red = '\033[41m'
        green = '\033[42m'
        orange = '\033[43m'
        blue = '\033[44m'
        purple = '\033[45m'
        cyan = '\033[46m'
        lightgrey = '\033[47m'


def printc(text, color):
    print(f"{color}{text}{colors.reset}")

############################################

def get_image_from_url(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert("RGB")
    return img

def url_to_torch(url, size=(384, 384)):
    img = get_image_from_url(url)
    img = img.resize(size, Image.ANTIALIAS)
    img = torch.from_numpy(np.asarray(img)).float()
    img = img.permute(2, 0, 1)
    img.div_(255)
    return img

def pil_to_batched_tensor(img):
    return ToTensor()(img).unsqueeze(0)

def save_raw_16bit(depth, fpath="raw.png"):
    if isinstance(depth, torch.Tensor):
        depth = depth.squeeze().cpu().numpy()
    
    assert isinstance(depth, np.ndarray), "Depth must be a torch tensor or numpy array"
    assert depth.ndim == 2, "Depth must be 2D"
    depth = depth * 256  # scale for 16-bit png
    depth = depth.astype(np.uint16)
    depth = Image.fromarray(depth)
    depth.save(fpath)
    print("Saved raw depth to", fpath)