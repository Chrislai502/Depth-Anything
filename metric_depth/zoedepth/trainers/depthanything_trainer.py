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

import torch
# import torch.cuda.amp as amp # Depricated
import torch.amp as amp
import torch.nn as nn

from zoedepth.trainers.loss import GradL1Loss, SILogLoss
from zoedepth.utils.config import DATASETS_CONFIG
from zoedepth.utils.misc import compute_metrics, compute_metrics_and_save
from zoedepth.data.preprocess import get_black_border

from .base_trainer import BaseTrainer
from torchvision import transforms
from PIL import Image
import numpy as np
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config

# from mmseg.apis import inference_model, init_model, show_result_pyplot

class Trainer(BaseTrainer):
    def __init__(self, config, model, train_loader, test_loader=None, device=None):
        super().__init__(config, model, train_loader,
                         test_loader=test_loader, device=device)
        self.device = device
        self.silog_loss = SILogLoss()
        self.grad_loss = GradL1Loss()
        self.scaler = amp.GradScaler('cuda', enabled=self.config.use_amp)
        self.dense_depth = config.dense_depth
        self.use_segmentation = config.use_segmentation
        
        # Create Best model for inference
        if self.dense_depth:
            if config.teacher_checkpoint is None:
                raise ValueError("teacher_checkpoint for Depth Anything Inference model cannot be None")
            overwrite = {"pretrained_resource": config.teacher_checkpoint, "bs": config.bs}
            zoe_best_config = get_config(config.teacher_checkpoint_model, "eval", config.dataset, **overwrite)
            self.zoe_model = build_model(zoe_best_config)
            self.zoe_model.to(self.device)
            self.zoe_model.eval()
            
        # TODO: Segmentation model Initialization
        if config.use_segmentation:
            # initialize segmentation model using the config and checkpoint file 
            self._device = torch.device('cuda:0' if self.device == 0 else 'cpu')
            self.segmentation_model = init_model(config.segmentation_config, config.segmentation_checkpoint, device=self._device)
            
        self.thres_min, self.thres_max = config.teacher_anchors
        
    def train_on_batch(self, batch, train_step):
        """
        Expects a batch of images and depth as input
        batch[0]["image"].shape : batch_size, c, h, w
        batch[0]["depth"].shape : batch_size, 1, h, w
        batch[1]["image"].shape : batch_size_1, c, h', w'
        batch[1]["depth"].shape : batch_size_1, 1, h', w'
        ...
        batch[n]["image"].shape : ...
        batch[n]["depth"].shape : ...
        """
        # Infer with the Zoe Model
        images = batch['image'].to(self.device)
        dataset = batch['dataset'][0]
        focal = batch['focal'].to(self.device)
        
        if self.dense_depth:
            with torch.no_grad():
                depths_gt = self.zoe_model(images, dataset=dataset, focal=focal)
                depths_gt = self.get_depth_from_prediction(depths_gt) # not replacing dataset mem due to possible distributed race conditions
                depths_gt = self.clip_and_invert(depths_gt) 
                # depths_gt = torch.clamp(depths_gt, min=self.thres_min, max = self.thres_max)
        else: # Else, prepare mask and depth map
            mask = batch["mask"].to(self.device).to(torch.bool)
            depths_gt = batch['depth'].to(self.device)
            depths_gt = self.clip_and_invert(depths_gt) 
        b, c, h, w = images.size()

        losses = {}
        
        if self.use_segmentation:
            # if using segmentation, create a segmentation map for the batch 
            batch_segmentation_result = inference_model(self.segmentation_model, images, device=self._device)
            breakpoint()

        with amp.autocast('cuda', enabled=self.config.use_amp):
            
            # Infer with the Inference Model
            output = self.model(images)
            pred_depths = output
            pred_depths = pred_depths.unsqueeze(1)
            if self.dense_depth:
                l_si, pred = self.silog_loss(
                    pred_depths, depths_gt, interpolate=True, return_interpolated=True)
            else:           
                l_si, pred = self.silog_loss(
                    pred_depths, depths_gt, mask=mask, interpolate=True, return_interpolated=True)
            
            loss = self.config.w_si * l_si
            losses[self.silog_loss.name] = l_si

            if self.config.w_grad > 0:
                if self.dense_depth:
                    l_grad = self.grad_loss(pred, depths_gt)
                else:
                    l_grad = self.grad_loss(pred, depths_gt, mask=mask)
                loss = loss + self.config.w_grad * l_grad
                losses[self.grad_loss.name] = l_grad
            else:
                l_grad = torch.Tensor([0])

        self.scaler.scale(loss).backward()
        
        if self.config.clip_grad > 0:
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.clip_grad)

        self.scaler.step(self.optimizer)

        if self.should_log and (self.step % int(self.config.log_images_every * self.iters_per_epoch)) == 0:
            # -99 is treated as invalid depth in the log_images function and is colored grey.
            if not self.dense_depth:
                depths_gt[torch.logical_not(mask)] = -99
            idx = np.random.randint(0, b)
            self.log_images(rgb={"Input": images[idx, ...]}, depth={"GT": depths_gt[idx], "PredictedMono": pred[idx]}, prefix="Train",
                            min_depth=DATASETS_CONFIG[dataset]['min_depth'], max_depth=DATASETS_CONFIG[dataset]['max_depth'],
                            scalar_cmap="magma")

            if self.config.get("log_rel", False):
                self.log_images(
                    scalar_field={"RelPred": output["relative_depth"][idx]}, prefix="TrainRel",
                    scalar_cmap="magma")

        self.scaler.update()
        self.optimizer.zero_grad()

        return losses
    
    def validate_on_batch(self, batch, val_step, idx_to_log):
        images = batch['image'].to(self.device)
        depths_gt = self.clip_and_invert(batch['depth'])
        depths_gt = depths_gt.to(self.device)
        depths_gt = depths_gt.squeeze().unsqueeze(0).unsqueeze(0)
        dataset = batch['dataset'][0]

        if not self.dense_depth:
            mask = batch["mask"].to(self.device).to(torch.bool)
            mask = mask.squeeze().unsqueeze(0).unsqueeze(0)
            
        if 'has_valid_depth' in batch:
            if not batch['has_valid_depth']:
                return None, None
        pred_depths = self.eval_infer(images)
        pred_depths = pred_depths.squeeze().unsqueeze(0).unsqueeze(0)

        with amp.autocast('cuda', enabled=self.config.use_amp):
            if not self.dense_depth:
                l_depth = self.silog_loss(
                    pred_depths, depths_gt, mask=mask.to(torch.bool), interpolate=True)
            else:
                l_depth = self.silog_loss(pred_depths, depths_gt, interpolate=True)
        # If ground truth and prediction sizes do not match, and interpolation is requested, interpolate prediction
        if depths_gt.shape[-2:] != pred_depths.shape[-2:]:
            pred_depths = nn.functional.interpolate(pred_depths, depths_gt.shape[-2:], mode='bilinear', align_corners=True)

        if not self.should_log:
            metrics = compute_metrics(depths_gt, pred_depths, **self.config)
        else:
            if not self.dense_depth:
                if not self.dense_depth:
                    depths_gt[torch.logical_not(mask)] = -99
                    
                metrics, val_fields= compute_metrics_and_save(depths_gt, pred_depths, 
                                                            save_err_img=False, 
                                                            max_depth_eval=self.thres_max, 
                                                            min_depth_eval=self.thres_min)
        losses = {f"{self.silog_loss.name}": l_depth.item()}

        if val_step == idx_to_log and self.should_log:
            idx = 0
            self.log_val_images(rgb={"Input": images[idx]}, depth={"GT": depths_gt[idx], "PredictedMono": pred_depths[idx]}, 
                                val_fields=val_fields, prefix="Test", scalar_cmap="magma",
                            min_depth=self.thres_min, max_depth=self.thres_max)

        return metrics, losses

    
    def clip_and_normalize(self, tensor):
        clipped_tensor = torch.clamp(tensor, min=self.thres_min, max=self.thres_max)
        normalized_tensor = (clipped_tensor - self.thres_min) / (self.thres_max - self.thres_min)
        return normalized_tensor
    
    def clip_and_invert(self, tensor):
        clipped_tensor = torch.clamp(tensor, min=self.thres_min, max=self.thres_max)
        return self.thres_max - clipped_tensor
    
    def clip_normalize_and_invert(self, tensor):
        clipped_tensor = torch.clamp(tensor, min=self.thres_min, max=self.thres_max)
        normalized_tensor = (clipped_tensor - self.thres_min) / (self.thres_max - self.thres_min)
        return 1.0 - normalized_tensor # make 0 furthest
    
    def invert_undo_normalization(self, normalized_tensor):
        return (1.0 - normalized_tensor) * (self.thres_max - self.thres_min) + self.thres_min
    
    def undo_normalization(self, normalized_tensor):
        return normalized_tensor * (self.thres_max - self.thres_min) + self.thres_min
    
    def get_depth_from_prediction(self, pred):
        if isinstance(pred, torch.Tensor):
            pred = pred  # pass
        elif isinstance(pred, (list, tuple)):
            pred = pred[-1]
        elif isinstance(pred, dict):
            pred = pred['metric_depth'] if 'metric_depth' in pred else pred['out']
        else:
            raise NotImplementedError(f"Unknown output type {type(pred)}")
        return pred
    
    @torch.no_grad()
    def zoe_infer(self, model, images, **kwargs):
        pred1 = model(images, **kwargs)
        pred1 = self.get_depth_from_prediction(pred1)
        return pred1
        
    @torch.no_grad()
    def eval_infer(self, x):
        with amp.autocast('cuda', enabled=self.config.use_amp):
            m = self.model.module if self.config.multigpu else self.model
            if self.config.model == "depthanything":
                pred_depths = m(x)
            else:
                pred_depths = m(x)['metric_depth']
        return pred_depths

    @torch.no_grad()
    def crop_aware_infer(self, x):
        # if we are not avoiding the black border, we can just use the normal inference
        if not self.config.get("avoid_boundary", False):
            return self.eval_infer(x)
        
        # otherwise, we need to crop the image to avoid the black border
        # For now, this may be a bit slow due to converting to numpy and back
        # We assume no normalization is done on the input image

        # get the black border
        assert x.shape[0] == 1, "Only batch size 1 is supported for now"
        x_pil = transforms.ToPILImage()(x[0].cpu())
        x_np = np.array(x_pil, dtype=np.uint8)
        black_border_params = get_black_border(x_np)
        top, bottom, left, right = black_border_params.top, black_border_params.bottom, black_border_params.left, black_border_params.right
        x_np_cropped = x_np[top:bottom, left:right, :]
        x_cropped = transforms.ToTensor()(Image.fromarray(x_np_cropped))

        # run inference on the cropped image
        pred_depths_cropped = self.eval_infer(x_cropped.unsqueeze(0).to(self.device))

        # resize the prediction to x_np_cropped's size
        pred_depths_cropped = nn.functional.interpolate(
            pred_depths_cropped, size=(x_np_cropped.shape[0], x_np_cropped.shape[1]), mode="bilinear", align_corners=False)
        

        # pad the prediction back to the original size
        pred_depths = torch.zeros((1, 1, x_np.shape[0], x_np.shape[1]), device=pred_depths_cropped.device, dtype=pred_depths_cropped.dtype)
        pred_depths[:, :, top:bottom, left:right] = pred_depths_cropped

        return pred_depths



