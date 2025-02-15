import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin, hf_hub_download

from depth_anything.blocks import FeatureFusionBlock, _make_scratch
from depth_anything.dpt import DPT_DINOv2
from yolov7.models.yolo import Detect

class v7_DPT(nn.Module):
    def __init__(self, dino: DPT_DINOv2):
        
        self.detection_head = Detect(
            nc=80, 
            anchors=[ # for now
                [10, 13, 16, 30, 33, 23], 
                [30, 61, 62, 45, 59, 119], 
                [116, 90, 156, 198, 373, 326]
            ],
            ch=[
                *dino.backbone[-1].out_channels, 
                *dino.backbone[-2].out_channels, 
                *dino.backbone[-3].out_channels
            ]
        )
    
    def forward(self, x):
        x = self.dino(x)
        y = self.detection_head(x) # TODO: Check for correctness
        return {"dino":x, "yolo":y}