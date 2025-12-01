# models/yolov12_backbone.py
import torch
import torch.nn as nn
from pretrain.yolov12_conv import Conv
from pretrain.yolov12_block import C3k2, A2C2f

class YOLOv12Backbone(nn.Module):
    """YOLOv12-turbo backbone implementation."""

    def __init__(self, in_ch=3):
        super().__init__()
        self.layers = nn.ModuleList([
            Conv(in_ch, 64, 3, 2),               # 0 - P1/2
            Conv(64, 128, 3, 2, 1, 2),           # 1 - P2/4
            C3k2(128, 256, n=2, c3k=False, e=0.25),
            Conv(256, 256, 3, 2, 1, 4),          # 3 - P3/8
            C3k2(256, 512, n=2, c3k=False, e=0.25),
            Conv(512, 512, 3, 2),                # 5 - P4/16
            A2C2f(512, 512, n=4, a2=True, area=4),
            Conv(512, 1024, 3, 2),               # 7 - P5/32
            A2C2f(1024, 1024, n=4, a2=True, area=1),
        ])

        self.num_features = 1024

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
