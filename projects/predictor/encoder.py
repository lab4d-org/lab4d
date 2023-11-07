# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import numpy as np
import pdb
import torch
from torch import nn
import torch.nn.functional as F
import torchvision


class ResNetConv(nn.Module):
    """
    adapted from https://github.com/shubhtuls/factored3d/blob/master/nnutils/net_blocks.py
    """

    def __init__(self, in_channels):
        super(ResNetConv, self).__init__()
        self.resnet = torchvision.models.resnet18(
            weights="ResNet18_Weights.IMAGENET1K_V1"
        )
        if in_channels != 3:
            self.resnet.conv1 = nn.Conv2d(
                in_channels,
                64,
                kernel_size=(7, 7),
                stride=(2, 2),
                padding=(3, 3),
                bias=False,
            )
        self.resnet.fc = None

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        return x


class Encoder(nn.Module):
    """
    adapted from https://github.com/shubhtuls/factored3d/blob/master/nnutils/net_blocks.py
    Current:
    Resnet with 4 blocks (x32 spatial dim reduction)
    Another conv with stride 2 (x64)
    This is sent to 2 fc layers with final output nz_feat.
    """

    def __init__(self, in_channels=3, out_channels=128, batch_norm=True):
        super(Encoder, self).__init__()
        self.resnet_conv = ResNetConv(in_channels=in_channels)
        self.conv1 = conv2d(batch_norm, 512, out_channels, stride=1, kernel_size=3)
        # net_init(self.conv1)

    def forward(self, img):
        feat = self.resnet_conv.forward(img)  # 512,4,4
        feat = self.conv1(feat)  # 128,4,4
        feat = F.max_pool2d(feat, 4, 4)
        feat = feat.view(img.size(0), -1)
        return feat


## 2D convolution layers
def conv2d(batch_norm, in_planes, out_planes, kernel_size=3, stride=1):
    """
    adapted from https://github.com/shubhtuls/factored3d/blob/master/nnutils/net_blocks.py
    """
    if batch_norm:
        return nn.Sequential(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=(kernel_size - 1) // 2,
                bias=True,
            ),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.2, inplace=True),
        )
    else:
        return nn.Sequential(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=(kernel_size - 1) // 2,
                bias=True,
            ),
            nn.LeakyReLU(0.2, inplace=True),
        )
