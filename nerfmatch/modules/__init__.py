# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

from torch import nn
import timm
import numpy as np


class MetaFormer_MS(nn.Module):
    SUPPORTED = {
        "convformer": ("convformer_b36.sail_in1k", [128, 256]),
        "convformer384": ("convformer_b36.sail_in1k_384", [128, 256]),
        "caformer": ("caformer_b36.sail_in1k", [128, 256]),
        "caformer384": ("caformer_b36.sail_in1k_384", [128, 256]),
    }

    def __init__(self, name, pretrained=True):
        super().__init__()

        self.use_fpn = "_fpn" in name
        self.model_tag, self.layer_dims = self.SUPPORTED[name.replace("_fpn", "")]
        self.feat_dim = [self.layer_dims[-1], self.layer_dims[0]]
        model = timm.create_model(
            self.model_tag,
            features_only=True,
            pretrained=pretrained,
            out_indices=[0, 1],
        )
        model.stem.conv.stride = (2, 2)
        model.stem.conv.padding = (3, 3)
        model.stages_1.downsample.conv.stride = (4, 4)

        self.model = model
        if self.use_fpn:
            self.init_fpn(self.layer_dims)

    def init_fpn(self, block_dims):

        self.layer2_outconv = nn.Conv2d(
            block_dims[1], block_dims[1], kernel_size=1, stride=1, padding=0, bias=False
        )
        self.layer1_outconv = nn.Conv2d(
            block_dims[0], block_dims[1], kernel_size=1, stride=1, padding=0, bias=False
        )
        self.layer1_outconv2 = nn.Sequential(
            nn.Conv2d(
                block_dims[1],
                block_dims[1],
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(block_dims[1]),
            nn.LeakyReLU(),
            nn.Conv2d(
                block_dims[1],
                block_dims[0],
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
        )

        # Custom initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1, x2 = self.model(x)
        if not self.use_fpn:
            return x2, x1

        # FPN
        x2_out = self.layer2_outconv(x2)
        x2_out_4x = F.interpolate(
            x2_out, scale_factor=4.0, mode="bilinear", align_corners=True
        )
        x1_out = self.layer1_outconv(x1)
        x1_out = self.layer1_outconv2(x1_out + x2_out_4x)
        return [x2_out, x1_out]


def init_backbone(name, downsample=8, pretrained=False):
    if name.startswith("conv"):
        idx = int(np.log2(downsample)) - 2
        convformers = {
            "convformer": "convformer_b36.sail_in1k",
            "convformer384": "convformer_b36.sail_in1k_384",
        }
        if name in convformers:
            name = convformers[name]
        backbone = timm.create_model(
            name, features_only=True, pretrained=pretrained, out_indices=[idx]
        )
        backbone.feat_dim = backbone.feature_info.channels()[0]
    return backbone


def init_backbone_8_2(name, pretrained=False):
    if name.startswith("convformer") or name.startswith("caformer"):
        backbone = MetaFormer_MS(name, pretrained=pretrained)
    return backbone
