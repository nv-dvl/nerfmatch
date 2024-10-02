# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import torch
import torch.nn as nn
from nerfmatch.utils import update_configs


class NeRF(nn.Module):
    default_config = {
        "layer_num": 8,
        "hid_dim": 256,
        "xyz_dim": 3,
        "dirs_dim": 3,
        "app_dim": 0,
        "output_dim": 4,
        "skips": [4],
        "use_viewdirs": False,
        "out_3d_pnt": False,
        "out_add_ch": 0,
        "stop_layer": -1,
    }

    def __init__(self, config):
        super().__init__()

        config = update_configs(self.default_config, config)
        self.layer_num = config.layer_num
        hid_dim = config.hid_dim
        self.xyz_dim = config.xyz_dim
        self.dirs_dim = config.dirs_dim
        self.app_dim = config.app_dim
        self.output_dim = config.output_dim
        self.skips = config.skips
        self.use_viewdirs = config.use_viewdirs
        self.out_3d_pnt = config.out_3d_pnt
        self.stop_layer = config.stop_layer

        # Init model layers
        self.pts_linears = nn.ModuleList(
            [nn.Linear(self.xyz_dim, hid_dim)]
            + [
                (
                    nn.Linear(hid_dim, hid_dim)
                    if i not in self.skips
                    else nn.Linear(hid_dim + self.xyz_dim, hid_dim)
                )
                for i in range(self.layer_num - 1)
            ]
        )

        if self.use_viewdirs:
            self.views_linears = nn.ModuleList(
                [nn.Linear(self.dirs_dim + hid_dim + self.app_dim, hid_dim // 2)]
            )
            self.feature_linear = nn.Linear(hid_dim, hid_dim)
            self.alpha_linear = nn.Linear(hid_dim, 1)
            self.rgb_linear = nn.Linear(hid_dim // 2, self.output_dim - 1)
        else:
            self.output_linear = nn.Linear(hid_dim, self.output_dim)

        self.stop = -1
        if self.out_3d_pnt:
            out_ch = config.out_add_ch
            if "viewdir" in self.out_3d_pnt:
                self.pnt_block = torch.nn.Sequential(
                    torch.nn.Linear(hid_dim // 2, hid_dim // 2),
                    torch.nn.ReLU(hid_dim // 2),
                    nn.Linear(hid_dim // 2, out_ch),
                )
            elif self.out_3d_pnt == "short":
                self.pnt_block = torch.nn.Sequential(
                    nn.Linear(hid_dim, out_ch),
                )
            elif "begin" in self.out_3d_pnt:
                self.pnt_block = torch.nn.Sequential(
                    torch.nn.Linear(hid_dim, hid_dim // 2),
                    torch.nn.ReLU(hid_dim // 2),
                    nn.Linear(hid_dim // 2, out_ch),
                )
                self.stop = 4
            else:
                self.pnt_block = torch.nn.Sequential(
                    torch.nn.Linear(hid_dim, hid_dim // 2),
                    torch.nn.ReLU(hid_dim // 2),
                    nn.Linear(hid_dim // 2, out_ch),
                )

    def forward(self, x, ret_pfeat=0, pfeat_mask=None, val=False):
        input_pts, input_views, input_app = torch.split(
            x, [self.xyz_dim, self.dirs_dim, self.app_dim], dim=-1
        )
        h = input_pts
        stop_layer = self.stop_layer if self.stop_layer >= 0 else self.stop
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = nn.functional.relu(h)
            if i == stop_layer:
                out_feat = h
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)
        pt_feat = h

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h_rgb = torch.cat([feature, input_views, input_app], -1)
            for i, l in enumerate(self.views_linears):
                h_rgb = self.views_linears[i](h_rgb)
                h_rgb = nn.functional.relu(h_rgb)
            rgb = torch.sigmoid(self.rgb_linear(h_rgb))
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        if self.out_3d_pnt and not val:
            if "viewdir" in self.out_3d_pnt:
                pnt_out = self.pnt_block(h_rgb)
                outputs = torch.cat([outputs, pnt_out], -1)
            elif "begin" in self.out_3d_pnt:
                pnt_out = self.pnt_block(out_feat)
                outputs = torch.cat([outputs, pnt_out], -1)
            else:
                pnt_out = self.pnt_block(h)
                outputs = torch.cat([outputs, pnt_out], -1)

        if ret_pfeat > 0:
            if self.out_3d_pnt and "viewdir" in self.out_3d_pnt:
                out_feats = h_rgb
            elif self.out_3d_pnt and "begin" in self.out_3d_pnt:
                out_feats = out_feat
            else:
                out_feats = pt_feat
            if pfeat_mask is not None:
                out_feats = out_feats[..., pfeat_mask, :]
            if self.stop_layer >= 0:
                out_feats = out_feat
            return outputs, out_feats
        return outputs
