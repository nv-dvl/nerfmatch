# Code: https://github.com/zju3dv/LoFTR/blob/master/src/loftr/utils/fine_matching.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange, repeat

from kornia.geometry.subpix import dsnt
from kornia.utils.grid import create_meshgrid


class FinePreprocess(nn.Module):
    def __init__(
        self, win_sz=5, stride=4, d_model_f=128, d_model_c=256, cat_c_feat=True
    ):
        super().__init__()

        self.W = win_sz
        self.stride = stride
        self.cat_c_feat = cat_c_feat
        self.d_model_f = d_model_f
        if self.cat_c_feat:
            self.down_proj = nn.Linear(d_model_c, d_model_f, bias=True)
            self.merge_feat = nn.Linear(2 * d_model_f, d_model_f, bias=True)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.kaiming_normal_(p, mode="fan_out", nonlinearity="relu")

    def forward(self, match_ids, feat_f, pt_ffeat, feat_c=None):
        # stride: feat_f.shape // feat_c.shape
        b_ids, i_ids, j_ids = match_ids

        if b_ids.shape[0] == 0:
            feat0 = torch.empty(0, self.W**2, self.d_model_f, device=feat_f.device)
            feat1 = torch.empty(0, self.W**2, self.d_model_f, device=feat_f.device)
            return feat0, feat1

        pt_ffeat = pt_ffeat[b_ids, j_ids]

        # 1. unfold(crop) all local windows
        feat_f_unfold = F.unfold(
            feat_f,
            kernel_size=(self.W, self.W),
            stride=self.stride,
            padding=self.W // 2,
        )
        feat_f_unfold = rearrange(feat_f_unfold, "n (c ww) l -> n l ww c", ww=self.W**2)

        # 2. select only the predicted matches
        feat_f_unfold = feat_f_unfold[b_ids, i_ids]  # [n, ww, cf]

        # option: use coarse-level loftr feature as context: concat and linear
        if self.cat_c_feat:
            feat_c_win = self.down_proj(feat_c[b_ids, i_ids])  # [n, c]
            feat_cf_win = self.merge_feat(
                torch.cat(
                    [
                        feat_f_unfold,  # [n, ww, cf]
                        repeat(
                            feat_c_win, "n c -> n ww c", ww=self.W**2
                        ),  # [n, ww, cf]
                    ],
                    -1,
                )
            )
        return feat_f_unfold, pt_ffeat


class FineMatching(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, feat_f0, feat_f1):
        """
        Args:
            feat0 (torch.Tensor): [M, C]
            feat1 (torch.Tensor): [M, WW, C]
        Update:
            expec_f (torch.Tensor): [M, 3]
            coords_delta (torch.Tensor): [M, 2]
        """

        M, WW, C = feat_f1.shape
        W = int(math.sqrt(WW))
        # self.M, self.W, self.WW, self.C = M, W, WW, C

        # corner case: if no coarse matches found
        if M == 0:
            expec_f = torch.empty(0, 3, device=feat_f0.device)
            return expec_f

        # Pick the center point of the wxw patch
        sim_matrix = torch.einsum("mc,mrc->mr", feat_f0, feat_f1)
        softmax_temp = 1.0 / C**0.5
        heatmap = torch.softmax(softmax_temp * sim_matrix, dim=1).view(-1, W, W)

        # compute coordinates from heatmap
        coords_normalized = dsnt.spatial_expectation2d(heatmap[None], True)[
            0
        ]  # [M, 2], range [-1, 1]
        grid_normalized = create_meshgrid(W, W, True, heatmap.device).reshape(
            1, -1, 2
        )  # [1, WW, 2]

        # compute std over <x, y>
        var = (
            torch.sum(grid_normalized**2 * heatmap.view(-1, WW, 1), dim=1)
            - coords_normalized**2
        )  # [M, 2]
        std = torch.sum(
            torch.sqrt(torch.clamp(var, min=1e-10)), -1
        )  # [M]  clamp needed for numerical stability

        # for fine-level supervision
        expec_f = torch.cat([coords_normalized, std.unsqueeze(1)], -1)
        return expec_f
