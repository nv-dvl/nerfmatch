# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import numpy as np
import torch


def extract_mutual_matches(
    conf_matrix,
    mutual=True,
    threshold=0.0,
    conf_gt=None,
    coarse_percent=0.3,
    train_percent=0.3,
):
    b, d2, d3 = conf_matrix.shape

    # Confidence thresholding
    mask = conf_matrix > threshold

    # Mutual nearest
    max_w = conf_matrix == conf_matrix.max(dim=2, keepdim=True)[0]
    max_h = conf_matrix == conf_matrix.max(dim=1, keepdim=True)[0]
    mask = mask * max_w * max_h if mutual else mask * max_w

    # Apply filter to conf_matrix
    mask_v, all_j_ids = mask.max(dim=2)
    b_ids, i_ids = torch.where(mask_v)
    j_ids = all_j_ids[b_ids, i_ids]
    mconf = conf_matrix[b_ids, i_ids, j_ids]
    pred_num = len(b_ids)

    if conf_gt is not None:
        total_pts = b * min(d2, d3)

        # Count GT coarse matches
        b_ids_gt, i_ids_gt, j_ids_gt = torch.where(conf_gt)

        # We collect this amount for training
        train_num = int(total_pts * train_percent)
        pred_num = min(int(train_num * coarse_percent), pred_num)
        gt_num = train_num - pred_num

        # Indexing
        mconf_gt = torch.zeros(gt_num).to(mconf)
        pred_idx = np.random.choice(len(b_ids), pred_num)
        gt_idx = np.random.choice(len(b_ids_gt), gt_num)
        b_ids = torch.cat([b_ids[pred_idx], b_ids_gt[gt_idx]])
        i_ids = torch.cat([i_ids[pred_idx], i_ids_gt[gt_idx]])
        j_ids = torch.cat([j_ids[pred_idx], j_ids_gt[gt_idx]])
        mconf = torch.cat([mconf[pred_idx], mconf_gt])
    return (b_ids, i_ids, j_ids), mconf, pred_num
