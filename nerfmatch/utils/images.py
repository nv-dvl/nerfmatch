# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import torch
import torchvision.transforms as T
import numpy as np

from PIL import Image
import cv2
import imageio


def colorize_depth(
    depth, cmap=cv2.COLORMAP_JET, return_origin=False, force_min=None, force_max=None
):
    if isinstance(depth, torch.Tensor):
        depth = depth.cpu().numpy()
    depth = np.nan_to_num(depth)
    mi = np.min(depth) if force_min is None else force_min
    ma = np.max(depth) if force_max is None else force_max
    depth = (depth - mi) / max(ma - mi, 1e-8)
    depth_im = (255 * np.clip(depth, 0, 1)).astype(np.uint8)
    depth_im = Image.fromarray(cv2.applyColorMap(depth_im, cmap))
    if return_origin:
        return depth_im
    return T.functional.to_tensor(depth_im)


def img2int8(img):
    if isinstance(img, torch.Tensor):
        img = img.cpu().data.numpy()
    img = img[..., :3]
    img = (255 * np.clip(img, 0, 1)).astype(np.uint8)
    return img


def depth2img(depth, max_val):
    if isinstance(depth, torch.Tensor):
        depth = depth.cpu().data.numpy()
    depth = depth.squeeze()
    return (255 - depth / max_val * 255).astype(np.uint8)


def img2depth(depth_img, max_val, bg_val=0.0, bg_mask=None):
    if isinstance(depth_img, Image.Image):
        depth_img = np.array(depth_img)
    if len(depth_img.shape) > 2:
        depth_img = depth_img[..., 0]
    depth_img = max_val * ((255 - depth_img) / 255)
    if bg_mask is not None:
        depth_img[~bg_mask] = bg_val
    return depth_img


def save_depth_as_img(path, raw_depth, max_val=None):
    if isinstance(raw_depth, torch.Tensor):
        raw_depth = raw_depth.cpu().data.numpy()
    if max_val:
        depth = depth2img(raw_depth, max_val)
    else:
        depth = np.array(colorize_depth(raw_depth, return_origin=True))
    imageio.imsave(path, depth)


def load_depth_from_img(depth_path, max_val, img_wh=None, bg_val=0.0, bg_mask=None):
    """Following github issue to decode the nerf-synthetic lego gt depth map:
    https://github.com/bmild/nerf/issues/77
    The decoded depth map looks different from predictions and leads
    to inconsistent geometry.
    """

    depth = Image.open(depth_path)
    if img_wh:
        depth = depth.resize(img_wh, Image.LANCZOS)
    depth = img2depth(depth, max_val, bg_val=bg_val, bg_mask=bg_mask)
    return depth


def inside_image(pts, w, h):
    mask = ((pts > 0).all(-1)) & (pts[:, 0] < w) & (pts[:, 1] < h)
    return mask
