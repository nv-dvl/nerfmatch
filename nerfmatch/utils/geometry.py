# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import torch
from torch import Tensor

import numpy as np
import torch

import cv2
import pycolmap


def to_numpy(data):
    if isinstance(data, torch.Tensor):
        data = data.cpu().data.numpy()
    return data


def to_tensor(data):
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    return data


def qvec2rotmat(qvec):
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )


def skew(v):
    mat = torch.FloatTensor([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return mat


def get_K(f, w, h):
    K = torch.FloatTensor([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])
    return K


def get_Rt(pose):
    # Pose: 4x4
    R, t = pose[:3, :3], pose[:3, 3:4]
    return R, t


def get_pose(R, t):
    pose = np.eye(4)
    pose[:3, :3] = R
    pose[:3, 3] = t.squeeze()
    return pose.astype(np.float32)


def unnormaliz_pts(pt3d_normed, unnorm_mat):
    # pt3d_normed: b, n, 3
    # unnorm_scene: b, 4, 4
    pt3d_normed = torch.cat(
        [pt3d_normed, torch.ones_like(pt3d_normed[..., 0:1])], dim=-1
    )
    pt3d = torch.bmm(unnorm_mat, pt3d_normed.transpose(-1, -2)).transpose(-1, -2)[
        ..., :3
    ]
    return pt3d


def get_pixel_coords_grid_np(w, h, ds=1):
    grid = np.meshgrid(np.arange(w // ds), np.arange(h // ds), indexing="xy")
    pts = np.dstack(grid) * ds + ds / 2
    return pts.astype(np.float32)


def get_pixel_coords_grid(w, h, ds=1, center_shift=True, homo=False):
    w = int(w)
    h = int(h)
    ys, xs = torch.meshgrid(torch.arange(h // ds), torch.arange(w // ds), indexing="ij")
    # pts = torch.stack([xs, ys], dim=-1) * ds + ds / 2
    pts = torch.stack([xs, ys], dim=-1) * ds
    if center_shift:
        pts = pts + ds / 2
    if homo:
        pts = torch.cat([pts, torch.ones_like(pts[..., :1])], dim=-1)
    return pts.float()


def compute_point3d_from_depth(c2w, K, depth, ds=1):
    H, W = depth.shape
    xys = get_pixel_coords_grid(W, H, homo=True)
    cam_coords = torch.matmul(K.inverse(), xys.reshape(-1, 3).T) * depth.flatten()
    cam_coords_h = torch.cat([cam_coords, torch.ones_like(cam_coords[0:1, ...])], dim=0)
    gt_coords = torch.matmul(c2w, cam_coords_h)[:3, ...].reshape(-1, H, W)

    # Subsample patch center
    gt_coords = gt_coords[:, ds // 2 :: ds, ds // 2 :: ds]
    return gt_coords


def project_points3d(K, R, t, pts3d, ret_depth=False):
    """Project 3D points to 2D points using extrinsics and intrinsics
    Args:
        - K: camera intrinc matrix (3, 3)
        - R: world to camera rotation (3, 3)
        - t: world to camera translation (3,)
        - pts3d: 3D points (N, 3)
    Return:
        - pts2d: projected 2D points (N, 2)
    """

    pts3d_cam = pts3d @ R.T + t.flatten()
    depth = pts3d_cam[:, -1]
    pts2d_norm = pts3d_cam / depth[..., None]
    pixels = pts2d_norm @ K.T
    if ret_depth:
        return pixels[:, :2], depth
    return pixels[:, :2]


def to_homogeneous(tensor, dim=1):
    """Raise a tensor to homogeneous coordinates on the specified dim."""

    ones = torch.ones_like(tensor.select(dim, 0).unsqueeze(dim))
    return torch.cat([tensor, ones], dim=dim)


def expand_homo_ones(arr2d, axis=1):
    """Raise 2D array to homogenous coordinates
    Args:
        - arr2d: (N, 2) or (2, N)
        - axis: the axis to append the ones
    """

    if axis == 0:
        ones = np.ones((1, arr2d.shape[1]))
    else:
        ones = np.ones((arr2d.shape[0], 1))
    return np.concatenate([arr2d, ones], axis=axis)


def mutual_nn_matching(desc1, desc2, threshold=None, eps=1e-9):
    if len(desc1) == 0 or len(desc2) == 0:
        return torch.empty((0, 2), dtype=torch.int64), torch.empty(
            (0, 2), dtype=torch.int64
        )

    device = desc1.device
    desc1 = desc1 / (desc1.norm(dim=1, keepdim=True) + eps)
    desc2 = desc2 / (desc2.norm(dim=1, keepdim=True) + eps)
    similarity = torch.einsum("id, jd->ij", desc1, desc2)
    nn12 = similarity.max(dim=1)[1]
    nn21 = similarity.max(dim=0)[1]
    ids1 = torch.arange(0, similarity.shape[0], device=device)
    mask = ids1 == nn21[nn12]
    matches = torch.stack([ids1[mask], nn12[mask]]).t()
    scores = similarity.max(dim=1)[0][mask]
    if threshold:
        mask = scores > threshold
        matches = matches[mask]
        scores = scores[mask]
    return matches, scores


def to_numpy(data):
    if isinstance(data, Tensor):
        data = data.cpu().data.numpy()
    return data


def estimate_pose(pts2d, pts3d, K, ransac_thres=1):
    if len(pts2d) < 4:
        return None

    pts2d = to_numpy(pts2d).astype(np.float32)
    pts3d = to_numpy(pts3d)
    K = to_numpy(K)
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        pts3d,
        pts2d,
        cameraMatrix=K,
        distCoeffs=None,
        reprojectionError=ransac_thres,
        flags=cv2.SOLVEPNP_AP3P,
    )
    if not success or np.any(np.isnan(tvec)):
        return None

    # Refinement with just inliers
    inliers = inliers.ravel()
    rvec, tvec = cv2.solvePnPRefineLM(
        pts3d[inliers],
        pts2d[inliers],
        cameraMatrix=K,
        distCoeffs=None,
        rvec=rvec,
        tvec=tvec,
    )
    R = cv2.Rodrigues(rvec)[0]
    t = tvec.ravel()
    return R, t, inliers


def estimate_pose_pycolmap(
    pts2d,
    pts3d,
    K,
    img_wh=None,
    ransac_thres=1,
    center_subpixel=False,
    camera_model="PINHOLE",
):
    pts2d = to_numpy(pts2d)
    pts3d = to_numpy(pts3d)
    if center_subpixel:
        pts2d = pts2d + np.array([[0.5, 0.5]], dtype=np.float32)
    K = to_numpy(K)

    if len(pts2d) < 4:
        return None

    if not img_wh:
        img_wh = (K[0, 2] * 2, K[1, 2] * 2)

    # Initialize camera
    camera = pycolmap.Camera(
        model=camera_model,
        width=int(img_wh[0]),
        height=int(img_wh[1]),
        params=[K[0, 0], K[1, 1], K[0, 2], K[1, 2]],
    )

    # Pose estimation
    results = pycolmap.absolute_pose_estimation(
        pts2d,
        pts3d,
        camera,
        max_error_px=ransac_thres,
    )

    # Parsing
    if not results["success"]:
        return None
    t = results["tvec"]
    R = qvec2rotmat(results["qvec"])
    inliers = np.where(results["inliers"])[0]
    return R, t, inliers
