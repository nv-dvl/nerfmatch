# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import torch
from einops import repeat
import json


def frustum_world_bounds(HWs, Ks, cam2worlds, max_depth, format="bbox"):
    """Compute bounds defined by the frustums provided cameras

    Args:
        HWs (N,2): heights,widths of cameras
        Ks (N,3,3): intrinsics (unnormalized, hence HW required)
        cam2worlds (N,4,4): camera to world transformations
        max_depth (float): depth of all frustums
        format (str): bbox: convex bounding box, sphere: convex bounding sphere

    """
    # unproject corner points
    h_img_corners = torch.Tensor(
        [[0, 0, 0], [0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]]
    )
    Ks_inv = torch.linalg.inv(Ks[:, [1, 0, 2]])  # K in WH -> convert to WH
    k = len(h_img_corners)
    n = len(HWs)
    rep_HWds = repeat(torch.cat([HWs, torch.ones((n, 1))], 1), "n c -> n k c", k=k)
    skel_pts = rep_HWds * repeat(h_img_corners, "k c -> n k c", n=n)  # (N,K,(hwd))
    corners_cam = (
        torch.einsum("nkij,nkj->nki", repeat(Ks_inv, "n x y -> n k x y", k=k), skel_pts)
        * max_depth
    )
    corners_cam_h = torch.cat(
        [corners_cam, torch.ones(corners_cam.shape[0], corners_cam.shape[1], 1)], -1
    )
    corners_world_h = torch.einsum("nij,nkj->nki", cam2worlds, corners_cam_h)
    corners_world_flat = corners_world_h.reshape(-1, 4)[:, :3]

    if format == "bbox":
        bounds = torch.stack(
            [corners_world_flat.min(0).values, corners_world_flat.max(0).values]
        )
        return bounds
    elif format == "sphere":
        corners_world_center = torch.mean(corners_world_flat, 0)
        sphere_radius = torch.max(
            torch.norm((corners_world_flat - corners_world_center), dim=1)
        )
        return corners_world_center, sphere_radius
    else:
        raise Exception("Not implemented yet: Ellipsoid for example")


def compute_world2nscene(HWs, Ks, cam2worlds, max_depth, rescale_factor=1.0):
    """Compute transform converting world to a normalized space enclosing all
    cameras frustums (given depth) into a unit sphere
    Note: max_depth=0 -> camera positions only are contained (like NeRF++ does it)

    Args:
        HWs (N,2): heights,widths of cameras
        Ks (N,3,3): intrinsics (unnormalized, hence HW required)
        cam2worlds (N,4,4): camera to world transformations
        max_depth (float): depth of all frustums
        rescale_factor (float)>=1.0: factor to scale the world space even further so no camera is close too close to the unit sphere surface
    """
    assert rescale_factor >= 1.0, "prevent cameras outside of unit sphere"

    sphere_center, sphere_radius = frustum_world_bounds(
        HWs, Ks, cam2worlds, max_depth, "sphere"
    )
    sphere_radius = rescale_factor * sphere_radius
    T = torch.eye(4)
    T[:3, :3] = torch.eye(3) / sphere_radius
    T[:3, 3] = -sphere_center / sphere_radius
    return T


def compute_scene_normalization_fst(
    transform_json, max_frustum_depth=10, rescale_factor=1.0
):
    with open(transform_json, "r") as f:
        meta_dict = json.load(f)
    cam2scenes = torch.stack(
        [torch.FloatTensor(f["transform_matrix"]) for f in meta_dict["frames"]]
    )
    Ks = torch.stack([torch.FloatTensor(f["intrinsics"]) for f in meta_dict["frames"]])
    HWs = torch.stack(
        [torch.FloatTensor([f["height"], f["width"]]) for f in meta_dict["frames"]]
    )
    scene2s_scene = compute_world2nscene(
        HWs, Ks, cam2scenes, max_frustum_depth, rescale_factor
    )
    return scene2s_scene


def rays_intersect_sphere(rays_o, rays_d, r=1):
    """
    Solve for t such that a=ro+trd with ||a||=r
    Quad -> r^2 = ||ro||^2 + 2t (ro.rd) + t^2||rd||^2
    -> t = (-b +- sqrt(b^2 - 4ac))/(2a) with
       a = ||rd||^2
       b = 2(ro.rd)
       c = ||ro||^2 - r^2
       => (forward intersection) t= (sqrt(D) - (ro.rd))/||rd||^2
       with D = (ro.rd)^2 - (r^2 - ||ro||^2) * ||rd||^2
    """

    odotd = torch.sum(rays_o * rays_d, 1)
    d_norm_sq = torch.sum(rays_d**2, 1)
    o_norm_sq = torch.sum(rays_o**2, 1)
    determinant = odotd**2 + (r**2 - o_norm_sq) * d_norm_sq
    assert torch.all(
        determinant >= 0
    ), "Not all your cameras are bounded by the unit sphere; please make sure the cameras are normalized properly!"
    return (torch.sqrt(determinant) - odotd) / d_norm_sq
