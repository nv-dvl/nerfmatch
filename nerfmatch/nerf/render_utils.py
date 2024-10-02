# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import torch
import torch.nn.functional as F
import numpy as np

from .scene_utils import rays_intersect_sphere


def get_K(H, W, focal_x, focal_y):
    K = torch.tensor(
        [[focal_x, 0, 0.5 * W], [0, focal_y, 0.5 * H], [0, 0, 1]], dtype=torch.float32
    )
    return K


def get_ray_dirs(H, W, K, flipped_yz=False, return_xys=False):
    """Ray directions computed from all image pixels."""

    ys, xs = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    xys = torch.stack([xs, ys, torch.ones_like(xs)], dim=-1).float()
    dirs = xys @ torch.linalg.inv(K).T
    if flipped_yz:
        # This operation is explicitly needed for some datasets
        dirs = dirs * torch.tensor([1, -1, -1])
    if return_xys:
        return dirs, xys
    return dirs


def get_rays_c2w(dirs, c2w):
    rays_d = dirs.to(c2w) @ c2w[:3, :3].T
    rays_o = c2w[:3, 3].expand_as(rays_d)
    viewdirs = rays_d / rays_d.norm(dim=-1, keepdim=True)
    return rays_o, rays_d, viewdirs


def prepare_rays_from_pose(
    c2w, K, near=0.0, far=1.0, flipped_yz=False, comp_radii=False
):
    if isinstance(c2w, np.ndarray):
        c2w = torch.from_numpy(c2w)
    w, h = K[:2, 2].numpy().astype(np.int32) * 2
    ray_dirs_c = get_ray_dirs(h, w, K, flipped_yz=flipped_yz)
    rays_o, rays_d, viewdirs = get_rays_c2w(ray_dirs_c, c2w)
    rays = prepare_rays_data(rays_o, rays_d, viewdirs, near, far, comp_radii=comp_radii)
    return rays.float()


def sample_nerf_rays(H, W, K, c2w, ds=8, embed_type="mip"):
    directions = get_ray_dirs(H, W, K, return_xys=False)
    rays_o, rays_d, viewdirs = get_rays_c2w(directions, c2w)
    rays_d = viewdirs

    # Compute far plane dynamically
    try:
        far = rays_intersect_sphere(
            rays_o.view(-1, 3), viewdirs.view(-1, 3), r=1
        ).reshape(H, W, 1)
    except Exception as e:
        far = torch.ones((H, W, 1)).float().to(rays_o)
        print(f"Fail to find far plane: {e}! Set far to 1.")

    # Construct rays
    rays = prepare_rays_data(
        rays_o, rays_d, viewdirs, 0.01, far, comp_radii=(embed_type == "mip")
    )
    rays = rays.reshape(H, W, -1)
    _, _, c = rays.shape
    rays = rays[ds // 2 :: ds, ds // 2 :: ds]
    rays = rays.reshape(-1, c)
    return rays


def prepare_rays_data(
    rays_o, rays_d, viewdirs=None, near=0.0, far=1.0, flatten=True, comp_radii=False
):
    if not isinstance(near, torch.Tensor):
        near = near * torch.ones_like(rays_d[..., :1])
    if not isinstance(far, torch.Tensor):
        far = far * torch.ones_like(rays_d[..., :1])

    rays = torch.cat([rays_o, rays_d, near, far], dim=-1)
    if viewdirs is not None:
        rays = torch.cat((rays, viewdirs), dim=-1)
    if comp_radii:
        # Distance from each unit-norm direction vector to its x-axis neighbor
        dx = torch.sqrt(torch.sum((rays_d[:-1, :, :] - rays_d[1:, :, :]) ** 2, -1))
        dx = torch.cat([dx, dx[-2:-1, :]], 0)

        # Cut the distance in half, and then round it out so that it's
        # halfway between inscribed by / circumscribed about the pixel.
        radii = dx[..., None] * 2 / np.sqrt(12)
        rays = torch.cat([rays, radii], dim=-1)

    if flatten:
        rays = rays.view(-1, rays.shape[-1])
    return rays


def sample_pts_init(
    rays_o, rays_d, near, far, num_pts=64, use_disp=False, perturb=False
):
    t_vals = torch.linspace(0.0, 1.0, steps=num_pts).to(rays_o)
    if use_disp:
        # Sample linearly in inverse depth (disparity)
        z_vals = 1.0 / (1.0 / near * (1.0 - t_vals) + 1.0 / far * t_vals)
    else:
        # Linearly sample pts between near and far
        z_vals = near * (1.0 - t_vals) + far * t_vals
    z_vals = z_vals.expand([len(rays_o), num_pts])

    if perturb:
        # Perturb sampled z_vals
        mids = 0.5 * (z_vals[..., :-1] + z_vals[..., 1:])
        upper = torch.cat((mids, z_vals[..., -1:]), dim=-1)
        lower = torch.cat((z_vals[..., :1], mids), dim=-1)
        t_rand = torch.rand_like(z_vals)
        z_vals = lower + (upper - lower) * t_rand

    # Compute 3D points
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
    return pts, z_vals


def sample_pts_fine(rays_o, rays_d, z_vals, weights, num_pts=64, perturb=False):
    # Sample new z_vals based on the coarse z_vals
    z_vals_mid = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
    z_samples = sample_pdf(
        z_vals_mid,
        weights[..., 1:-1],
        num_pts,
        det=(not perturb),
    )

    # Important otherwise  coarse can not be trained
    z_samples = z_samples.detach()
    z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], dim=-1), dim=-1)

    # Re-compute pts
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
    return pts, z_vals


def sample_pts_along_rays(
    rays, num_pts=64, z_vals=None, weights=None, use_disp=True, perturb=False
):
    rays_o, rays_d = rays[..., :3], rays[..., 3:6]
    near, far = rays[..., 6:7], rays[..., 7:8]

    if z_vals is None:
        # Initial z sampling
        return sample_pts_init(
            rays_o,
            rays_d,
            near,
            far,
            num_pts=num_pts,
            use_disp=use_disp,
            perturb=perturb,
        )
    else:
        # Finer sampling
        return sample_pts_fine(
            rays_o, rays_d, z_vals, weights, num_pts=num_pts, perturb=perturb
        )
    return pts, z_vals


def volume_render_radiance_field(
    radiance_field,
    z_vals,
    rays_d,
    noise_std=0.0,
    white_bg=True,
    embed_type=False,
    out_last=False,
    input_dim=4,
):
    # Parse RGB and density
    rgb = radiance_field[..., : input_dim - 1]
    raw_density = radiance_field[..., input_dim - 1]
    noise = (
        torch.randn_like(raw_density) * noise_std
        if noise_std > 0.0
        else torch.zeros_like(raw_density)
    )
    density = F.relu(raw_density + noise)
    if out_last:
        last_feat = radiance_field[..., input_dim:]
        # density = (last_feat.argmax(-1)!=30)*density
        # density = density*(last_feat[:,:,0]>0.95)

    # Compute sample distances
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    if embed_type[:3] == "mip":
        z_mids = 0.5 * (z_vals[..., :-1] + z_vals[..., 1:])
    else:
        dists = torch.cat([dists, 1e10 * torch.ones_like(dists[..., :1])], dim=-1)
        z_mids = z_vals
    dists = dists * rays_d[..., None, :].norm(dim=-1)
    alpha = 1.0 - (-density * dists).exp()

    # Replicate tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = (
        alpha
        * torch.cumprod(
            torch.cat([torch.ones_like(alpha[:, 0:1]), 1.0 - alpha + 1e-10], dim=-1),
            dim=-1,
        )[:, :-1]
    )

    # Compute rendered maps
    rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)  # [N_rays, 3]
    depth_map = torch.sum(weights * z_mids, dim=-1)
    acc_map = torch.sum(weights, dim=-1)
    disp_map = 1.0 / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / acc_map)
    if white_bg:
        rgb_map = rgb_map + (1.0 - acc_map[..., None])

    last_map = (
        torch.sum(weights[..., None].detach() * last_feat, dim=-2) if out_last else None
    )
    return rgb_map, disp_map, acc_map, weights, depth_map, last_map


def sample_pdf(bins, weights, N_samples, det=False, eps=1e-5):
    """Hierarchical sampling (section 5.2)"""

    # Get pdf
    weights = weights + eps
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0.0, 1.0, steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Invert CDF
    u = u.contiguous().to(cdf)
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])
    return samples


def sample_smth_along_rays(
    rays,
    num_pts=64,
    z_vals=None,
    weights=None,
    use_disp=True,
    perturb=False,
    embed_type="normal",
    model_type="coarse",
    randomized=True,
    resample_padding=0.01,
    scale_var=-1,
):
    rays_o, rays_d = rays[..., :3], rays[..., 3:6]

    if embed_type == "mip":
        near, far = rays[..., 6:7], rays[..., 7:8]
        radii = rays[..., 11:12]

        if model_type == "coarse":
            z_vals, (mean, var) = sample_gaus_along_rays(
                rays_o,
                rays_d,
                radii,
                num_pts,
                near,
                far,
                randomized=randomized,
                lindisp=False,
                ray_shape="cone",
            )

        else:  # fine grain sample/s
            z_vals, (mean, var) = resample_gaus_along_rays(
                rays_o,
                rays_d,
                radii,
                z_vals.to(rays_o.device),
                weights.to(rays_o.device),
                randomized=randomized,
                stop_grad=True,
                resample_padding=resample_padding,
                ray_shape="cone",
            )
        if scale_var > 0:
            var = scale_var * var

        return (mean, var), z_vals
    else:
        return sample_pts_along_rays(
            rays,
            num_pts=num_pts,
            z_vals=z_vals,
            weights=weights,
            use_disp=use_disp,
            perturb=perturb,
        )


def lift_gaussian(d, t_mean, t_var, r_var, diag):
    """Lift a Gaussian defined along a ray to 3D coordinates."""
    mean = d[..., None, :] * t_mean[..., None]

    eps = torch.tensor(1e-10)
    d_mag_sq = torch.maximum(eps, torch.sum(d**2, dim=-1, keepdim=True))

    if diag:
        d_outer_diag = d**2
        null_outer_diag = 1 - d_outer_diag / d_mag_sq
        t_cov_diag = t_var[..., None] * d_outer_diag[..., None, :]
        xy_cov_diag = r_var[..., None] * null_outer_diag[..., None, :]
        cov_diag = t_cov_diag + xy_cov_diag
        return mean, cov_diag
    else:
        d_outer = d[..., :, None] * d[..., None, :]
        eye = torch.eye(d.shape[-1], device=d.device)
        null_outer = eye - d[..., :, None] * (d / d_mag_sq)[..., None, :]
        t_cov = t_var[..., None, None] * d_outer[..., None, :, :]
        xy_cov = r_var[..., None, None] * null_outer[..., None, :, :]
        cov = t_cov + xy_cov
        return mean, cov


def conical_frustum_to_gaussian(d, t0, t1, base_radius, diag, stable=True):
    """Approximate a conical frustum as a Gaussian distribution (mean+cov).
    Assumes the ray is originating from the origin, and base_radius is the
    radius at dist=1. Doesn't assume `d` is normalized.
    Args:
    d: torch.float32 3-vector, the axis of the cone
    t0: float, the starting distance of the frustum.
    t1: float, the ending distance of the frustum.
    base_radius: float, the scale of the radius as a function of distance.
    diag: boolean, whether or the Gaussian will be diagonal or full-covariance.
    stable: boolean, whether or not to use the stable computation described in
      the paper (setting this to False will cause catastrophic failure).
    Returns:
    a Gaussian (mean and covariance).
    """
    if stable:
        mu = (t0 + t1) / 2
        hw = (t1 - t0) / 2
        eps = torch.tensor(torch.finfo(torch.float32).eps)
        t_mean = mu + (2 * mu * hw**2) / torch.maximum(eps, 3 * mu**2 + hw**2)
        denom = torch.maximum(eps, 3 * mu**2 + hw**2)
        t_var = (hw**2) / 3 - (4 / 15) * ((hw**4 * (12 * mu**2 - hw**2)) / denom**2)
        r_var = base_radius**2 * (
            (mu**2) / 4 + (5 / 12) * hw**2 - 4 / 15 * (hw**4) / denom
        )
    else:
        t_mean = (3 * (t1**4 - t0**4)) / (4 * (t1**3 - t0**3))
        r_var = base_radius**2 * (3 / 20 * (t1**5 - t0**5) / (t1**3 - t0**3))
        t_mosq = 3 / 5 * (t1**5 - t0**5) / (t1**3 - t0**3)
        t_var = t_mosq - t_mean**2
    return lift_gaussian(d, t_mean, t_var, r_var, diag)


def cast_rays(t_vals, origins, directions, radii, ray_shape="cone", diag=True):
    """Cast rays (cone- or cylinder-shaped) and featurize sections of it.
    Args:
      t_vals: float array, the "fencepost" distances along the ray.
      origins: float array, the ray origin coordinates.
      directions: float array, the ray direction vectors.
      radii: float array, the radii (base radii for cones) of the rays.
      diag: boolean, whether or not the covariance matrices should be diagonal.
    Returns:
      a tuple of arrays of means and covariances.
    """
    t0 = t_vals[..., :-1]
    t1 = t_vals[..., 1:]
    if ray_shape == "cone":
        gaussian_fn = conical_frustum_to_gaussian
    else:
        assert False
    means, covs = gaussian_fn(directions, t0, t1, radii, diag)
    means = means + origins[..., None, :]
    return means, covs


def sample_gaus_along_rays(
    origins,
    directions,
    radii,
    num_samples,
    near,
    far,
    randomized,
    lindisp,
    ray_shape,
    diag=True,
):
    """Stratified sampling along the rays.
    Args:
      origins: torch.tensor(float32), [batch_size, 3], ray origins.
      directions: torch.tensor(float32), [batch_size, 3], ray directions.
      radii: torch.tensor(float32), [batch_size, 3], ray radii.
      num_samples: int.
      near: torch.tensor, [batch_size, 1], near clip.
      far: torch.tensor, [batch_size, 1], far clip.
      randomized: bool, use randomized stratified sampling.
      lindisp: bool, sampling linearly in disparity rather than depth.
    Returns:
      t_vals: torch.tensor, [batch_size, num_samples], sampled z values.
      means: torch.tensor, [batch_size, num_samples, 3], sampled means.
      covs: torch.tensor, [batch_size, num_samples, 3, 3], sampled covariances.
    """
    batch_size = origins.shape[0]

    t_vals = torch.linspace(0.0, 1.0, num_samples + 1, device=origins.device)
    if lindisp:
        t_vals = 1.0 / (1.0 / near * (1.0 - t_vals) + 1.0 / far * t_vals)
    else:
        t_vals = near * (1.0 - t_vals) + far * t_vals

    if randomized:
        mids = 0.5 * (t_vals[..., 1:] + t_vals[..., :-1])
        upper = torch.cat([mids, t_vals[..., -1:]], -1)
        lower = torch.cat([t_vals[..., :1], mids], -1)
        t_rand = torch.rand(batch_size, num_samples + 1, device=origins.device)
        t_vals = lower + (upper - lower) * t_rand
    else:
        # Broadcast t_vals to make the returned shape consistent.
        t_vals = torch.broadcast_to(t_vals, [batch_size, num_samples + 1])
    means, covs = cast_rays(t_vals, origins, directions, radii, ray_shape, diag=diag)
    return t_vals, (means, covs)


def sorted_piecewise_constant_pdf(bins, weights, num_samples, randomized):
    # Pad each weight vector (only if necessary) to bring its sum to `eps`. This
    # avoids NaNs when the input is zeros or small, but has no effect otherwise.
    eps = 1e-5
    weight_sum = torch.sum(weights, dim=-1, keepdim=True)
    padding = torch.maximum(torch.zeros_like(weight_sum), eps - weight_sum)
    weights += padding / weights.shape[-1]
    weight_sum += padding

    # Compute the PDF and CDF for each weight vector, while ensuring that the CDF
    # starts with exactly 0 and ends with exactly 1.
    pdf = weights / weight_sum
    cdf = torch.cumsum(pdf[..., :-1], dim=-1)
    cdf = torch.minimum(torch.ones_like(cdf), cdf)
    cdf = torch.cat(
        [
            torch.zeros(list(cdf.shape[:-1]) + [1], device=cdf.device),
            cdf,
            torch.ones(list(cdf.shape[:-1]) + [1], device=cdf.device),
        ],
        dim=-1,
    )

    # Draw uniform samples.
    if randomized:
        s = 1 / num_samples
        u = (torch.arange(num_samples, device=cdf.device) * s)[None, ...]
        u = (
            u
            + u
            + torch.empty(
                list(cdf.shape[:-1]) + [num_samples], device=cdf.device
            ).uniform_(to=(s - torch.finfo(torch.float32).eps))
        )
        # `u` is in [0, 1) --- it can be zero, but it can never be 1.
        u = torch.minimum(
            u, torch.full_like(u, 1.0 - torch.finfo(torch.float32).eps, device=u.device)
        )
    else:
        # Match the behavior of jax.random.uniform() by spanning [0, 1-eps].
        u = torch.linspace(
            0.0, 1.0 - torch.finfo(torch.float32).eps, num_samples, device=cdf.device
        )
        u = torch.broadcast_to(u, list(cdf.shape[:-1]) + [num_samples])

    # Identify the location in `cdf` that corresponds to a random sample.
    # The final `True` index in `mask` will be the start of the sampled interval.
    mask = u[..., None, :] >= cdf[..., :, None]

    def find_interval(x, y):
        # Grab the value where `mask` switches from True to False, and vice versa.
        # This approach takes advantage of the fact that `x` is sorted.
        btch_sz = 1000
        samples = torch.ones_like(x)
        for i in range(x.shape[0] // btch_sz + 1):
            x0, _ = torch.max(
                torch.where(
                    mask[btch_sz * i : btch_sz * (i + 1)],
                    x[btch_sz * i : btch_sz * (i + 1), ..., None],
                    x[btch_sz * i : btch_sz * (i + 1), ..., :1, None],
                ),
                -2,
            )
            x1, _ = torch.min(
                torch.where(
                    ~mask[btch_sz * i : btch_sz * (i + 1)],
                    x[btch_sz * i : btch_sz * (i + 1), ..., None],
                    x[btch_sz * i : btch_sz * (i + 1), ..., -1:, None],
                ),
                -2,
            )

            y0, _ = torch.max(
                torch.where(
                    mask[btch_sz * i : btch_sz * (i + 1)],
                    y[btch_sz * i : btch_sz * (i + 1), ..., None],
                    y[btch_sz * i : btch_sz * (i + 1), ..., :1, None],
                ),
                -2,
            )
            y1, _ = torch.min(
                torch.where(
                    ~mask[btch_sz * i : btch_sz * (i + 1)],
                    y[btch_sz * i : btch_sz * (i + 1), ..., None],
                    y[btch_sz * i : btch_sz * (i + 1), ..., -1:, None],
                ),
                -2,
            )

            t = torch.clip(
                torch.nan_to_num(
                    (u[btch_sz * i : btch_sz * (i + 1)] - y0) / (y1 - y0), 0
                ),
                0,
                1,
            )
            samples[btch_sz * i : btch_sz * (i + 1)] = x0 + t * (x1 - x0)
        return samples

    return find_interval(bins, cdf)


def resample_gaus_along_rays(
    origins,
    directions,
    radii,
    t_vals,
    weights,
    randomized,
    stop_grad,
    resample_padding,
    ray_shape,
    diag=True,
):
    """Resampling.
    Args:
      origins: torch.tensor(float32), [batch_size, 3], ray origins.
      directions: torch.tensor(float32), [batch_size, 3], ray directions.
      radii: torch.tensor(float32), [batch_size, 3], ray radii.
      t_vals: torch.tensor(float32), [batch_size, num_samples+1].
      weights: torch.tensor(float32), weights for t_vals
      randomized: bool, use randomized samples.
      stop_grad: bool, whether or not to backprop through sampling.
      resample_padding: float, added to the weights before normalizing.
    Returns:
      t_vals: torch.tensor(float32), [batch_size, num_samples+1].
      points: torch.tensor(float32), [batch_size, num_samples, 3].
    """
    if stop_grad:
        with torch.no_grad():
            weights_pad = torch.cat(
                [weights[..., :1], weights, weights[..., -1:]], dim=-1
            )
            weights_max = torch.maximum(weights_pad[..., :-1], weights_pad[..., 1:])
            weights_blur = 0.5 * (weights_max[..., :-1] + weights_max[..., 1:])

            # Add in a constant (the sampling function will renormalize the PDF).
            weights = weights_blur + resample_padding

            new_t_vals = sorted_piecewise_constant_pdf(
                t_vals,
                weights,
                t_vals.shape[-1],
                randomized,
            )
    else:
        weights_pad = torch.cat([weights[..., :1], weights, weights[..., -1:]], dim=-1)
        weights_max = torch.maximum(weights_pad[..., :-1], weights_pad[..., 1:])
        weights_blur = 0.5 * (weights_max[..., :-1] + weights_max[..., 1:])

        # Add in a constant (the sampling function will renormalize the PDF).
        weights = weights_blur + resample_padding

        new_t_vals = sorted_piecewise_constant_pdf(
            t_vals,
            weights,
            t_vals.shape[-1],
            randomized,
        )
    means, covs = cast_rays(
        new_t_vals, origins, directions, radii, ray_shape, diag=diag
    )
    return new_t_vals, (means, covs)


def t_to_s(t_vals, near, far):
    """transform t to s:using the formula in the paper"""
    s_vals = (g(t_vals) - g(near)) / (g(far) - g(near))
    return s_vals


def s_to_t(s_vals, near, far):
    """transform s to t:using the formula in the paper"""
    t_vals = g(s_vals * g(far) + (1 - s_vals) * g(near))
    return t_vals


def g(x):
    """compute the disparity of x:g(x)=1/x"""
    # pad the tensor to avoid dividing zero
    eps = 1e-6
    x += eps
    s = 1 / x
    return s


def contract(x):
    """Contracts points towards the origin (Eq 10 of arxiv.org/abs/2111.12077)."""
    # Clamping to eps prevents non-finite gradients when x == 0.
    eps = torch.tensor(1e-10)
    x_mag_sq = torch.max(eps, torch.sum(x**2, dim=-1, keepdims=True))
    z = torch.where(x_mag_sq <= 1, x, ((2 * torch.sqrt(x_mag_sq) - 1) / x_mag_sq) * x)
    return z
