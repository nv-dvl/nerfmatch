# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

from collections import defaultdict
import torch
from torch.utils.data import DataLoader
import numpy as np
from argparse import Namespace
from pathlib import Path
from pytorch_lightning import seed_everything

import os
from tqdm import tqdm
import time

from .nerf.scene_utils import rays_intersect_sphere
from .nerf_evaluator import GenericModelEvaluator, load_nerf_render_from_ckpt
from .nerf.render_utils import sample_smth_along_rays, volume_render_radiance_field

from .nerfmatch_coarse_trainer import NeRFMatcherCoarse
from .nerfmatch_c2f_trainer import NeRFMatcherMS
from .data_loaders import init_data_loader, init_multiscene_dataset, init_mixed_dataset
from .utils.geometry import (
    estimate_pose,
    estimate_pose_pycolmap,
    get_pose,
    unnormaliz_pts,
)
from .utils.metrics import (
    POSE_THRES,
    pose_err,
    summarize_pose_statis,
    average_pose_metrics,
    compute_matching_loss,
)

import cv2
import imageio

from .utils import data_to_device, get_logger, merge_configs

logger = get_logger(level="INFO", name="nerfmatch_eval")

to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)


def update_paths(conf, root_dir):
    # Update data dir
    conf.data_dir = os.path.join(root_dir, conf.data_dir)
    conf.scene_dir = os.path.join(root_dir, conf.scene_dir)
    conf.train_pair_txt = os.path.join(root_dir, conf.train_pair_txt)
    conf.test_pair_txt = os.path.join(root_dir, conf.test_pair_txt)


def parse_nerf_stop_layer(scene_dir):
    splited = scene_dir.split("inter_layer")
    if len(splited) == 2:
        stop_layer = int(splited[1].split("/")[0])
    else:
        stop_layer = -1
    return stop_layer


def load_nerfmatch_from_ckpt(ckpt_path, args=None, root_dir=".", arg_mask=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device)
    config = Namespace(**ckpt["hyper_parameters"])
    config.ckpt = ckpt_path

    # Update data dir
    if getattr(config.data, "datasets", None):
        for dt_name, dt_config in config.data.datasets.__dict__.items():
            update_paths(dt_config, root_dir)
    else:
        update_paths(config.data, root_dir)

    # Update config if given
    if args:
        config = merge_configs(config, args)
        if getattr(args, "img_wh", None):
            config.data.img_wh = config.img_wh
        if getattr(args, "pair_topk", None):
            if getattr(config.data, "datasets", None):
                for dt_name, dt_config in config.data.datasets.__dict__.items():
                    dt_config.pair_topk = config.pair_topk
            config.data.pair_topk = args.pair_topk

        if "downsample" in config:
            config.data.downsample = config.downsample
        if getattr(args, "scene_dir", None) is not None:
            config.data.scene_dir = args.scene_dir
        if getattr(args, "scene", None) is not None:
            config.data.scenes = [args.scene]

        if arg_mask == "default":
            print("config.data.use_msk", getattr(config.data, "use_msk", None))
        if arg_mask == "no mask":
            config.data.use_msk = False
        else:
            config.data.use_msk = arg_mask

    logger.info(config)

    # Init model
    evaluator = NeRFMatchEvaluator(config)
    state = evaluator.load_state_dict(ckpt["state_dict"], strict=False)
    logger.info(
        f"Load ckpt from {ckpt_path}: {state} epochs={ckpt['epoch']} step={ckpt['global_step']}"
    )
    return evaluator


class NeRFMatchEvaluator(GenericModelEvaluator):
    def __init__(self, config, data_loader=None):
        super().__init__(config)
        self.seed = config.exp.seed
        if getattr(config, "iters", 1) > 1:
            torch.manual_seed(self.seed)
        self.use_rgb = getattr(config.model, "use_rgb", False)

        # Init model
        model_conf = config.model
        if "ffeat_dim" not in model_conf:
            self.model = NeRFMatcherCoarse(model_conf)
            self.coarse_only = True
        else:
            self.model = NeRFMatcherMS(model_conf)
            self.coarse_only = False

        self.model.to(self.device)
        self.model.eval()

        # Init dataset
        if not data_loader:
            self.data_loader = init_data_loader(
                config.data, split=getattr(config, "split", "test")
            )
        else:
            self.data_loader = data_loader

        self.cache_dir = Path(
            config.ckpt.replace("checkpoints/", "").replace(".ckpt", "_eval_results")
        )

    def eval_match_pose(
        self,
        batch,
        mutual=True,
        match_thres=0.0,
        solver="colmap",
        rthres=1,
        center_subpixel=False,
        match_oracle=False,
    ):
        K = batch["K"].cpu()
        c2w_gt = batch["c2w"].cpu()

        if match_oracle:
            pt3d = batch["pt3d"].cpu().reshape(len(K), -1, 3)
            conf_gt = batch["conf_gt"]
            bid, i2d, i3d = torch.where(conf_gt)
            bid, i2d, i3d = bid.cpu(), i2d.cpu(), i3d.cpu()
            i2d_ = i2d[bid == 0]
            i3d_ = i3d[bid == 0]
            pt3d = pt3d[0][i3d_]
            if not self.coarse_only:
                pt2d = batch["pt2d_proj"].cpu()[0][i3d_]
            else:
                pt2d = batch["pt2d"].cpu()[0][i2d_]
        else:
            # Image to nerf matching
            t0 = time.time()
            self.model.forward(batch, mutual=mutual, match_thres=match_thres)
            match_time = time.time() - t0
            self.timer["match_time"].append(match_time / batch["pt3d"].shape[-3])

            if self.coarse_only:
                pt2d = batch["pt2d"].cpu()
                pt3d = batch["pt3d"].cpu().reshape(len(K), -1, 3)
                bid, i2d, i3d = batch["match_ids"]
                bid, i2d, i3d = bid.cpu(), i2d.cpu(), i3d.cpu()
                i2d_ = i2d[bid == 0]
                i3d_ = i3d[bid == 0]
                pt2d = pt2d[0][i2d_]
                pt3d = pt3d[0][i3d_]
            else:
                pt2d = batch["mpt2d_f"].detach().cpu()
                pt3d = batch["mpt3d"].cpu()

        # Pose estimation
        if solver == "colmap":
            # Pycolmap solver
            pose_res = estimate_pose_pycolmap(
                pt2d,
                pt3d,
                K.squeeze(),
                ransac_thres=rthres,
                center_subpixel=center_subpixel,
            )
        elif solver == "cv2":
            # Opencv solver
            pose_res = estimate_pose(
                pt2d,
                pt3d,
                K.squeeze(),
                ransac_thres=rthres,
            )
        else:
            raise ValueError(f"{solver} is not supported!")

        if not pose_res:
            print(f"Failed to predict pose, matches={len(pt2d)}")
            R_err, t_err = torch.tensor(torch.inf), torch.tensor(torch.inf)
            c2w_est = None
            inliers = []
        else:
            R, t, inliers = pose_res
            w2c_est = torch.from_numpy(get_pose(R, t))
            c2w_est = w2c_est.inverse()

            # Eval pose err in c2w
            R_err, t_err = pose_err(c2w_est, c2w_gt.squeeze())

        num_matches = len(pt2d)
        return c2w_est, R_err, t_err, num_matches

    def gen_rays(self, poses, width, height, z_near, z_far, K, ds=8, c=None, ndc=False):
        """
        Generate camera rays
        :return (B, H, W, 8)
        """

        num_images = poses.shape[0]
        device = poses.device

        ys, xs = torch.meshgrid(
            torch.arange(height), torch.arange(width), indexing="ij"
        )
        xys = torch.stack([xs, ys, torch.ones_like(xs)], dim=-1).float()
        dirs = xys @ torch.linalg.inv(K.squeeze()).T
        # dirs /= torch.norm(dirs, dim=-1).unsqueeze(-1)

        cam_centers = poses[:, None, None, :3, 3].expand(-1, height, width, -1)[0]
        cam_raydir = torch.matmul(
            poses[:, None, None, :3, :3], dirs.unsqueeze(-1).to(device)
        )[0, :, :, :, 0]

        viewdirs = cam_raydir / cam_raydir.norm(dim=-1, keepdim=True)

        if ndc:
            if not (z_near == 0 and z_far == 1):
                warnings.warn(
                    "dataset z near and z_far not compatible with NDC, setting them to 0, 1 NOW"
                )
            z_near, z_far = 0.0, 1.0
            cam_centers, cam_raydir = ndc_rays(
                width, height, focal, 1.0, cam_centers, cam_raydir
            )

        cam_nears = (
            torch.tensor(z_near, device=device).view(1, 1, 1).expand(height, width, -1)
        )
        try:
            z_far = rays_intersect_sphere(
                cam_centers.view(-1, 3), viewdirs.view(-1, 3), r=1
            )
        except Exception as e:
            z_far = torch.ones((height, width, 1)).float().to(device)
            print(f"Fail to find far plane: {e}! Set far to 1.")
        cam_fars = z_far.view(height, width, -1)

        dx = torch.sqrt(torch.sum((viewdirs[:-1, :, :] - viewdirs[1:, :, :]) ** 2, -1))
        dx = torch.cat([dx, dx[-2:-1, :]], 0)
        radii = dx[..., None] * 2 / np.sqrt(12)

        rays = torch.cat(
            (cam_centers, viewdirs, cam_nears, cam_fars, viewdirs, radii), dim=-1
        )
        rays = rays[ds // 2 :: ds, ds // 2 :: ds].reshape((-1, rays.shape[-1]))
        pts2d = xys[ds // 2 :: ds, ds // 2 :: ds, 0:2].reshape(-1, 2)
        return rays, pts2d

    def inerf_refinement(
        self,
        batch,
        renderer,
        unnorm_scene,
        c2w_est,
        inerf_conf,
        mutual=True,
        match_thres=0.0,
        solver="colmap",
        rthres=1,
        center_subpixel=False,
        visualize=False,
        overlay_ims=None,
        cache_iters=False,
        iter_t_errs=None,
        iter_R_errs=None,
        debug=False,
    ):
        # c2w_est in un-normalized scale
        mse_loss = torch.nn.MSELoss(reduction="mean")
        batch_sz = 3600  # Hard coded to clean
        lrate = getattr(inerf_conf, "lrate", 0.001)
        lrdecay = getattr(inerf_conf, "lrdecay", False)
        num_optim = getattr(inerf_conf, "num_optim", 5)
        eval_pose = getattr(inerf_conf, "eval_pose", False)
        use_match_loss = getattr(inerf_conf, "use_match_loss", False)
        ds = getattr(inerf_conf, "ds", 8)

        c2w_gt = batch["c2w"].cpu()
        K = batch["K"].cpu()

        # Image gt
        img = batch["image"].clone()[0].permute(1, 2, 0)
        H, W, _ = img.shape
        img_ds = img[ds // 2 :: ds, ds // 2 :: ds]
        img_ds = img_ds.contiguous().view(-1, 3)

        # Init camera pose
        scene_norm = unnorm_scene.inverse()
        cam_pose = torch.clone(scene_norm @ c2w_est.detach().to(self.device)).unsqueeze(
            0
        )
        cam_pose.requires_grad = True

        # Nerf update
        optimizer = torch.optim.Adam(params=[cam_pose], lr=lrate)
        for j in range(num_optim):
            tj = time.time()
            with torch.enable_grad():
                if lrdecay:
                    # Update lr
                    new_lrate = lrate * (1 + np.cos(np.pi * j / num_optim)) / 2
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = new_lrate

                all_rays, pts2d_render = self.gen_rays(cam_pose, W, H, 0.01, 1.0, K, ds)
                img_btch_num = int(all_rays.shape[0] // batch_sz)

                rays = all_rays
                rays_d = rays[..., 3:6]
                viewdirs = rays[..., 8:11]

                z_vals = None
                weights = None
                models = [renderer.nerf_coarse, renderer.nerf_fine]
                num_pts = [128, 128]
                names = ["coarse", "fine"]

                for i, (model, npts, key) in enumerate(zip(models, num_pts, names)):
                    pts, z_vals = sample_smth_along_rays(
                        rays.detach(),
                        num_pts=128,
                        z_vals=z_vals,
                        weights=weights,
                        use_disp=False,
                        perturb=False,
                        embed_type="mip",
                        model_type=key,
                        scale_var=1,
                    )
                    viewdirs_flat = viewdirs[..., None, :].expand(
                        *pts[0].shape[:-1], viewdirs.shape[-1]
                    )

                    mu = (z_vals[..., :-1] + z_vals[..., 1:]) / 2
                    hw = (z_vals[..., :-1] - z_vals[..., 1:]) / 2
                    eps = torch.tensor(torch.finfo(torch.float32).eps)
                    t_mean = mu + (2 * mu * hw**2) / torch.maximum(
                        eps, 3 * mu**2 + hw**2
                    )
                    var_flat = pts[1].contiguous().view(-1, 3)

                    pts_z = rays[:, None, :3].expand(*pts[0].shape[:-1], 3) + (
                        t_mean[:, :, None] * viewdirs_flat
                    )
                    pts_flat = pts_z.contiguous().view(-1, 3)
                    inputs = renderer.xyz_encoder(pts_flat, y=var_flat)[0]

                    viewdirs_flat = viewdirs_flat.contiguous().view(-1, 3)
                    idirs_emb = renderer.dirs_encoder(viewdirs_flat)
                    inputs = torch.cat((inputs, idirs_emb), dim=-1)

                    if renderer.embedding_a is not None:
                        ray_id = torch.zeros((rays.shape[0])).long().to(rays.device) + 1
                        app_emb = renderer.embedding_a(ray_id)
                        app_emb = app_emb.view(-1, app_emb.shape[-1])[
                            ..., None, :
                        ].expand(*pts[0].shape[:-1], app_emb.shape[-1])
                        app_emb = app_emb.contiguous().view(-1, app_emb.shape[-1])
                        inputs = torch.cat((inputs, app_emb), dim=-1)

                    if key == "coarse":
                        with torch.no_grad():
                            raw_outs, feats = model.forward(
                                inputs, ret_pfeat=1, val=True
                            )
                    else:
                        raw_outs, feats = model.forward(inputs, ret_pfeat=1, val=True)
                    raw_outs = raw_outs.view(-1, 128, raw_outs.shape[-1])
                    feats = feats.view(-1, 128, feats.shape[-1])

                    # Rendering
                    rendered = volume_render_radiance_field(
                        raw_outs[..., :],
                        z_vals,
                        rays_d,
                        noise_std=0.0,
                        white_bg=True,
                        embed_type="mip",
                        input_dim=4,
                    )
                    rgb_map, disp_map, acc_map, weights, depth_map, last_map = rendered

                loss = 0

                # RGB loss
                rgb_loss = mse_loss(rgb_map, img_ds)
                loss += rgb_loss

                # Matching loss
                match_loss = 0.0
                if use_match_loss:
                    pt_feat = torch.sum(weights[..., None] * feats, dim=-2).unsqueeze(0)
                    pt3d = unnormaliz_pts(
                        torch.sum(weights[..., None] * pts[0], dim=-2)[None],
                        unnorm_scene[None],
                    )
                    preds = self.model.forward_match(
                        batch["image"],
                        pt_feat,
                        pt3d,
                        im_mask=batch["im_mask"],
                        pt_mask=batch["pt_mask"],
                        ret_feats=False,
                        mutual=True,
                    )
                    conf_matrix = preds["conf_matrix"]
                    conf_gt = torch.eye(len(pts2d_render))[None]
                    match_loss = compute_matching_loss(conf_matrix, conf_gt)
                    loss += match_loss

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            self.timer["inerf_step_time"].append(time.time() - tj)

            # Overlay image
            if visualize:
                rendered = to8b(rgb_map.reshape(60, 60, 3).cpu().detach().numpy())
                query = to8b(img_ds.reshape(60, 60, 3).cpu().detach().numpy())
                overlayed = cv2.addWeighted(rendered, 0.7, query, 0.3, 0)
                overlay_ims.append(overlayed)

            # Cache intermediate results
            if debug or cache_iters or j == num_optim - 1:
                if eval_pose:
                    c2w_est = (unnorm_scene @ cam_pose.detach().squeeze()).cpu()
                    R_err, t_err = pose_err(c2w_gt.squeeze(), c2w_est)

                else:
                    # 1 iteration less than the pose
                    # Compute updated pts and feats
                    pred_pnts = torch.sum(weights[..., None] * pts[0], dim=-2)
                    pt3d = unnormaliz_pts(pred_pnts[None], unnorm_scene[None])[0]
                    feats_rays = torch.sum(weights[..., None] * feats, dim=-2)

                    # Update data
                    batch["pt3d"] = pt3d.unsqueeze(0)
                    batch["pt_feat"] = feats_rays.unsqueeze(0)
                    batch["pt_mask"] = torch.ones_like(pt3d[..., 0]).unsqueeze(0)

                    # Image to nerf matching
                    c2w_est, R_err, t_err, _ = self.eval_match_pose(
                        batch,
                        mutual=mutual,
                        match_thres=match_thres,
                        solver=solver,
                        rthres=rthres,
                        center_subpixel=center_subpixel,
                    )

                if cache_iters and j > 0 and j != (num_optim - 1):
                    iter_t_errs.append(t_err)
                    iter_R_errs.append(R_err)

                if debug:
                    print(
                        f"  inerf step={j} loss={loss.item():2f} rgb_loss={rgb_loss:.2f} match_loss={match_loss:.2f} t={t_err*100:.4f}cm R={R_err:.4f}"
                    )

        return c2w_est, R_err, t_err

    def eval_batch(
        self,
        batch,
        renderer=None,
        inerf_conf=None,
        iters=1,
        mutual=True,
        match_thres=0.0,
        match_oracle=False,
        solver="colmap",
        rthres=1,
        center_subpixel=False,
        visualize=False,
        overlay_ims=None,
        query2query=False,
        retrieval_only=False,
        cached_pt=True,
        cache_iters=False,
        debug=False,
    ):
        data_to_device(batch, self.device)

        # Parse data
        img = batch["image"]
        K = batch["K"].cpu()
        if "unnorm_scene" in batch:
            unnorm_scene = batch["unnorm_scene"].squeeze()
        else:
            unnorm_scene = renderer.unnorm_scene
        if isinstance(unnorm_scene, np.ndarray):
            unnorm_scene = torch.from_numpy(unnorm_scene)

        iter_t_errs = []
        iter_R_errs = []
        ts = time.time()

        # For nerf rendering init
        if query2query:
            # Set init pose as query pose (oracle checking)
            c2w_est = batch["c2w"].squeeze()
        elif (not cached_pt) or retrieval_only:
            # Set init pose as ref pose
            c2w_est = batch["rc2w"].squeeze()
        else:
            c2w_est = None

        for itr in range(iters):
            if retrieval_only:
                num_matches = 0
                R_err, t_err = pose_err(batch["c2w"].squeeze().cpu(), c2w_est.cpu())
            else:
                # Pose initialization
                with torch.no_grad():
                    if c2w_est is not None:
                        # Nerf update
                        t0 = time.time()
                        outs = renderer.render_novel_view(
                            img.shape[-2:],
                            K.squeeze(),
                            c2w_est.to(self.device),
                            unnorm_scene.to(self.device),
                            self.device,
                            downsample=8,
                        )
                        pt3d = outs["pt3d"].unsqueeze(0)
                        pt_feat = outs["pt_feat"].unsqueeze(0)
                        pt_mask = torch.ones_like(pt3d[..., 0])

                        # Update data
                        batch["pt3d"] = pt3d
                        batch["pt_feat"] = pt_feat
                        batch["pt_mask"] = pt_mask

                    c2w_est, R_err, t_err, num_matches = self.eval_match_pose(
                        batch,
                        mutual=mutual,
                        match_thres=match_thres,
                        solver=solver,
                        rthres=rthres,
                        center_subpixel=center_subpixel,
                        match_oracle=match_oracle,
                    )
                    if inerf_conf and cache_iters:
                        iter_t_errs.append(t_err)
                        iter_R_errs.append(R_err)

            if c2w_est is not None and inerf_conf:
                # inerf pose refinement
                inerf_res = self.inerf_refinement(
                    batch,
                    renderer,
                    unnorm_scene,
                    c2w_est,
                    inerf_conf,
                    mutual=mutual,
                    match_thres=match_thres,
                    solver=solver,
                    rthres=rthres,
                    center_subpixel=center_subpixel,
                    visualize=visualize,
                    overlay_ims=overlay_ims,
                    cache_iters=cache_iters,
                    iter_t_errs=iter_t_errs,
                    iter_R_errs=iter_R_errs,
                    debug=debug,
                )
                if inerf_res[1] != torch.tensor(torch.inf):
                    # Take the inerf pose only if it manages to repdict a pose
                    c2w_est, R_err, t_err = inerf_res

            if cache_iters:
                iter_t_errs.append(t_err)
                iter_R_errs.append(R_err)

            if debug:
                print(
                    f">> iter={itr} matches={num_matches} t={t_err*100:.4f}cm R={R_err:.4f} "
                )

        self.timer["localize_time"].append(time.time() - ts)
        metrics = dict(
            R_err=[R_err],
            t_err=[t_err],
            iter_t_errs=iter_t_errs,
            iter_R_errs=iter_R_errs,
        )

        return metrics

    def eval_data_loader(
        self,
        renderer=None,
        iters=1,
        rthres=1,
        center_subpixel=False,
        solver="colmap",
        mutual=True,
        match_thres=0.0,
        match_oracle=False,
        data_loader=None,
        query2query=False,
        cached_pt=True,
        debug=False,
        inerf_conf=None,
        retrieval_only=False,
        cache_iters=False,
        visualize=False,
    ):
        if data_loader is None:
            data_loader = self.data_loader
        scene = data_loader.dataset.scene
        vis_dir = self.cache_dir / "visualization" / scene
        vis_dir.mkdir(exist_ok=True, parents=True)
        overlay_ims = []

        metrics = defaultdict(list)
        logger.info(f"Start evaluating rthres={rthres} ...")
        count = 0
        for i, batch in enumerate(tqdm(data_loader, total=len(data_loader))):
            imetric = self.eval_batch(
                batch,
                renderer,
                inerf_conf,
                iters=iters,
                rthres=rthres,
                center_subpixel=center_subpixel,
                solver=solver,
                mutual=mutual,
                match_thres=match_thres,
                match_oracle=match_oracle,
                query2query=query2query,
                retrieval_only=retrieval_only,
                cached_pt=cached_pt,
                cache_iters=cache_iters,
                visualize=visualize,
                overlay_ims=overlay_ims,
                debug=debug,
            )

            for k, v in imetric.items():
                if k not in [
                    "R_err",
                    "t_err",
                    "num_matches",
                    "num_inls",
                    "iter_t_errs",
                    "iter_R_errs",
                ]:
                    continue
                if isinstance(v, torch.Tensor):
                    v = v.cpu().numpy()
                metrics[k].append(v)

            if visualize:
                R_err = imetric["R_err"][0]
                t_err = imetric["t_err"][0]
                name = str(batch["qim_path"][0]).split(scene + "/")[1].replace("/", "_")
                print(">>>", name)
                if t_err > 50:
                    if len(overlay_ims) > 0:
                        imageio.mimwrite(
                            vis_dir / f"{i}_{name}_t{t_err:.1f}cm_R{R_err:.1f}deg.gif",
                            overlay_ims,
                            duration=250,
                        )

            if debug:
                logger.info(
                    f"{i} t={imetric['t_err'][0]*100:.1f}cm r={imetric['R_err'][0]:.3f}deg"
                )
                if i >= 5:
                    break

        # Summarize
        try:
            for k, v in metrics.items():
                if "iter" in k:
                    metrics[k] = np.stack(metrics[k])
                else:
                    metrics[k] = np.concatenate(metrics[k]).squeeze()
        except:
            return metrics
        return metrics

    def eval_multi_scenes(
        self,
        split="test",
        batch_size=1,
        rthres=1,
        center_subpixel=False,
        solver="colmap",
        mutual=True,
        match_thres=0.0,
        iters=1,
        nerf_path=None,
        inerf_conf=None,
        test_pair_txt=None,
        scene_dir=None,
        ow_cache=False,
        data_conf=None,
        query2query=False,
        cached_pt=True,
        stop_layer=-1,
        debug=False,
        visualize=False,
        cache_dir=None,
        cache_iters=False,
        retrieval_only=False,
        match_oracle=False,
        seed=None,
    ):
        if cache_dir:
            self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        conf = self.config.data
        if data_conf is not None:
            conf = merge_configs(conf, data_conf)

        if test_pair_txt:
            conf.test_pair_txt = test_pair_txt
        if scene_dir:
            conf.scene_dir = scene_dir

        if "datasets" in conf:
            datasets = init_mixed_dataset(conf, split=split, concat=False)
        elif "scenes" in conf:
            datasets = init_multiscene_dataset(conf, split=split, concat=False)

        metr_all = []
        for dataset in datasets:
            if seed:
                # Set random seed
                seed_everything(seed, workers=True)
                # torch.manual_seed(cibfseed)

            self.timer = defaultdict(list)

            # Define cache path
            cache_path = str(
                self.cache_dir / f"{dataset.scene}_rth{rthres:.0f}{split}.npy"
            )
            if self.coarse_only:
                cache_path = cache_path.replace(".npy", "_coarse.npy")
            if not mutual:
                cache_path = cache_path.replace(".npy", "_no_mutual.npy")
            if match_thres > 0:
                cache_path = cache_path.replace(".npy", f"_sc{match_thres:.2f}.npy")
            if solver != "cv":
                cache_path = cache_path.replace(".npy", f"_{solver}.npy")
            if center_subpixel:
                cache_path = cache_path.replace(".npy", "_subpx.npy")
            if retrieval_only:
                cache_path = cache_path.replace(".npy", f"_IR.npy")
                assert iters == 1
            if inerf_conf:
                print(f">> inerf_conf:{inerf_conf}")
                lrate = getattr(inerf_conf, "lrate", 0.001)
                lrdecay = getattr(inerf_conf, "lrdecay", False)
                num_optim = getattr(inerf_conf, "num_optim", 5)
                eval_pose = getattr(inerf_conf, "eval_pose", False)
                ds = getattr(inerf_conf, "ds", 8)
                inerf_tag = f"_itr{iters}ds{ds}inerf{num_optim}lr{lrate}"
                if lrdecay > 0:
                    inerf_tag += f"lrdcos"
                if eval_pose:
                    inerf_tag += "pose"
                else:
                    inerf_tag += "match"
                cache_path = cache_path.replace(".npy", f"{inerf_tag}.npy")
            else:
                cache_path = cache_path.replace(".npy", f"_itr{iters}.npy")

            if conf.dataset == "NeRFMatchMultiPair":
                cache_path = cache_path.replace(
                    ".npy", f"_top{conf.pair_topk}pt{conf.sample_pts}.npy"
                )
                if conf.sample_mode:
                    cache_path = cache_path.replace(".npy", f"_{conf.sample_mode}.npy")

            if test_pair_txt:
                pair_tag = test_pair_txt.split("netvlad10-")[1].replace(
                    ".txt", "_pairs"
                )
                cache_path = cache_path.replace(".npy", f".{pair_tag}.npy")

            if not cached_pt:
                cache_path = cache_path.replace(".npy", "_nocache.npy")

            if query2query:
                cache_path = cache_path.replace(".npy", ".query2query.npy")
            if cache_iters:
                cache_path = cache_path.replace(".npy", ".itercache.npy")
            if match_oracle:
                cache_path = cache_path.replace(".npy", ".match_oracle.npy")
            if debug:
                cache_path = cache_path.replace(".npy", ".debug.npy")

            logger.info(f"\n####Cache path: {cache_path}.")
            if os.path.exists(cache_path) and not ow_cache:
                logger.info(f"Found existing cache! Skip evaluation.")
                metrics = np.load(cache_path, allow_pickle=True).item()
                if dataset.scene in POSE_THRES:
                    metr = summarize_pose_statis(
                        metrics,
                        pose_thres=POSE_THRES[dataset.scene],
                        t_unit="cm",
                        t_scale=1e2,
                    )
                else:
                    metr = summarize_pose_statis(
                        metrics, pose_thres=[(5, 5)], t_unit="cm", t_scale=1e2
                    )

                metr_all.append(metr)
                continue

            logger.info(f"\n>>>> Eval dataset {dataset.scene}:\n{dataset}")

            # Init datat loader
            data_loader = DataLoader(
                dataset,
                shuffle=False,
                batch_size=batch_size,
                pin_memory=True,
            )

            # Init nerf renderer
            renderer = None
            if (not cached_pt) or query2query or (iters > 1) or inerf_conf:
                stop_layer = (
                    stop_layer
                    if stop_layer > 0
                    else parse_nerf_stop_layer(dataset.scene_dir)
                )
                print(f"Init NeRF renderer with stop layer: {stop_layer}.")
                renderer = load_nerf_render_from_ckpt(
                    nerf_path.replace("$scene", dataset.scene).replace(
                        "#scene", dataset.scene
                    ),
                    self.device,
                    stop_layer=stop_layer,
                )

            # Run evaluation
            metrics = self.eval_data_loader(
                renderer=renderer,
                iters=iters,
                rthres=rthres,
                center_subpixel=center_subpixel,
                solver=solver,
                mutual=mutual,
                match_thres=match_thres,
                match_oracle=match_oracle,
                data_loader=data_loader,
                query2query=query2query,
                cached_pt=cached_pt,
                debug=debug,
                inerf_conf=inerf_conf,
                retrieval_only=retrieval_only,
                cache_iters=cache_iters,
                visualize=visualize,
            )

            # Record runtime
            for k, v in self.timer.items():
                v = np.array(v)
                metrics[k] = v

            # Cache statis
            np.save(cache_path, metrics)

            # Summarize metrics
            if dataset.scene in POSE_THRES:
                metr = summarize_pose_statis(
                    metrics,
                    pose_thres=POSE_THRES[dataset.scene],
                    t_unit="cm",
                    t_scale=1e2,
                )
            else:
                metr = summarize_pose_statis(
                    metrics, pose_thres=[(5, 5)], t_unit="cm", t_scale=1e2
                )

            metr_all.append(metr)

        if metr_all:
            # Average
            average_pose_metrics(metr_all)
