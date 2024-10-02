# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import torch
import torch.nn as nn
from einops import rearrange
import numpy as np

from .embedding import FourierEmbedding, PositionalEncodingMIP
from .render_utils import (
    sample_smth_along_rays,
    volume_render_radiance_field,
    sample_nerf_rays,
    t_to_s,
)

from nerfmatch.utils.geometry import unnormaliz_pts
import nerfmatch.nerf.models as nerf_models


class NerfRenderer(nn.Module):
    def __init__(self, config, num_frames=None, training=True, stop_layer=-1):
        super().__init__()
        self.training = training

        # Set rendering configs
        self.chunksize = config.render.chunksize
        self.use_disp = config.render.use_disp
        self.perturb = config.render.perturb
        self.noise_std = config.render.noise_std
        self.white_bg = config.render.white_bg or getattr(
            config.data, "white_bg", False
        )
        self.use_viewdirs = config.render.use_viewdirs
        self.embed_type = getattr(config.embedding, "type", "normal")
        self.bg_color = None
        if self.white_bg:
            self.bg_color = [1.0, 1.0, 1.0]
        in_channels = 3
        self.img_wh = config.data.img_wh
        self.mip_var_scale = getattr(config.embedding, "mip_var_scale", -1)

        self.num_out_ch = 0
        self.out_scr = getattr(config.data, "out_scr", False)
        if self.out_scr:
            self.num_out_ch = 3

        # Init embedding funcs (execute before init nerf)
        if self.embed_type == "mip":
            self.xyz_encoder = PositionalEncodingMIP(config.embedding.xyz_num_freqs)
            xyz_emb_dim = self.xyz_encoder.get_embedding_dim(in_channels) - in_channels
        else:
            self.xyz_encoder = FourierEmbedding(config.embedding.xyz_num_freqs)
            xyz_emb_dim = self.xyz_encoder.get_embedding_dim(in_channels)

        if self.use_viewdirs:
            if self.embed_type == "mip":
                self.dirs_encoder = PositionalEncodingMIP(
                    config.embedding.dirs_num_freqs
                )
            else:
                self.dirs_encoder = FourierEmbedding(config.embedding.dirs_num_freqs)
            dirs_emb_dim = self.dirs_encoder.get_embedding_dim(3)
        else:
            dirs_emb_dim = 0

        self.appearance_embedding = getattr(config.embedding, "appearance_embed", False)
        embed_sz = 16
        unbounded = True
        self.single_model = getattr(config.render, "single_model", False)

        if self.single_model == False:
            # Init coarse nerf model
            coarse_conf = config.coarse_nerf
            coarse_conf.use_viewdirs = self.use_viewdirs
            coarse_conf.xyz_dim = xyz_emb_dim
            coarse_conf.dirs_dim = dirs_emb_dim
            coarse_conf.app_dim = embed_sz if self.appearance_embedding else 0
            coarse_conf.out_3d_pnt = self.out_scr
            coarse_conf.out_add_ch = self.num_out_ch
            self.num_pts_coarse = coarse_conf.num_pts
            self.nerf_coarse = getattr(nerf_models, coarse_conf.method)(coarse_conf)

        # Init fine model
        fine_conf = getattr(config, "fine_nerf", None)
        self.nerf_fine = self.num_pts_fine = None
        if fine_conf:
            fine_conf.use_viewdirs = self.use_viewdirs
            fine_conf.xyz_dim = xyz_emb_dim
            fine_conf.dirs_dim = dirs_emb_dim
            fine_conf.app_dim = embed_sz if self.appearance_embedding else 0
            fine_conf.out_3d_pnt = self.out_scr
            fine_conf.out_add_ch = self.num_out_ch
            fine_conf.stop_layer = stop_layer
            self.num_pts_fine = fine_conf.num_pts
            self.nerf_fine = getattr(nerf_models, fine_conf.method)(fine_conf)

        self.output_dim = getattr(config.fine_nerf, "output_dim", 4)
        self.embedding_a = None
        vocab_num = num_frames
        if self.appearance_embedding:
            self.embedding_a = torch.nn.Embedding(vocab_num, embed_sz)

        # Nerf feat rendering
        self.ret_pfeat = False
        self.pfeat_mask = None
        self.feat_comb = "lin"

        self.weight_dir = getattr(config.loss, "weight_dir", 1)

    def set_training_mode(self, state):
        self.training = state

    def forward_nerf(
        self, model, pts, viewdirs=None, ret_pfeat=False, app_emb=None, validation=False
    ):
        # pts :  (n, m, 3), n rays, m samples per ray
        # viewdirs: (n, 3)

        if self.embed_type == "mip":
            pts_flat = rearrange(pts[0], "n m c -> (n m) c")
            var_flat = rearrange(pts[1], "n m c -> (n m) c")
            pts = pts[0]
        else:
            pts_flat = rearrange(pts, "n m c -> (n m) c")
        if self.use_viewdirs:
            viewdirs = viewdirs[..., None, :].expand(
                *pts.shape[:-1], viewdirs.shape[-1]
            )
            viewdirs = rearrange(viewdirs, "n m c -> (n m) c")
        if app_emb is not None:
            app_emb = app_emb.view(-1, app_emb.shape[-1])[..., None, :].expand(
                *pts.shape[:-1], app_emb.shape[-1]
            )
            app_emb = rearrange(app_emb, "n m c -> (n m) c")

        # Init mask
        pfeat_mask = None
        n, m, _ = pts.shape
        if ret_pfeat and (self.pfeat_mask is not None):
            pfeat_mask = self.pfeat_mask.flatten()[..., None]
            pfeat_mask = pfeat_mask.expand(n, m).flatten()

        # Mini-batching
        N = self.chunksize
        ray_outs = []
        feat_outs = []
        for i in range(0, len(pts_flat), N):
            ipts = pts_flat[i : i + N]
            inputs = (
                self.xyz_encoder(ipts, y=var_flat[i : i + N])[0]
                if self.embed_type == "mip"
                else self.xyz_encoder(ipts)
            )
            if self.use_viewdirs:
                idirs_emb = self.dirs_encoder(viewdirs[i : i + N])
                inputs = torch.cat((inputs, idirs_emb), dim=-1)
            if app_emb is not None:
                inputs = torch.cat((inputs, app_emb[i : i + N]), dim=-1)

            imask = pfeat_mask[i : i + N] if pfeat_mask is not None else None
            outs_ = model.forward(
                inputs, ret_pfeat=ret_pfeat, pfeat_mask=imask, val=validation
            )
            if ret_pfeat:
                outs_, pfeat_outs_ = outs_
                feat_outs.append(pfeat_outs_)
            ray_outs.append(outs_)
        ray_outs = torch.cat(ray_outs, dim=0)
        ray_outs = ray_outs.reshape(*pts.shape[:-1], -1)
        if ret_pfeat:
            feat_outs = torch.cat(feat_outs, dim=0)
            feat_outs = feat_outs.reshape(-1, pts.shape[-2], feat_outs.shape[-1])
            return ray_outs, feat_outs, pfeat_mask
        return ray_outs

    def render_rays(self, rays, ray_id=None, validation=False):
        """Hierachical neural rendering from coarse to fine."""

        rays_d = rays[..., 3:6]
        viewdirs = None
        if self.use_viewdirs:
            viewdirs = rays[..., 8:11] if rays.shape[1] >= 11 else rays_d

        # Coarse to fine rendering
        models = (
            [self.nerf_fine, self.nerf_fine]
            if self.single_model
            else [self.nerf_coarse, self.nerf_fine]
        )
        num_pts = (
            [self.num_pts_fine, self.num_pts_fine]
            if self.single_model
            else [self.num_pts_coarse, self.num_pts_fine]
        )
        names = ["coarse", "fine"]

        preds = {}
        z_vals = None
        weights = None
        for i, (model, npts, key) in enumerate(zip(models, num_pts, names)):
            if model is None:
                continue

            # Sample pts along the ray
            pts, z_vals = sample_smth_along_rays(
                rays,
                num_pts=npts,
                z_vals=z_vals,
                weights=weights,
                use_disp=self.use_disp,
                perturb=self.perturb if self.training else False,
                embed_type=self.embed_type,
                model_type=key,
                scale_var=self.mip_var_scale,
            )

            # Nerf prediction
            ret_pfeat = self.ret_pfeat  # and (key=='fine')
            app_emb = self.embedding_a(ray_id) if self.appearance_embedding else None
            raw_outs = self.forward_nerf(
                model,
                pts,
                viewdirs,
                ret_pfeat=ret_pfeat,
                app_emb=app_emb,
                validation=validation,
            )
            if ret_pfeat:
                raw_outs, feats, pfeat_mask = raw_outs

            # Rendering
            rendered = volume_render_radiance_field(
                raw_outs[..., : self.output_dim + self.num_out_ch + 3],
                z_vals,
                rays_d,
                noise_std=self.noise_std if self.training else 0.0,
                white_bg=self.white_bg,
                embed_type=self.embed_type,
                out_last=self.num_out_ch > 0,  # if key=='fine' else False
                input_dim=self.output_dim,
            )
            rgb_map, disp_map, acc_map, weights, depth_map, last_map = rendered

            if ret_pfeat:
                if pfeat_mask is not None:
                    pfeat_weights = weights.flatten()[pfeat_mask.flatten()]
                    pfeat_weights = pfeat_weights.reshape(-1, weights.shape[-1])
                else:
                    pfeat_weights = weights

                if self.feat_comb == "max":
                    # Surface point feature only
                    max_ids = pfeat_weights.max(dim=-1)[1]
                    preds[f"feat_{key}"] = feats[torch.arange(len(feats)), max_ids, :]
                else:
                    # Weighted combination along ray
                    preds[f"feat_{key}"] = torch.sum(
                        pfeat_weights[..., None] * feats, dim=-2
                    )

            if self.out_scr and not validation:
                preds[f"scr_{key}"] = (
                    rays[:, :3] + rays_d * depth_map.unsqueeze(1).detach() - last_map
                )

            # Pred pts
            if self.embed_type == "mip":
                pts = pts[0]

            if validation:
                if self.feat_comb == "max":
                    max_ids = weights.max(dim=-1)[1]
                    preds[f"pts_{key}"] = pts[torch.arange(len(pts)), max_ids, :]
                else:
                    preds[f"pts_{key}"] = torch.sum(weights[..., None] * pts, dim=-2)

            if key == "fine" and not validation:
                s_vals = t_to_s(t_vals=z_vals, near=z_vals.min(), far=z_vals.max())
                preds[f"s_{key}"] = s_vals
                preds[f"weights_{key}"] = weights

            # Construct output dict
            preds[f"rgb_{key}"] = rgb_map
            preds[f"depth_{key}"] = depth_map

            if validation:
                del raw_outs
                torch.cuda.empty_cache()
        return preds

    def forward(self, rays, step=0, ray_id=None, validation=False):
        if ray_id is None:
            ray_id = torch.zeros((rays.shape[0])).long().to(rays.device) + 1
        return self.render_rays(rays, ray_id, validation=validation)

    def predict(self, rays, w, h, out_raw=False, ray_id=None):
        self.set_training_mode(False)
        preds = self.forward(rays, validation=True, ray_id=ray_id)
        img_keys = ["rgb_coarse", "depth_coarse", "rgb_fine", "depth_fine"]
        if out_raw:
            return preds
        for k, v in preds.items():
            if k in img_keys:
                if h * w == v.shape[0]:
                    v = v.reshape(h, w, -1)
            preds[k] = v
        return preds

    def render_novel_view(self, img_hw, K, c2w, unnorm_scene, device, downsample=8):
        self.ret_pfeat = True

        H, W = img_hw
        if isinstance(unnorm_scene, np.ndarray):
            unnorm_scene = torch.from_numpy(unnorm_scene)
        scene_norm = unnorm_scene.inverse()
        c2w = scene_norm @ c2w
        rays = sample_nerf_rays(
            H, W, K, c2w, ds=downsample, embed_type=self.embed_type
        ).to(device)
        preds = self.predict(rays, W // downsample, H // downsample)
        pt3d = unnormaliz_pts(preds["pts_fine"][None], unnorm_scene[None])[0]
        outs = dict(
            im_pred=preds["rgb_fine"],
            pt3d=pt3d,
            pt_feat=preds["feat_fine"],
        )
        return outs
