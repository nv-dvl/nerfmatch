# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

from collections import defaultdict
import os
from pathlib import Path
from argparse import Namespace
from nerfmatch.nerf.scene_utils import compute_scene_normalization_fst
import numpy as np
from tqdm import tqdm
import imageio

import torch
from pytorch_lightning import seed_everything

from .nerf_trainer import init_data_loader
from .utils import merge_configs

from .nerf.renderer import NerfRenderer
from .utils.metrics import compute_nerf_metrics
from .nerf.render_utils import prepare_rays_from_pose
from .utils import get_logger, img2int8, save_depth_as_img
from collections import defaultdict

logger = get_logger(level="INFO", name="nerf_eval")


def load_nerf_from_ckpt(
    ckpt_path, args=None, root_dir=".", mask=False, frame_num=-1, seq=False
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device)
    config = Namespace(**ckpt["hyper_parameters"])
    config.ckpt = ckpt_path
    print("config", config)

    # Update data dir
    config.data.data_dir = os.path.join(root_dir, config.data.data_dir)
    if getattr(args, "scene_anno_path", None):
        config.data.scene_anno_path = args.scene_anno_path
    if getattr(args, "snorm_json", None):
        config.data.snorm_json = args.snorm_json
    if mask:
        config.data.mask_dir = getattr(
            config.data, "mask_dir", "data/mask_preprocessed/cambridge"
        )
        config.data.mask_dir = os.path.join(root_dir, config.data.mask_dir)

    # Always test the full dataset
    if not seq:
        config.data.scene_seq = None

    # Update config if given
    if args:
        config = merge_configs(config, args)
        if args.img_wh:
            config.data.img_wh = config.img_wh
        if "downsample" in config:
            config.data.downsample = config.downsample
        if "mip_var_scale" in args:
            config.embedding.mip_var_scale = args.mip_var_scale

    if config.split != "train":
        config.data.max_sample_num = None

    # Init model
    vocab_num = (
        ckpt["state_dict"]["model.embedding_a.weight"].shape[0]
        if "model.embedding_a.weight" in ckpt["state_dict"].keys()
        else -1
    )
    vocab_num = max(
        vocab_num,
        (
            ckpt["state_dict"]["model.kernelblur.img_embed"].shape[0]
            if "model.kernelblur.img_embed" in ckpt["state_dict"].keys()
            else -1
        ),
    )
    evaluator = NerfEvaluator(
        config,
        mask=mask,
        frame_num=frame_num,
        vocab_num=vocab_num,
        stop_layer=args.stop_layer,
    )
    state = evaluator.load_state_dict(ckpt["state_dict"], strict=False)
    logger.info(
        f"Load ckpt from {ckpt_path}: {state} epochs={ckpt['epoch']} step={ckpt['global_step']}"
    )
    return evaluator


def load_scene_normalization(config, root_dir="."):
    # Compute scene normalization
    assert config.snorm_type == "fst"
    if "scene_anno_path" in config:
        train_json = Path(
            config.scene_anno_path.replace("#scene", config.scene).replace(
                "#split", "train"
            )
        )
    else:
        train_json = Path(config.data_dir) / config.scene / "transforms_train.json"

    print(f"Compute scene norm for {train_json}")
    scene2s_scene = compute_scene_normalization_fst(
        root_dir / train_json, config.max_frustum_depth, config.rescale_factor
    )
    unnorm_scene = scene2s_scene.inverse()
    return unnorm_scene


def load_nerf_render_from_ckpt(ckpt_path, device, stop_layer=-1):
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt["state_dict"]
    vocab_num = (
        state_dict["model.embedding_a.weight"].shape[0]
        if "model.embedding_a.weight" in state_dict
        else -1
    )
    config = Namespace(**ckpt["hyper_parameters"])
    print(config)
    render = NerfRenderer(
        config, num_frames=vocab_num, training=False, stop_layer=stop_layer
    )
    render.to(device)
    render.eval()

    # Parse state dict
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            new_state_dict[k.replace("model.", "")] = v
    state = render.load_state_dict(new_state_dict, strict=False)
    logger.info(
        f"Load ckpt from {ckpt_path}: {state} epochs={ckpt['epoch']} step={ckpt['global_step']}"
    )

    render.unnorm_scene = load_scene_normalization(config.data)
    return render


class GenericModelEvaluator(torch.nn.Module):
    def __init__(self, config):
        super().__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logger.info(self.device)
        torch.set_grad_enabled(False)
        self.config = config


class NerfEvaluator(GenericModelEvaluator):
    def __init__(self, config, mask=False, frame_num=-1, vocab_num=100, stop_layer=-1):
        super().__init__(config)
        self.seed = config.exp.seed

        if mask:
            self.config.data.mask_transient = True
            self.config.data.white_bg = True
        else:
            self.config.data.mask_transient = False
            self.config.data.white_bg = False
        if frame_num > 0:
            self.config.data.max_sample_num = frame_num

        # Init model
        self.model = NerfRenderer(
            self.config, num_frames=vocab_num, training=False, stop_layer=stop_layer
        )
        print(f"Init NeRF Render stop_layer={stop_layer}")
        self.model.to(self.device)
        self.model.eval()
        self.comp_radii = self.model.embed_type == "mip"

        # Init dataset
        self.data_loader = init_data_loader(
            config.data,
            split=getattr(config, "split", "test"),
            num_workers=0 if getattr(config, "cache_scene_rays", False) else 8,
        )

        self.cache_dir = Path(
            config.ckpt.replace("checkpoints/", "").replace(
                ".ckpt",
                f"_rendered_{config.data.img_wh[0]}-{config.data.img_wh[1]}_{config.split}",
            )
        )
        if self.model.mip_var_scale > -1:
            self.cache_dir = self.cache_dir / f"mip_var{self.model.mip_var_scale}"
        logger.info(f"Init caching dir: {self.cache_dir}")
        self.split = config.split

    def eval_batch(self, batch, comp_metric=True):
        w, h = batch["img_wh"].squeeze(0)
        if len(batch["rgbs"].squeeze(0).shape) >= 3:
            rgb_gt = batch["rgbs"].squeeze().to(self.device)
            rays = (
                batch["rays"]
                .reshape(-1, batch["rays"].shape[-1])
                .squeeze()
                .to(self.device)
            )
        else:
            if comp_metric:
                rgb_gt = batch["rgbs"].reshape(h, w, -1).to(self.device)
            rays = batch["rays"].squeeze(0).to(self.device)
        preds = self.model.predict(
            rays,
            w,
            h,
            ray_id=(
                batch["ts"].squeeze().to(self.device) if "ts" in batch.keys() else None
            ),
        )
        if comp_metric:
            masks = (
                batch["mask"].reshape(h, w, -1).to(self.device)
                if "mask" in batch.keys()
                else None
            )
            metrics = compute_nerf_metrics(
                preds, rgb_gt, validation_mode=True, mask_loss=masks
            )
            return preds, metrics
        return preds

    def unnorm(self, unnorm_scene, org_mat):
        mat = org_mat.reshape(-1, 3).cpu()
        mat = torch.cat([mat, torch.ones_like(mat[:, 0:1])], dim=-1)
        mat = (unnorm_scene @ mat.T).T[:, :3]
        return mat.reshape(org_mat.shape)

    def eval_data_loader(
        self, data_loader=None, save_depth=False, cache_dir=None, debug=False
    ):
        if data_loader is None:
            data_loader = self.data_loader

        seed_everything(self.seed, workers=True)
        cache_dir = Path(cache_dir if cache_dir else self.cache_dir)
        if debug:
            cache_dir = cache_dir / "debug"
        (cache_dir / "rgb").mkdir(parents=True, exist_ok=True)
        if save_depth:
            (cache_dir / "depth").mkdir(parents=True, exist_ok=True)
        logger.info(f"Set cache dir to {cache_dir}")

        logger.info("Start evaluating...")
        results = defaultdict(list)
        for i, batch in enumerate(tqdm(data_loader)):
            preds, metrics = self.eval_batch(batch)
            psnr = metrics["rgb_fine_psnr"]
            results["psnr"].append(psnr.item())

            if "org_est_fine" in preds.keys() and "unnorm_scene" in batch:
                unnorm_scene = batch["unnorm_scene"][0]
                org_gt = self.unnorm(unnorm_scene, preds["org_gt_fine"])
                org_est = self.unnorm(unnorm_scene, preds["org_est_fine"])
                dist_er = torch.sqrt(torch.sum((org_est - org_gt) ** 2, -1))
                results["translation_l2"].append(dist_er.mean().item())
            img_idx = batch["img_idx"][0]

            # Cache rendered images
            if cache_dir:
                img_path = cache_dir / "rgb" / f"{img_idx}.png"
                rgb = (
                    preds["rgb_fine"]
                    if "rgb_fine" in preds.keys()
                    else preds["rgb_coarse"]
                )
                imageio.imwrite(img_path, img2int8(rgb))

                if save_depth:
                    depth_pred = (
                        preds["depth_fine"]
                        if "depth_fine" in preds.keys()
                        else preds["depth_coarse"]
                    )
                    depth_pred = depth_pred.squeeze().cpu().numpy()
                    depth_path = cache_dir / "depth" / f"{img_idx}.png"
                    max_val = getattr(data_loader.dataset, "far", None)

                    # Global max plane known
                    save_depth_as_img(depth_path, depth_pred, max_val)

            if debug:
                logger.info(f"{i} psnr={psnr:.3f}")
                if i > 10:
                    break

        # Summarize
        logger.info(f"Summary:")
        for k, v in results.items():
            logger.info(f"Average {k}={np.mean(v):.4f}")

        # Save results
        if cache_dir:
            np.save(cache_dir / "results.npy", results)
        return results

    def cache_scene_pts(self, feat_comb="lin", debug=False, cache_dir=None):
        # Set true to return scene pt & feat
        self.model.ret_pfeat = True
        self.model.feat_comb = feat_comb

        seed_everything(self.seed, workers=True)
        key = "fine"

        if cache_dir is None:
            cache_dir = Path(self.cache_dir)
            path_parts = list(self.cache_dir.parts)
            path_parts[1] = "scene_dirs"
            del path_parts[-2]
            cache_dir = Path(os.path.join(*path_parts))
            if debug:
                cache_dir = cache_dir / "debug"

            postfix = ""
            # Setting tag
            tag = f"ds{self.config.downsample}{feat_comb}"
            scene_fld = "scene_msk" if key == "fine" else "scene_msk_" + key
            scene_fld = scene_fld + postfix if postfix else scene_fld
            scene_dir = cache_dir / scene_fld / tag
        else:
            cache_dir = Path(cache_dir)
            scene_dir = cache_dir / "ds8lin"

        scene_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Set scene dir to {scene_dir}")

        if "val" in self.split:
            (cache_dir / "rgb").mkdir(parents=True, exist_ok=True)
            (cache_dir / "feat").mkdir(parents=True, exist_ok=True)
            (cache_dir / "msk").mkdir(parents=True, exist_ok=True)
            (cache_dir / "corr").mkdir(parents=True, exist_ok=True)
            (cache_dir / "corr_rgb").mkdir(parents=True, exist_ok=True)

        logger.info("Start evaluating...")
        for i, batch in enumerate(tqdm(self.data_loader)):
            img_idx = batch["img_idx"][0]
            preds = self.eval_batch(batch, comp_metric=False)

            # Extract scene pts
            pt3d = preds["pts_fine"].cpu()
            unnorm_scene = torch.eye(4)
            if "unnorm_scene" in batch:
                # Scene un-normalize
                unnorm_scene = batch["unnorm_scene"][0].cpu()
                pt3d = self.unnorm(unnorm_scene, pt3d)

            scene_pts = dict(
                pt3d=pt3d.numpy(),
                unnorm_scene=unnorm_scene.numpy(),
                pt_feat=preds["feat_" + key].cpu().numpy(),
                pt_color=preds["rgb_" + key].reshape(-1, 3).clamp(0, 1).cpu().numpy(),
            )
            if "sky_mask" in batch.keys():
                scene_pts["sky_mask"] = batch["sky_mask"].cpu().numpy()

            if "valid_mask" in batch:
                # Necessary for synthetic dataset
                scene_pts["mask"] = batch["valid_mask"].squeeze().numpy()

            # Cache scene pts (make sure frame names are within 4 digits)
            np.save(scene_dir / f"{img_idx}.npy", scene_pts)

            if "val" in self.split:
                # compute and output different statistics. variance, correlation
                img_path = cache_dir / "rgb" / f"{img_idx}.png"
                rgb = (
                    preds["rgb_fine"]
                    if "rgb_fine" in preds.keys()
                    else preds["rgb_coarse"]
                )
                imageio.imwrite(img_path, img2int8(rgb))

                feat_path = cache_dir / "feat" / f"{img_idx}.png"
                feat = preds["feat_fine"]
                feat = feat.reshape(rgb.shape[0], rgb.shape[1], -1)
                feat = torch.var(feat, dim=-1)

                def tensor2img(a):
                    if isinstance(a, torch.Tensor):
                        a = a.cpu().data.numpy()
                    return (255 * np.clip(a, 0, 1)).astype(np.uint8)

                imageio.imwrite(feat_path, tensor2img(feat))

            if debug:
                logger.info(f"{i} {scene_pts.keys()}")
                if i > 10:
                    break

        # Set back to False
        self.model.ret_pfeat = False

    def render_single_view(self, pose, K, near=0.0, far=1.0, flipped_yz=False):
        w, h = K[:2, 2].numpy().astype(np.int32) * 2
        rays = prepare_rays_from_pose(
            pose, K, near, far, flipped_yz, comp_radii=self.comp_radii
        ).to(self.device)
        preds = self.model.predict(rays, h, w)
        rgb_coarse = preds.get("rgb_coarse")
        rgb_fine = preds.get("rgb_fine", None)
        rgb = rgb_fine if rgb_fine is not None else rgb_coarse
        return rgb.cpu().data.numpy(), preds

    def eval_on_scaled_poses(
        self, dataset=None, pose_scale=1, pose_shift=[0, 0, 0], debug=False
    ):
        if not dataset:
            dataset = self.data_loader.dataset
        sav_dir = self.cache_dir / f"rgb_pose_scale{pose_scale}"
        sav_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Eval on scaled pose, save to {sav_dir}")
        scaled_poses = dataset.load_poses(scale=pose_scale, shift=pose_shift)
        np.save(sav_dir / "scaled_poses.npy", scaled_poses)
        for i, c2w in enumerate(tqdm(scaled_poses)):
            rgb, preds = self.render_single_view(
                c2w, dataset.K, dataset.near, dataset.far, dataset.flip_pose_yz
            )
            if debug and i > 5:
                break
            img_path = sav_dir / f"{i:04d}.png"
            imageio.imwrite(img_path, img2int8(rgb))
        return rgb
