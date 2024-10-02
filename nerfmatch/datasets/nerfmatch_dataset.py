# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import torch
from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
from pathlib import Path
import json
import random
from einops import rearrange

from ..utils.geometry import (
    get_pose,
    project_points3d,
    get_pixel_coords_grid,
    estimate_pose,
)
from ..utils.metrics import pose_err
from .data_loading import (
    load_topk_retrieval_pairs,
    load_retrieval_pairs,
    parse_pair_ids,
    parse_pair_ids_balanced,
    parse_multipair_ids_balanced,
    load_frame_3d,
)


def process_img(img_wh, img_path, imagenet_norm=False, ret_orig=False):
    # Load img
    img = Image.open(img_path)

    # Compute intrinsics scaler before resizing
    sK = torch.from_numpy(
        np.diag([img_wh[0] / img.size[0], img_wh[1] / img.size[1], 1])
    ).float()

    # Resize -> normalize -> reshape
    img = img.resize(img_wh, Image.LANCZOS)
    if ret_orig:
        return img
    img = np.array(img) / 255.0

    # Normalization
    if imagenet_norm:
        imagenet_mean = np.array([0.485, 0.456, 0.406])
        imagenet_std = np.array([0.229, 0.224, 0.225])
        img = img - imagenet_mean
        img = img / imagenet_std

    # Convert to torch tensor
    img = torch.tensor(img)
    img = rearrange(img, "h w c -> c h w").float()
    return img, sK


class NeRFMatchBase(Dataset):
    def __init__(self, config, split="train", val_num=100, debug=False):
        super().__init__()

        self.config = config
        self.split = split
        self.scene = config.scene
        self.root_dir = Path(config.data_dir) / self.scene
        self.scene_dir = config.scene_dir.replace("#scene", self.scene)
        self.model_ds = getattr(config, "model_ds", 1)
        self.img_wh = config.img_wh
        self.val_num = val_num
        self.use_msk = getattr(config, "use_msk", False)
        # Load scene data
        self.load_scene_data()

    def process_img(self, img_path, imagenet_norm=False, ret_orig=False):
        return process_img(
            self.img_wh, img_path, imagenet_norm=imagenet_norm, ret_orig=ret_orig
        )

    def load_scene_data(self):
        anno_tag = "test" if self.split == "test" else "train"
        anno_path = self.root_dir / f"transforms_{anno_tag}.json"
        with open(anno_path, "r") as f:
            frames = json.load(f)["frames"]
        self.frames = sorted(frames, key=lambda x: x["file_path"])
        print(f"Number of frames:", len(frames))

    def load_sample(self, idx):
        frame = self.frames[idx]
        fname = frame["file_path"]
        w, h = self.img_wh

        # Load image & intrinsics
        image_path = str(self.root_dir / fname)
        img, sK = self.process_img(image_path)
        K = sK @ torch.tensor(frame["intrinsics"], dtype=torch.float32)
        pt2d = get_pixel_coords_grid(w, h, ds=self.model_ds).reshape(-1, 2)

        # Load scene pts
        pt_name = fname.replace("/", "_").replace(".color", "").replace(".png", "")
        pt_path = os.path.join(self.scene_dir, f"{pt_name}.npy")
        scene_pts = np.load(pt_path, allow_pickle=True).item()
        pt3d = scene_pts["pt3d"]
        pt_color = scene_pts["pt_color"]
        pt_feat = scene_pts["pt_feat"]
        pt_mask = np.ones(len(pt3d)).astype(np.bool_)

        # Match gt
        conf_gt = np.eye(len(pt3d))

        # GT pose
        c2w = torch.tensor(frame["transform_matrix"], dtype=torch.float32)

        # Pack
        sample = {
            "image_path": image_path,
            "image": img,
            "im_mask": pt_mask,
            "pt2d": pt2d,
            "pt3d": pt3d,
            "pt_color": pt_color,
            "pt_feat": pt_feat,
            "pt_mask": pt_mask,
            "c2w": c2w,
            "K": K,
            "conf_gt": conf_gt,
        }
        return sample

    def sanity_check_sample(self, sample):
        rpt2d = qpt2d = sample["pt2d"]
        rpt3d = sample["pt3d"]
        qc2w = sample["c2w"]
        match_gt = sample["conf_gt"]
        K = sample["K"]
        qids, rids = np.where(match_gt)

        matches = np.concatenate([qpt2d[qids], rpt2d[rids]], axis=-1)

        # Localize im2 with pts3d im1 and pts2d im2
        pose_res = estimate_pose(qpt2d[qids], rpt3d[rids], K, ransac_thres=1)
        if not pose_res:
            R_err = t_err = np.inf
        else:
            R, t, inliers = pose_res
            qw2c_est = torch.from_numpy(get_pose(R, t))
            R_err, t_err = pose_err(qc2w, qw2c_est.inverse())
        print(f"R={R_err:.3f}, t={t_err:.3f} #matches={match_gt.sum()}")

    def __getitem__(self, idx):
        return self.load_sample(idx)

    def __len__(self):
        return len(self.frames)

    def __repr__(self):
        fmt_str = f"NeRFMatchBase(\n"
        fmt_str += f" split={self.split} samples={self.__len__()}"
        fmt_str += f" img_wh={self.img_wh}\n"
        fmt_str += f" scene_dir={self.scene_dir}\n"
        fmt_str += f"\n)\n"
        return fmt_str


class NeRFMatchPair(NeRFMatchBase):
    def __init__(self, config, split="train", val_num=500, debug=False):
        self.anno_tag = "test" if split == "test" else "train"
        self.pair_txt = getattr(config, f"{self.anno_tag}_pair_txt").replace(
            "#scene", config.scene
        )
        self.pair_topk = getattr(config, "pair_topk", 10)
        self.imagenet_norm = getattr(config, "imagenet_norm", False)
        self.balanced_pair = getattr(config, "balanced_pair", False)
        if self.balanced_pair and split == "val":
            # To gaurantee the same val split for different topk
            self.pair_topk = -1

        # Training augmentation
        self.aug_self_pairs = (
            getattr(config, "aug_self_pairs", False) if split == "train" else False
        )

        super().__init__(config, split=split, val_num=val_num, debug=debug)

        self.im_dir = self.root_dir
        self.epoch_sample_num = (
            getattr(config, "epoch_sample_num", -1) if split == "train" else -1
        )

    def load_scene_data(self):
        if "scene_anno_path" in self.config:
            scene_anno_path = self.config.scene_anno_path.replace("#scene", self.scene)
            self.ref_json = scene_anno_path.replace("#split", "train")
            self.query_json = scene_anno_path.replace("#split", self.anno_tag)
        else:
            self.ref_json = self.root_dir / "transforms_train.json"
            self.query_json = self.root_dir / f"transforms_{self.anno_tag}.json"

        # Load ref frames
        with open(self.ref_json, "r") as f:
            rframes = json.load(f)["frames"]
            self.rframes = sorted(rframes, key=lambda x: x["file_path"])

        # Load query frames
        if self.query_json == self.ref_json:
            self.qframes = self.rframes
        else:
            with open(self.query_json, "r") as f:
                qframes = json.load(f)["frames"]
                self.qframes = sorted(qframes, key=lambda x: x["file_path"])

        print(f"Loaded {len(self.qframes)} query and {len(self.rframes)} ref frames.")

        # Load retrieval pairs
        pairs = load_topk_retrieval_pairs(self.pair_txt, kmax=self.pair_topk)

        if self.balanced_pair:
            self.pair_ids = parse_pair_ids_balanced(
                self.qframes,
                self.rframes,
                pairs,
                split=self.split,
                val_num=self.val_num,
            )
        else:
            self.pair_ids = parse_pair_ids(
                self.qframes,
                self.rframes,
                pairs,
                split=self.split,
                val_num=self.val_num,
            )
        print(f"Parsed {self.split} pairs {len(self.pair_ids)}.\n")

        # Augment with self2self pairs
        if self.aug_self_pairs:
            self_pairs = [(i, i) for i, _ in enumerate(self.qframes)] * int(
                self.aug_self_pairs
            )
            self.pair_ids += self_pairs
            print(
                f"##Augment self-to-self pairs {len(self_pairs)}, updated pairs: {len(self.pair_ids)}.\n"
            )

    def load_sample(self, idx):
        if self.epoch_sample_num > 0:
            pidx = np.random.randint(len(self.pair_ids))
        else:
            pidx = idx
        qid, rid = self.pair_ids[pidx]
        qframe = self.qframes[qid]
        ds = self.model_ds
        w, h = self.img_wh

        # GT pose
        qc2w = torch.tensor(qframe["transform_matrix"], dtype=torch.float32)
        qw2c = qc2w.inverse()

        # Load query image
        qname = qframe["file_path"]
        qim_path = str(self.im_dir / qname)
        qim, sK = self.process_img(qim_path, imagenet_norm=self.imagenet_norm)
        qK = sK @ torch.tensor(qframe["intrinsics"], dtype=torch.float32)
        qpt2d = get_pixel_coords_grid(w, h, ds=ds).reshape(-1, 2)
        if self.split != "test":
            qpt3d, _, qmask, _ = load_frame_3d(
                qframe,
                self.scene_dir,
                use_msk=self.use_msk,
            )
        else:
            qmask = np.ones(len(qpt2d)).astype(np.bool_)

        # Load ref im & scene pts & poses
        rframe = self.rframes[rid]
        rname = rframe["file_path"]
        rim_path = str(self.im_dir / rname)
        rc2w = torch.tensor(rframe["transform_matrix"], dtype=torch.float32)
        if os.path.exists(self.scene_dir):
            rpt3d, rpt_feat, rmask, unnorm_scene = load_frame_3d(
                rframe,
                self.scene_dir,
                use_msk=self.use_msk,
            )
        else:
            return {
                "rim_path": rim_path,
                "qim_path": qim_path,
                "image": qim,
                "im_mask": qmask,
                "K": qK,
                "c2w": qc2w,
                "rc2w": rc2w,
                "pt2d": qpt2d,
            }

        # Proj 3D pts to query
        qpt2d_proj = project_points3d(
            qK.numpy(),
            qw2c[:3, :3].numpy(),
            qw2c[:3, 3].numpy(),
            rpt3d,
        )

        # Pack
        sample = {
            "rim_path": rim_path,
            "qim_path": qim_path,
            "image": qim,
            "im_mask": qmask,
            "K": qK,
            "c2w": qc2w,
            "rc2w": rc2w,
            "pt2d": qpt2d,
            "pt2d_proj": qpt2d_proj,  # Used for oracle check
            "pt3d": rpt3d,
            "pt_feat": rpt_feat,
            "pt_mask": rmask,
            "unnorm_scene": unnorm_scene,
        }

        if self.split != "test":
            # Map to patch centers
            qpt2d_proj_ds = np.floor(qpt2d_proj / ds).astype(np.int64)
            rpt3d_visible = (
                (qpt2d_proj_ds.min(-1) > 0)
                & (qpt2d_proj_ds[:, 0] < (w // ds))
                & (qpt2d_proj_ds[:, 1] < (h // ds))
            )
            qpt2d_ids = qpt2d_proj_ds[:, 0] + qpt2d_proj_ds[:, 1] * (w // ds)

            # Mask out 3D points not visible by query
            qpt2d_ids = qpt2d_ids.clip(0, len(qpt2d) - 1)

            # Compute match matrix
            rpt3d_ids = np.arange(len(rpt3d))
            match_gt = np.zeros((len(qpt2d), len(rpt3d_ids))).astype(np.float32)
            match_gt[qpt2d_ids, rpt3d_ids] = 1.0
            match_gt = (
                qmask[:, None] * rmask[None, :] * rpt3d_visible[None, :] * match_gt
            )
            if match_gt.sum() < 1:
                match_gt[
                    int(random.random() * (match_gt.shape[0] - 1)),
                    int(random.random() * (match_gt.shape[0] - 1)),
                ] = 1.0

            sample["conf_gt"] = match_gt
            sample["qpt3d"] = qpt3d

        return sample

    def sanity_check_sample(self, sample):
        # Load ref im
        qpt2d = rpt2d = sample["pt2d"]
        rpt3d = sample["pt3d"]
        match_gt = sample["conf_gt"]
        K = sample["K"]
        qc2w = sample["c2w"]

        # Matches
        qids, rids = np.where(match_gt)
        matches = np.concatenate([qpt2d[qids], rpt2d[rids]], axis=-1)

        # Localize im2 with pts3d im1 and pts2d im2
        pose_res = estimate_pose(qpt2d[qids], rpt3d[rids], K, ransac_thres=1)
        if not pose_res:
            R_err = t_err = np.inf
        else:
            R, t, inliers = pose_res
            qw2c_est = torch.from_numpy(get_pose(R, t))
            R_err, t_err = pose_err(qc2w, qw2c_est.inverse())
        print(f"R={R_err:.3f}, t={t_err:.3f} #matches={match_gt.sum()}")
        return sample

    def __getitem__(self, idx):
        return self.load_sample(idx)

    def __len__(self):
        if self.epoch_sample_num > 0:
            return self.epoch_sample_num
        return len(self.pair_ids)

    def __repr__(self):
        fmt_str = f"NeRFMatchPair(\n"
        fmt_str += f" split={self.split} samples={self.__len__()} epoch_sample_num={self.epoch_sample_num}"
        fmt_str += f" img_wh={self.img_wh} imagenet_norm={self.imagenet_norm} \n"
        fmt_str += f" scene_dir={self.scene_dir}\n query_json={self.query_json}\n"
        fmt_str += f" aug_self_pairs={self.aug_self_pairs} \n"
        fmt_str += f" im_dir={self.im_dir}\n"
        fmt_str += f" pairs={self.pair_txt} topk={self.pair_topk} balanced_pair={self.balanced_pair}\n"
        fmt_str += f"\n)\n"
        return fmt_str


class NeRFMatchMultiPair(NeRFMatchPair):
    def __init__(self, config, split="train", val_num=500, debug=False):
        super().__init__(config, split=split, val_num=val_num, debug=debug)
        self.sample_pts = getattr(config, "sample_pts", -1)
        self.sample_mode = getattr(config, "sample_mode", None)
        self.pair_topk = getattr(config, "pair_topk", 10)

    def load_scene_data(self):
        if "scene_anno_path" in self.config:
            scene_anno_path = self.config.scene_anno_path.replace("#scene", self.scene)
            self.ref_json = scene_anno_path.replace("#split", "train")
            self.query_json = scene_anno_path.replace("#split", self.anno_tag)
        else:
            self.ref_json = self.root_dir / "transforms_train.json"
            self.query_json = self.root_dir / f"transforms_{self.anno_tag}.json"

        # Load ref frames
        with open(self.ref_json, "r") as f:
            rframes = json.load(f)["frames"]
            self.rframes = sorted(rframes, key=lambda x: x["file_path"])

        # Load query frames
        if self.query_json == self.ref_json:
            self.qframes = self.rframes
        else:
            with open(self.query_json, "r") as f:
                qframes = json.load(f)["frames"]
                self.qframes = sorted(qframes, key=lambda x: x["file_path"])

        print(f"Loaded {len(self.qframes)} query and {len(self.rframes)} ref frames.")

        # Load pairs
        # pair_ids: {qid:rid_list}
        pairs = load_retrieval_pairs(self.pair_txt)
        self.pair_ids = parse_multipair_ids_balanced(
            self.qframes, self.rframes, pairs, split=self.split, val_num=self.val_num
        )
        self.pair_ids_keys = list(self.pair_ids.keys())
        print(f"Parsed {self.split} pairs {len(self.pair_ids)}.\n")

    def load_ref_pts(self, rids):
        all_rpt3d = []
        all_rmask = []
        all_rpt_feat = []
        all_rK = []
        if self.split == "train":
            rids_ = np.random.choice(rids, self.pair_topk)
        else:
            rids_ = rids[: self.pair_topk]
        for i, rid in enumerate(rids_):
            rframe = self.rframes[rid]
            if i == 0:
                rc2w = torch.tensor(rframe["transform_matrix"], dtype=torch.float32)
            assert os.path.exists(self.scene_dir)
            rpt3d, rpt_feat, rmask, unnorm_scene = load_frame_3d(
                rframe,
                self.scene_dir,
                use_msk=self.use_msk,
            )
            all_rpt3d.append(rpt3d)
            all_rpt_feat.append(rpt_feat)
            all_rmask.append(rmask)

        # Merge
        rpt3d = np.transpose(np.dstack(all_rpt3d), (2, 0, 1))
        rpt_feat = np.transpose(np.dstack(all_rpt_feat), (2, 0, 1))
        rmask = np.transpose(np.dstack(all_rmask), (2, 1, 0))

        # Reshape
        rpt3d = rearrange(rpt3d, "b n d -> (b n) d")
        rpt_feat = rearrange(rpt_feat, "b n d -> (b n) d")
        rmask = rearrange(rmask, "b n d -> (b n) d").squeeze()

        if not self.sample_mode:
            return rpt3d, rpt_feat, rmask, unnorm_scene, rc2w

        # Keep pt3d visible in all references
        visible = torch.ones(len(rpt3d)).bool()
        all_rpt2d = []
        for i, rid in enumerate(rids_):
            rframe = self.rframes[rid]
            rc2w = torch.tensor(rframe["transform_matrix"], dtype=torch.float32)
            rw2c = rc2w.inverse()

            # Check visibility at target resolution
            WH = torch.tensor(self.img_wh, dtype=torch.float32)
            sK = torch.from_numpy(
                np.diag([WH[0] / rframe["width"], WH[1] / rframe["height"], 1])
            ).float()
            rK = torch.tensor(rframe["intrinsics"], dtype=torch.float32)
            rpt2d = project_points3d(
                sK @ rK,
                rw2c[:3, :3],
                rw2c[:3, 3],
                torch.tensor(rpt3d),
            )
            all_rpt2d.append(rpt2d)
            i_visible = (rpt2d >= 0).all(-1) & (rpt2d < WH).all(-1)
            intersect = visible & i_visible
            union = visible | i_visible
            # visible = union if intersect.sum() < 10 else intersect
            visible = union if intersect.sum() < visible.sum() / 3 else intersect
        rpt2d = torch.stack(all_rpt2d)[:, visible, :]
        rpt3d = rpt3d[visible]
        rpt_feat = rpt_feat[visible]
        rmask = rmask[visible]

        # Sub-sampling
        if self.sample_mode == "rand":
            N = len(rpt3d)
            idx = torch.randperm(N)
            if self.sample_pts > 0:
                idx = idx.repeat((self.sample_pts // N) + 1)[: self.sample_pts]
            rpt3d = rpt3d[idx]
            rpt_feat = rpt_feat[idx]
            rmask = rmask[idx]

        return rpt3d, rpt_feat, rmask, unnorm_scene, rc2w

    def load_sample(self, idx):
        if self.epoch_sample_num > 0:
            pidx = np.random.randint(len(self.pair_ids))
        else:
            pidx = idx
        qid = self.pair_ids_keys[pidx]
        qframe = self.qframes[qid]
        ds = self.model_ds
        w, h = self.img_wh

        # GT pose
        qc2w = torch.tensor(qframe["transform_matrix"], dtype=torch.float32)
        qw2c = qc2w.inverse()

        # Load query image
        qim_path = str(self.root_dir / qframe["file_path"])
        qim, sK = self.process_img(qim_path, imagenet_norm=self.imagenet_norm)
        qK = sK @ torch.tensor(qframe["intrinsics"], dtype=torch.float32)
        qpt2d = get_pixel_coords_grid(w, h, ds=ds).reshape(-1, 2)
        if self.split != "test":
            qpt3d, _, qmask, _ = load_frame_3d(
                qframe,
                self.scene_dir,
                use_msk=self.use_msk,
            )
        else:
            qmask = np.ones(len(qpt2d)).astype(np.bool_)

        # Load ref data
        rpt3d, rpt_feat, rmask, unnorm_scene, rc2w = self.load_ref_pts(
            self.pair_ids[qid]
        )

        # Proj 3D pts to query
        qpt2d_proj = project_points3d(
            qK.numpy(),
            qw2c[:3, :3].numpy(),
            qw2c[:3, 3].numpy(),
            rpt3d,
        )

        # Map to patch centers
        qpt2d_proj_ds = np.floor(qpt2d_proj / ds).astype(np.int64)
        rpt3d_visible = (
            (qpt2d_proj_ds.min(-1) > 0)
            & (qpt2d_proj_ds[:, 0] < (w // ds))
            & (qpt2d_proj_ds[:, 1] < (h // ds))
        )
        qpt2d_ids = qpt2d_proj_ds[:, 0] + qpt2d_proj_ds[:, 1] * (w // ds)

        # Mask out 3D points not visible by query
        qpt2d_ids = qpt2d_ids.clip(0, len(qpt2d) - 1)

        # Compute match matrix
        rpt3d_ids = np.arange(len(rpt3d))
        match_gt = np.zeros((len(qpt2d), len(rpt3d_ids))).astype(np.float32)
        match_gt[qpt2d_ids, rpt3d_ids] = 1.0
        match_gt = qmask[:, None] * rmask[None, :] * rpt3d_visible[None, :] * match_gt

        if match_gt.sum() < 1:
            match_gt[
                int(random.random() * (match_gt.shape[0] - 1)),
                int(random.random() * (match_gt.shape[0] - 1)),
            ] = 1.0

        # Reshape multiple pairs when no subsample
        if not self.sample_mode:
            n = len(rpt3d) // self.pair_topk
            rpt3d = rpt3d.reshape(self.pair_topk, n, -1)
            rpt_feat = rpt_feat.reshape(self.pair_topk, n, -1)
            rmask = rmask.reshape(self.pair_topk, n)

        # Pack
        sample = {
            "qim_path": qim_path,
            "image": qim,
            "im_mask": qmask,
            "K": qK,
            "c2w": qc2w,
            "rc2w": rc2w,
            "pt2d": qpt2d,
            "pt2d_proj": qpt2d_proj,
            "pt3d": rpt3d,
            "pt_feat": rpt_feat,
            "pt_mask": rmask,
            "conf_gt": match_gt,
            "unnorm_scene": unnorm_scene,
        }

        if self.split != "test":
            sample["qpt3d"] = qpt3d

        return sample

    def __getitem__(self, idx):
        return self.load_sample(idx)

    def __len__(self):
        if self.epoch_sample_num > 0:
            return self.epoch_sample_num
        return len(self.pair_ids)

    def __repr__(self):
        fmt_str = f"NeRFMatchMultiPair(\n"
        fmt_str += f" split={self.split} samples={self.__len__()} epoch_sample_num={self.epoch_sample_num}"
        fmt_str += f" img_wh={self.img_wh} imagenet_norm={self.imagenet_norm}\n"
        fmt_str += f" scene_dir={self.scene_dir} query_json={self.query_json}\n"
        fmt_str += f" im_dir={self.im_dir}\n"
        fmt_str += f" pairs={self.pair_txt} topk={self.pair_topk} sample_pts={self.sample_pts} sample_mode={self.sample_mode} \n"
        fmt_str += f"\n)\n"
        return fmt_str
