# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

import numpy as np
from PIL import Image
from pathlib import Path
import json
from tqdm import tqdm

from ..nerf.render_utils import (
    get_ray_dirs,
    get_rays_c2w,
    prepare_rays_data,
)
from .data_loading import load_retrieval_pair_ids
from ..nerf.scene_utils import compute_scene_normalization_fst, rays_intersect_sphere


class NerfBaseDataset(Dataset):
    def __init__(self, config, split="train", val_num=8, debug=False):
        self.config = config

        # Dataset paths
        self.split = split
        self.scene = config.scene
        self.root_dir = Path(config.data_dir) / self.scene
        self.max_sample_num = getattr(config, "max_sample_num", None)
        self.val_num = 3 if debug else val_num

        # Ray sampling
        self.img_wh = config.img_wh
        self.ray_type = getattr(config, "ray_type", "normal")
        self.norm_ray_dir = getattr(config, "norm_ray_dir", True)

        # Inference caching
        self.downsample = getattr(config, "downsample", 1)

        # Load frames
        frames = self.load_scene_frames(config)

        # Train / val / test split indices
        self.init_split_indices(self.dataset_size)

        # Nomralize scene cameras
        self.init_scene_normalization(config)

        # Image masking
        self.init_masks(config, frames)

        # Retrieval pair for matching training metrics
        self.init_retrieval_pair(frames, config)

        # Load all rays and rgb data at once
        if self.split == "train":
            self.process_train_data()

        # For debugging
        self.frame_inds = {}
        for i in range(len(self.split_inds)):
            self.frame_inds["_".join(frames[i]["file_path"].split("/"))[:-4]] = (
                self.split_inds[i]
            )

    def load_scene_frames(self, config, sort=True):
        if "scene_anno_path" in config:
            scene_anno_path = config.scene_anno_path.replace("#scene", self.scene)
            self.train_json = scene_anno_path.replace("#split", "train")
            self.test_json = scene_anno_path.replace("#split", "test")
        else:
            self.train_json = self.root_dir / "transforms_train.json"
            self.test_json = self.root_dir / "transforms_test.json"
        self.scene_anno_path = (
            self.test_json if self.split == "test" else self.train_json
        )
        self.scene_seq = (
            None if self.split == "test" else getattr(config, "scene_seq", None)
        )

        # Load frames
        with open(self.scene_anno_path, "r") as f:
            scene_anno = json.load(f)
            frames = scene_anno["frames"]

        # Keep only frames within specific seq
        if self.scene_seq is not None:
            frames = [
                f for f in frames if f["file_path"].split("/")[0] == self.scene_seq
            ]

        # Sort based on the frame names
        if sort:
            frames = sorted(frames, key=lambda x: x["file_path"])

        # Parse annotations
        seq_ind = [f["file_path"].split("/")[0] for f in frames]
        seq_map = {s: i for i, s in enumerate(np.unique(seq_ind))}
        self.seq_ind = [seq_map[i] for i in seq_ind]
        self.img_paths = [self.root_dir / f["file_path"] for f in frames]
        self.img_idxs = [
            f["file_path"].replace("/", "_").replace(".color", "").replace(".png", "")
            for f in frames
        ]
        self.cam2scenes = [torch.FloatTensor(f["transform_matrix"]) for f in frames]
        self.org_Ks = [torch.FloatTensor(f["intrinsics"]) for f in frames]
        self.dataset_size = len(frames)
        print(f"Number of frames: {self.dataset_size}")
        return frames

    def init_retrieval_pair(self, frames, config):
        self.pair_txt = (
            getattr(config, f"train_pair_txt", None) if self.split == "val" else None
        )
        if not self.pair_txt:
            return
        self.pair_txt = self.pair_txt.replace("$scene", config.scene)
        self.pair_txt = self.pair_txt.replace("#scene", config.scene)
        self.pair_ids = load_retrieval_pair_ids(frames, self.pair_txt, topk=10)

    def init_scene_normalization(self, config):
        # Compute scene normalization
        self.snorm_type = getattr(config, "snorm_type", "fst")
        self.rescale_factor = getattr(config, "rescale_factor", 1.0)
        if self.snorm_type == "fst":
            self.max_frustum_depth = getattr(config, "max_frustum_depth", 10)
            self.scale_tag = f"snfst_dep{self.max_frustum_depth}rs{self.rescale_factor}"
            self.snorm_json = (
                config.snorm_json
                if getattr(config, "snorm_json", None)
                else self.train_json
            )
            print(f"Compute scene norm for {self.snorm_json}")
            self.scene2s_scene = compute_scene_normalization_fst(
                self.snorm_json, self.max_frustum_depth, self.rescale_factor
            )
        if self.scene2s_scene is not None:
            print(f"Scene normalization ({self.scale_tag}):{self.scene2s_scene} ")
            self.unnorm_scene = self.scene2s_scene.inverse()
            self.s_scaling = self.scene2s_scene[0, 0]
        else:
            self.unnorm_scene = 1

        # Scale camera poses
        self.cam2s_scenes = {}
        for idx, c2w in enumerate(self.cam2scenes):
            self.cam2s_scenes[idx] = self.scene2s_scene @ c2w

    def init_masks(self, config, frames):
        self.exclude_masks = getattr(config, "exclude_masks", True)
        self.white_bg = getattr(config, "white_bg", False)
        self.load_transient = getattr(config, "mask_transient", False)

        # Init mask paths
        mask_dir = Path(getattr(config, "mask_dir", "data"))
        self.root_trnz_mask = mask_dir / "masks_trnz_cars" / self.scene
        self.root_sem_mask = mask_dir / "masks_semantic" / self.scene
        self.root_bg_mask = mask_dir / "masks_bg" / self.scene
        self.mask_trnz_paths = [self.root_trnz_mask / f["file_path"] for f in frames]
        self.mask_sem_paths = [
            self.root_sem_mask / (f["file_path"][:-3] + "npy") for f in frames
        ]
        self.mask_bg_paths = [self.root_bg_mask / f["file_path"] for f in frames]

    def init_split_indices(self, num_samples):
        sample_inds = np.arange(num_samples)
        if self.split in ["train", "val", "val_check"]:
            frame_skip = len(sample_inds) // self.val_num
            val_inds = sample_inds[:: max(1, frame_skip)][: self.val_num]
            train_inds = [i for i in sample_inds if i not in val_inds]

            # Check maximum limit
            if self.max_sample_num and len(train_inds) > self.max_sample_num:
                np.random.seed(1357)
                train_inds = np.random.choice(train_inds, self.max_sample_num)
            print(
                f"Scene={len(sample_inds)} train={len(train_inds)} val={len(val_inds)}"
            )
            self.split_inds = (
                val_inds if self.split in ["val", "val_check"] else train_inds
            )
        else:
            if self.max_sample_num:
                self.split_inds = sample_inds[: self.max_sample_num]
            else:
                self.split_inds = sample_inds
            print(f"Scene={len(sample_inds)} test={len(self.split_inds)}")
        self.split_inds.sort()

    def process_img(self, img_path, load_mask=False):
        if "_aug" in str(img_path):
            name = str(img_path).split("_aug")
            img_path = name[0] + "." + name[1].split(".")[-1]

        # Load img
        img = Image.open(img_path)
        if load_mask:
            img = img.convert("L")

        # Compute intrinsics scaler
        sK = torch.from_numpy(
            np.diag([self.img_wh[0] / img.size[0], self.img_wh[1] / img.size[1], 1])
        ).float()

        # Resize -> normalize -> reshape (N, 3)
        img = img.resize(self.img_wh, Image.LANCZOS)
        img = T.functional.to_tensor(img)
        img = img.permute(1, 2, 0)
        return img, sK

    def mask_img_bg(self, img, sample_idx):
        bg_color = torch.Tensor([1.0, 1.0, 1.0])
        bg_mask, _ = self.process_img(self.mask_bg_paths[sample_idx], load_mask=True)
        bg_mask = torch.round(bg_mask)
        img = img[:, :] * (1 - bg_mask) + bg_mask * bg_color
        return img

    def mask_transient(self, sample_data, sample_idx, exclude_mask=True):
        mask, _ = self.process_img(self.mask_trnz_paths[sample_idx], load_mask=True)
        mask = torch.round(mask).reshape(-1, 1)
        sample_data["mask"] = 1 - mask.clone()

        if exclude_mask:
            mask = (1 - mask).type(torch.BoolTensor)
            non_msk_ind = [i for i in range(mask.size(0)) if mask[i, 0]]
            main_rays_num = sample_data["rgbs"].size(0)
            for k in sample_data.keys():
                if torch.is_tensor(sample_data[k]):
                    if sample_data[k].size(0) == main_rays_num:
                        sample_data[k] = sample_data[k][non_msk_ind, :]

    def data_downsample(self, sample_data):
        ds = self.downsample
        img_w, img_h = sample_data["img_wh"]
        sample_data["r_orig"] = sample_data["rays"]

        for k, v in sample_data.items():
            if k in ["rgbs", "rays", "img_ijs", "ts", "mask"]:
                v_ = v.reshape(img_h, img_w, -1)
                v_ = v_[ds // 2 :: ds, ds // 2 :: ds]
                sample_data[k] = v_

        sample_data["img_wh"] = sample_data["img_wh"] // ds
        if self.white_bg:
            bg_mask = (bg_mask + mask.reshape(bg_mask.shape)).cpu().numpy()
            sample_data["sky_mask"] = bg_mask[ds // 2 :: ds, ds // 2 :: ds]

    def load_sample(
        self,
        sample_idx,
        exclude_mask=True,
        validation=False,
        camera_only=False,
        camera_mat=None,
    ):
        cam2s_scene = self.cam2s_scenes[sample_idx].float()
        if camera_only:
            return cam2s_scene
        if camera_mat is not None:
            cam2s_scene = camera_mat
        cam2scene = self.cam2scenes[sample_idx].float()

        img, sK = self.process_img(self.img_paths[sample_idx])
        img_org = img.clone()
        K = sK @ self.org_Ks[sample_idx]
        img_w, img_h = self.img_wh

        if self.white_bg:
            img = self.mask_img_bg(img, sample_idx)

        # Compute sample data
        img_ijs = torch.from_numpy(np.argwhere(np.ones_like(img[..., 0])))
        rgbs = img.reshape(-1, 3)

        directions, xys = get_ray_dirs(img_h, img_w, K, return_xys=True)
        rays_o, rays_d, viewdirs = get_rays_c2w(directions, cam2s_scene)
        rays_d = viewdirs if self.norm_ray_dir else rays_d

        # Compute far plane dynamically
        try:
            far = rays_intersect_sphere(
                rays_o.view(-1, 3), viewdirs.view(-1, 3), r=1
            ).reshape(img_h, img_w, 1)
        except Exception as e:
            far = torch.ones((img_h, img_w, 1)).float()
            print(f"Fail to find far plane: {e}! Set far to 1.")

        # Construct rays
        rays = prepare_rays_data(
            rays_o, rays_d, viewdirs, 0.01, far, comp_radii=self.ray_type == "mip"
        )

        sample_data = {
            "img_idx": self.img_idxs[sample_idx],
            "rgbs": rgbs,
            "rays": rays,
            "img_ijs": img_ijs,
            "img_wh": torch.LongTensor([img_w, img_h]),
            "K": K,
            "ts": self.seq_ind[sample_idx]
            * torch.ones(len(rays), 1).long(),  # sample_idx
            "unnorm_scene": self.unnorm_scene,
            "seq_ind": self.seq_ind[sample_idx],
            "cam2scene": cam2s_scene,
            "cam2scene_org": cam2scene,
        }

        if self.load_transient:
            self.mask_transient(sample_data, sample_idx, exclude_mask=exclude_mask)

        # Downsample
        if self.downsample > 1:
            self.data_downsample(sample_data)
        return sample_data

    def load_retrieval_pair_sample(self, sample_idx, validation=True):
        # Determinisitic random sampling
        kid = sample_idx % len(self.pair_ids[sample_idx])
        ret_idx = self.pair_ids[sample_idx][kid]
        sample1 = self.load_sample(
            sample_idx, exclude_mask=False, validation=validation
        )
        sample2 = self.load_sample(ret_idx, exclude_mask=False, validation=validation)

        sample = {}
        sample["img_idx"] = [sample1["img_idx"], sample2["img_idx"]]
        sample["rays"] = torch.cat([sample1["rays"], sample2["rays"]], dim=0)
        if "mask" in sample1:
            sample["mask"] = torch.cat([sample1["mask"], sample2["mask"]], dim=0)
        sample["rgbs"] = torch.cat([sample1["rgbs"], sample2["rgbs"]], dim=0)
        sample["img_wh"] = torch.cat([sample1["img_wh"], sample2["img_wh"]], dim=0)
        sample["K"] = torch.cat([sample1["K"], sample2["K"]], dim=0)
        sample["seq_ind"] = [sample1["seq_ind"], sample2["seq_ind"]]
        sample["c2w"] = torch.cat(
            [
                sample1["unnorm_scene"] @ sample1["cam2scene"],
                sample2["unnorm_scene"] @ sample2["cam2scene"],
            ],
            dim=0,
        )
        sample["unnorm_scene"] = self.unnorm_scene
        return sample

    def process_train_data(self):
        self.all_rgbs = []
        self.all_rays = []
        self.all_img_ijs = []
        self.all_ts = []
        self.all_ind = []
        self.all_msks = []
        self.all_wh = []
        self.all_K = []
        self.all_dirs = []
        self.all_c2s = []

        print("Pre-Loading Data:")
        for i, sample_idx in enumerate(tqdm(self.split_inds)):
            sample_data = self.load_sample(sample_idx, exclude_mask=self.exclude_masks)
            self.all_rays.append(sample_data["rays"])
            self.all_rgbs.append(sample_data["rgbs"])
            self.all_img_ijs.append(sample_data["img_ijs"])

            # Sample_idx
            # self.all_ts.append(sample_data["ts"])
            self.all_ts.append(
                torch.ones(len(sample_data["rays"]), 1).long() * sample_data["seq_ind"]
            )
            if "mask" in sample_data.keys():
                self.all_msks.append(sample_data["mask"])
        self.all_wh = sample_data["img_wh"] if "img_wh" in sample_data.keys() else None

        # Concate all samples
        self.all_rays = torch.cat(self.all_rays, 0)
        self.all_rgbs = torch.cat(self.all_rgbs, 0)
        self.all_img_ijs = torch.cat(self.all_img_ijs, 0)
        self.all_ts = torch.cat(self.all_ts, 0)
        if "mask" in sample_data.keys():
            self.all_msks = torch.cat(self.all_msks, 0)

    def getframe(self, frame_name, camera_only=False, id=False, camera_input=None):
        if camera_only:
            if id:
                sample = self.load_sample(frame_name, camera_only=True)
            elif frame_name in self.frame_inds:
                sample = self.load_sample(self.frame_inds[frame_name], camera_only=True)
            else:
                return None
        elif camera_input is not None:
            sample = self.load_sample(
                0, exclude_mask=False, validation=True, camera_mat=camera_input
            )
        else:
            sample = self.load_sample(
                self.frame_inds[frame_name], exclude_mask=False, validation=True
            )
        return sample

    def __len__(self):
        if self.split == "train":
            return len(self.all_rays)
        return len(self.split_inds)

    def __getitem__(self, idx):
        if self.split in ["train", "all"]:
            sample = {
                "rays": self.all_rays[idx],
                "rgbs": self.all_rgbs[idx],
                "ts": self.all_ts[idx],
                "img_ijs": self.all_img_ijs[idx],
                "img_wh": self.all_wh,
            }
            if self.load_transient and len(self.all_msks) > 0:
                sample["mask"] = self.all_msks[idx]
        else:
            if self.pair_txt:
                sample = self.load_retrieval_pair_sample(
                    self.split_inds[idx], validation=True
                )
            else:
                sample = self.load_sample(
                    self.split_inds[idx], exclude_mask=False, validation=True
                )
        return sample

    def __repr__(self):
        fmt_str = f"NerfBaseDataset(split={self.split} samples={self.__len__()} "
        fmt_str += f"img_wh={self.img_wh} downsample={self.downsample} \n annotations={self.scene_anno_path} tag={self.scale_tag} scene_seq={self.scene_seq} max_sample={self.max_sample_num})\n pair_txt={self.pair_txt}"
        return fmt_str
