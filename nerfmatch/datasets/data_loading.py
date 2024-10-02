# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import torch
import json
import os
import numpy as np
from pathlib import Path
import glob
from os.path import join as osp
import json
from collections import defaultdict
from scipy.spatial.transform import Rotation as Rotation
from transforms3d.quaternions import quat2mat

from ..utils.geometry import get_pose

import random

SEVEN_SCENES = ["heads", "chess", "fire", "office", "pumpkin", "redkitchen", "stairs"]

CAMBRIDGE_LANDMARKS = [
    "KingsCollege",
    "OldHospital",
    "ShopFacade",
    "StMarysChurch",
    "GreatCourt",
]


def load_frame_3d(frame, scene_dir, use_msk=None, return_pose=False):
    fname = frame["file_path"]

    # Load scene cache
    pt_name = fname.replace("/", "_").replace(".color", "").replace(".png", "")
    pt_path = os.path.join(scene_dir, f"{pt_name}.npy")
    scene_pts = np.load(pt_path, allow_pickle=True).item()
    pt3d = scene_pts["pt3d"]
    unnorm_scene = scene_pts["unnorm_scene"]
    c2w = None
    if "cam2scene" in scene_pts:
        cam2scene = scene_pts["cam2scene"]
        c2w = torch.from_numpy(unnorm_scene @ cam2scene)

    pt_feat = scene_pts["pt_feat"]

    # different type of masking
    mask = np.ones(len(pt3d)).astype(np.bool_)
    if "pt_mask" in scene_pts.keys() and use_msk:
        if use_msk == "sky":
            mask = (
                1
                - scene_pts["sky_mask"][0].reshape(
                    -1,
                )
            ).astype(np.bool_)
        elif use_msk == "corr":
            mask = (
                1
                - scene_pts["corr_mask"].reshape(
                    -1,
                )
            ).astype(np.bool_)
        else:
            mask = (
                1
                - scene_pts["pt_mask"][0].reshape(
                    -1,
                )
            ).astype(np.bool_)

    if return_pose:
        return pt3d, pt_feat, mask, unnorm_scene, c2w

    return pt3d, pt_feat, mask, unnorm_scene


def split_val_ids(total_num, chunck_size=4, val_percent=0.1):
    """Select uniform chuncks to compose validation subset."""

    chunck_num = total_num // chunck_size
    val_num = int(val_percent * total_num)
    ids = np.array_split(np.arange(total_num), chunck_num)
    skip = len(ids) // (val_num // chunck_size)
    val_ids = np.concatenate(ids[::skip])[:val_num]
    return val_ids


def load_topk_retrieval_pairs(pair_txt, kmax=5, mode="top"):
    k_count = defaultdict(int)
    pairs = []
    all_pairs = defaultdict(list)
    with open(pair_txt, "r") as f:
        for line in f.readlines():
            pair = line.split()[:2]
            if mode == "random":
                all_pairs[pair[0]].append(pair)
            if k_count[pair[0]] >= kmax and kmax > 0:
                continue
            pairs.append(pair)
            k_count[pair[0]] += 1

    if mode == "random":
        pairs = []
        for k in all_pairs.keys():
            pairs += random.sample(all_pairs[k], kmax)

    print(
        f"Load {len(pairs)} pairs  from {pair_txt}: {len(k_count)} queries, topk={kmax} "
    )
    return pairs


def load_retrieval_pairs(pair_txt):
    pairs = defaultdict(list)
    with open(pair_txt, "r") as f:
        for line in f.readlines():
            pair = line.split()
            pairs[pair[0]].append(pair[1])

    print(f"Load {len(pairs)} samples  from {pair_txt}")
    return pairs


def parse_multipair_ids_balanced(qframes, rframes, pairs, split="train", val_num=500):
    np.random.seed(val_num)

    rname2ids = {f["file_path"]: i for i, f in enumerate(rframes)}
    qname2ids = {f["file_path"]: i for i, f in enumerate(qframes)}

    if split == "test":
        pair_ids = defaultdict(list)
        for qname, rnames in pairs.items():
            if qname not in qname2ids:
                continue
            qid = qname2ids[qname]
            rids = []
            for rname in rnames:
                if rname not in rname2ids:
                    continue
                rids.append(rname2ids[rname])
            pair_ids[qid] = rids
    else:
        # Define ids for validation split
        val_qids = split_val_ids(len(qframes), val_percent=0.1)

        # Collect the corresponding pairs
        train_pairs = defaultdict(list)
        val_pairs = defaultdict(list)
        for qname, rnames in pairs.items():
            if qname not in qname2ids:
                continue
            qid = qname2ids[qname]
            rids = []
            for rname in rnames:
                if rname not in rname2ids:
                    continue
                rids.append(rname2ids[rname])
            if qid in val_qids:
                val_pairs[qid] = rids
            else:
                train_pairs[qid] = rids

        if val_num < len(val_pairs):
            val_keys = list(val_pairs.keys())
            ids = np.random.permutation(len(val_keys))
            val_pairs = {val_keys[i]: val_pairs[val_keys[i]] for i in ids[:val_num]}
        pair_ids = train_pairs if split == "train" else val_pairs
    return pair_ids


def parse_pair_ids_balanced(qframes, rframes, pairs, split="train", val_num=500):
    np.random.seed(val_num)

    rname2ids = {f["file_path"]: i for i, f in enumerate(rframes)}
    qname2ids = {f["file_path"]: i for i, f in enumerate(qframes)}

    if split == "test":
        pair_ids = []
        for i, (qname, rname) in enumerate(pairs):
            if not (qname in qname2ids and rname in rname2ids):
                continue
            pair_ids.append((qname2ids[qname], rname2ids[rname]))
    else:
        # Define ids for validation split
        val_qids = split_val_ids(len(qframes), val_percent=0.1)

        # Collect the corresponding pairs
        train_pairs = []
        val_pairs = []
        for qname, rname in pairs:
            if not (qname in qname2ids):  # and rname in rname2ids):
                continue
            qid = qname2ids[qname]
            if qid in val_qids:
                # Val split
                if rname in rname2ids:
                    # Normal retrieval pairs
                    ids = (qid, rname2ids[rname])
                else:
                    continue
                val_pairs.append(ids)
            else:
                # Train split
                if rname in rname2ids:
                    # Normal retrieval pairs
                    ids = (qid, rname2ids[rname])
                elif "_aug" in rname:
                    # Augmentation pairs
                    ids = (qid, rname)
                else:
                    continue
                train_pairs.append(ids)

        if val_num < len(val_pairs):
            ids = np.random.permutation(len(val_pairs))
            val_pairs = [val_pairs[i] for i in ids[:val_num]]
        pair_ids = train_pairs if split == "train" else val_pairs
    return pair_ids


def parse_pair_ids(qframes, rframes, pairs, split="train", val_num=500):
    rname2ids = {f["file_path"]: i for i, f in enumerate(rframes)}
    qname2ids = {f["file_path"]: i for i, f in enumerate(qframes)}

    if split == "test":
        pair_ids = []
        for i, (qname, rname) in enumerate(pairs):
            if not (qname in qname2ids and rname in rname2ids):
                continue
            pair_ids.append((qname2ids[qname], rname2ids[rname]))
    else:
        val_num = min(len(pairs) // 5, val_num)
        indices = np.arange(len(pairs))
        skip = len(pairs) // val_num
        val_indices = indices[::skip][:val_num]
        train_ids = []
        val_ids = []
        for i, (qname, rname) in enumerate(pairs):
            if not (qname in qname2ids and rname in rname2ids):
                continue
            ids = (qname2ids[qname], rname2ids[rname])
            if i in val_indices:
                val_ids.append(ids)
            else:
                train_ids.append(ids)
        if split == "train":
            pair_ids = train_ids
        elif split == "val":
            pair_ids = val_ids
    return pair_ids


def load_retrieval_pair_ids(frames, pair_txt, topk=1):
    im2ids = {f["file_path"]: i for i, f in enumerate(frames)}
    pair_ids = defaultdict(list)

    with open(pair_txt, "r") as f:
        for line in f.readlines():
            qim, rim = line.split()
            if qim not in im2ids or rim not in im2ids:
                continue
            qlist = pair_ids[im2ids[qim]]
            if len(qlist) >= topk:
                continue
            qlist.append(im2ids[rim])
    return pair_ids


def load_scene_cache(scene_cache_dir, masked=True):
    pts = []
    colors = []
    for pt_path in glob.glob(osp(scene_cache_dir, "*.npy")):
        scene_pts = np.load(pt_path, allow_pickle=True).item()
        pts_i = scene_pts["pt3d"]
        color_i = scene_pts["pt_color"]
        if masked and "mask" in scene_pts:
            mask = scene_pts["mask"]
            pts_i = pts_i[mask]
            color_i = color_i[mask]
        pts.append(pts_i)
        colors.append(color_i)
    print(f"Load {len(pts)} pts files from {scene_cache_dir}")
    return pts, colors


def generate_7scenes_annotations(root_dir, cache_dir=None, overwrite=False):
    """Generate 7scenes annotations cache file in custom format."""

    # Scene prior
    # https://github.com/tsattler/visloc_pseudo_gt_limitations/blob/main/setup_7scenes.py#L11-L18
    H = 480
    W = 640
    focal = 525.0
    K = [[focal, 0, W / 2], [0, focal, H / 2], [0, 0, 1]]

    cache_dir = Path(cache_dir if cache_dir else root_dir)
    cache_dir.mkdir(exist_ok=True, parents=True)
    for scene in SEVEN_SCENES:
        data_dir = Path(root_dir) / scene
        for split in ["train", "test"]:
            meta_cache_path = cache_dir / f"transforms_{scene}_{split}.json"
            if meta_cache_path.exists() and not overwrite:
                continue

            # Parse split sequences
            split_file = "TrainSplit.txt" if split == "train" else "TestSplit.txt"
            with open(data_dir / split_file, "r") as f:
                seqs = [
                    "seq-" + l.strip().split("sequence")[-1].zfill(2)
                    for l in f
                    if not l.startswith("#")
                ]

            # Load scene poses
            poses_paths = []
            for seq in seqs:
                poses_paths += glob.glob(str(data_dir / seq / "*.pose.txt"))

            # Construct metadata dict
            meta_dict = {}
            meta_dict["frames"] = []
            for pose_file in sorted(poses_paths):
                frame_path = "seq" + pose_file.split("seq")[-1].replace(
                    "pose.txt", "color.png"
                )
                frame_meta = dict(
                    file_path=frame_path,
                    intrinsics=K,
                    height=H,
                    width=W,
                    transform_matrix=np.loadtxt(pose_file).tolist(),
                )
                meta_dict["frames"].append(frame_meta)

            # Dump as json
            with open(meta_cache_path, "w") as fp:
                json.dump(meta_dict, fp, indent=4)
            print(f"Generated annotations: {meta_cache_path}")


def convert_7scenes_pgt_annoations(pgt_dir, cache_dir, overwrite=False):
    """Convert pose annotations to predefined json format.
    Pesudo-ground truth files are downloaded from https://github.com/tsattler/visloc_pseudo_gt_limitations/tree/main/pgt.
    """

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(exist_ok=True, parents=True)
    for pgt_txt in glob.iglob(os.path.join(pgt_dir, "*.txt")):
        print(pgt_txt)
        basename = os.path.basename(pgt_txt)
        meta_cache_path = cache_dir / f"transforms_{basename.replace('.txt', '.json')}"
        if meta_cache_path.exists() and not overwrite:
            continue

        with open(pgt_txt, "r") as f:
            # Pose format: file qw qx qy qz tx ty tz (f)
            pose_data = f.readlines()

        meta_dict = {}
        meta_dict["frames"] = []
        for pose_string in pose_data:
            pose_string = pose_string.split()
            file_name = pose_string[0]

            pose_q = np.array(pose_string[1:5])
            pose_q = np.array([pose_q[1], pose_q[2], pose_q[3], pose_q[0]])
            pose_t = np.array(pose_string[5:8])
            pose_R = Rotation.from_quat(pose_q).as_matrix()

            # Convert world->cam to cam->world
            w2c = np.identity(4)
            w2c[0:3, 0:3] = pose_R
            w2c[0:3, 3] = pose_t
            c2w = np.linalg.inv(w2c)

            # Intrinsics
            H = 480
            W = 640
            if len(pose_string) > 8:
                focal = float(pose_string[8])
            else:
                focal = 525.0
            K = [[focal, 0, W / 2], [0, focal, H / 2], [0, 0, 1]]

            # Construct metadata dict
            frame_meta = dict(
                file_path=file_name,
                intrinsics=K,
                height=H,
                width=W,
                transform_matrix=c2w.tolist(),
            )
            meta_dict["frames"].append(frame_meta)

        # Dump as json
        with open(meta_cache_path, "w") as fp:
            json.dump(meta_dict, fp, indent=4)
        print(f"Generated annotations: {meta_cache_path}")


def generate_cambridge_annotations(
    root_dir,
    cache_dir=None,
):
    root_dir = Path(root_dir)
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

    for scene in CAMBRIDGE_LANDMARKS:
        data_dir = Path(root_dir) / scene
        frame_dict = parse_cambridge_nvm(data_dir / "reconstruction.nvm")
        for split in ["train", "test"]:
            meta_dict = {}
            if cache_dir is not None:
                meta_cache_path = cache_dir / f"transforms_{scene}_{split}.json"
            else:
                meta_cache_path = data_dir / f"transforms_{split}.json"
            print(f"Generate annotations: {data_dir / split} ...")

            # Load train / test splits
            ims = [
                line.split(" ")[0]
                for line in open(data_dir / f"dataset_{split}.txt").readlines()[3::]
            ]
            print(f"Split={split} ims={len(ims)}")

            # Construct metadata dict
            meta_dict["frames"] = [frame_dict[k] for k in ims if k in frame_dict]

            # Dump as json
            with open(meta_cache_path, "w") as fp:
                json.dump(meta_dict, fp, indent=4)
            print(f"Generated annotations: {meta_cache_path}")


def parse_cambridge_nvm(nvm):
    meta_dict = {}
    W, H = 1920, 1080
    with open(nvm, "r") as f:
        # Skip headding lines
        next(f)
        next(f)
        cam_num = int(f.readline().split()[0])
        print(f"Loading cameras={cam_num}")
        for i in range(cam_num):
            # fname f w p q r x y z r e
            line = f.readline()
            cur = line.split()[0:9]
            frame_path = cur[0].replace("jpg", "png")

            # Intrinsics
            focal = float(cur[1])
            K = [[focal, 0, W / 2], [0, focal, H / 2], [0, 0, 1]]

            # Camera pose -> camera2world
            q = np.array([float(v) for v in cur[2:6]], dtype=np.float32)
            c = np.array([float(v) for v in cur[6:9]], dtype=np.float32)

            # Skip ill-samples (happened in GreatCourt
            if np.abs(np.max(c)) > 1e5:
                print(f"Skip problematic sample {frame_path} with c={c}")
                continue
            c2w = get_pose(quat2mat(q).T, c)

            frame_meta = dict(
                file_path=frame_path,
                intrinsics=K,
                height=H,
                width=W,
                transform_matrix=c2w.tolist(),
            )
            meta_dict[frame_path] = frame_meta
    return meta_dict
