# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import argparse
import os

from nerfmatch.nerf_evaluator import load_nerf_from_ckpt
from nerfmatch.datasets.data_loading import (
    CAMBRIDGE_LANDMARKS,
    SEVEN_SCENES,
)

SCENES = {
    "cambridge": CAMBRIDGE_LANDMARKS,
    "7scenes": SEVEN_SCENES,
}


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Init model
    evaluator = load_nerf_from_ckpt(
        args.ckpt, args, mask=args.mask, frame_num=args.nums
    )
    print(evaluator.config)

    if args.cache_scene_pts:
        evaluator.cache_scene_pts(
            cache_dir=args.cache_dir, feat_comb=args.feat_comb, debug=args.debug
        )
    elif args.scale_pose:
        evaluator.eval_on_scaled_poses(pose_scale=args.scale_pose, debug=args.debug)
    else:
        evaluator.eval_data_loader(
            None, save_depth=args.save_depth, cache_dir=args.cache_dir, debug=args.debug
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--scene_anno_path", type=str, default=None)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--feat_comb", type=str, default="lin")
    parser.add_argument("--img_wh", type=int, nargs="*", default=[480, 480])
    parser.add_argument("--downsample", type=int, default=1)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--scale_pose", type=float, default=None)
    parser.add_argument("--cache_scene_pts", action="store_true")
    parser.add_argument("--save_depth", action="store_true")
    parser.add_argument("--mask", action="store_true")
    parser.add_argument("--nums", type=int, default=-1)
    parser.add_argument("--stop_layer", type=int, default=3)
    args = parser.parse_args()

    if not args.dataset:
        main(args)
    else:
        ckpt = args.ckpt
        cache_dir = args.cache_dir
        for scene in SCENES[args.dataset]:
            args.ckpt = ckpt.replace("#scene", scene)
            args.cache_dir = cache_dir.replace("#scene", scene)
            print(f"{scene}: ckpt={args.ckpt} cache_dir={args.cache_dir}")
            if os.path.exists(args.ckpt):
                main(args)
