# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

from collections import defaultdict
from argparse import Namespace
import argparse
from pathlib import Path
import numpy as np

from nerfmatch.nerfmatch_evaluator import load_nerfmatch_from_ckpt
from nerfmatch.utils.metrics import (
    summarize_pose_statis,
    average_pose_metrics,
    POSE_THRES,
)


def merge_scene_metrics(
    cache_root,
    scenes,
    conf="rth10test_coarse_colmap",
    runs=["v100_16GB_results"],
    feats=None,
    print_out=False,
):
    scores = defaultdict(list)
    if not feats:
        feats = [
            "pt3d",
            "pe3d",
            "layer1",
            "layer2",
            "layer3",
            "layer4",
            "layer5",
            "layer6",
            "layer7",
        ]

    for feat in feats:
        feat_dir = Path(cache_root) / feat
        if not feat_dir.exists():
            print(f"{feat_dir} does not exist!")
            continue

        print(f">>>>>> Feature dir: {feat_dir} conf:{conf}")
        print(f"Scenes: {scenes}")

        # Collect across runs
        run_scs = defaultdict(list)
        for tag in runs:
            cache_dir = feat_dir / tag
            metr_all = []
            for scene in scenes:
                if print_out:
                    print(f"\n>>scenes={scene}")
                cache_path = cache_dir / f"{scene}_{conf}.npy"
                if not cache_path.exists():
                    print(f"{cache_path} doesn't exist!")
                    continue

                metrics = np.load(cache_path, allow_pickle=True).item()
                metr = summarize_pose_statis(
                    metrics,
                    pose_thres=POSE_THRES[scene],
                    t_unit="cm",
                    t_scale=1e2,
                    print_out=print_out,
                )
                metr_all.append(metr)

            if metr_all:
                # For easy put into a table
                table_cell = [
                    "/".join(
                        map(lambda x: f"{x:.1f}", [f["t_med"], f["r_med"], f["recall"]])
                    )
                    for f in metr_all
                ]
                print(table_cell)

                # Average across scenes
                avg = average_pose_metrics(metr_all)

                # Collect across runs
                for k, v in avg.items():
                    run_scs[k].append(v)
                    scores[k].append(v)
    return scores


def eval_ckpt(args):
    nerfmatch_evaluator = load_nerfmatch_from_ckpt(args.ckpt, args, arg_mask=args.mask)
    if not nerfmatch_evaluator.coarse_only:
        nerfmatch_evaluator.coarse_only = args.coarse_only

    data_conf = Namespace()
    if args.pair_topk > 1:
        data_conf = Namespace(
            dataset="NeRFMatchMultiPair",
            sample_mode=args.sample_mode,
            sample_pts=args.sample_pts,
            pair_topk=args.pair_topk,
        )
    if args.scene and "allscenes" in args.ckpt:
        print("### Set scene to : ", args.scene)
        data_conf.scenes = [args.scene]

    if args.scene_anno_path:
        print("### Set scene annotation to : ", args.scene_anno_path)
        data_conf.scene_anno_path = args.scene_anno_path

    inerf_conf = None
    if args.inerf:
        inerf_conf = Namespace(
            num_optim=args.inerf_optim,
            lrate=args.inerf_lr,
            lrdecay=args.inerf_lrd,
            eval_pose=args.inerf_pose,
            ds=args.inerf_ds,
            use_match_loss=args.inerf_match_loss,
        )

    nerfmatch_evaluator.eval_multi_scenes(
        rthres=args.rthres,
        center_subpixel=args.center_subpixel,
        solver=args.solver,
        split=args.split,
        mutual=args.mutual,
        match_thres=args.match_thres,
        iters=args.iters,
        nerf_path=args.nerf_path,
        test_pair_txt=args.test_pair_txt,
        scene_dir=args.scene_dir,
        data_conf=data_conf,
        query2query=args.query2query,
        ow_cache=args.ow_cache,
        inerf_conf=inerf_conf,
        debug=args.debug,
        cached_pt=not args.no_cache_pt,
        cache_dir=args.cache_dir,
        cache_iters=args.cache_iters,
        retrieval_only=args.retrieval_only,
        match_oracle=args.match_oracle,
        visualize=args.visualize,
        seed=args.seed,
    )


def benchmark(args):
    if args.ckpts:
        ckpts = [Path(c) for c in args.ckpts]
    else:
        # Search ckpts
        ckpt_dir = Path(args.ckpt_dir)
        if "allscenes" in str(ckpt_dir):
            model_pattern = f"{args.model_name}.ckpt"
        else:
            model_pattern = f"*_{args.model_name}.ckpt"

        if args.feats:
            ckpts = []
            for k in args.feats:
                ckpts += list(ckpt_dir.glob(f"{k}/{model_pattern}"))
        else:
            ckpts = list(ckpt_dir.glob(f"*/{model_pattern}"))

        # Keep only target scene
        if args.scene:
            ckpts = [c for c in ckpts if args.scene in str(c)]

    ckpt_str = "\n".join([str(i) for i in ckpts])
    print(f"Found the following {len(ckpts)} ckpts:\n{ckpt_str}.")

    cache_tag = ""
    if args.cache_tag:
        cache_tag = f"{args.cache_tag}_"
    if args.model_name != "best":
        cache_tag += f"{args.model_name}_"
    run_seeds = args.seeds
    for ckpt in ckpts:
        cach_root = ckpt.parent
        if run_seeds:
            for i, seed in enumerate(run_seeds):
                print(f"\n>>> Benchmark {ckpt} - Run {i} - Seed {seed}.")
                cache_dir = cach_root / f"{cache_tag}run{i}"

                # Run evaluation per seed
                args.ckpt = str(ckpt)
                args.cache_dir = cache_dir
                args.seed = seed
                eval_ckpt(args)
        else:
            print(f"\n>>> Benchmark {ckpt}.")
            cache_dir = cach_root / f"{cache_tag}results"

            # Run evaluation per seed
            args.ckpt = str(ckpt)
            args.cache_dir = cache_dir
            args.seed = None
            eval_ckpt(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--ckpt_dir", type=str, default=None)
    parser.add_argument("--scene_anno_path", type=str, default=None)
    parser.add_argument("--ckpts", type=str, nargs="*", default=[])
    parser.add_argument("--model_name", type=str, default="best_tmed")
    parser.add_argument("--coarse_only", action="store_true")
    parser.add_argument("--mutual", action="store_true")
    parser.add_argument("--query2query", action="store_true")
    parser.add_argument("--match_thres", type=float, default=0.0)
    parser.add_argument("--ow_cache", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--solver", type=str, default="colmap")
    parser.add_argument("--rthres", type=float, default=10)
    parser.add_argument("--center_subpixel", action="store_true")
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument("--nerf_path", type=str, default=None)
    parser.add_argument("--test_pair_txt", type=str, default=None)
    parser.add_argument("--scene_dir", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--scene", type=str, default=None)
    parser.add_argument("--pair_topk", type=int, default=1)
    parser.add_argument("--sample_pts", type=int, default=-1)
    parser.add_argument("--sample_mode", type=str, default=None)
    parser.add_argument("--mask", type=str, default="default")
    parser.add_argument("--cache_tag", type=str, default=None)
    parser.add_argument("--inerf", action="store_true")
    parser.add_argument("--inerf_optim", type=int, default=5)
    parser.add_argument("--inerf_lr", type=float, default=0.001)
    parser.add_argument("--inerf_lrd", action="store_true")
    parser.add_argument("--inerf_ds", type=int, default=8)
    parser.add_argument("--inerf_pose", action="store_true")
    parser.add_argument("--inerf_match_loss", action="store_true")
    parser.add_argument("--cache_iters", action="store_true")
    parser.add_argument("--no_cache_pt", action="store_true")
    parser.add_argument("--retrieval_only", action="store_true")
    parser.add_argument("--match_oracle", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="*", default=[])
    parser.add_argument("--feats", type=str, nargs="*", default=[])
    args = parser.parse_args()
    benchmark(args)
