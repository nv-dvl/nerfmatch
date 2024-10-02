# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import argparse
import os

from nerfmatch.nerfmatch_coarse_trainer import train
from nerfmatch.utils import load_yaml_config, merge_configs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=int, nargs="*", default=-1)
    parser.add_argument("--prefix", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to (.yml) config file."
    )

    # Arch
    parser.add_argument("--backbone", type=str, default="convformer384")
    parser.add_argument("--cformer_type", type=str, default="crs")
    parser.add_argument("--coarse_layers", type=int, default=1)
    parser.add_argument("--pt_sa", type=int, default=3)
    parser.add_argument("--im_sa", type=int, default=3)
    parser.add_argument("--pt_dim", type=int, default=256)
    parser.add_argument("--cfeat_dim", type=int, default=256)
    parser.add_argument("--no_pretrain", dest="pretrained", action="store_false")
    parser.add_argument("--post_pt_pe", action="store_true")
    parser.add_argument("--no_pt_pe", dest="pt_pe", action="store_false")
    parser.add_argument("--no_im_pe", dest="im_pe", action="store_false")
    parser.add_argument("--im_sa_type", type=str, default="share")
    parser.add_argument("--pt_sa_type", type=str, default="full")
    parser.add_argument("--pt_ftype", type=str, default="nerf")
    parser.add_argument("--pt_pe_type", type=str, default="fourier")
    parser.add_argument("--temp_type", type=str, default="mul")
    parser.add_argument("--pt_feat_norm", action="store_true")
    parser.add_argument("--finetune", type=str, default=None)

    # Optim
    parser.add_argument("--update_conf", action="store_true")
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--clr", type=float, default=0.0008)
    parser.add_argument("--cbs", type=int, default=16)
    parser.add_argument("--max_epochs", type=int, default=30)

    # Data
    parser.add_argument("--epoch_sample_num", type=int, default=10000)
    parser.add_argument("--pair_topk", type=int, default=20)
    parser.add_argument("--sample_pts", type=int, default=3600)
    parser.add_argument("--aug_self_pairs", type=int, default=0)
    parser.add_argument("--train_pair_txt", type=str, default=None)
    parser.add_argument("--scene_dir", type=str, default=None)
    parser.add_argument("--scenes", type=str, nargs="*", default=None)
    parser.add_argument("--resume_version", type=str, default=None)

    args = parser.parse_args()
    config, _ = load_yaml_config(args.config)
    config = merge_configs(config, args)
    if args.update_conf:
        config.model.backbone = args.backbone
        config.model.pt_dim = args.pt_dim
        config.model.pt_sa = args.pt_sa
        config.model.im_sa = args.im_sa
        config.model.pt_sa_type = args.pt_sa_type
        config.model.coarse_layers = args.coarse_layers
        config.model.cformer_type = args.cformer_type
        config.model.cfeat_dim = args.cfeat_dim
        config.model.pretrained = args.pretrained
        config.model.post_pt_pe = args.post_pt_pe
        config.model.pt_pe = args.pt_pe
        config.model.im_pe = args.im_pe
        config.model.pt_ftype = args.pt_ftype
        config.model.temp_type = args.temp_type
        config.model.pt_feat_norm = args.pt_feat_norm
        config.model.pt_pe_type = args.pt_pe_type
        config.model.finetune = args.finetune

        config.exp.batch_size = args.batch_size
        config.exp.max_epochs = args.max_epochs
        config.optim.clr = args.clr
        config.optim.cbs = args.cbs
        config.data.epoch_sample_num = args.epoch_sample_num
        config.data.pair_topk = args.pair_topk
        config.data.aug_self_pairs = args.aug_self_pairs
        config.data.sample_pts = args.sample_pts
        if args.train_pair_txt:
            config.data.train_pair_txt = args.train_pair_txt

        config.exp.prefix = args.prefix
        if args.scene_dir:
            config.data.scene_dir = args.scene_dir
        if args.scenes:
            config.data.scenes = args.scenes
        if args.resume_version:
            config.exp.resume_version = args.resume_version

    if args.debug:
        config.exp.debug = True
    train(config)


if __name__ == "__main__":
    main()
