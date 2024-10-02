# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import argparse
import os

from nerfmatch.nerf_trainer import train
from nerfmatch.utils import load_yaml_config, merge_configs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=int, nargs="*", default=-1)
    parser.add_argument("--scene", type=str, default=None)
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--prefix", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to (.yml) config file."
    )

    args = parser.parse_args()
    config, _ = load_yaml_config(args.config)
    config = merge_configs(config, args)
    if args.scene is not None:
        config.data.scene = args.scene
    if args.max_epochs is not None:
        config.exp.max_epochs = args.max_epochs
    if args.batch_size is not None:
        config.exp.batch_size = args.batch_size

    if args.debug:
        config.exp.debug = True
        config.exp.prefix = "debug"

    train(config)


if __name__ == "__main__":
    main()
