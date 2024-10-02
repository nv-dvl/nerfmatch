# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

from argparse import Namespace
from torch.utils.data import ConcatDataset, DataLoader

import nerfmatch.datasets as datasets
from .utils import get_logger, merge_configs

logger = get_logger(level="INFO", name="loader")


def init_mixed_dataset(config, split="train", concat=True, debug=False):
    mixed_datasets = []
    for dt_name, dt_config in config.datasets.__dict__.items():
        dataset_config = merge_configs(config, dt_config)
        mixed_datasets += init_multiscene_dataset(
            dataset_config, split=split, concat=False, debug=debug
        )
    if not concat:
        return mixed_datasets
    dataset = ConcatDataset(mixed_datasets)
    print(f">>> Concated Dataset: split={split} samples={len(dataset)}.")
    return dataset


def init_multiscene_dataset(config, split="train", concat=True, debug=False):
    ms_datasets = []
    for scene in config.scenes:
        sconf = {"scene": scene}
        for k, v in vars(config).items():
            if k == "scenes":
                continue
            if k in ["scene_dir", "train_pair_txt", "test_pair_txt"]:
                if "#" in v:
                    sconf[k] = v.replace("#scene", scene)
                else:
                    sconf[k] = v
            else:
                sconf[k] = v

        sdata = getattr(datasets, config.dataset)(
            Namespace(**sconf), split=split, debug=debug
        )
        print(sdata)
        ms_datasets.append(sdata)
    if not concat:
        return ms_datasets
    dataset = ConcatDataset(ms_datasets)
    print(f">>> Concated Dataset: split={split} samples={len(dataset)}.")
    return dataset


def init_data_loader(config, num_workers=1, batch_size=1, split="train", debug=False):
    # Initialize dataset
    if "datasets" in config:
        dataset = init_mixed_dataset(config, split=split, debug=debug)
    elif "scenes" in config:
        dataset = init_multiscene_dataset(config, split=split, debug=debug)
    else:
        dataset = getattr(datasets, config.dataset)(config, split=split, debug=debug)
    if split == "train":
        data_loader = DataLoader(
            dataset,
            shuffle=True,
            num_workers=num_workers,
            batch_size=batch_size,
            pin_memory=True,
        )
    else:
        data_loader = DataLoader(
            dataset,
            shuffle=False,
            num_workers=num_workers,
            batch_size=1,
            pin_memory=True,
        )
    logger.info(f"\nLoading:\n{dataset}")
    return data_loader
