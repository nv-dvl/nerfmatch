# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.


import logging
from .config import load_yaml_config, save_config, update_configs, merge_configs
from .optim import init_optimizer, init_scheduler, get_lr
from .images import *


def data_to_device(data, device):
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            data[k] = v.to(device)


def get_model_size(model):
    learnable_size = 0
    param_size = 0

    for param in model.parameters():
        psize = param.numel() * param.element_size()
        param_size += psize
        if param.requires_grad:
            learnable_size += psize

    param_size_mb = param_size / 1024**2
    learn_size_mb = learnable_size / 1024**2
    print(f"model params: {param_size_mb:.3f}MB learnable: {learn_size_mb:.3f} MB")
    return param_size_mb


def get_logger(level="INFO", log_path=None, name=None):
    logger = logging.getLogger(name)

    if not logger.hasHandlers():
        # Initialize the logger
        level = logging.__dict__[level]
        logger.setLevel(level)
        formatter = logging.Formatter(
            "[%(asctime)s|%(name)s|%(levelname)s]: %(message)s", "%Y-%m-%d %H:%M:%S"
        )

        # Add console handler
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        if log_path:
            # Add log file handler
            fh = logging.FileHandler(log_path)
            fh.setLevel(level)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
    return logger
