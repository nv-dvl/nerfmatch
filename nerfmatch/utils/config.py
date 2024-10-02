# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

from argparse import Namespace
import yaml
import random
import numpy as np
from pathlib import Path


def config2pretty_str(config):
    return (
        str(config)
        .replace("{", "{\n")
        .replace("},", "\n}\n")
        .replace("(", "(\n")
        .replace(")", "\n)")
    )


def dict2namespace(data_dict):
    """Recursively convert dicts into namespaces."""

    data_ns = Namespace(**data_dict)
    for k, v in data_ns.__dict__.items():
        if isinstance(v, dict):
            data_ns.__dict__[k] = dict2namespace(v)
    return data_ns


def namespace2dict(data_ns):
    """Recursively convert namespaces into dicts."""

    data_dict = vars(data_ns)
    for k, v in data_dict.items():
        if isinstance(v, Namespace):
            data_dict[k] = namespace2dict(v)
    return data_dict


def config_as_dict(conf):
    if isinstance(conf, dict):
        return conf
    if isinstance(conf, Namespace):
        return vars(conf)


def config_as_namespace(conf):
    if isinstance(conf, Namespace):
        return conf
    if isinstance(conf, dict):
        return Namespace(**conf)


def merge_configs(old_conf, new_conf):
    merged_conf = {**config_as_dict(old_conf), **config_as_dict(new_conf)}
    return Namespace(**merged_conf)


def update_configs(old_conf, new_conf):
    old_conf = dict(config_as_dict(old_conf))
    new_conf = dict(config_as_dict(new_conf))
    for k in old_conf:
        if k in new_conf:
            old_conf[k] = new_conf[k]
    return Namespace(**old_conf)


def load_yaml_config(cfg_path):
    cfg_path = Path(cfg_path)
    with open(cfg_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Customized to inherit default setting
    if "inherit" in config:
        parent_cfg_path = cfg_path.parent / config["inherit"]["path"]
        print(parent_cfg_path, config["inherit"])
        with open(parent_cfg_path) as f:
            parent = yaml.load(f, Loader=yaml.FullLoader)
        if "key" in config["inherit"]:
            inherit_key = config["inherit"]["key"]
            parent = parent[inherit_key]
        config.pop("inherit")
        config = dict(**parent, **config)
    return dict2namespace(config), config


def save_config(cfg_path, cfg_dict):
    with open(cfg_path, "w") as f:
        yaml.dump(cfg_dict, f)
