# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

from pathlib import Path
import torch
import numpy as np
import pytorch_lightning as pl

from .data_loaders import init_data_loader
from .nerf.renderer import NerfRenderer
from .utils.metrics import compute_nerf_metrics, compute_nerf_pose_metrics
from .utils import (
    colorize_depth,
    get_logger,
    init_optimizer,
    init_scheduler,
    get_lr,
)

logger = get_logger(level="INFO", name="trainer")


def init_pfeat_mask(img_wh, ds=8, sample_num=1):
    pfeat_mask = torch.zeros(sample_num, *img_wh, 1).bool()
    pfeat_mask[:, ds // 2 :: ds, ds // 2 :: ds] = 1
    print(f"Init pfeat_mask: {pfeat_mask.shape}")
    return pfeat_mask


class NerfTrainer(pl.LightningModule):
    def __init__(self, config, num_frames=300, closest_ind=None):
        super().__init__()
        self.save_hyperparameters(config)
        self.gpu_num = config.gpu_num
        self.closest_ind = closest_ind

        # Init model
        sample_num = 2 if getattr(config.data, f"train_pair_txt", None) else 1
        self.model = NerfRenderer(config, num_frames)
        self.model.pfeat_mask = init_pfeat_mask(
            config.data.img_wh, ds=8, sample_num=sample_num
        )

        # Loss settings
        self.mask_loss = False
        self.cnfg_loss = getattr(config, "loss", None)
        if self.cnfg_loss:
            self.mask_loss = getattr(self.cnfg_loss, "use_sem_mask", False)

    def configure_optimizers(self):
        params = self.model.parameters()
        config = self.hparams.optim
        self.optimizer = init_optimizer(config, params)
        if config.lr_scheduler is not None:
            scheduler = init_scheduler(config, self.optimizer)
            print(self.optimizer, scheduler)
            return [self.optimizer], [scheduler]
        return [self.optimizer]

    def compute_loss_and_metrics(
        self, preds, rgb_gt, masks=None, validation_mode=False
    ):
        metrics = compute_nerf_metrics(
            preds,
            rgb_gt,
            mask_loss=masks,
            validation_mode=validation_mode,
            cnfg_loss=self.cnfg_loss,
        )
        loss = metrics["loss"]
        return metrics, loss

    def log_step(self, preds, data, metrics, training=True, debug=False):
        prefix = "train" if training else "val"
        log_dict = {}

        # Log scalars
        for k, v in metrics.items():
            self.log(f"{prefix}/{k}", v.item(), sync_dist=True)

        if training:
            self.log("lr", get_lr(self.optimizer), sync_dist=True)

        # Log images (only for val)
        if not training:
            # image: gt / coarse / fine
            # depth: coarse / fine
            nsample = len(data["img_idx"])
            w, h = data["img_wh"][0][:2].cpu()
            rgb_gt = np.split(data["rgbs"].squeeze(), nsample, axis=0)[0]
            log_dict[f"{prefix}/rgb_gt"] = (
                rgb_gt.reshape(h, w, -1).permute(2, 0, 1).cpu()
            )
            for k, v in preds.items():
                if v is None:
                    continue

                # Split the query image
                v = np.split(v, nsample, axis=0)[0]

                # Add images
                if "rgb" in k:
                    v_ = v.detach().reshape(h, w, -1).permute(2, 0, 1).cpu()
                    log_dict[f"{prefix}/{k}"] = v_

                if "sem" in k:
                    v_ = (
                        self.color_map[v.detach().cpu().argmax(-1)]
                        .reshape(h, w, -1)
                        .permute(2, 0, 1)
                    )
                    log_dict[f"{prefix}/{k}"] = v_

                # Add depth
                if "depth" in k:
                    div = 1
                    v_ = colorize_depth(v.detach().reshape(h // div, w // div, -1))
                    log_dict[f"{prefix}/{k}"] = v_

            if nsample > 1:
                # Compute camera pose metrics
                pt_mask = self.model.pfeat_mask[0, ..., 0]
                pts_feat = preds["feat_fine"]
                pose_metrics = compute_nerf_pose_metrics(
                    preds["pts_fine"], pt_mask, pts_feat, data
                )
                for k, v in pose_metrics.items():
                    self.log(f"{prefix}/{k}", v, sync_dist=True)

                if debug:
                    print(f"Pose metrics: ", pose_metrics)

        return log_dict

    def training_step(self, data, batch_idx):
        self.model.ret_pfeat = False
        self.model.set_training_mode(True)
        preds = self.model.forward(
            data["rays"].squeeze(),
            step=self.global_step,
            ray_id=data["ts"].squeeze() if "ts" in data.keys() else None,
        )

        metrics, loss = self.compute_loss_and_metrics(
            preds,
            data["rgbs"].squeeze(),
            masks=data["mask"] if self.mask_loss else None,
        )

        # Logging
        if self.global_step % getattr(self.hparams.exp, "log_step", 100):
            self.log_step(preds, data, metrics, training=True)
        return loss

    def validation_step(self, data, batch_idx, debug=False):
        self.model.ret_pfeat = True
        self.model.set_training_mode(False)
        for k in range(len(data["seq_ind"])):
            ray_id_part = (
                torch.zeros((data["rays"].shape[1] // len(data["seq_ind"])))
                .long()
                .to(data["rays"].device)
                + data["seq_ind"][k]
            )
            ray_id = torch.cat((ray_id, ray_id_part), dim=0) if k > 0 else ray_id_part
        preds = self.model.forward(
            data["rays"].squeeze(), ray_id=ray_id, validation=True
        )

        metrics, loss = self.compute_loss_and_metrics(
            preds,
            data["rgbs"].squeeze(),
            masks=data["mask"] if self.mask_loss else None,
            validation_mode=True,
        )
        if debug:
            print(preds.keys())
            print(metrics, loss)
            return preds, data, metrics

        # Logging
        log_dict = self.log_step(preds, data, metrics, training=False)
        return log_dict

    def validation_epoch_end(self, outputs):
        if self.gpu_num > 1:
            # gather results from all processes
            all_outputs = [None for _ in range(self.gpu_num)]
            torch.distributed.all_gather_object(all_outputs, outputs)
            all_outputs = [
                output for proc_outputs in all_outputs for output in proc_outputs
            ]
            outputs = all_outputs

        num_max = getattr(self.hparams.exp, "log_num_max", 5)
        keys = outputs[0].keys()
        num_batches = len(outputs)
        idx = torch.arange(num_batches)
        skip = max(1, num_batches // num_max)
        idx = idx[::skip][:num_max]
        for k in keys:
            k_list = [outputs[i][k] for i in idx]
            img_stack = torch.stack(k_list)
            self.logger.experiment.add_images(k, img_stack, self.global_step)


def parse_optim_tag(config):
    tag = f"{config.optimizer}{config.lr}"
    if config.weight_decay > 0:
        tag += f"wd{config.weight_decay}"
    if config.lr_scheduler == "steplr":
        if getattr(config, "decay_per_step", None):
            tag += f"sp{config.decay_per_step}-{config.decay_gamma}"
        elif getattr(config, "decay_step", None):
            tag += f"sp{'-'.join(map(str, config.decay_step))}-{config.decay_gamma}"
    if config.lr_scheduler == "cosine":
        tag += "cosine" + str(config.max_epochs)
    if config.lr_scheduler == "chained":
        tag += "chainep" + str(config.max_epochs)
    return tag


def init_config_odir(config):
    # Data tag
    data = config.data
    data_tag = f"{data.dataset}_{data.scene}"
    if "scene_seq" in data and data.scene_seq is not None:
        data_tag += f"-{data.scene_seq}"
    data_tag += f"_wh{data.img_wh[0]}-{data.img_wh[1]}"
    if "focal" in data:
        data_tag += f"f{data.focal}"

    # Scene normalization
    if getattr(data, "normalize_scene", False):
        snorm_type = getattr(data, "snorm_type", "fst")
        if snorm_type == "fst":
            data_tag += f"snfst_dep{data.max_frustum_depth}rs{data.rescale_factor}"

    if getattr(data, "far", False):
        data_tag += f"far{data.far}"

    if getattr(data, "max_sample_num", None):
        data_tag += f"nmax{data.max_sample_num}"
    if "train_skip" in data:
        data_tag += f"skip{data.train_skip}"
    if "ray_num" in data:
        data_tag += f"ray{data.ray_num}"
        if data.random_rays:
            data_tag += "random"
        if data.norm_ray_dir:
            data_tag += "_nrd"

    if getattr(data, "mask_transient", False):
        data_tag += "_trans"
    if getattr(data, "out_scr", False):
        data_tag += f"_scr{data.out_scr}"

    # Model tag
    model_tag = f"{config.fine_nerf.method}"
    if "fine_nerf" in config:
        model_tag += "_c2f"
    if getattr(config.embedding, "appearance_embed", False):
        model_tag += "_app"
    if getattr(config.embedding, "mip_var_scale", -1) > -1:
        model_tag += f"_mvar{config.embedding.mip_var_scale}"

    # Loss
    if getattr(config, "loss", None):
        if getattr(config.loss, "use_sem_mask", False):
            model_tag += "_lsem"
        if getattr(config.loss, "ray_reg_weight", 0):
            model_tag += f"_lreg{config.loss.ray_reg_weight}"
    exp = config.exp
    config.optim.max_epochs = exp.max_epochs
    exp.name = ""
    if getattr(config, "prefix", False):
        exp.prefix = config.prefix
    if getattr(exp, "debug", False):
        exp.prefix = "debug"
        # To fasten the data loading
        config.data.max_sample_num = 10
    if exp.prefix:
        exp.name += f"{exp.prefix}/"
    exp.name += f"{data_tag}/{model_tag}/gpu{config.gpu_num}batch{exp.batch_size}{parse_optim_tag(config.optim)}"
    exp.resume_version = getattr(exp, "resume_version", "version_0")
    exp.odir = Path(exp.odir)


def find_closest(val_split, trn_split):
    k = 0
    close_split = []
    for i, _ in enumerate(trn_split[:-1]):
        if abs(trn_split[i] - val_split[k]) < abs(trn_split[i + 1] - val_split[k]):
            close_split.append(trn_split[i])
            k += 1
            if k >= len(val_split):
                break
    close_split = close_split + list(val_split[k:])
    return close_split


def train(config):
    gpu_num = torch.cuda.device_count() if config.gpus == -1 else len(config.gpus)
    torch.backends.cudnn.benchmark = True

    major, minor = torch.cuda.get_device_capability()
    system_compute_capability = major * 10 + minor
    print("system_compute_capability", system_compute_capability)

    # Make training determinisitic
    pl.seed_everything(config.exp.seed, workers=True)
    config.gpu_num = gpu_num
    init_config_odir(config)
    logger.info(config)
    exp = config.exp
    debug = getattr(exp, "debug", False)

    # Tensorbpard Logger
    tb_logger = pl.loggers.TensorBoardLogger(
        exp.odir,
        name=exp.name,
        version=exp.resume_version,
        default_hp_metric=False,
    )
    tb_logger.experiment.add_text("Exp config", str(config))

    # Checkpoint
    metric = "val/rgb_fine_psnr"
    mode = "max"
    if getattr(config.data, "train_pair_txt", False):
        metric = "val/t_err_match"
        mode = "min"

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_last=True,
        monitor=metric,
        mode=mode,
        filename="best",
        save_top_k=3,
    )

    last_ckpt = None
    if getattr(exp, "resume_version", None):
        last_ckpt = exp.odir / exp.name / exp.resume_version / "checkpoints/last.ckpt"
        last_ckpt = last_ckpt if last_ckpt.exists() else None
    num_sanity_val_steps = 0 if last_ckpt else -1
    if debug:
        num_sanity_val_steps = 1

    # Multi-gpu training
    find_unused_parameters = False
    if pl.__version__ < "1.6":
        strategy = pl.plugins.DDPPlugin(find_unused_parameters=find_unused_parameters)
    else:
        strategy = pl.strategies.ddp.DDPStrategy(
            find_unused_parameters=find_unused_parameters
        )

    # Init Trainer
    logger.info(f"# GPUs={gpu_num} {strategy}\n")
    trainer = pl.Trainer.from_argparse_args(
        config,
        logger=tb_logger,
        deterministic=False,
        max_epochs=exp.max_epochs,
        check_val_every_n_epoch=exp.check_epochs,
        callbacks=[checkpoint_callback],
        strategy=strategy,
        num_sanity_val_steps=num_sanity_val_steps,
    )

    # Load data loader
    train_loader = init_data_loader(
        config.data, exp.num_workers, exp.batch_size, split="train"
    )
    val_loader = init_data_loader(config.data, split="val", debug=debug)
    tb_logger.experiment.add_text("Dataset/train", str(train_loader.dataset))
    tb_logger.experiment.add_text("Dataset/val", str(val_loader.dataset))

    # Init model
    num_frames = train_loader.dataset.dataset_size
    closest_ind = (
        find_closest(val_loader.dataset.split_inds, train_loader.dataset.split_inds)
        if getattr(config.embedding, "appearance_embed", False)
        and getattr(config.embedding, "use_close_val", True)
        else list(val_loader.dataset.split_inds)
    )
    model = NerfTrainer(config, num_frames, closest_ind)

    # Train
    trainer.fit(model, train_loader, val_loader, ckpt_path=last_ckpt)
    return config
