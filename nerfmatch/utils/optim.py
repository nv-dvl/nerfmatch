# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

from torch.optim.lr_scheduler import (
    _LRScheduler,
    ReduceLROnPlateau,
    CosineAnnealingLR,
    MultiStepLR,
    ChainedScheduler,
    LinearLR,
)
import torch.optim as optim


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def init_optimizer(config, parameters, eps=1e-8):
    eps = float(getattr(config, "eps", eps))
    if config.optimizer == "sgd":
        optimizer = optim.SGD(
            parameters,
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer == "adam":
        optimizer = optim.Adam(
            parameters, lr=config.lr, eps=eps, weight_decay=config.weight_decay
        )
    elif config.optimizer == "adamw":
        optimizer = optim.AdamW(
            parameters, lr=config.lr, eps=eps, weight_decay=config.weight_decay
        )
    elif config.optimizer == "rmsprop":
        optimizer = optim.RMSprop(
            parameters,
            lr=config.lr,
            eps=eps,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer == "radam":
        optimizer = optim.RAdam(
            parameters, lr=config.lr, eps=eps, weight_decay=config.weight_decay
        )
    elif config.optimizer == "ranger":
        optimizer = optim.Ranger(
            parameters, lr=config.lr, eps=eps, weight_decay=config.weight_decay
        )
    else:
        raise ValueError("optimizer not recognized!")
    return optimizer


def init_scheduler(config, optimizer):
    interval = "epoch"
    if config.lr_scheduler == "steplr":
        if getattr(config, "decay_per_step"):
            step = config.decay_per_step
            milestones = [i for i in range(step, config.max_epochs, step)]
        else:
            milestones = config.decay_step
        scheduler = MultiStepLR(
            optimizer, milestones=milestones, gamma=config.decay_gamma
        )
        print(f"Init MultiStepLR: milestones={milestones} decay={config.decay_gamma}")
    elif config.lr_scheduler == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=config.max_epochs, eta_min=1e-8)
    elif config.lr_scheduler == "poly":
        scheduler = LambdaLR(
            optimizer, lambda epoch: (1 - epoch / config.max_epochs) ** config.poly_exp
        )
    elif config.lr_scheduler == "chained":
        scheduler = ChainedScheduler(
            [
                LinearLR(optimizer, start_factor=0.01, total_iters=100),
                MultiStepLR(
                    optimizer,
                    milestones=[
                        config.max_epochs // 2,
                        config.max_epochs * 3 // 4,
                        config.max_epochs * 9 // 10,
                    ],
                    gamma=0.33,
                ),
            ]
        )
    else:
        raise ValueError("scheduler not recognized!")

    if getattr(config, "warmup_epochs", 0) > 0 and config.optimizer not in [
        "radam",
        "ranger",
    ]:
        scheduler = GradualWarmupScheduler(
            optimizer,
            multiplier=config.warmup_multiplier,
            total_epoch=config.warmup_epochs,
            after_scheduler=scheduler,
        )
    return {"interval": interval, "scheduler": scheduler}


class GradualWarmupScheduler(_LRScheduler):
    """Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.0:
            raise ValueError("multiplier should be greater thant or equal to 1.")
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [
                        base_lr * self.multiplier for base_lr in self.base_lrs
                    ]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        return [
            base_lr
            * ((self.multiplier - 1.0) * self.last_epoch / self.total_epoch + 1.0)
            for base_lr in self.base_lrs
        ]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = (
            epoch if epoch != 0 else 1
        )  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [
                base_lr
                * ((self.multiplier - 1.0) * self.last_epoch / self.total_epoch + 1.0)
                for base_lr in self.base_lrs
            ]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group["lr"] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)
