# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

from collections import defaultdict
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

import pytorch_lightning as pl

from third_party.loftr.position_encoding import PositionEncodingSine
from third_party.loftr.fine_matching import FinePreprocess, FineMatching

from .modules.attention import SelfAttentionBlock, GenericEncoderLayer
from .modules import init_backbone_8_2
from .modules.extract_matches import extract_mutual_matches
from .nerf.embedding import FourierEmbedding

from .data_loaders import init_data_loader
from .utils import (
    get_logger,
    init_optimizer,
    init_scheduler,
)
from .utils.metrics import (
    compute_pose_metrics,
    compute_fine_pose_metrics,
    compute_feat_l2,
    compute_matching_loss,
    compute_fine_loss_l2_std,
    compute_fine_match_loss_l2_std,
)

logger = get_logger(level="INFO", name="trainer")


def init_pretrained_coarse_matcher(model, ckpt_path):
    if not os.path.exists(ckpt_path):
        return
    ckpt = torch.load(ckpt_path, map_location=torch.device("cpu"))
    new_state = {}
    for k, v in ckpt["state_dict"].items():
        if k.startswith("model."):
            k = k.replace("model.", "")
        if k.startswith("backbone"):
            k = k.replace("backbone", "backbone.model")
        new_state[k] = v
    state = model.load_state_dict(new_state, strict=False)
    print(
        f"Load coarse ckpt from {ckpt_path}: epochs={ckpt['epoch']} step={ckpt['global_step']}\n{state}"
    )


def init_pretrained_matcher(model, ckpt_path):
    if not os.path.exists(ckpt_path):
        return
    ckpt = torch.load(ckpt_path, map_location=torch.device("cpu"))
    new_state = {}
    for k, v in ckpt["state_dict"].items():
        if k.startswith("model."):
            k = k[6:]
        new_state[k] = v
    state = model.load_state_dict(new_state, strict=False)
    print(
        f"Load pretrained ckpt from {ckpt_path}: epochs={ckpt['epoch']} step={ckpt['global_step']}\n{state}"
    )


class NeRFMatcherMS(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Backbone
        self.coarse_ds = 8
        self.fine_ds = 2
        self.backbone = init_backbone_8_2(
            config.backbone,
            pretrained=config.pretrained,
        )
        self.cfeat_dim = getattr(config, "cfeat_dim", 256)
        self.ffeat_dim = getattr(config, "ffeat_dim", 128)
        backbone_dim = self.backbone.feat_dim

        # Image feat projection layers
        self.cfeat_proj, self.ffeat_proj = None, None
        if backbone_dim[0] != self.cfeat_dim:
            self.cfeat_proj = nn.Linear(backbone_dim[0], self.cfeat_dim, bias=True)
        if backbone_dim[1] != self.ffeat_dim:
            self.ffeat_proj = nn.Linear(backbone_dim[1], self.ffeat_dim, bias=True)

        # Temperature
        self.temp_type = getattr(config, "temp_type", "mul")
        if self.temp_type == "div":
            # LoFTR
            self.temperature = nn.parameter.Parameter(
                torch.tensor(0.1), requires_grad=False
            )
            self.apply_temperature = lambda sim_matrix: sim_matrix / self.temperature
        elif self.temp_type == "mul":
            # Aspanformer
            self.temperature = nn.parameter.Parameter(
                torch.tensor(10.0), requires_grad=True
            )
            self.apply_temperature = lambda sim_matrix: sim_matrix * self.temperature

        self.norm_feat = lambda feat: (feat / (feat.norm(dim=-1, keepdim=True) + 1e-6))

        # Image PE
        self.im_pe = None
        if getattr(config, "im_pe", True):
            self.im_pe = PositionEncodingSine(self.cfeat_dim)

        # Point PE
        pt_pe = getattr(config, "pt_pe", True)
        self.post_pt_pe = getattr(config, "post_pt_pe", False)
        self.pt_dim = getattr(config, "pt_dim", self.cfeat_dim)
        self.pt_ftype = getattr(config, "pt_ftype", "nerf")
        self.pt_proj = None

        if self.pt_ftype == "pe3d":
            self.pt_enc = FourierEmbedding(15)
            self.pt_dim = self.pt_enc.get_embedding_dim(3)
        elif self.pt_ftype == "pt3d":
            self.pt_dim = 3
        if self.pt_dim == 3:
            assert self.pt_ftype == "pt3d"

        if self.pt_dim != self.cfeat_dim:
            print(
                f"### pt_dim={self.pt_dim}, cfeat_dim={self.cfeat_dim}, init pt_projection"
            )
            self.pt_proj = nn.Linear(self.pt_dim, self.cfeat_dim, bias=True)

        # Positional encoding
        self.pt_pe_dim = 0
        if pt_pe:
            self.pt_pe_type = getattr(config, "pt_pe_type", "fourier")
            if self.pt_pe_type == "id":
                assert self.post_pt_pe
                self.pt_pe_dim = self.pt_dim
            else:
                self.pt_pe = FourierEmbedding(15)
                self.pt_pe_dim = self.pt_pe.get_embedding_dim(3)
            self.pt_pe_proj = nn.Linear(self.cfeat_dim + self.pt_pe_dim, self.cfeat_dim)
            print(f"### pt_pe_type={self.pt_pe_type} pt_pe_dim={self.pt_pe_dim}.")

        self.pt_ffeat_proj = nn.Sequential(
            *[
                nn.Linear(self.cfeat_dim, self.ffeat_dim),
                nn.Linear(self.ffeat_dim, self.ffeat_dim),
            ]
        )

        # Point sa
        pt_sa_type = getattr(config, "pt_sa_type", "full")
        pt_sa = getattr(config, "pt_sa", 3)
        if pt_sa_type == None or pt_sa == 0:
            self.pt_sa = None
        if pt_sa_type == "full" and pt_sa > 0:
            self.pt_sa = SelfAttentionBlock(
                config.pt_sa,
                self.cfeat_dim,
                att_type="full",
                head_dim=(self.cfeat_dim // 8),
            )

        # Image sa
        im_sa_type = getattr(config, "im_sa_type", None)
        im_sa = getattr(config, "im_sa", 3)
        if im_sa_type == None or im_sa == 0:
            self.im_sa = None
        elif im_sa_type == "share":
            self.im_sa = self.pt_sa
            print(f">>>> Initialize im self-attention as pt self-attention.")
        elif im_sa_type == "full" and im_sa > 0:
            self.im_sa = SelfAttentionBlock(
                im_sa, self.cfeat_dim, att_type="full", head_dim=(self.cfeat_dim // 8)
            )

        # Coarse former
        self.cformer_type = getattr(config, "cformer_type", "crs")
        self.coarse_layers = getattr(config, "coarse_layers", 1)
        self.coarse_former = None
        if self.cformer_type.startswith("crs") and self.coarse_layers > 0:
            self.coarse_former = GenericEncoderLayer(
                model_dim=self.cfeat_dim,
                context_dim=self.cfeat_dim,
                head_dim=self.cfeat_dim // 8,
                att_mode="cross",
                att_type="full",
            )

        # Fine matcher
        self.coarse_percent = getattr(config, "coarse_percent", 0.3)
        self.coarse_dthres = getattr(config, "coarse_dthres", 20)
        self.fine_loss = getattr(config, "fine_loss", "match")
        self.win_sz = int(getattr(config, "win_sz", 5))
        self.cat_c_feat = getattr(config, "cat_c_feat", True)
        self.fine_preprocess = FinePreprocess(
            win_sz=self.win_sz,
            stride=4,
            d_model_f=self.ffeat_dim,
            d_model_c=self.cfeat_dim,
            cat_c_feat=self.cat_c_feat,
        )

        # Fine former
        fsa_type = getattr(config, "fsa_type", "full")
        print(f">>>>Fine sa : {fsa_type}")
        if fsa_type in ["full", "lsa"]:
            self.fine_sa = SelfAttentionBlock(
                config.fine_sa,
                self.ffeat_dim,
                att_type=fsa_type,
                head_dim=(self.ffeat_dim // 8),
            )

        self.fine_matching = FineMatching()

        # Load coarse model if available
        self.coarse_ckpt = getattr(config, "coarse_ckpt", None)
        if self.coarse_ckpt:
            init_pretrained_coarse_matcher(self, self.coarse_ckpt)

        # Load fine model if available
        if getattr(config, "c2f_ckpt", None):
            init_pretrained_matcher(self, config.c2f_ckpt)

    def extract_im_feat(self, img):
        cfeat, ffeat = self.backbone.forward(img)
        b, c, h, w = cfeat.shape
        cfeat = cfeat.flatten(-2).permute(0, 2, 1)
        if self.cfeat_proj is not None:
            cfeat = self.cfeat_proj(cfeat)
        if self.ffeat_proj is not None:
            b, f, hf, wf = ffeat.shape
            ffeat = ffeat.flatten(-2).permute(0, 2, 1)
            ffeat = self.ffeat_proj(ffeat)
            ffeat = rearrange(ffeat, "n (h w) c -> n c h w", h=hf, w=wf)

        if self.im_pe:
            # Only apply PE to coarse feature
            cfeat = rearrange(cfeat, "n (h w) c -> n c h w", h=h, w=w)
            cfeat = rearrange(self.im_pe(cfeat), "n c h w -> n (h w) c")

        if self.im_sa is not None:
            cfeat = self.im_sa(cfeat)
        return cfeat, ffeat

    def cat_pe(self, pt_feat, pt_feat_in, pt3d):
        pt_emb = pt_feat_in if self.pt_pe_type == "id" else self.pt_pe(pt3d)
        pt_feat = self.pt_pe_proj(torch.cat([pt_feat, pt_emb], dim=-1))
        return pt_feat

    def extract_pt_feat(self, pt_feat, pt3d):
        if self.pt_ftype == "pt3d":
            pt_feat = pt3d
        if self.pt_ftype == "rand":
            pt_feat = torch.randn_like(pt_feat)
        elif self.pt_ftype == "pe3d":
            pt_feat = self.pt_enc(pt3d)

        pt_feat_in = pt_feat

        # Projection
        if self.pt_proj is not None:
            pt_feat = self.pt_proj(pt_feat)

        # Pt forward
        if self.pt_pe_dim > 0 and not self.post_pt_pe:
            pt_feat = self.cat_pe(pt_feat, pt_feat_in, pt3d)

        # Self attention on point feature
        if self.pt_sa is not None:
            pt_feat = self.pt_sa(pt_feat)

        if self.pt_pe_dim > 0 and self.post_pt_pe:
            pt_feat = self.cat_pe(pt_feat, pt_feat_in, pt3d)
        return pt_feat

    def coarse_matching(self, im_feat, pt_feat, im_mask=None, pt_mask=None):
        # Normalize
        im_feat, pt_feat = map(self.norm_feat, [im_feat, pt_feat])

        # Dual softmax
        sim_matrix = torch.einsum("bmd,bnd->bmn", im_feat, pt_feat)
        sim_matrix = self.apply_temperature(sim_matrix)
        im_mask_ = torch.ones_like(im_feat[..., 0]) if im_mask is None else im_mask
        pt_mask_ = torch.ones_like(pt_feat[..., 0]) if pt_mask is None else pt_mask
        sim_matrix.masked_fill_(~(im_mask_[..., None] * pt_mask_[:, None]).bool(), -1e9)
        conf_matrix = F.softmax(sim_matrix, 1) * F.softmax(sim_matrix, 2)
        return conf_matrix, im_feat, pt_feat

    def forward_match(
        self,
        img,
        pt_feat,
        pt3d,
        im_mask=None,
        pt_mask=None,
        conf_gt=None,
        ret_feats=False,
        mutual=False,
        match_thres=0.0,
    ):
        # Feature extraction
        im_cfeat, im_ffeat = self.extract_im_feat(img)
        pt_cfeat = self.extract_pt_feat(pt_feat, pt3d)

        # Coarse matching
        if self.coarse_former:
            if self.cformer_type == "crs":
                im_cfeat = self.coarse_former(im_cfeat, pt_cfeat)
                pt_cfeat = self.coarse_former(pt_cfeat, im_cfeat)
            elif self.cformer_type == "crsv2":
                im_cfeat, pt_cfeat = self.coarse_former(
                    im_cfeat, pt_cfeat
                ), self.coarse_former(pt_cfeat, im_cfeat)
            else:
                im_cfeat, pt_cfeat = self.coarse_former(im_cfeat, pt_cfeat)

        conf_matrix, im_cfeat_, pt_cfeat_ = self.coarse_matching(
            im_cfeat, pt_cfeat, im_mask=im_mask, pt_mask=pt_mask
        )

        # Extract coarse matches (in image resolution)
        match_ids, mconf, pred_num = extract_mutual_matches(
            conf_matrix,
            mutual=mutual,
            threshold=match_thres,
            conf_gt=conf_gt,
            coarse_percent=self.coarse_percent,
        )

        # Fine matching
        pt_ffeat = pt_cfeat
        pt_ffeat = self.pt_ffeat_proj(pt_ffeat)
        im_ffeat, pt_ffeat = self.fine_preprocess(
            match_ids, im_ffeat, pt_ffeat, im_cfeat
        )
        if im_ffeat.size(0) != 0:
            im_ffeat = self.fine_sa(im_ffeat)
        expec_f = self.fine_matching(pt_ffeat, im_ffeat)

        # Compose outputs
        preds = dict(
            conf_matrix=conf_matrix,
            expec_f=expec_f,
            match_ids=match_ids,
            mconf=mconf,
            pred_mask=mconf != 0,
            pred_num=pred_num,
        )
        if ret_feats:
            preds.update(
                dict(
                    im_cfeat=im_cfeat_,
                    pt_cfeat=pt_cfeat_,
                )
            )
        return preds

    def forward_multi_pair(self, data, mutual=False, match_thres=0.0):
        # Data parsing
        img = data["image"]
        im_mask = data["im_mask"]
        pt3d = data["pt3d"]
        pt_feat = data["pt_feat"]
        pt_mask = data["pt_mask"]
        pt2d = data["pt2d"]

        mpt2d_f = []
        mpt2d_c = []
        mpt3d = []
        m_bids = []
        mconf = []
        for ipt3d, ipt_feat, ipt_mask in zip(
            pt3d.permute(1, 0, 2, 3),
            pt_feat.permute(1, 0, 2, 3),
            pt_mask.permute(1, 0, 2),
        ):
            # Forward pass
            preds = self.forward_match(
                img,
                ipt_feat,
                ipt3d,
                im_mask=im_mask,
                pt_mask=ipt_mask,
                mutual=mutual,
                match_thres=match_thres,
            )
            imconf = preds["mconf"]
            expec_f = preds["expec_f"]
            b_ids, i_ids, j_ids = preds["match_ids"]

            # Compute matches in image resolution
            b_ids, i_ids, j_ids = preds["match_ids"]
            impt2d_c = pt2d[b_ids, i_ids]
            impt3d = ipt3d[b_ids, j_ids]

            # Extract fine matches in image resolution
            impt2d_f = impt2d_c + expec_f[:, :2] * self.win_sz / 2 * self.fine_ds

            # Save pred matches
            mpt2d_c.append(impt2d_c)
            mpt2d_f.append(impt2d_f)
            mpt3d.append(impt3d)
            m_bids.append(b_ids)
            mconf.append(imconf)

        data.update(
            dict(
                mpt2d_f=torch.cat(mpt2d_f),
                mpt2d_c=torch.cat(mpt2d_c),
                mpt3d=torch.cat(mpt3d),
                m_bids=torch.cat(m_bids),
                mconf=torch.cat(mconf),
            )
        )

    def forward(
        self, data, training=False, ret_feats=False, mutual=False, match_thres=0.0
    ):
        # Data parsing
        img = data["image"]
        im_mask = data["im_mask"]
        pt3d = data["pt3d"]
        pt_feat = data["pt_feat"]
        pt_mask = data["pt_mask"]
        pt2d = data["pt2d"]

        # Only for inference
        if len(pt3d.shape) == 4:
            return self.forward_multi_pair(data, mutual=mutual, match_thres=match_thres)

        # Forward pass
        preds = self.forward_match(
            img,
            pt_feat,
            pt3d,
            im_mask=im_mask,
            pt_mask=pt_mask,
            conf_gt=data["conf_gt"] if training else None,
            ret_feats=ret_feats,
            mutual=mutual,
            match_thres=match_thres,
        )
        data.update(preds)
        expec_f = preds["expec_f"]

        # Compute matches in image resolution
        b_ids, i_ids, j_ids = preds["match_ids"]
        mpt2d_c = pt2d[b_ids, i_ids]
        mpt3d = pt3d[b_ids, j_ids]

        # Extract fine matches in image resolution
        mpt2d_f = mpt2d_c + expec_f[:, :2] * self.win_sz / 2 * self.fine_ds

        # Save both aug + true matches
        data.update(
            mpt2d_c_train=mpt2d_c,
            mpt3d_train=mpt3d,
            mpt2d_f_train=mpt2d_f,
        )

        # Save pred matches
        pred_mask = preds["pred_mask"]
        data.update(
            dict(
                m_bids=b_ids[pred_mask],
                mpt2d_c=mpt2d_c[pred_mask],
                mpt2d_f=mpt2d_f[pred_mask],
                mpt3d=mpt3d[pred_mask],
            )
        )

        if "pt2d_proj" in data:
            mpt2d_f_gt = data["pt2d_proj"][b_ids, j_ids]
            data["mpt2d_f_gt_train"] = mpt2d_f_gt
            data["mpt2d_f_gt"] = mpt2d_f_gt[pred_mask]

    def forward_with_metrics(
        self, data, rthres=1, training=False, coarse_only=False, oracle=False
    ):
        # torch.autograd.set_detect_anomaly(True)

        metrics = defaultdict(list)

        # Data parsing
        pt3d = data["pt3d"]
        conf_gt = data["conf_gt"]
        pt2d_proj = data["pt2d_proj"]

        # Forward pass
        self.forward(data, training=training, ret_feats=True)

        # Feature l2
        feat_l2 = compute_feat_l2(data.pop("im_cfeat"), data.pop("pt_cfeat"), conf_gt)
        metrics["feat_l2"] = feat_l2

        # Compute match losses
        conf_matrix = data["conf_matrix"]
        coarse_loss = compute_matching_loss(conf_matrix, conf_gt)

        metrics["coarse_loss"] = coarse_loss
        if len(data["match_ids"][1]) == 0 or coarse_only:
            # Compute coarse pose error
            metrics.update(compute_pose_metrics(data, rthres=rthres, oracle=oracle))
            metrics["loss"] = coarse_loss
            return metrics

        # Compute fine pose error
        mpt2d_f_gt = data["mpt2d_f_gt_train"]
        mpt2d_f = data["mpt2d_f_train"]
        mpt2d_c = data["mpt2d_c_train"]
        expec_f = data["expec_f"]
        metrics.update(
            compute_fine_pose_metrics(
                data,
                rthres=rthres,
                oracle=oracle,
            )
        )
        coarse_dist = (mpt2d_f_gt - mpt2d_c).norm(dim=-1)
        coarse_pos = coarse_dist < self.coarse_dthres
        metrics["coarse_dist"] = coarse_dist.mean()
        metrics["coarse_pos_ratio"] = coarse_pos.float().mean() * 100

        if self.fine_loss == "match":
            fine_loss = compute_fine_match_loss_l2_std(
                mpt2d_f,
                mpt2d_f_gt,
                expec_f[:, 2],
                mask=coarse_pos,
            )
        elif self.fine_loss == "exp":
            radius = self.fine_ds * self.win_sz // 2
            expec_gt = (mpt2d_f_gt - mpt2d_c) / radius
            fine_loss = compute_fine_loss_l2_std(expec_f, expec_gt, training=training)
        metrics["fine_loss"] = fine_loss
        loss = coarse_loss + fine_loss
        metrics["loss"] = loss
        return metrics


class NeRFMatchMSTrainer(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)

        # Init model
        model_conf = config.model
        self.model = NeRFMatcherMS(model_conf)
        self.win_sz = self.model.win_sz
        self.rthres = getattr(model_conf, "rthres", 1)
        self.gpu_num = getattr(config, "gpu_num", 1)
        self.coarse_only_epochs = config.optim.coarse_only_epochs

    def configure_optimizers(self):
        params = self.model.parameters()
        config = self.hparams.optim
        self.optimizer = init_optimizer(config, params)
        if config.lr_scheduler is not None:
            scheduler = init_scheduler(config, self.optimizer)
            print(self.optimizer, scheduler)
            return [self.optimizer], [scheduler]
        return [self.optimizer]

    def model_forward(self, batch, training=False, oracle=False):
        # Mode pred
        coarse_only = self.current_epoch < self.coarse_only_epochs
        metrics = self.model.forward_with_metrics(
            batch,
            rthres=self.rthres,
            training=training,
            coarse_only=coarse_only,
            oracle=oracle,
        )
        return metrics

    def log_step(self, metrics, training=True):
        prefix = "train" if training else "val"

        # Log scalars
        for k, v in metrics.items():
            if isinstance(v, list):
                v_ = torch.FloatTensor(v)

                # Take mean by ignoring the invalid ones
                mask = ~(v_.isinf() | v_.isnan())
                v_ = v_[mask].mean() if mask.sum() > 0 else torch.tensor(float("inf"))
            elif isinstance(v, torch.Tensor):
                v_ = v.item()
            else:
                v_ = v_
            self.log(f"{prefix}/neum_{k}", v_, on_epoch=True, sync_dist=True)

    def training_step(self, data, batch_idx):
        metrics = self.model_forward(data, training=True)

        # Logging
        self.log_step(metrics, training=True)
        return metrics["loss"]

    def validation_step(self, data, batch_idx):
        metrics = self.model_forward(data, training=False)

        # Logging
        self.log_step(metrics, training=False)
        return metrics

    def validation_epoch_end(self, outputs):
        if self.gpu_num > 1:
            # gather results from all processes
            all_outputs = [None for _ in range(self.gpu_num)]
            torch.distributed.all_gather_object(all_outputs, outputs)
            all_outputs = [
                output for proc_outputs in all_outputs for output in proc_outputs
            ]
            outputs = all_outputs

        log_dict = {}

        # Gather pose errs
        r_errs = []
        t_errs = []
        for odict in outputs:
            r_errs += odict["R_err"]
            t_errs += odict["t_err"]
        r_errs = torch.FloatTensor(r_errs)
        t_errs = torch.FloatTensor(t_errs)

        # Compute median pose
        log_dict["hp/neum_tmed"] = t_errs.median()
        log_dict["hp/neum_Rmed"] = r_errs.median()
        log_dict["hp/neum_tmean"] = t_errs[~t_errs.isinf()].mean()
        log_dict["hp/neum_Rmean"] = r_errs[~r_errs.isinf()].mean()

        # Log dict
        for k, v in log_dict.items():
            self.log(k, v, sync_dist=True)


def parse_optim_tag(config):
    tag = f"{config.optimizer}"
    if config.weight_decay > 0:
        tag += f"wd{config.weight_decay}"
    if config.lr_scheduler == "steplr":
        if getattr(config, "decay_per_step", None):
            tag += f"sp{config.decay_per_step}-{config.decay_gamma}"
        elif getattr(config, "decay_step", None):
            tag += f"sp{'-'.join(map(str, config.decay_step))}-{config.decay_gamma}"
    if config.lr_scheduler == "cosine":
        tag += "cosine"
    return tag


def config_adaptive_lr(config):
    gpu_num = config.gpu_num
    true_batch = gpu_num * config.exp.batch_size
    true_lr = config.optim.clr * true_batch / config.optim.cbs
    print(f"True batch: {true_batch} lr: {true_lr} ")
    return true_lr, true_batch


def init_config_odir(config):
    # Parse experiment name from config
    data = config.data

    if "datasets" in data:
        data_tag = f"wh{data.img_wh[0]}-{data.img_wh[1]}"
    else:
        if "scene" in data:
            scene = data.scene
        elif "scenes" in data:
            if len(data.scenes) == 1:
                scene = data.scenes[0]
            else:
                scene = "all"
        data_tag = f"{data.dataset}_{scene}_wh{data.img_wh[0]}-{data.img_wh[1]}"
        data_tag += os.path.basename(data.scene_dir)
        if data.dataset == "NeRFMatchPair":
            data_tag += f"_top{data.pair_topk}"
        if data.dataset == "NeRFMatchMultiPair":
            data_tag += f"_top{data.pair_topk}pt{data.sample_pts}"
        if getattr(data, "use_msk", False):
            data_tag += "_msk"

    # Samples per epoch
    if getattr(data, "epoch_sample_num", -1) > 0:
        data_tag += f"ep{data.epoch_sample_num}"
    if getattr(data, "balanced_pair", False):
        data_tag += "_bala"
    if getattr(data, "imagenet_norm", False):
        data_tag += "_imn"
    if getattr(data, "aug_self_pairs", 0) > 0:
        data_tag += "_slfaug"
        if data.aug_self_pairs > 1:
            data_tag += f"{data.aug_self_pairs}"

    # Model tag
    mconf = config.model
    model_tag = f"{mconf.backbone}"
    if mconf.pretrained:
        model_tag += "_pre"

    # PEs
    if not mconf.im_pe and not mconf.pt_pe:
        model_tag += "_nopes"
    else:
        if mconf.im_pe:
            model_tag += "_imp"
        if mconf.pt_pe:
            model_tag += "_ptp"
            if getattr(mconf, "pt_pe_type", "fourier") != "fourier":
                model_tag += f"_{mconf.pt_pe_type}"

    # Point sa
    if getattr(mconf, "pt_sa", 0):
        model_tag += f"_pt{mconf.pt_sa_type}{mconf.pt_sa}"
    if getattr(mconf, "post_pt_pe", False) and mconf.pt_pe:
        model_tag += f"_pepos"
    # point feat type
    if getattr(mconf, "pt_ftype", "nerf") != "nerf":
        model_tag += f"_{mconf.pt_ftype}"

    # Img sa
    if getattr(mconf, "im_sa_type", None):
        if mconf.im_sa_type == "share" and mconf.im_sa > 0:
            model_tag += "_imsa_share"
        elif mconf.im_sa > 0:
            model_tag += f"_im{mconf.im_sa_type}{mconf.im_sa}"

    # Matcher conf
    model_tag += "_cf"
    if mconf.coarse_layers > 0:
        model_tag += f"{mconf.cformer_type}{mconf.coarse_layers}"
    model_tag += f"d{mconf.cfeat_dim}"
    model_tag += f"_{mconf.temp_type}tmp"
    model_tag += f"_fsa{mconf.fsa_type}{mconf.fine_sa}d{mconf.ffeat_dim}w{mconf.win_sz}"
    if getattr(mconf, "cat_c_feat", True):
        model_tag += "_catc"

    # Supervision
    model_tag += f"_{mconf.fine_loss}{mconf.coarse_percent}d{mconf.coarse_dthres}"

    # Coarse pretrained
    if getattr(mconf, "coarse_ckpt", None):
        mconf.coarse_ckpt = mconf.coarse_ckpt.replace("#scene", scene)
        if os.path.exists(mconf.coarse_ckpt):
            ckpt_tag = os.path.basename(mconf.coarse_ckpt).replace(".ckpt", "")
            model_tag = model_tag + f"/ft_coarse_{ckpt_tag}"

    # C2f pretrained
    if getattr(mconf, "c2f_ckpt", None) and os.path.exists(mconf.c2f_ckpt):
        model_tag = "ft_" + model_tag

    # Exp
    exp = config.exp
    config.optim.max_epochs = exp.max_epochs
    exp.name = ""
    if getattr(config, "prefix", False):
        exp.prefix = config.prefix
    if getattr(exp, "debug", False):
        exp.prefix = "debug"
    if exp.prefix:
        exp.name += f"{exp.prefix}/"

    # Parse optim
    if getattr(config.optim, "adapt_lr", True):
        true_lr, true_batch = config_adaptive_lr(config)
        config.optim.lr = true_lr
        batch_tag = f"g{config.gpu_num}clr{config.optim.clr}cbs{config.optim.cbs}"
    else:
        config.optim.lr = config.optim.clr
        batch_tag = f"gpu{config.gpu_num}lr{config.optim.lr}batch{exp.batch_size}"
    exp.name += f"{data_tag}/{model_tag}/{batch_tag}{parse_optim_tag(config.optim)}_ep{config.exp.max_epochs}"
    if getattr(config.optim, "coarse_only_epochs", 0) > 0:
        exp.name += f"coarse{config.optim.coarse_only_epochs}"

    exp.resume_version = getattr(exp, "resume_version", "version_0")
    exp.odir = Path(exp.odir)


def train(config):
    gpu_num = torch.cuda.device_count() if config.gpus == -1 else len(config.gpus)
    if gpu_num > 1:
        local_rank = int(os.environ["LOCAL_RANK"])

    # Make training determinisitic
    pl.seed_everything(config.exp.seed, workers=True)
    config.gpu_num = gpu_num

    # Update lr based on the batch size
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
    loss_ckpt = pl.callbacks.ModelCheckpoint(
        save_last=True,
        monitor="val/neum_loss",
        mode="min",
        filename="best",
        save_top_k=1,
    )

    tmed_ckpt = pl.callbacks.ModelCheckpoint(
        monitor="hp/neum_tmed",
        filename="best_tmed",
        auto_insert_metric_name=False,
        mode="min",
        verbose=True,
    )

    pl_callbacks = [loss_ckpt, tmed_ckpt]

    last_ckpt = None
    if getattr(exp, "resume_version", None):
        last_ckpt = exp.odir / exp.name / exp.resume_version / "checkpoints/last.ckpt"
        last_ckpt = last_ckpt if last_ckpt.exists() else None
    num_sanity_val_steps = 0 if last_ckpt else -1
    if debug:
        num_sanity_val_steps = 5

    # Multi-gpu training
    strategy = (
        None if gpu_num <= 1 else pl.plugins.DDPPlugin(find_unused_parameters=True)
    )

    # Init Trainer
    logger.info(f"# GPUs={gpu_num} {strategy}\n")
    trainer = pl.Trainer.from_argparse_args(
        config,
        logger=tb_logger,
        deterministic=False,
        max_epochs=exp.max_epochs,
        check_val_every_n_epoch=exp.check_epochs,
        callbacks=pl_callbacks,
        strategy=strategy,
        num_sanity_val_steps=num_sanity_val_steps,
    )

    # Init model
    model = NeRFMatchMSTrainer(config)

    # Load data loader
    train_loader = init_data_loader(
        config.data, exp.num_workers, exp.batch_size, split="train"
    )
    val_loader = init_data_loader(
        config.data,
        exp.num_workers,
        split="val",
        debug=getattr(config.exp, "debug", False),
    )
    tb_logger.experiment.add_text("Dataset/train", str(train_loader.dataset))
    tb_logger.experiment.add_text("Dataset/val", str(val_loader.dataset))

    # Train
    trainer.fit(model, train_loader, val_loader, ckpt_path=last_ckpt)
    return config
