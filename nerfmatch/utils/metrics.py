# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

from collections import defaultdict
import os
from argparse import Namespace
import numpy as np
import torch
from cv2 import Rodrigues


from .geometry import (
    get_pose,
    estimate_pose,
    estimate_pose_pycolmap,
    project_points3d,
    unnormaliz_pts,
    mutual_nn_matching,
)

# Scene-dependent thresholds following dsac*
POSE_THRES = {
    # Cambridge
    "GreatCourt": [(5, 45)],
    "KingsCollege": [(5, 38)],
    "OldHospital": [(5, 22)],
    "ShopFacade": [(5, 15)],
    "StMarysChurch": [(5, 35)],
    # 7Scenes
    "chess": [(5, 5)],
    "fire": [(5, 5)],
    "heads": [(5, 5)],
    "office": [(5, 5)],
    "pumpkin": [(5, 5)],
    "redkitchen": [(5, 5)],
    "stairs": [(5, 5)],
}


def mse(img_pred, img_gt, mask=None):
    dists = (img_pred - img_gt) ** 2
    dists = dists[mask] if mask is not None else dists
    return torch.mean(dists)


def mse2psnr(mse):
    return -10 * torch.log10(mse)


def psnr(img_pred, img_gt, mask=None):
    return mse2psnr(mse(img_pred, img_gt, mask))


def compute_nerf_metrics(
    preds, rgb_gt, validation_mode=False, mask_loss=None, cnfg_loss=None
):
    metrics = {}
    loss = 0
    if mask_loss is not None:
        if validation_mode:
            mask_loss = torch.round(mask_loss)
    else:
        mask_loss = 1

    if "rgb_coarse" in preds:
        coarse_weight = getattr(cnfg_loss, "coarse_weight", 1.0)
        if "app_coarse" in preds.keys() and not validation_mode:
            loss += l2_regularize(preds["app_coarse"]) * 1e-5
        rgb_coarse_mse = 0.5 * (mask_loss * (preds["rgb_coarse"] - rgb_gt) ** 2).mean()
        loss += rgb_coarse_mse * coarse_weight
        metrics["rgb_coarse_mse"] = rgb_coarse_mse
        metrics["rgb_coarse_psnr"] = mse2psnr(rgb_coarse_mse)

    if "rgb_fine" in preds:
        rgb_fine_mse = 0.5 * (mask_loss * (preds["rgb_fine"] - rgb_gt) ** 2).mean()
        loss += rgb_fine_mse
        metrics["rgb_fine_mse"] = rgb_fine_mse
        metrics["rgb_fine_psnr"] = mse2psnr(rgb_fine_mse)
    else:
        metrics["rgb_fine_mse"] = metrics["rgb_coarse_mse"]
        metrics["rgb_fine_psnr"] = metrics["rgb_coarse_psnr"]

    if not validation_mode:
        ray_reg_weight = getattr(cnfg_loss, "ray_reg_weight", None)
        if "s_fine" in preds and ray_reg_weight:
            loss += (
                distortion_loss(preds["s_fine"], preds["weights_fine"]) * ray_reg_weight
            )

    metrics["loss"] = loss
    return metrics


def compute_nerf_pose_metrics(pts_fine, pt_mask, pts_feat, data, ds=8, ransac_thres=1):
    nsample = len(data["img_idx"])
    w, h = data["img_wh"][0][:2].cpu()

    # Compute camera pose
    c2w_gt1, c2w_gt2 = np.split(data["c2w"].squeeze(), nsample, axis=0)
    K1, K2 = np.split(data["K"].squeeze(), nsample, axis=0)

    # Compute 3D pts in world reference
    unnorm_scene = data["unnorm_scene"].cpu()
    pt3d_1, pt3d_2 = np.split(pts_fine.cpu(), nsample, axis=0)
    pt3d_2 = unnormaliz_pts(pt3d_2.reshape(1, -1, 3), unnorm_scene).squeeze().numpy()
    pt3d_1 = unnormaliz_pts(pt3d_1.reshape(1, -1, 3), unnorm_scene).squeeze().numpy()

    # Part 1. Pose metrics
    # Compute camera pose error for im1
    R_err1, t_err1 = compute_reproj_pose_metrics(
        (w, h),
        K1,
        c2w_gt1.cpu(),
        pt3d_2,
        ds=ds,
        ransac_thres=ransac_thres,
    )

    # Compute camera pose error for im2
    R_err2, t_err2 = compute_reproj_pose_metrics(
        (w, h),
        K2,
        c2w_gt2.cpu(),
        pt3d_1,
        ds=ds,
        ransac_thres=ransac_thres,
    )

    R_err_depth = 0.5 * (R_err1 + R_err2)
    t_err_depth = 0.5 * (t_err1 + t_err2) * 100

    # Part 2. feature metrics
    # Parse feats  & corresponding pts
    pt_mask_ = pt_mask.flatten()
    pfeat_1, pfeat_2 = np.split(pts_feat.cpu(), nsample, axis=0)
    pt3d_2 = pt3d_2[pt_mask_]
    pt3d_1 = pt3d_1[pt_mask_]
    ys, xs = np.where(pt_mask)
    pts2d = np.dstack([xs, ys]).squeeze()

    # Compute mutual matches
    matches, scores = mutual_nn_matching(pfeat_1, pfeat_2)
    match_score = scores.mean()

    # Eval matches via pose estimation
    R_err1, t_err1 = compute_pose_errs(
        K1,
        c2w_gt1.cpu(),
        pt3d_2[matches[:, 1]],
        pts2d[matches[:, 0]],
        ransac_thres=ransac_thres,
    )

    R_err2, t_err2 = compute_pose_errs(
        K2,
        c2w_gt2.cpu(),
        pt3d_1[matches[:, 0]],
        pts2d[matches[:, 1]],
        ransac_thres=ransac_thres,
    )
    R_err_match = 0.5 * (R_err1 + R_err2)
    t_err_match = 0.5 * (t_err1 + t_err2) * 100

    metrics = dict(
        R_err_depth=R_err_depth,
        t_err_depth=t_err_depth,
        R_err_match=R_err_match,
        t_err_match=t_err_match,
        match_score=match_score,
        num_matches=len(matches),
    )
    return metrics


def compute_reproj_pose_metrics(img_wh, K, c2w_gt, pt3d, ds=8, ransac_thres=1):
    w, h = img_wh
    w2c_gt = c2w_gt.inverse()
    # Re-project 3D onto image
    pt2d_proj = project_points3d(
        K.cpu().numpy(), w2c_gt[:3, :3].numpy(), w2c_gt[:3, 3].numpy(), pt3d
    )

    # Subsample 2D-3D correspondences
    pt2d_proj_int = pt2d_proj.astype(np.int32)
    pt2d_samp = pt2d_proj_int.reshape(h, w, 2)[ds // 2 :: ds, ds // 2 :: ds].reshape(
        -1, 2
    )
    pt3d_samp = pt3d.reshape(h, w, 3)[ds // 2 :: ds, ds // 2 :: ds].reshape(-1, 3)

    # Compute camera pose
    R_err, t_err = compute_pose_errs(
        K, c2w_gt, pt3d_samp, pt2d_samp, ransac_thres=ransac_thres
    )
    return R_err, t_err


def compute_pose_errs(K, c2w_gt, pt3d, pt2d, solver="cv", ransac_thres=1):
    # Compute camera pose
    if solver == "colmap":
        pose_res = estimate_pose_pycolmap(pt2d, pt3d, K, ransac_thres=ransac_thres)
    else:
        pose_res = estimate_pose(pt2d, pt3d, K, ransac_thres=ransac_thres)

    if not pose_res:
        R_err, t_err = torch.tensor(torch.inf), torch.tensor(torch.inf)
        inliers = []
    else:
        R, t, inliers = pose_res
        w2c_est = torch.from_numpy(get_pose(R, t))

        # Eval pose err in c2w
        R_err, t_err = pose_err(c2w_gt, w2c_est.inverse())
    return R_err, t_err


def compute_pose_metrics(batch, rthres=1, solver="cv", oracle=False, debug=False):
    metrics = defaultdict(list)

    # Load data
    im_mask = batch["im_mask"].cpu()
    if oracle:
        bid, i2d, i3d = torch.where(batch["conf_gt"].cpu())
    else:
        bid, i2d, i3d = batch["match_ids"]
        bid, i2d, i3d = bid.cpu(), i2d.cpu(), i3d.cpu()
    K = batch["K"].cpu()
    pt2d = batch["pt2d"].cpu()
    pt3d = batch["pt3d"].cpu().reshape(len(K), -1, 3)
    c2w_gt = batch["c2w"].cpu()

    # Compute pose errors
    for i in range(len(K)):
        # Extract matches
        i2d_ = i2d[bid == i]
        i3d_ = i3d[bid == i]
        if solver == "colmap":
            # Pycolmap solver
            pose_res = estimate_pose_pycolmap(
                pt2d[i][i2d_],
                pt3d[i][i3d_],
                K[i],
                ransac_thres=rthres,
            )
        else:
            # Opencv solver
            pose_res = estimate_pose(
                pt2d[i][i2d_],
                pt3d[i][i3d_],
                K[i],
                ransac_thres=rthres,
            )
        if not pose_res:
            R_err, t_err = torch.tensor(torch.inf), torch.tensor(torch.inf)
            inliers = []
        else:
            R, t, inliers = pose_res
            w2c_est = torch.from_numpy(get_pose(R, t))

            # Eval pose err in c2w
            R_err, t_err = pose_err(c2w_gt[i], w2c_est.inverse())
            if debug:
                print(f"Matches: {len(i2d_)} R={R_err:.4f}, t={t_err:.4f}")

        metrics["num_matches"].append(len(i2d_))
        metrics["R_err"].append(R_err)
        metrics["t_err"].append(t_err)
    return metrics


def compute_fine_pose_metrics(
    data,
    rthres=1,
    solver="cv",
    oracle=False,
):
    metrics = defaultdict(list)

    m_bids = data["m_bids"].cpu()
    pt2d = data["mpt2d_f"].detach().cpu()
    pt3d = data["mpt3d"].cpu()
    mconf = data["mconf"].cpu()

    # Load data
    K = data["K"].cpu()
    c2w_gt = data["c2w"].cpu()
    if oracle:
        pt2d = data["mpt2d_f_gt"].cpu()

    # Compute pose errors
    for i in range(len(K)):
        imask = m_bids == i
        if solver == "colmap":
            # Pycolmap solver
            pose_res = estimate_pose_pycolmap(
                pt2d[imask],
                pt3d[imask],
                K[i],
                ransac_thres=rthres,
            )
        else:
            # Opencv solver
            pose_res = estimate_pose(
                pt2d[imask],
                pt3d[imask],
                K[i],
                ransac_thres=rthres,
            )
        if not pose_res:
            R_err, t_err = torch.tensor(torch.inf), torch.tensor(torch.inf)
            inliers = []
        else:
            R, t, inliers = pose_res
            w2c_est = torch.from_numpy(get_pose(R, t))

            # Eval pose err in c2w
            R_err, t_err = pose_err(c2w_gt[i], w2c_est.inverse())

        metrics["num_matches"].append(imask.sum())
        metrics["num_inls"].append(len(inliers))
        metrics["R_err"].append(R_err)
        metrics["t_err"].append(t_err)
    return metrics


def compute_mean_recall(errs, thres):
    sample_rec = []
    for err in errs:
        if isinstance(err, torch.Tensor):
            err = err.cpu().numpy()
        sample_rec.append([(err < th).mean() for th in thres])
    mean_rec = np.array(sample_rec).mean(0) * 100
    return mean_rec


def cal_error_auc(errors, thresholds):
    if len(errors) == 0:
        return np.zeros(len(thresholds))
    N = len(errors)
    errors = np.append([0.0], np.sort(errors))
    recalls = np.arange(N + 1) / N
    aucs = []
    for thres in thresholds:
        last_index = np.searchsorted(errors, thres)
        rcs_ = np.append(recalls[:last_index], recalls[last_index - 1])
        errs_ = np.append(errors[:last_index], thres)
        aucs.append(np.trapz(rcs_, x=errs_) / thres)
    return np.array(aucs) * 100


def pose_recall(r_errs, t_errs, r_thres, t_thres):
    return ((np.array(r_errs) < r_thres) & (np.array(t_errs) < t_thres)).mean() * 100


def pose_err(gt_pose, est_pose):
    # pose in c2w
    t_err = float(torch.norm(gt_pose[0:3, 3] - est_pose[0:3, 3]))

    gt_R = gt_pose[0:3, 0:3].numpy()
    est_R = est_pose[0:3, 0:3].numpy()

    r_err = np.matmul(est_R, np.transpose(gt_R))
    r_err = Rodrigues(r_err)[0]
    r_err = np.linalg.norm(r_err) * 180 / np.pi
    return r_err, t_err


def compute_matching_loss(conf, conf_gt, alpha=0.25, gamma=2.0, clamp=True):
    # To avoid nans in backward pass
    if clamp:
        conf = torch.clamp(conf, 1e-6, 1 - 1e-6)
    pos_mask, neg_mask = conf_gt == 1, conf_gt == 0
    loss_pos = -alpha * torch.pow(1 - conf[pos_mask], gamma) * (conf[pos_mask]).log()
    loss_neg = -alpha * torch.pow(conf[neg_mask], gamma) * (1 - conf[neg_mask]).log()
    loss = loss_pos.mean() + loss_neg.mean()
    return loss


def compute_feat_l2(im_feat, pt_feat, conf_gt):
    feat_l2 = []
    for i, iconf in enumerate(conf_gt):
        im_ids, pt_ids = torch.where(iconf)
        ifeat_l2 = (im_feat[i][im_ids] - pt_feat[i][pt_ids]).norm(dim=-1).mean()
        feat_l2.append(ifeat_l2)
    feat_l2 = torch.stack(feat_l2).mean()
    return feat_l2


def compute_fine_loss_l2_std(expec_f, expec_f_gt, training=True):
    """
    Original version in loftr, compute distance at local level.
    Args:
        expec_f (torch.Tensor): [M, 3] <x, y, std>
        expec_f_gt (torch.Tensor): [M, 2] <x, y>
    """
    correct_mask = torch.linalg.norm(expec_f_gt, ord=float("inf"), dim=1) < 1

    # Use std as weight that measures uncertainty
    std = expec_f[:, 2]
    inverse_std = 1.0 / torch.clamp(std, min=1e-10)
    weight = (inverse_std / torch.mean(inverse_std)).detach()

    # corner case: no correct coarse match found
    if not correct_mask.any():
        # this seldomly happen during training, since we pad prediction with gt
        # sometimes there is not coarse-level gt at all.
        print("assign a false supervision to avoid ddp deadlock")
        correct_mask[0] = True
        weight[0] = 0.0

    # l2 loss with std
    flow_l2 = ((expec_f_gt[correct_mask] - expec_f[correct_mask, :2]) ** 2).sum(-1)
    loss = (flow_l2 * weight[correct_mask]).mean()
    return loss


def compute_fine_match_loss_l2_std(mpt2d_f, mpt2d_f_gt, std, mask=None):
    """Adapted version, directly compute distance at global level.
    Args:
        expec_f (torch.Tensor): [M, 3] <x, y, std>
        expec_f_gt (torch.Tensor): [M, 2] <x, y>
    """
    # Use std as weight that measures uncertainty
    inverse_std = 1.0 / torch.clamp(std, min=1e-10)
    weight = (inverse_std / torch.mean(inverse_std)).detach()

    if mask is None:
        mask = torch.ones_like(weight)

    # corner case: no correct coarse match found
    if mask.sum() == 0:
        # this seldomly happen during training, since we pad prediction with gt
        # sometimes there is not coarse-level gt at all.
        print("assign a false supervision to avoid ddp deadlock")
        mask[0] = True
        weight[0] = 0.0

    # l2 loss with std
    flow_l2 = ((mpt2d_f - mpt2d_f_gt) ** 2).sum(-1)
    loss = (flow_l2 * weight * mask).mean()
    return loss


def distortion_loss(s, w):
    """Computes the distortion loss regularizer defined in mip-NeRF 360."""
    return torch.mean(lossfun_distortion(s, w))


def lossfun_distortion(t, w):
    """Compute iint w[i] w[j] |t[i] - t[j]| di dj."""
    # The loss incurred between all pairs of intervals.
    if w.shape[-1] == t.shape[-1]:
        t = torch.hstack((t[:, :1] * 0, t))
    ut = (t[..., 1:] + t[..., :-1]) / 2
    dut = torch.abs(ut[..., :, None] - ut[..., None, :])
    loss_inter = torch.sum(w * torch.sum(w[..., None, :] * dut, dim=-1), dim=-1)

    # The loss incurred within each individual interval with itself.
    loss_intra = torch.sum(w**2 * (t[..., 1:] - t[..., :-1]), dim=-1) / 3

    return loss_inter + loss_intra


def summarize_refinement_curves(cache_path, scenes):
    t_meds = []
    R_meds = []
    pose_recalls = []
    for scene in scenes:
        r_thres, t_thres = POSE_THRES[scene][0]
        statis = np.load(cache_path.replace("#scene", scene), allow_pickle=True).item()
        t_errs_iter = statis["iter_t_errs"]
        R_errs_iter = statis["iter_R_errs"]

        if isinstance(t_errs_iter, list):
            # Handle cases where iterative caching didn't go until the end
            niter = np.max([len(k) for k in t_errs_iter])
            for idx in np.where(np.isinf(statis["t_err"]))[0]:
                t_errs_iter[idx] += [np.inf] * (niter - len(t_errs_iter[idx]))
                R_errs_iter[idx] += [np.inf] * (niter - len(R_errs_iter[idx]))
            t_errs_iter = np.stack(t_errs_iter)
            R_errs_iter = np.stack(R_errs_iter)

        t_errs_iter = t_errs_iter * 100
        t_errs = np.median(t_errs_iter, axis=0)
        R_errs = np.median(R_errs_iter, axis=0)
        recall = ((R_errs_iter < r_thres) & (t_errs_iter < t_thres)).mean(0) * 100

        t_meds.append(t_errs)
        R_meds.append(R_errs)
        pose_recalls.append(recall)

    t_curve = np.stack(t_meds).mean(0)
    r_curve = np.stack(R_meds).mean(0)
    recall_curve = np.stack(pose_recalls).mean(0)
    return recall_curve, t_curve, r_curve


def summarize_statis(statis, pose_thres=[1, 2, 5, 10], reproj_thres=[1, 5, 10]):
    np.set_printoptions(precision=4)
    if isinstance(statis, str):
        if not os.path.exists(statis):
            return
        print(f"Load statis from {statis}")
        statis = np.load(statis, allow_pickle=True).item()
    if isinstance(statis, dict):
        statis = Namespace(**statis)

    # Compute pose errors
    r_errs, t_errs = statis.r_errs, statis.t_errs
    print(f"Samples: {len(r_errs)}")
    print(f"Median Error: {np.median(r_errs):.1f}deg, {np.median(t_errs):.1f}cm")
    pose_rec = np.array([pose_recall(r_errs, t_errs, th, th) for th in pose_thres])
    print(f"Recall@{pose_thres}cm/deg: {pose_rec}%")
    pose_auc = cal_error_auc(np.maximum(t_errs, r_errs), pose_thres)
    print(f"AUC@{pose_thres}cm/deg: {pose_auc}%")

    # Compute reproj recall
    if "reproj_errs" in statis:
        reproj_rec = compute_mean_recall(statis.reproj_errs, reproj_thres)
        print(f"Reproj<{reproj_thres}px avg recall={reproj_rec}%")

    # Timing
    print(f"Avg. processing time: {np.mean(statis.times) * 1000:.1f}ms\n\n")


def load_pos(cache_path, summarize=False):
    statis = np.load(cache_path, allow_pickle=True).item()
    if summarize:
        summarize_statis(statis)
    gt_pos = np.array([p.numpy()[:3, 3] for p in statis["gt_poses"]])
    est_pos = np.array([p.numpy()[:3, 3] for p in statis["est_poses"]])
    return est_pos, gt_pos


def l2_regularize(mu):
    mu_2 = torch.pow(mu, 2)
    encoding_loss = torch.mean(mu_2)
    return encoding_loss


def summarize_pose_statis(
    statis,
    pose_thres=[1, 2, 5, 10],
    auc_thres=[1, 2, 5, 10],
    t_unit="?",
    t_scale=1,
    print_out=True,
):
    if print_out:
        printf = print
    else:
        printf = lambda x: None

    np.set_printoptions(precision=1)
    if isinstance(statis, dict):
        statis = Namespace(**statis)

    if isinstance(pose_thres[0], int):
        pose_thres = [(th, th) for th in pose_thres]

    r_errs, t_errs = statis.R_err, statis.t_err
    t_errs = t_scale * t_errs

    printf(f"\nSamples: {len(r_errs)} t_unit={t_unit} t_scale={t_scale}")
    if "num_matches" in statis:
        printf(f"Mean matches: {np.mean(statis.num_matches):.0f}")
    if "num_inls" in statis:
        printf(f"Ransac inliers:{np.mean(statis.num_inls):.0f}")

    t_med = np.median(t_errs)
    r_med = np.median(r_errs)
    printf(f"Median Error: {t_med:.1f}/{r_med:.1f} {t_unit}/deg")
    pose_rec = np.array(
        [pose_recall(r_errs, t_errs, rth, tth) for rth, tth in pose_thres]
    )
    printf(f"Recall@{pose_thres}{t_unit}/deg: {pose_rec}%")
    pose_auc = cal_error_auc(np.maximum(t_errs, r_errs), auc_thres)
    printf(f"AUC@{auc_thres}{t_unit}/deg: {pose_auc}%")

    summary_dict = {
        f"t_med": t_med,
        f"r_med": r_med,
        f"recall": pose_rec[0],
    }
    if "match_time" in statis:
        # Timing
        match_time = np.mean(statis.match_time) * 1000
        summary_dict["match_time"] = match_time
        printf(f"Avg match time: {match_time:.1f}ms")

    return summary_dict


def average_pose_metrics(metr_all):
    print(f"\nAverage metrics of {len(metr_all)} (scene) caches:")
    avg = {}
    for k in metr_all[0]:
        avg[k] = np.mean([metr[k] for metr in metr_all])

    print(f"Median pose error(cm/deg): {avg['t_med']:.1f}/{avg['r_med']:.1f}")
    print(f"Recall(%): {avg['recall']:.1f}")
    print(f"Table: {avg['t_med']:.1f}/{avg['r_med']:.1f}/{avg['recall']:.1f}")
    for k, v in avg.items():
        if "time" in k:
            print(f"{k}:{v:.1f} ms")
    print("---------\n")
    return avg
