import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import sys

sys.path.append("../pips/")
import pips_utils.improc


def draw_thumbnails(
    rgbs,
    deps,
    mask_locs,
    trajs,
    trajs_rev=None,
    anchor_idx=None,
    ncol=10,
    keypoints_to_plot=0,
):
    """
    Draw thumbnail for of each keypoint for each frame.

    Inputs:
        rgbs: array of rgb frames (n_timestep, 3, W, H)
        deps: array of dep frames (n_timestep, 3, W, H)
        mask_locs: list of mask indices (from mask R-CNN)
        trajs: array of trajectories (n_timestep, n_keypoints, 2)
        trajs_rev: array of reverse trajectories in order (n_timestep, n_keypoints, 2)
        anchor_idx: index of anchor frame (if applicable, default = None)
        ncol: number of columns in thumbnail grid

    Outputs:
        thumbnails: list of thumbnails for each keypoint
    """
    # Resize rgbs
    # rgbs = np.transpose(rgbs, [0, 2, 3, 1])
    rgbs = rgbs[:, :, :, [2, 1, 0]]
    # deps = np.transpose(deps, [0, 2, 3, 1])
    pad = np.full(
        (50, (20 + rgbs.shape[2]) * ncol, rgbs.shape[3] + 1),
        (255, 255, 255, 255),
    )

    # Annotate frames
    def ann_keypoint(k):
        anns, row = [], []
        for i in range(rgbs.shape[0]):
            rgb = draw_masked(rgbs[i], mask_locs[i])
            dep = draw_masked(deps[i], mask_locs[i])
            ann_fwd = draw_keypoint(rgb, trajs[i, k], radius=15)
            dep = draw_keypoint(dep, trajs[i, k], radius=15)
            frame = np.concatenate([dep, ann_fwd], axis=0)
            if trajs_rev is not None:
                ann_rev = draw_keypoint(rgb, trajs_rev[i, k], radius=15)
                frame = np.concatenate([frame, ann_rev], axis=0)
            frame = cv2.copyMakeBorder(
                frame,
                10,
                10,
                10,
                10,
                cv2.BORDER_CONSTANT,
                value=(0, 0, 255, 255) if i == anchor_idx else (255, 255, 255, 255),
            )
            row += [frame]
            if i < rgbs.shape[0] - 1 and len(row) == ncol:
                strip = np.concatenate(row, axis=1)
                anns += [np.concatenate([strip, pad], axis=0)]
                row = []

        # Add last row
        strip = np.concatenate(row, axis=1)
        empty = np.full(
            (strip.shape[0], pad.shape[1] - strip.shape[1], strip.shape[2]),
            (255, 255, 255, 255),
        )
        anns += [np.concatenate([strip, empty], axis=1)]

        anns = np.concatenate(anns, axis=0)
        return anns

    thumbnails = []
    idx_list = range(trajs.shape[1]) if keypoints_to_plot is None else keypoints_to_plot
    for k in idx_list:
        thumbnails.append(ann_keypoint(k))

    return thumbnails


def draw_keypoint(rgb, pt, radius=4, color=(255, 0, 255, 255), fill=True):
    """
    Draw keypoint on frame.

    Inputs:
        rgb: rgb frame
        pt: array (2,) specifying point center
        radius: radius of point
        color: color of point

    Outputs:
        rgb: annotated rgb frame
    """
    if not np.isnan(pt).any():
        pt = pt.astype(int)
        rgb = cv2.circle(
            rgb.copy(),
            pt,
            radius,
            color,
            -1 if fill else 1,
        )
    return rgb


def draw_frame_multi_pts(rgb, pts, radius=4, color=(255, 0, 255)):
    """
    Draw keypoint on frame.

    Inputs:
        rgb: rgb frame
        pts: array (num,2) specifying point center
        radius: radius of point
        color: color of point

    Outputs:
        rgb: annotated rgb frame
    """
    for pt_idx in range(pts.shape[0]):
        pt = pts[pt_idx]
        if not np.isnan(pt).any():
            rgb = cv2.circle(
                rgb.copy(),
                (round(pt[0]), round(pt[1])),
                radius,
                color,
                -1,
            )
    return rgb


def draw_trajectory(rgb, traj, color=(255, 0, 255), arrow_size=4):
    """
    Draw trajectory on frame.
    Inputs:
        rgb: rgb frame
        traj: array (n_timestep, n_keypoints, 2) of trajectories
        color: color of arrow lines
        arrow_size: size of arrow tip (in px)
    Outputs:
        rgb: annotated rgb frame
    """
    for k in range(traj.shape[1]):
        for i in range(1, traj.shape[0]):
            src = traj[i - 1, k]
            dst = traj[i, k]
            if not np.isnan(src).any():
                arrow_frac = arrow_size / max(1, np.linalg.norm(dst - src, axis=0))
                if not np.isnan(dst).any():
                    rgb = cv2.arrowedLine(
                        rgb.copy(),
                        (round(src[0]), round(src[1])),
                        (round(dst[0]), round(dst[1])),
                        color,
                        2,
                        tipLength=arrow_frac,
                    )
    return rgb


def draw_masked(im, mask_locs):
    im = cv2.cvtColor(im, cv2.COLOR_BGR2BGRA)
    im[:, :, 3] = 100
    im[mask_locs[0], mask_locs[1], 3] = 255
    return im


def draw_pips_animation(pips, trajs, rgbs, deps, linewidth=1, pad=50):
    # Add extra dimension if needed
    if trajs.dim() < 4:
        trajs = trajs.unsqueeze(0)

    _, S, _, H, W = rgbs.shape
    rgbs = F.pad(
        rgbs.reshape(S, 3, H, W),
        (pad, pad, pad, pad),
        "constant",
        0,
    ).reshape(1, S, 3, H + pad * 2, W + pad * 2)
    deps = F.pad(
        deps.reshape(1 * S, 3, H, W),
        (pad, pad, pad, pad),
        "constant",
        0,
    ).reshape(1, S, 3, H + pad * 2, W + pad * 2)
    trajs = trajs + pad

    # visualize the input RGB
    o1 = pips.sw.summ_rgbs(
        "inputs/rgbs",
        pips_utils.improc.preprocess_color(rgbs[0:1]).unbind(1),
        only_return=True,
    )

    # visualize trajs overlaid on the disparity map
    o1a = pips.sw.summ_traj2ds_on_rgbs(
        "inputs/trajs_on_dep",
        trajs[0:1],
        pips_utils.improc.preprocess_color(deps[0:1]),
        cmap="spring",
        linewidth=linewidth,
        only_return=True,
    )

    # visualize the trajs overlaid on the rgbs
    o2 = pips.sw.summ_traj2ds_on_rgbs(
        "outputs/trajs_on_rgbs",
        trajs[0:1],
        pips_utils.improc.preprocess_color(rgbs[0:1]),
        cmap="spring",
        linewidth=linewidth,
        only_return=True,
    )

    # visualize the trajs alone
    o3 = pips.sw.summ_traj2ds_on_rgbs(
        "outputs/trajs_on_black",
        trajs[0:1],
        torch.ones_like(rgbs[0:1]) * -0.5,
        cmap="spring",
        linewidth=linewidth,
        only_return=True,
    )

    # concat these for a synced wide vis
    wide_cat = torch.cat([o1, o1a, o2, o3], dim=-1)

    # write to disk, in case that's more convenient
    wide_list = list(wide_cat.unbind(1))
    wide_list = [wide[0].permute(1, 2, 0).cpu().numpy() for wide in wide_list]
    wide_list = [Image.fromarray(wide) for wide in wide_list]

    return wide_list

