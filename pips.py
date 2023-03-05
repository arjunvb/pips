import torch
import torch.nn.functional as F
import numpy as np
import random
import draw

import sys
import os

sys.path.append(os.path.dirname(__file__))
from nets.pips import Pips

import pips_utils.improc
import saverloader

torch.manual_seed(4)
random.seed(4)
np.random.seed(4)


def is_topk(a, k=1):
    _, rix = np.unique(-a, return_inverse=True)
    return np.where(rix < k, 1, 0).reshape(a.shape)


def clip_trajectories_oob(trajs):
    """
    Clip trajectories that run OOB.

    Inputs:
        trajs: array (pips.flow_len, n_keypoints, 2), trajectory output from PIPs

    Outputs:
        trajs: array (pips.flow_len, n_keypoints, 2), where OOB entries are replaced by nan
    """
    for k in range(trajs.shape[1]):
        neg_oob = np.where(trajs[:, k] < 0)[0]
        if neg_oob.size > 0:
            max_neg_oob = np.max(neg_oob)
            trajs[0 : max_neg_oob + 1, k] = np.nan
        pos_oob = np.where(trajs[:, k] > 255)[0]
        if pos_oob.size > 0:
            min_pos_oob = np.min(pos_oob)
            trajs[min_pos_oob:, k] = np.nan
    return trajs


def draw_thumbnails(rgbs, trajs, trajs_rev, ncol=10):
    """
    Draw thumbnail for of each keypoint for each frame.

    Inputs:
        rgbs: array of rgb frames (n_timestep, 3, W, H)
        trajs: array of trajectories (n_timestep, n_keypoints, 2)
        trajs: array of reverse trajectories in order (n_timestep, n_keypoints, 2)
        ncol: number of columns in thumbnail grid

    Outputs:
        thumbnails: list of thumbnails for each keypoint
    """
    # Resize rgbs
    rgbs = np.transpose(rgbs, [0, 2, 3, 1])
    rgbs = rgbs[:, :, :, [2, 1, 0]]
    pad = np.full((50, rgbs.shape[2] * ncol, rgbs.shape[3]), (255, 255, 255))

    # Annotate frames
    def ann_keypoint(k):
        anns, row = [], []
        for i in range(rgbs.shape[0]):
            ann_fwd = draw.draw_frame(rgbs[i], trajs[i, k])
            ann_rev = draw.draw_frame(rgbs[i], trajs_rev[i, k])
            row += [np.concatenate([ann_fwd, ann_rev], axis=0)]
            if len(row) == ncol:
                strip = np.concatenate(row, axis=1)
                anns += [np.concatenate([strip, pad], axis=0)]
                row = []

        # Add last row
        strip = np.concatenate(row, axis=1)
        empty = np.full(
            (strip.shape[0], pad.shape[1] - strip.shape[1], strip.shape[2]),
            (255, 255, 255),
        )
        anns += [np.concatenate([strip, empty], axis=1)]

        anns = np.concatenate(anns, axis=0)
        return anns

    # Annotate frames
    def ann_keypoint_all():
        anns, row = [], []
        for i in range(rgbs.shape[0]):
            ann_fwd = draw.draw_frame_multi_pts(rgbs[i], trajs[i])
            ann_rev = draw.draw_frame_multi_pts(rgbs[i], trajs_rev[i])
            row += [np.concatenate([ann_fwd, ann_rev], axis=0)]
            if len(row) == ncol:
                strip = np.concatenate(row, axis=1)
                anns += [np.concatenate([strip, pad], axis=0)]
                row = []

        # Add last row
        strip = np.concatenate(row, axis=1)
        empty = np.full(
            (strip.shape[0], pad.shape[1] - strip.shape[1], strip.shape[2]),
            (255, 255, 255),
        )
        anns += [np.concatenate([strip, empty], axis=1)]

        anns = np.concatenate(anns, axis=0)
        return anns

    thumbnails = []
    for k in range(trajs.shape[1]):
        thumbnails += [ann_keypoint(k)]

    thumbnails += [ann_keypoint_all()]

    return thumbnails


class PipsFlow:
    """Wrapper for PIPs optical flow tracker."""

    def __init__(self, pips_path, H=192, W=256):
        # Parameters
        self.flow_len = 8
        self.B = 1
        self.H, self.W = H, W

        # Init PIPs model
        self.model = Pips(stride=4, S=self.flow_len).cuda()
        _ = saverloader.load(pips_path, self.model)
        self.model.eval()

    def track(self, keypoints, rgbs):
        """
        Run PIPs tracker on rgb sequence.

        Inputs:
            keypoints: np array of size (n_keypoints, 2)
            rgbs: tensor of rgb frames (1, flow_len, 3, H, W)

        Outputs:
            trajs: keypoint trajectories (1, flow_len, n_keypoints, 2)
        """
        keypoints = keypoints.reshape(self.B, keypoints.shape[0], 2)
        keypoints = keypoints.cuda()
        preds, _, _, _ = self.model(keypoints, rgbs, iters=6)
        trajs = preds[-1].cpu()
        return trajs

    def downsample(self, rgbs, deps):
        """
        Downsample (and reshape) rgb sequence to H x W.

        Inputs:
            rgbs: tensor of rgb frames (1, flow_len, 3, H, W)
            deps: tensor of dep frames (1, flow_len, 3, H, W)

        Output:
            rgbs, deps resized
        """
        rgbs = rgbs.cuda().float()  # B, S, C, H, W
        deps = deps.cuda().float()
        B, S, C, H, W = rgbs.shape
        rgbs_ = rgbs.reshape(B * S, C, H, W)
        deps_ = deps.reshape(B * S, C, H, W)
        # rgbs_ = F.interpolate(rgbs_, (self.H, self.W), mode="bilinear")
        # deps_ = F.interpolate(deps_, (self.H, self.W), mode="bilinear")
        rgbs = rgbs_.reshape(B, S, C, self.H, self.W)
        deps = deps_.reshape(B, S, C, self.H, self.W)
        return rgbs, deps
