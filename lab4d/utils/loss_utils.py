# Copyright (c) 2023 Gengshan Yang, Carnegie Mellon University.
import torch
import numpy as np
import torch.nn.functional as F

from lab4d.utils.geom_utils import rot_angle


def compute_se3_smooth_loss(rtk_all, data_offset, vid=None):
    """
    2nd order loss
    """
    rot_sm_loss = []
    trn_sm_loss = []
    for didx in range(len(data_offset) - 1):
        if vid is not None and didx not in vid:
            continue
        stt_idx = data_offset[didx]
        end_idx = data_offset[didx + 1]

        stt_rtk = rtk_all[stt_idx : end_idx - 2]
        mid_rtk = rtk_all[stt_idx + 1 : end_idx - 1]
        end_rtk = rtk_all[stt_idx + 2 : end_idx]

        rot_sub1 = stt_rtk[:, :3, :3].matmul(mid_rtk[:, :3, :3].permute(0, 2, 1))
        rot_sub2 = mid_rtk[:, :3, :3].matmul(end_rtk[:, :3, :3].permute(0, 2, 1))

        trn_sub1 = stt_rtk[:, :3, 3] - mid_rtk[:, :3, 3]
        trn_sub2 = mid_rtk[:, :3, 3] - end_rtk[:, :3, 3]

        rot_sm_sub = rot_sub1.matmul(rot_sub2.permute(0, 2, 1))
        trn_sm_sub = trn_sub1 - trn_sub2

        rot_sm_loss.append(rot_sm_sub)
        trn_sm_loss.append(trn_sm_sub)
    rot_sm_loss = torch.cat(rot_sm_loss, 0)
    rot_sm_loss = rot_angle(rot_sm_loss).mean() * 1e-1
    trn_sm_loss = torch.cat(trn_sm_loss, 0)
    trn_sm_loss = trn_sm_loss.norm(2, -1).mean()
    root_sm_loss = rot_sm_loss + trn_sm_loss
    root_sm_loss = root_sm_loss * 0.1
    return root_sm_loss


def entropy_loss(prob, dim=-1):
    """Compute entropy of a probability distribution
    In the case of skinning weights, each column is a distribution over assignment to B bones.
    We want to encourage low entropy, i.e. each point is assigned to fewer bones.

    Args:
        prob: (..., B) Probability distribution
    Returns:
        entropy (...,) Entropy of each distribution
    """
    entropy = -(prob * (prob + 1e-9).log()).sum(dim)
    return entropy


def cross_entropy_skin_loss(skin):
    """Compute entropy of a probability distribution
    In the case of skinning weights, each column is a distribution over assignment to B bones.
    We want to encourage low entropy, i.e. each point is assigned to fewer bones.

    Args:
        skin: (..., B) un-normalized skinning weights
    """
    shape = skin.shape
    nbones = shape[-1]
    full_skin = skin.clone()

    # find the most likely bone assignment
    score, indices = skin.max(-1, keepdim=True)
    skin = torch.zeros_like(skin).fill_(0)
    skin = skin.scatter(-1, indices, torch.ones_like(score))

    cross_entropy = F.cross_entropy(
        full_skin.view(-1, nbones), skin.view(-1, nbones), reduction="none"
    )
    cross_entropy = cross_entropy.view(shape[:-1])
    return cross_entropy


def align_tensors(v1, v2, dim=None):
    """Return the scale that best aligns v1 to v2 in the L2 sense:
    min || kv1-v2 ||^2

    Args:
        v1: (...,) Source vector
        v2: (...,) Target vector
        dim: Dimension to align. If None, return a scalar
    Returns:
        scale_fac (1,): Scale factor
    """
    if dim is None:
        scale = (v1 * v2).sum() / (v1 * v1).sum()
        if scale < 0:
            scale = torch.tensor([1.0], device=scale.device)
        return scale
    else:
        scale = (v1 * v2).sum(dim, keepdim=True) / (v1 * v1).sum(dim, keepdim=True)
        scale[scale < 0] = 1.0
        return scale
