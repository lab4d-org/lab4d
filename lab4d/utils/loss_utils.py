# Copyright (c) 2023 Gengshan Yang, Carnegie Mellon University.
import torch
import trimesh


def align_vectors(v1, v2):
    """Return the scale that best aligns v1 to v2 in the L2 sense:
    min || kv1-v2 ||^2

    Args:
        v1: (...,) Source vector
        v2: (...,) Target vector
    Returns:
        scale_fac (1,): Scale factor
    """
    scale_fac = (v1 * v2).sum() / (v1 * v1).sum()
    if scale_fac < 0:
        scale_fac = torch.tensor([1.0], device=scale_fac.device)
    return scale_fac
