# Copyright (c) 2023 Gengshan Yang, Carnegie Mellon University.
import numpy as np
import os, sys
import numpy as np

sys.path.insert(0, os.getcwd())
from lab4d.utils.geom_utils import K2inv


def depth_to_xyz(depth, intrinsics, xy_homo=None):
    """
    Args:
        depth: (H,W)
        intrinsics: (4,) fx,fy,px,py
    Returns:
        xyz: (H,W,3)
    """
    # depth to xyz
    H, W = depth.shape
    Kinv = K2inv(intrinsics)
    if xy_homo is None:
        xy = np.stack(np.meshgrid(np.arange(W), np.arange(H)), axis=-1)
        xy = xy.reshape((-1, 2))
        xy_homo = np.hstack((xy, np.ones((xy.shape[0], 1))))
    xyz = xy_homo @ Kinv.T * depth.reshape((-1, 1))
    xyz = xyz.reshape((H, W, 3))
    return xyz


def xyz_to_canonical(xyz, extrinsics):
    """
    Args:
        depth: (H,W)
        extrinsics: (4,4)
    Returns:
        xyz: (H,W,3)
    """
    # depth to xyz
    H, W, _ = xyz.shape
    xyz = xyz.reshape((-1, 3))

    # xyz to world
    xyz = np.hstack((xyz, np.ones((xyz.shape[0], 1))))
    xyz = xyz @ np.linalg.inv(extrinsics).T
    xyz = xyz[:, :3].reshape((H, W, 3))
    return xyz


def depth_to_canonical(depth, intrinsics, extrinsics):
    """
    Args:
        depth: (H,W)
        intrinsics: (4,) fx,fy,px,py
        extrinsics: (4,4)
    Returns:
        xyz: (H,W,3)
    """
    xyz = depth_to_xyz(depth, intrinsics)
    xyz = xyz_to_canonical(xyz, extrinsics)
    return xyz
