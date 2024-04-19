# Copyright (c) 2023 Jeff Tan, Carnegie Mellon University.
import os
import sys

import cv2
import pdb
import numpy as np
import torch

sys.path.insert(0, os.getcwd())

from lab4d.tests.utils import check_func


def knn_cuda(pts, k):
    import open3d.core as o3c

    pts = o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(pts))
    nns = o3c.nns.NearestNeighborSearch(pts)
    nns.knn_index()

    # Single query point.
    query_points = pts
    indices, distances = nns.knn_search(query_points, k)
    indices = torch.utils.dlpack.from_dlpack(indices.to_dlpack())
    distances = torch.utils.dlpack.from_dlpack(distances.to_dlpack())
    return distances, indices


def test_knn(pts):
    """Test quaternion to matrix operation"""
    from projects.diffgs.gs_renderer import o3d_knn

    o3d_knn_cpu = lambda pts, k: o3d_knn(pts.cpu(), k)[0]

    check_func(
        knn_cuda,
        knn_cuda,
        (pts, 100),
        name="knn",
    )


if __name__ == "__main__":
    test_knn(torch.randn(4096, 3, dtype=torch.float32, device="cuda"))
