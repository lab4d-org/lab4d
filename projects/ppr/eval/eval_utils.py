"""utility functions for evaluation. some functions are taken from DASR: 
https://github.com/jefftan969/dasr/blob/main/eval_utils.py#L86

pip install -e third_party/ChamferDistancePytorch/chamfer3D/
"""

import sys, os
import pdb
import trimesh
from copy import deepcopy
import cv2
import numpy as np
import torch
import tqdm
from matplotlib import pyplot as plt

cmap = plt.get_cmap("plasma")

sys.path.insert(
    0,
    "%s/third_party/ChamferDistancePytorch/" % os.path.join(os.path.dirname(__file__)),
)


from chamfer3D.dist_chamfer_3D import chamfer_3DDist

import pytorch3d
from pytorch3d.ops.knn import _KNN
from pytorch3d.ops.points_alignment import (
    ICPSolution,
    SimilarityTransform,
    corresponding_points_alignment,
    _apply_similarity_transform,
)
from pytorch3d.ops.utils import wmean

from lab4d.utils.vis_utils import visualize_trajectory


def load_ama_intrinsics(path):
    pmat = np.loadtxt(path)
    K, R, T, _, _, _, _ = cv2.decomposeProjectionMatrix(pmat)
    Rmat_gt = R
    Tmat_gt = T[:3, 0] / T[-1, 0]
    Tmat_gt = Rmat_gt.dot(-Tmat_gt[..., None])[..., 0]
    K = K / K[-1, -1]
    intrinscs_gt = np.asarray([K[0, 0], K[1, 1], K[0, 2], K[1, 2]])
    Gmat_gt = np.eye(4)
    Gmat_gt[:3, :3] = Rmat_gt
    Gmat_gt[:3, 3] = Tmat_gt
    return intrinscs_gt, Gmat_gt


def ama_eval(
    pred_mesh_dict,
    gt_mesh_dict,
    verbose=False,
    device="cuda",
    shape_scale=1,
):
    """Evaluate a sequence of AMA videos
    Modified from DASR: https://github.com/jefftan969/dasr/blob/main/eval_utils.py#L86

    Args
        load_dir [str]: Directory to load predicted meshes from
        seqname [str]: Name of sequence (e.g. T_samba)
        vidid [int]: Video identifier (e.g. 1)
        verbose [bool]: Whether to print eval metrics
        render_vid [str]: If provided, output an error video to this path

    Returns:
        cd_avg [float]: Chamfer distance (cm), averaged across all frames
        f010_avg [float]: F-score at 10cm threshold, averaged across all frames
        f005_avg [float]: F-score at 5cm threshold, averaged across all frames
        f002_avg [float]: F-score at 2cm threshold, averaged across all frames
    """
    all_verts_pred = [v.vertices for k, v in pred_mesh_dict.items()]
    all_verts_gt = [v.vertices for k, v in gt_mesh_dict.items()]

    # Evaluate metrics: chamfer distance and f-score (@10cm, @5cm, @2cm)
    nframes = len(all_verts_gt)
    metrics = torch.zeros(nframes, 4, dtype=torch.float32, device=device)  # nframes, 4
    all_verts_pred = torch.tensor(all_verts_pred, device=device, dtype=torch.float32)
    all_verts_gt = torch.tensor(all_verts_gt, device=device, dtype=torch.float32)

    # visualize_trajectory(all_verts_pred, "pred")
    # global sim3 alignment, translation, rotation, scale
    all_verts_pred = align_seqs(
        all_verts_pred[:, None],
        all_verts_gt[:, None],
        align_se3=True,
        verbose=verbose,
    )
    all_verts_pred = [x[0] for x in all_verts_pred]
    # visualize_trajectory(all_verts_pred, "aligned")
    # visualize_trajectory(all_verts_gt, "gt")
    # pdb.set_trace()

    chamLoss = chamfer_3DDist()
    pred_cd_list = []
    gt_cd_list = []
    for idx in tqdm.trange(nframes, desc=f"Evaluating:"):
        raw_cd_fw, raw_cd_bw, _, _ = chamLoss(
            all_verts_gt[idx][None], all_verts_pred[idx][None]
        )  # 1, npts_gt | 1, npts_pred
        raw_cd_fw = raw_cd_fw.squeeze(0)  # npts_gt
        raw_cd_bw = raw_cd_bw.squeeze(0)  # npts_pred
        pred_cd_list.append(raw_cd_bw)
        gt_cd_list.append(raw_cd_fw)

        cd = torch.mean(torch.sqrt(raw_cd_fw)) + torch.mean(torch.sqrt(raw_cd_bw))
        f010, _, _ = fscore(raw_cd_fw, raw_cd_bw, threshold=(shape_scale * 0.10) ** 2)
        f005, _, _ = fscore(raw_cd_fw, raw_cd_bw, threshold=(shape_scale * 0.05) ** 2)
        f002, _, _ = fscore(raw_cd_fw, raw_cd_bw, threshold=(shape_scale * 0.02) ** 2)

        metrics[idx, 0] = cd
        metrics[idx, 1] = f010
        metrics[idx, 2] = f005
        metrics[idx, 3] = f002

        if verbose:
            print(
                f"Frame {idx}: CD={100 * cd:.2f}cm, f@10cm={100 * f010:.1f}%, "
                f"f@5cm={100 * f005:.1f}%, f@2cm={100 * f002:.1f}%"
            )

    metrics = torch.mean(metrics, dim=0)  # 4,
    cd_avg, f010_avg, f005_avg, f002_avg = tuple(float(x) for x in metrics)

    if verbose:
        print(f"Finished evaluation")
        print(f"  Avg chamfer dist: {100 * cd_avg:.2f}cm")
        print(f"  Avg f-score at d=10cm: {100 * f010_avg:.1f}%")
        print(f"  Avg f-score at d=5cm:  {100 * f005_avg:.1f}%")
        print(f"  Avg f-score at d=2cm:  {100 * f002_avg:.1f}%")

    # assign aligned vertices
    for fidx in pred_mesh_dict.keys():
        pred_mesh_dict[fidx].vertices = all_verts_pred[fidx].cpu().numpy()

    pred_cd_dict = deepcopy(pred_mesh_dict)
    gt_cd_dict = deepcopy(gt_mesh_dict)
    vis_err_max = 0.02  # 2cm
    for idx, fidx in tqdm.tqdm(enumerate(pred_cd_dict.keys()), desc=f"Evaluating:"):
        pred_cd_dict[fidx].visual.vertex_colors = 255 * cmap(
            pred_cd_list[idx].cpu().numpy() / vis_err_max
        )
        gt_cd_dict[fidx].visual.vertex_colors = 255 * cmap(
            gt_cd_list[idx].cpu().numpy() / vis_err_max
        )

    return (
        cd_avg,
        f010_avg,
        f005_avg,
        f002_avg,
        pred_mesh_dict,
        pred_cd_dict,
        gt_cd_dict,
    )


def fscore(dist1, dist2, threshold=0.001):
    """
    Calculates the F-score between two point clouds with the corresponding threshold value.
    modified from https://github.com/ThibaultGROUEIX/ChamferDistancePytorch
    :param dist1: N-Points
    :param dist2: N-Points
    :param th: float
    :return: fscore, precision, recall
    """
    # NB : In this depo, dist1 and dist2 are squared pointcloud euclidean distances, so you should adapt the threshold accordingly.
    precision_1 = torch.mean((dist1 < threshold).float())
    precision_2 = torch.mean((dist2 < threshold).float())
    fscore = 2 * precision_1 * precision_2 / (precision_1 + precision_2)
    fscore[torch.isnan(fscore)] = 0
    return fscore, precision_1, precision_2


def timeseries_pointclouds_to_tensor(X_pts):
    """Convert a time series of variable-length point clouds to a padded tensor

    Args:
        X_pts [List(bs, npts[t], dim)]: List of length T, containing a
            time-series of variable-length point cloud batches

    Returns:
        X [bs, T, npts, dim]: Padded pointcloud tensor
        num_points_X [bs, T]: Number of points in each point cloud
    """
    bs, _, dim = X_pts[0].shape
    T = len(X_pts)
    device = X_pts[0].device

    num_points_X = torch.tensor(
        [X_pts[t].shape[1] for t in range(T)], dtype=torch.int64
    )  # T,
    num_points_X = num_points_X.to(device)[None, :].repeat(bs, 1)  # bs, T

    npts = torch.max(num_points_X)
    X = X_pts[0].new_zeros(bs, T, npts, dim)  # bs, T, npts, dim
    for t in range(T):
        npts_t = X_pts[t].shape[1]
        X[:, t, :npts_t] = X_pts[t]  # bs, T, npts[t], dim

    return X, num_points_X


def timeseries_iterative_closest_point(
    X_pts,
    Y_pts,
    init_transform=None,
    max_iterations=100,
    relative_rmse_thr=1e-6,
    estimate_scale=False,
    allow_reflection=False,
    verbose=False,
):
    """Execute the ICP algorithm to find a similarity transform (R, T, s)
    between two time series of differently-sized point clouds

    Args:
        X_pts [List(bs, npts[t], dim)]: Time-series of variable-length
            point cloud batches
        Y_pts [List(bs, npts[t], dim)]: Time-series of variable-length
            point cloud batches
        init_transform [SimilarityTransform]: If provided, initialization for
            the similarity transform, containing orthonormal matrices
            R [bs, dim, dim], translations T [bs, dim], and scaling s[bs,]
        max_iterations (int): Maximum number of ICP iterations
        relative_rmse_thr (float): Threshold on relative root mean square error
            used to terminate the algorithm
        estimate_scale (bool): If True, estimate a scaling component of the
            transformation, otherwise assume identity scale
        allow_reflection (bool): If True, allow algorithm to return `R`
            which is orthonormal but has determinant -1
        verbose: If True, print status messages during each ICP iteration

    Returns: ICPSolution with the following fields
        converged (bool): Boolean flag denoting whether the algorithm converged
        rmse (float): Attained root mean squared error after termination
        Xt [bs, T, size_X, dim]: Point cloud X transformed with final similarity
            transformation (R, T, s)
        RTs (SimilarityTransform): Named tuple containing a batch of similarity transforms:
            R [bs, dim, dim] Orthonormal matrices
            T [bs, dim]: Translations
            s [bs,]: Scaling factors
        t_history (list(SimilarityTransform)): List of similarity transform
            parameters after each ICP iteration
    """
    # Convert input Pointclouds structures to padded tensors
    X, num_points_X = timeseries_pointclouds_to_tensor(
        X_pts
    )  # bs, T, size_X, dim  |  bs, T
    Y, num_points_Y = timeseries_pointclouds_to_tensor(
        Y_pts
    )  # bs, T, size_Y, dim  |  bs, T

    if (
        (X.shape[3] != Y.shape[3])
        or (X.shape[1] != Y.shape[1])
        or (X.shape[0] != Y.shape[0])
    ):
        raise ValueError(
            "X and Y should have same number of batch, time, and data dimensions"
        )
    bs, T, size_X, dim = X.shape
    bs, T, size_Y, dim = Y.shape

    # Handle heterogeneous input
    if ((num_points_Y < size_Y).any() or (num_points_X < size_X).any()) and (
        num_points_Y != num_points_X
    ).any():
        mask_X = (
            torch.arange(size_X, dtype=torch.int64, device=X.device)[None, None, :]
            < num_points_X[:, :, None]
        ).type_as(
            X
        )  # bs, T, size_X
    else:
        mask_X = X.new_ones(bs, T, size_X)  # bs, T, size_X

    X = X.reshape(bs, T * size_X, dim)  # bs, T*size_X, dim
    Y = Y.reshape(bs, T * size_Y, dim)  # bs, T*size_Y, dim
    mask_X = mask_X.reshape(bs, T * size_X)  # bs, T*size_X

    # Clone the initial point cloud
    X_init = X.clone()  # bs, T*size_X, dim

    # Initialize transformation with identity
    sim_R = torch.eye(dim, device=X.device, dtype=X.dtype)[None].repeat(
        bs, 1, 1
    )  # bs, 3, 3
    sim_T = X.new_zeros((bs, dim))  # bs, dim
    sim_s = X.new_ones(bs)  # bs,

    prev_rmse = None
    rmse = None
    iteration = -1
    converged = False
    t_history = []

    # Main loop over ICP iterations
    for iteration in range(max_iterations):
        X_nn_points = timeseries_knn_points(
            X.reshape(bs, T, size_X, dim),
            Y.reshape(bs, T, size_Y, dim),
            lengths_X=num_points_X,
            lengths_Y=num_points_Y,
            K=1,
            return_nn=True,
        ).knn[:, :, 0, :]

        # Get alignment of nearest neighbors from Y with X_init
        sim_R, sim_T, sim_s = corresponding_points_alignment(
            X_init,
            X_nn_points,
            weights=mask_X,
            estimate_scale=estimate_scale,
            allow_reflection=allow_reflection,
        )

        # Apply the estimated similarity transform to X_init
        X = _apply_similarity_transform(X_init, sim_R, sim_T, sim_s)

        # Add current transformation to history
        t_history.append(SimilarityTransform(sim_R, sim_T, sim_s))

        # Compute root mean squared error
        X_sq_diff = torch.sum((X - X_nn_points) ** 2, dim=2)
        rmse = wmean(X_sq_diff[:, :, None], mask_X).sqrt()[:, 0, 0]

        # Compute relative rmse change
        if prev_rmse is None:
            relative_rmse = rmse.new_ones(bs)
        else:
            relative_rmse = (prev_rmse - rmse) / prev_rmse

        if verbose:
            print(
                f"ICP iteration {iteration}: mean/max rmse={rmse.mean():1.2e}/{rmse.max():1.2e}; "
                f"mean relative rmse={relative_rmse.mean():1.2e}"
            )

        # Check for convergence
        if (relative_rmse <= relative_rmse_thr).all():
            converged = True
            break

        # Update the previous rmse
        prev_rmse = rmse

    X = X.reshape(bs, T, size_X, dim)  # bs, T, size_X, dim
    return ICPSolution(
        converged, rmse, X, SimilarityTransform(sim_R, sim_T, sim_s), t_history
    )


def align_seqs(all_verts_pred, all_verts_gt, align_se3=True, verbose=False):
    """Align predicted mesh sequence to the ground-truths
    Taken from DASR: https://github.com/jefftan969/dasr/blob/main/eval_utils.py#L86


    Args:
        all_verts_pred (List(bs, npts[t], 3)): Time-series of predicted mesh batches
        all_verts_gt (List(bs, npts[t], 3)): Time-series of ground-truth mesh batches
        verbose (bool): Whether to print ICP results

    Returns:
        out_verts_pred (List(bs, npts[t], 3)): Time-series of aligned predicted mesh batches
    """
    device = all_verts_pred[0].device
    nframes = len(all_verts_pred)

    # Compute coarse scale estimate (in the correct order of magnitude)
    fitted_scale = torch.zeros(nframes, dtype=torch.float32, device=device)  # nframes,
    for i in range(nframes):
        verts_pred = all_verts_pred[i]  # 1, npts_pred, 3
        verts_gt = all_verts_gt[i]  # 1, npts_gt, 3
        fitted_scale[i] = (
            torch.max(verts_gt[..., -1]) + torch.min(verts_gt[..., -1])
        ) / (torch.max(verts_pred[..., -1]) + torch.min(verts_pred[..., -1]))
    fitted_scale = torch.mean(fitted_scale)

    out_verts_pred = [verts_pred * fitted_scale for verts_pred in all_verts_pred]

    if align_se3:
        # Use ICP to align the first frame and fine-tune the scale estimate
        # scale estimation with ICP is not reliable
        frts0 = timeseries_iterative_closest_point(
            out_verts_pred[:1],
            all_verts_gt[:1],
            estimate_scale=False,
            max_iterations=100,
            verbose=verbose,
        )
        R_icp0, T_icp0, s_icp0 = frts0.RTs  # 1, 3, 3  |  1, 3  |  1, 1

        for i in range(nframes):
            out_verts_pred[i] = _apply_similarity_transform(
                out_verts_pred[i], R_icp0, T_icp0, s_icp0
            )

        # Run global ICP across the point cloud time-series
        frts = timeseries_iterative_closest_point(
            out_verts_pred,
            all_verts_gt,
            estimate_scale=True,
            max_iterations=100,
            verbose=verbose,
        )
        R_icp, T_icp, s_icp = frts.RTs  # 1, 3, 3  |  1, 3  |  1, 1

        for i in range(nframes):
            out_verts_pred[i] = _apply_similarity_transform(
                out_verts_pred[i], R_icp, T_icp, s_icp
            )

    return out_verts_pred


def timeseries_knn_points(
    X,
    Y,
    lengths_X=None,
    lengths_Y=None,
    K=1,
    version=-1,
    return_nn=False,
    return_sorted=True,
):
    """K-nearest neighbors on two time series of point clouds.

    Args:
        X [bs, T, size_X, dim]: A batch of `bs` time series, each with `T`
            point clouds containing `size_X` points of dimension `dim`
        Y [bs, T, size_Y, dim]: A batch of `bs` time series, each with `T`
            point clouds containing `size_Y` points of dimension `dim`
        lengths_X [bs, T]: Length of each point cloud in X, in range [0, size_X]
        lengths_Y [bs, T]: Length of each point cloud in Y, in range [0, size_Y]
        norm (int): Which norm to use, either 1 for L1-norm or 2 for L2-norm
        K (int): Number of nearest neighbors to return
        version (int): Which KNN implementation to use in the backend
        return_nn (bool): If True, returns K nearest neighbors in p2 for each point
        return_sorted (bool0: If True, return nearest neighbors sorted in
            ascending order of distance

    Returns:
        dists [bs, T*size_X, K]: Squared distances to nearest neighbors
        idx [bs, T*size_X, K]: Indices of K nearest neighbors from X to Y.
            If `X_idx[n, t, i, k] = j` then `Y[n, j]` is the k-th nearest
            neighbor to `X_idx[n, t, i]` in `Y[n]`.
        nn [bs, T*size_X, K, dim]: Coords of the K-nearest neighbors from X to Y.
    """
    if (
        (X.shape[3] != Y.shape[3])
        or (X.shape[1] != Y.shape[1])
        or (X.shape[0] != Y.shape[0])
    ):
        raise ValueError(
            "X and Y should have same number of batch, time, and data dimensions"
        )
    bs, T, size_X, dim = X.shape
    bs, T, size_Y, dim = Y.shape

    # Call knn_points, treating time as a batch dimension
    dists, idx, nn = pytorch3d.ops.knn_points(
        X.reshape(bs * T, size_X, dim),
        Y.reshape(bs * T, size_Y, dim),
        lengths1=lengths_X.reshape(bs * T),
        lengths2=lengths_Y.reshape(bs * T),
        K=K,
        version=version,
        return_nn=return_nn,
        return_sorted=return_sorted,
    )  # bs*T, size_X, K  |  bs*T, size_X, K  |  bs*T, size_X, K, dim

    # Reshape into batched time-series of points, and offset points along T-dimension
    dists = dists.reshape(bs, T * size_X, K)  # bs, T*size_X, K
    nn = (
        nn.reshape(bs, T * size_X, K, dim) if return_nn else None
    )  # bs, T*size_X, K, dim

    idx = idx.reshape(bs, T, size_X, K)  # bs, T, size_X, K
    offsets = torch.cumsum(lengths_Y, dim=-1) - lengths_Y  # bs, T
    idx += offsets[:, :, None, None].repeat(1, 1, size_X, K)  # bs, T, size_X, K
    idx = idx.reshape(bs, T * size_X, K)  # bs, T*size_X, K

    return _KNN(dists=dists, idx=idx, knn=nn)
