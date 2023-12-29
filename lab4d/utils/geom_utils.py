# Copyright (c) 2023 Gengshan Yang, Carnegie Mellon University.
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from scipy.spatial.transform import Rotation as R
from skimage import measure
import open3d as o3d
import point_cloud_utils as pcu

from lab4d.utils.quat_transform import (
    dual_quaternion_apply,
    quaternion_translation_apply,
    dual_quaternion_to_se3,
)


def fov_to_focal(fov, img_size=None):
    """Convert field of view to focal length

    Args:
        fov: float, Field of view
        img_size: int, Image size
    """
    if torch.is_tensor(fov):
        focal = 1 / (torch.tan(fov / 2))
    else:
        focal = 1 / (np.tan(fov / 2))
    if img_size is not None:
        focal = focal * img_size / 2
    return focal


def focal_to_fov(focal, img_size=None):
    """Convert focal length to field of view

    Args:
        focal: float, Focal length
        img_size: int, Image size
    """
    if img_size is not None:
        focal = 2 * focal / img_size
    if torch.is_tensor(focal):
        fov = 2 * torch.atan(1 / focal)
    else:
        fov = 2 * np.arctan(1 / focal)
    return fov


def Kmat_to_px(Kmat, res):
    """invserse of the following
    Kmat[..., :2, 2] = Kmat[..., :2, 2] - crop_size / 2
    Kmat[..., :2, :] = Kmat[..., :2, :] / crop_size * 2
    """
    Kmat[..., :2, :] = Kmat[..., :2, :] / 2 * res
    Kmat[..., :2, 2] = Kmat[..., :2, 2] + res / 2
    return Kmat


def pinhole_projection(Kmat, xyz_cam, keep_depth=False):
    """Project points from camera space to the image plane

    Args:
        Kmat: (M, 3, 3) Camera intrinsics
        xyz_cam: (M, ..., 3) Points in camera space
    Returns:
        hxy: (M, ..., 3) Homogeneous pixel coordinates on the image plane
    """
    if not torch.is_tensor(Kmat):
        Kmat = torch.tensor(Kmat, device=xyz_cam.device, dtype=xyz_cam.dtype)
    shape = xyz_cam.shape
    Kmat = Kmat.view(shape[:1] + (1,) * (len(shape) - 2) + (3, 3))
    hxy = torch.einsum("...ij,...j->...i", Kmat, xyz_cam)
    hxy = hxy / (hxy[..., -1:] + 1e-6)
    if keep_depth:
        hxy[..., 2] = xyz_cam[..., 2]
    return hxy


def rot_angle(mat):
    """Compute rotation angle of a rotation matrix

    Args:
        mat: (..., 3, 3) Rotation matrix
    Returns:
        angle: (...,) Rotation angle
    """
    eps = 1e-4
    cos = (mat[..., 0, 0] + mat[..., 1, 1] + mat[..., 2, 2] - 1) / 2
    cos = cos.clamp(-1 + eps, 1 - eps)
    angle = torch.acos(cos)
    return angle


def dual_quaternion_skinning(dual_quat, pts, skin):
    """Attach points to dual-quaternion bones according to skinning weights

    Args:
        dual_quat: ((M,B,4), (M,B,4)) per-bone SE(3) transforms,
            written as dual quaternions
        pts: (M, ..., 3) Points in object canonical space
        skin: (M, ..., B) Skinning weights from each point to each bone
    Returns:
        pts: (M, ..., 3) Articulated points
    """
    shape = pts.shape
    bs, B, _ = dual_quat[0].shape
    pts = pts.view(bs, -1, 3)
    skin = skin.view(bs, -1, B)  # M, N*D, B
    N = pts.shape[1]

    # (M, ND, B, 4)
    qr = dual_quat[0][:, None].repeat(1, N, 1, 1)
    qd = dual_quat[1][:, None].repeat(1, N, 1, 1)

    # make sure to blend in the same hemisphere
    anchor = skin.argmax(-1).view(shape[0], -1, 1, 1).repeat(1, 1, 1, 4)  # M, ND, 1, 4
    sign = (torch.gather(qr, 2, anchor) * qr).sum(-1) > 0  # M, ND, B
    sign = sign[..., None].float() * 2 - 1
    qr = sign * qr
    qd = sign * qd

    qr_w = torch.einsum("bnk,bnkl->bnl", skin, qr)
    qd_w = torch.einsum("bnk,bnkl->bnl", skin, qd)

    qr_mag_inv = qr_w.norm(p=2, dim=-1, keepdim=True).reciprocal()
    qr_w = qr_w * qr_mag_inv
    qd_w = qd_w * qr_mag_inv
    # apply
    pts = dual_quaternion_apply((qr_w, qd_w), pts)

    pts = pts.view(*shape)
    return pts


def linear_blend_skinning(dual_quat, xyz, skin_prob):
    """Attach points to SE(3) bones according to skinning weights

    Args:
        dual_quat: ((M,B,4), (M,B,4)) per-bone SE(3) transforms,
            written as dual quaternions
        xyz: (M, ..., 3) Points in object canonical space
        skin_prob: (M, ..., B) Skinning weights from each point to each bone
    Returns:
        pts: (M, ..., 3) Articulated points
    """
    shape = xyz.shape
    xyz = xyz.view(shape[0], -1, 3)  # M, N*D, 3
    skin_prob = skin_prob.view(shape[0], -1, skin_prob.shape[-1])  # M, N*D, B
    se3 = dual_quaternion_to_se3(dual_quat)  # M,B,4,4
    # M ND B 4 4
    out = se3[:, None, :, :3, :3] @ xyz[:, :, None, :, None]
    out = out + se3[:, None, :, :3, 3:4]  # M,ND,B,3,1
    out = (out[..., 0] * skin_prob[..., None]).sum(-2)  # M,ND,B,3
    out = out.view(shape)
    return out


def slerp(val, low, high, eps=1e-6):
    """
    Args:
        val: (M,) Interpolation value
        low: (M,4) Low quaternions
        high: (M,4) High quaternions
    Returns:
        out: (M,4) Interpolated quaternions
    """
    # Normalize input quaternions.
    low_norm = F.normalize(low, dim=1)
    high_norm = F.normalize(high, dim=1)

    # Compute cosine of angle between quaternions.
    cos_angle = torch.clamp((low_norm * high_norm).sum(dim=1), -1.0 + eps, 1.0 - eps)
    omega = torch.acos(cos_angle)

    so = torch.sin(omega)
    t1 = torch.sin((1.0 - val) * omega) / (so + eps)
    t2 = torch.sin(val * omega) / (so + eps)
    return t1.unsqueeze(-1) * low + t2.unsqueeze(-1) * high


def interpolate_slerp(y, idx_floor, idx_ceil, t_frac):
    """
    Args:
        y: (N,4) Quaternions
        idx_floor: (M,) Floor indices
        idx_ceil: (M,) Ceil indices
        t_frac: (M,) Fractional indices (0-1)
    Returns:
        y_interpolated: (M,4) Interpolated quaternions
    """
    # Use integer parts to index y
    idx_ceil.clamp_(max=len(y) - 1)
    y_floor = y[idx_floor]
    y_ceil = y[idx_ceil]

    # Check dot product to ensure the shortest path
    dp = torch.sum(y_floor * y_ceil, dim=-1, keepdim=True)
    y_ceil = torch.where(dp < 0.0, -y_ceil, y_ceil)

    # Normalize quaternions to be sure
    y_floor_norm = F.normalize(y_floor, dim=1)
    y_ceil_norm = F.normalize(y_ceil, dim=1)

    # Compute interpolated quaternion
    y_interpolated = slerp(t_frac, y_floor_norm, y_ceil_norm)
    y_interpolated_norm = F.normalize(y_interpolated, dim=1)
    return y_interpolated_norm


def interpolate_linear(y, idx_floor, idx_ceil, t_frac):
    """
    Args:
        y: (N,4) translation
        idx_floor: (M,) Floor indices
        idx_ceil: (M,) Ceil indices
        t_frac: (M,) Fractional indices (0-1)
    Returns:
        y_interpolated: (M,4) Interpolated translation
    """
    # Use integer parts to index y
    idx_ceil.clamp_(max=len(y) - 1)
    y_floor = y[idx_floor]
    y_ceil = y[idx_ceil]

    # Compute interpolated quaternion
    y_interpolated = y_floor + t_frac[..., None] * (y_ceil - y_floor)
    return y_interpolated


def hat_map(v):
    """Returns the skew-symmetric matrix corresponding to the last dimension of
    a PyTorch tensor.

    Args:
        v: (..., 3) Input vector
    Returns:
        V: (..., 3, 3) Output matrix
    """
    # Reshape the input tensor to have shape (..., 3)
    v = v.reshape(-1, 3)
    # Compute the skew-symmetric matrix using a vectorized implementation
    V = torch.zeros(v.shape[0], 3, 3, dtype=v.dtype, device=v.device)
    V[:, 0, 1] = -v[:, 2]
    V[:, 0, 2] = v[:, 1]
    V[:, 1, 0] = v[:, 2]
    V[:, 1, 2] = -v[:, 0]
    V[:, 2, 0] = -v[:, 1]
    V[:, 2, 1] = v[:, 0]
    # Reshape the output tensor to match the shape of the input tensor
    V = V.reshape(v.shape[:-1] + (3, 3))
    return V


def so3_to_exp_map(so3, eps=1e-6):
    """Converts a PyTorch tensor of shape (..., 3) representing an element of
    SO(3) to a PyTorch tensor of shape (..., 3, 3) representing the
    corresponding exponential map.

    Args:
        so3: (..., 3) Element of SO(3)
        eps (float): Small value to avoid division by zero
    Returns:
        exp_V: (..., 3, 3) Exponential map
    """
    shape = so3.shape
    so3 = so3.reshape(-1, 3)

    # Compute the magnitude and direction of the rotation vector
    theta = torch.norm(so3, p=2, dim=-1, keepdim=True)
    theta = torch.clamp(theta, eps)
    v = so3 / theta
    # Compute the skew-symmetric matrix of the rotation vector
    V = hat_map(v)
    # Broadcast theta along the last two dimensions
    theta = theta.unsqueeze(-1)
    # Compute the exponential map of the rotation vector
    exp_V = (
        torch.eye(3, dtype=V.dtype, device=V.device)
        + torch.sin(theta) * V
        + (1 - torch.cos(theta)) * torch.matmul(V, V)
    )
    # Reshape the output tensor to match the shape of the input tensor
    exp_V = exp_V.reshape(shape[:-1] + (3, 3))
    return exp_V


def compute_crop_params(mask, crop_factor=1.2, crop_size=256, use_full=False):
    """Compute camera intrinsics transform from cropped to raw images

    Args:
        mask: (H, W) segmentation mask
        crop_factor (float): Ratio between crop size and size of a tight crop
        crop_size (int): Target size of cropped images
        use_full (bool): If True, return a full image
    """
    if use_full or mask.min() < 0:  # no crop if no mask
        mask = np.ones_like(mask)
        crop_factor = 1
    # ss=time.time()
    indices = np.where(mask > 0)
    xid = indices[1]
    yid = indices[0]
    center = ((xid.max() + xid.min()) // 2, (yid.max() + yid.min()) // 2)
    length = (
        (xid.max() - xid.min()) // 2,
        (yid.max() - yid.min()) // 2,
    )  # half length
    length = (int(crop_factor * length[0]), int(crop_factor * length[1]))

    # print('center:%f'%(time.time()-ss))
    # transformation from augmented image to original image
    fls = [2 * length[0] / crop_size, 2 * length[1] / crop_size]
    pps = np.asarray([float(center[0] - length[0]), float(center[1] - length[1])])
    crop2raw = np.asarray([fls[0], fls[1], pps[0], pps[1]])
    return crop2raw


def se3_vec2mat(vec):
    """Convert an SE(3) quaternion or axis-angle vector into 4x4 matrix.

    Args:
        vec: (..., 7) quaternion real-last or (..., 6) axis angle
    Returns:
        mat: (..., 4, 4) SE(3) matrix
    """
    shape = vec.shape[:-1]
    if torch.is_tensor(vec):
        mat = torch.zeros(shape + (4, 4)).to(vec.device)
        if vec.shape[-1] == 6:
            rmat = transforms.axis_angle_to_matrix(vec[..., 3:6])
        else:
            vec = vec[..., [0, 1, 2, 6, 3, 4, 5]]  # xyzw => wxyz
            rmat = transforms.quaternion_to_matrix(vec[..., 3:7])
        tmat = vec[..., :3]
    else:
        mat = np.zeros(shape + (4, 4))
        vec = vec.reshape((-1, vec.shape[-1]))
        if vec.shape[-1] == 6:
            rmat = R.from_axis_angle(vec[..., 3:6]).as_matrix()  # xyzw
        else:
            rmat = R.from_quat(vec[..., 3:7]).as_matrix()  # xyzw
        tmat = np.asarray(vec[..., :3])
        rmat = rmat.reshape(shape + (3, 3))
        tmat = tmat.reshape(shape + (3,))
    mat[..., :3, :3] = rmat
    mat[..., :3, 3] = tmat
    mat[..., 3, 3] = 1
    return mat


def apply_se3mat(se3, pts):
    """Apply an SE(3) rotation and translation to points.

    Note:
        `se3` and `pts` have the same number of batch dimensions.
        During skinning there could be an additional dimension B
    Args:
        se3: (M,1,1,(B),4) Real-first quaternion and (M,1,1,(B),3) Translation
        pts: (M,N,D,(1),3) Points to transform
    Returns:
        pts_out: (M,N,D,(B),3) Transformed points
    """
    quat, trans = se3
    pts_out = quaternion_translation_apply(quat, trans, pts)
    return pts_out


def se3_mat2rt(mat):
    """Convert an SE(3) 4x4 matrix into rotation matrix and translation.

    Args:
        mat: (..., 4, 4) SE(3) matrix
    Returns:
        rmat: (..., 3, 3) Rotation
        tmat: (..., 3) Translation
    """
    rmat = mat[..., :3, :3]
    tmat = mat[..., :3, 3]
    return rmat, tmat


def se3_mat2vec(mat, outdim=7):
    """Convert SE(3) 4x4 matrix into a quaternion or axis-angle vector
    Args:
        mat: (..., 4, 4) SE(3) matrix
        outdim (int): 7 to output quaternion vector, 6 to output axis-angle
    Returns:
        vec: (..., outdim): Quaternion or axis-angle vector
    """
    shape = mat.shape[:-2]
    assert torch.is_tensor(mat)
    tmat = mat[..., :3, 3]
    quat = transforms.matrix_to_quaternion(mat[..., :3, :3])
    if outdim == 7:
        rot = quat[..., [1, 2, 3, 0]]  # xyzw <= wxyz
    elif outdim == 6:
        rot = transforms.quaternion_to_axis_angle(quat)
    else:
        print("error")
        exit()
    vec = torch.cat([tmat, rot], -1)
    return vec


def K2mat(K):
    """Convert camera intrinsics tuple into matrix

    Args:
        K: (..., 4) Camera intrinsics (fx, fy, cx, cy)

    Returns:
        Kmat: (..., 3, 3) Camera intrinsics matrix
    """
    if torch.is_tensor(K):
        Kmat = torch.zeros(K.shape[:-1] + (3, 3), device=K.device)
    else:
        Kmat = np.zeros(K.shape[:-1] + (3, 3))

    Kmat[..., 0, 0] = K[..., 0]
    Kmat[..., 1, 1] = K[..., 1]
    Kmat[..., 0, 2] = K[..., 2]
    Kmat[..., 1, 2] = K[..., 3]
    Kmat[..., 2, 2] = 1
    return Kmat


def mat2K(Kmat):
    """Convert camera intrinsics matrix into tuple

    Args:
        Kmat: (..., 3, 3) Camera intrinsics matrix

    Returns:
        K: (..., 4) Camera intrinsics (fx, fy, cx, cy)
    """
    shape = Kmat.shape[:-2]
    Kmat = Kmat.reshape((-1, 3, 3))
    bs = Kmat.shape[0]

    if torch.is_tensor(Kmat):
        K = torch.zeros(bs, 4, device=Kmat.device)
    else:
        K = np.zeros((bs, 4))
    K[:, 0] = Kmat[:, 0, 0]
    K[:, 1] = Kmat[:, 1, 1]
    K[:, 2] = Kmat[:, 0, 2]
    K[:, 3] = Kmat[:, 1, 2]
    K = K.reshape(shape + (4,))
    return K


def Kmatinv(Kmat):
    """Invert camera intrinsics matrix

    Args:
        Kmat: (..., 3, 3) Camera intrinsics matrix

    Returns:
        Kmatinv: (..., 3, 3) Inverse camera intrinsics matrix
    """
    K = mat2K(Kmat)
    Kmatinv = K2inv(K)
    Kmatinv = Kmatinv.reshape(Kmat.shape)
    return Kmatinv


def K2inv(K):
    """Compute the inverse camera intrinsics matrix from tuple

    Args:
        K: (..., 4) Camera intrinsics (fx, fy, cx, cy)

    Returns:
        Kmat: (..., 3, 3) Inverse camera intrinsics matrix
    """
    if torch.is_tensor(K):
        Kmat = torch.zeros(K.shape[:-1] + (3, 3), device=K.device)
    else:
        Kmat = np.zeros(K.shape[:-1] + (3, 3))
    Kmat[..., 0, 0] = 1.0 / K[..., 0]
    Kmat[..., 1, 1] = 1.0 / K[..., 1]
    Kmat[..., 0, 2] = -K[..., 2] / K[..., 0]
    Kmat[..., 1, 2] = -K[..., 3] / K[..., 1]
    Kmat[..., 2, 2] = 1
    return Kmat


def get_near_far(pts, rtmat, tol_fac=1.5, min_depth=1e-6):
    """
    Args:
        pts:        Point coordinate, (N,3), torch
        rtmat:      Object to camera transform, (M,4,4), torch
        tol_fac:    Tolerance factor
    """
    pts = obj_to_cam(pts, rtmat)

    pmax = pts[..., -1].max(-1)[0]
    pmin = pts[..., -1].min(-1)[0]
    delta = (pmax - pmin) * (tol_fac - 1)

    near = pmin - delta
    far = pmax + delta

    near_far = torch.stack([near, far], -1)
    near_far = torch.clamp(near_far, min=min_depth)
    return near_far


def obj_to_cam(pts, rtmat):
    """
    Args:
        pts:        Point coordinate, (M,N,3) or (N,3), torch or numpy
        rtmat:      Object to camera transform, M,4,4, torch or numpy
    Returns:
        verts:      Transformed points, (M,N,3), torch or numpy
    """
    if torch.is_tensor(pts):
        is_tensor = True
    else:
        is_tensor = False

    if not is_tensor:
        pts = torch.tensor(pts, device="cuda")
        rtmat = torch.tensor(rtmat, device="cuda")  # M,4,4

    pts = pts.view(-1, pts.shape[-2], 3)  # -1,N,3
    pts = torch.cat([pts, torch.ones_like(pts[..., :1])], -1)  # -1,N,4
    pts = torch.einsum("mij,mjk->mik", pts, rtmat.permute(0, 2, 1))  # M,N,4
    pts = pts[..., :3]

    if not is_tensor:
        pts = pts.cpu().numpy()
    return pts


def sample_grid(aabb, grid_size):
    """Densely sample points in a 3D grid

    Args:
        aabb: (2,3) Axis-aligned bounding box
        grid_size (int): Points to sample along each axis
    Returns:
        query_xyz: (grid_size^3,3) Dense xyz grid
    """
    device = aabb.device
    ptx = torch.linspace(aabb[0][0], aabb[1][0], grid_size, device=device)
    pty = torch.linspace(aabb[0][1], aabb[1][1], grid_size, device=device)
    ptz = torch.linspace(aabb[0][2], aabb[1][2], grid_size, device=device)
    query_xyz = torch.cartesian_prod(ptx, pty, ptz)  # (x,y,z)
    return query_xyz


def extend_aabb(aabb, factor=0.1):
    """Extend aabb along each side by factor of the previous size.
    If aabb = [-1,1] and factor = 1, the extended aabb will be [-3,3]

    Args:
        aabb: Axis-aligned bounding box, ((N,)2,3)
        factor (float): Amount to extend on each side
    Returns:
        aabb_new: Extended aabb, ((N,)2,3)
    """
    aabb_new = aabb.clone()
    size = (aabb[..., 1, :] - aabb[..., 0, :]) * factor
    aabb_new[..., 0, :] = aabb[..., 0, :] - size
    aabb_new[..., 1, :] = aabb[..., 1, :] + size
    return aabb_new


def eval_func_chunk(func, xyz, chunk_size):
    """Evaluate a function in chunks to avoid OOM.

    Args:
        func: (M,x) -> (M,y)
        xyz: (M,x)
        chunk_size: int
    Returns:
        vals: (M,y)
    """
    vals = []
    for i in range(0, xyz.shape[0], chunk_size):
        vals.append(func(xyz[i : i + chunk_size]))
    vals = torch.cat(vals, dim=0)
    return vals


@torch.no_grad()
def marching_cubes(
    sdf_func,
    aabb,
    visibility_func=None,
    grid_size=64,
    level=0,
    chunk_size=64**3,
    apply_connected_component=False,
):
    """Extract a mesh from a signed-distance function using marching cubes.
    For the multi-instance case, we use the mean shape/visibility

    Args:
        sdf_func (Function): Signed distance function
        aabb: (2,3) Axis-aligned bounding box
        visibility_func (Function): Returns visibility of each point from camera
        grid_size (int): Marching cubes resolution
        level (float): Contour value to search for isosurfaces on the signed
            distance function
        chunk_size (int): Chunk size to evaluate the sdf function
        apply_connected_component (bool): Whether to apply connected component
    Returns:
        mesh (Trimesh): Output mesh
    """
    # sample grid
    grid = sample_grid(aabb, grid_size)

    # evaluate sdf
    sdf = eval_func_chunk(sdf_func, grid, chunk_size=chunk_size)
    sdf = sdf.cpu().numpy().reshape(grid_size, grid_size, grid_size)

    # evaluate visibility: # ignore the points that are not sampled during optimization
    if visibility_func is not None:
        vis = eval_func_chunk(visibility_func, grid, chunk_size=chunk_size)
        vis = vis.cpu().numpy().reshape(grid_size, grid_size, grid_size)
    else:
        vis = np.ones_like(sdf).astype(bool)

    # extract mesh from sdf: 0-1
    try:
        verts, faces, _, _ = measure.marching_cubes(
            sdf,
            level=level,
            spacing=(1.0 / grid_size, 1.0 / grid_size, 1.0 / grid_size),
            mask=vis,
        )
    except:
        print("marching cubes failed")
        return trimesh.Trimesh()

    # transform from 0-1 to bbox
    aabb = aabb.cpu().numpy()
    verts = verts * (aabb[1:] - aabb[:1]) + aabb[:1]

    mesh = trimesh.Trimesh(verts, faces)
    if apply_connected_component:
        # fg: keep the largest connected component
        mesh = [i for i in mesh.split(only_watertight=False)]
        mesh = sorted(mesh, key=lambda x: x.vertices.shape[0])
        mesh = mesh[-1]
        res_f = 10000
        # decimation
        vw, fw = make_manifold(mesh, res_f=res_f)
        v, f, v_c, f_c = pcu.decimate_triangle_mesh(vw, fw, res_f)
        mesh = trimesh.Trimesh(v, f)
    else:
        pass
        # bg
        # TODO: isotropic remeshing
    return mesh


def make_manifold(mesh, res_f=10000):
    """
    from https://github.com/fwilliams/point-cloud-utils/issues/71
    """
    vw, fw = pcu.make_mesh_watertight(mesh.vertices, mesh.faces, res_f)
    # Compute the shortest distance between each point in p and the mesh:
    #   dists is a NumPy array of shape (P,) where dists[i] is the
    #   shortest distnace between the point p[i, :] and the mesh (v, f)
    dists, fid, bc = pcu.closest_points_on_mesh(vw, mesh.vertices, mesh.faces)

    # Interpolate the barycentric coordinates to get the coordinates of
    # the closest points on the mesh to each point in p
    vw = pcu.interpolate_barycentric_coords(mesh.faces, fid, bc, mesh.vertices)
    return vw, fw


def check_inside_aabb(xyz, aabb):
    """Return a mask of whether the input poins are inside the aabb

    Args:
        xyz: (N,...,3) Points in object canonical space to query
        aabb: (N,2,3) axis-aligned bounding box
    Returns:
        inside_aabb: (N,...,) Inside mask, bool
    """
    # check whether the point is inside the aabb
    shape = xyz.shape[:-1]
    aabb = aabb.view((aabb.shape[0], 2) + (1,) * (len(shape) - 1) + (3,))
    inside_aabb = ((xyz > aabb[:, 0]) & (xyz < aabb[:, 1])).all(-1)
    return inside_aabb


def compute_rectification_se3(mesh, up_direction, threshold=0.01, init_n=3, iter=2000):
    # run ransac to get plane
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(mesh.vertices)
    hypos = []
    n_hypo = 10
    for _ in range(n_hypo):
        if len(pcd.points) < 3:
            break
        best_eq, index = pcd.segment_plane(threshold, init_n, iter)
        # visibile plane given z direction
        if best_eq[2] > 0:
            best_eq = -1 * best_eq

        segmented_pts = pcd.select_by_index(index)
        pts_left = np.asarray(pcd.points)[~np.isin(np.arange(len(pcd.points)), index)]
        pcd.points = o3d.utility.Vector3dVector(pts_left)
        # print("segmented plane pts: ", len(segmented_pts.points) / len(mesh.vertices))
        score = np.asarray(up_direction).dot(best_eq[:3])
        hypos.append((best_eq, segmented_pts, score))
        # trimesh.Trimesh(segmented_pts.points).export("tmp/segmented_{}.obj".format(_))
    # find the one with best score
    best_eq, segmented_pts, score = sorted(hypos, key=lambda x: x[-1])[-1]

    # get se3
    plane_n = np.asarray(best_eq[:3])
    center = np.asarray(segmented_pts.points).mean(0)
    dist = (center * plane_n).sum() + best_eq[3]
    plane_o = center - plane_n * dist
    plane = np.concatenate([plane_o, plane_n])

    # hacky way to specify plane location
    # plane = np.asarray([0.11, 0.018, 0.05, 0, -1, 0])

    # xz plane
    bg2world = plane_transform(origin=plane[:3], normal=plane[3:6], axis=[0, -1, 0])

    # further transform the xz plane center to align with origin
    mesh_rectified = mesh.copy()
    mesh_rectified.apply_transform(bg2world)
    bounds = mesh_rectified.bounds
    center = (bounds[0] + bounds[1]) / 2
    bg2world[0, 3] -= center[0]
    bg2world[2, 3] -= center[2]

    # # DEBUG only
    # mesh.export("tmp/raw.obj")
    # mesh.apply_transform(bg2world)
    # mesh.export("tmp/rect.obj")
    # import pdb

    # pdb.set_trace()

    bg2world = torch.Tensor(bg2world)
    return bg2world


def plane_transform(origin, normal, axis=[0, 1, 0]):
    """
    # modified from https://github.com/mikedh/trimesh/blob/main/trimesh/geometry.py#L14
    Given the origin and normal of a plane find the transform
    that will move that plane to be coplanar with the XZ plane.
    Parameters
    ----------
    origin : (3,) float
        Point that lies on the plane
    normal : (3,) float
        Vector that points along normal of plane
    Returns
    ---------
    transform: (4,4) float
        Transformation matrix to move points onto XZ plane
    """
    normal = normal / (1e-6 + np.linalg.norm(normal))
    # transform = align_vectors(normal, axis)
    transform = np.eye(4)
    transform[:3, :3] = align_vector_a_to_b(normal, axis)
    if origin is not None:
        transform[:3, 3] = -np.dot(transform, np.append(origin, 1))[:3]
    return transform


def align_vector_a_to_b(a, b):
    """Find the rotation matrix that transforms one 3D vector
    to another.
    Args:
        a : (3,) float
          Unit vector
        b : (3,) float
          Unit vector
    Returns:
        matrix : (3, 3) float
          Rotation matrix to rotate from `a` to `b`
    """
    # Ensure the vectors are numpy arrays
    a = np.array(a)
    b = np.array(b)

    # Check if vectors are non-zero
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        raise ValueError("Vectors must be non-zero")

    # Normalize the vectors
    a_hat = a / np.linalg.norm(a)
    b_hat = b / np.linalg.norm(b)

    # Compute the rotation axis (normal to the plane formed by a and b)
    axis = np.cross(a_hat, b_hat)

    # Compute the cosine of the angle between a_hat and b_hat
    cos_angle = np.dot(a_hat, b_hat)

    # Handling numerical imprecision
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    # Compute the angle of rotation
    angle = np.arccos(cos_angle)

    # If vectors are parallel or anti-parallel, no axis is determined. Handle separately
    if np.isclose(angle, 0.0):
        return np.eye(3)  # Identity matrix, no rotation needed
    elif np.isclose(angle, np.pi):
        # Find a perpendicular vector
        axis = np.cross(a_hat, np.array([1, 0, 0]))
        if np.linalg.norm(axis) < 1e-10:
            axis = np.cross(a_hat, np.array([0, 1, 0]))
    axis = axis / np.linalg.norm(axis)  # Normalize axis

    # Compute the rotation matrix using the axis-angle representation
    axis_matrix = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )

    rotation_matrix = (
        np.eye(3)
        + np.sin(angle) * axis_matrix
        + (1 - np.cos(angle)) * np.dot(axis_matrix, axis_matrix)
    )

    return rotation_matrix


def align_vectors(a, b, return_angle=False):
    """
    # modified from https://github.com/mikedh/trimesh/blob/main/trimesh/geometry.py#L38
    Find the rotation matrix that transforms one 3D vector
    to another.
    Parameters
    ------------
    a : (3,) float
      Unit vector
    b : (3,) float
      Unit vector
    return_angle : bool
      Return the angle between vectors or not
    Returns
    -------------
    matrix : (4, 4) float
      Homogeneous transform to rotate from `a` to `b`
    angle : float
      If `return_angle` angle in radians between `a` and `b`
    """
    a = np.array(a, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    if a.shape != (3,) or b.shape != (3,):
        raise ValueError("vectors must be (3,)!")

    # find the SVD of the two vectors
    au = np.linalg.svd(a.reshape((-1, 1)))[0]
    bu = np.linalg.svd(b.reshape((-1, 1)))[0]

    if np.linalg.det(au) < 0:
        au[:, -1] *= -1.0
    if np.linalg.det(bu) < 0:
        bu[:, -1] *= -1.0

    # put rotation into homogeneous transformation
    matrix = np.eye(4)
    matrix[:3, :3] = bu.dot(au.T)

    if return_angle:
        # projection of a onto b
        # first row of SVD result is normalized source vector
        dot = np.dot(au[0], bu[0])
        # clip to avoid floating point error
        angle = np.arccos(np.clip(dot, -1.0, 1.0))
        if dot < -1e-5:
            angle += np.pi
        return matrix, angle

    return matrix


def se3_inv(rtmat):
    """Invert an SE(3) matrix

    Args:
        rtmat: (..., 4, 4) SE(3) matrix
    Returns:
        rtmat_inv: (..., 4, 4) Inverse SE(3) matrix
    """
    rmat, tmat = se3_mat2rt(rtmat)
    rmat = rmat.transpose(-1, -2)
    tmat = -rmat @ tmat[..., None]
    rtmat[..., :3, :3] = rmat
    rtmat[..., :3, 3] = tmat[..., 0]
    return rtmat


def rotation_over_plane(N, dim1, dim2, angle):
    """
    Create an N x N rotation matrix for a rotation over the plane defined by
    the two dimensions dim1 and dim2.

    Parameters:
    - N (int): The dimensionality.
    - dim1/2 (int): The axis around which to rotate (0-based index).
    - angle (float): The rotation angle in degrees.

    Returns:
    - ndarray: The rotation matrix.
    """

    # Basic error check
    if N == 1:
        return np.eye(1)
    if dim1 >= N or dim1 < 0 or dim2 >= N or dim2 < 0:
        raise ValueError("The axis index i is out of bounds for dimensionality N.")

    # Calculate cosine and sine values for the rotation angle
    c = np.cos(angle)
    s = np.sin(angle)

    # Create the 2D rotation block
    R_2D = np.array([[c, -s], [s, c]])

    # Insert the 2D rotation block into the top-left
    R_ND = np.eye(N)
    R_ND[:2, :2] = R_2D

    # If dim is not 0, create the permutation matrix and apply the axis swapping
    P = np.eye(N)
    P[0], P[dim1] = P[dim1].copy(), P[0].copy()  # Swap rows
    P[1], P[dim2] = P[dim2].copy(), P[1].copy()  # Swap columns
    R_ND = P @ R_ND @ P.T  # Apply permutation to the base rotation matrix

    return R_ND


def get_pre_rotation(in_channels):
    """Get the pre-rotation matrix for the input coordinates in positional encoding

    Args:
        in_channels (int): Number of input channels

    Returns:
        rot_mat (ndarray): Rotation matrix
    """
    rot_mat = [np.eye(in_channels)]
    angle = np.pi / 4
    for dim1 in range(in_channels):
        for dim2 in range(dim1):
            rot_mat.append(rotation_over_plane(in_channels, dim1, dim2, angle))
    rot_mat = np.concatenate(rot_mat, axis=0)
    return rot_mat
