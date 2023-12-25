import os, sys
import math
import numpy as np
from typing import NamedTuple
import trimesh
import open3d as o3d
import open3d.core as o3c
import pdb

# from plyfile import PlyData, PlyElement

import torch
from torch import nn

from simple_knn._C import distCUDA2

sys.path.insert(0, os.getcwd())
from projects.gsplat.sh_utils import eval_sh, SH2RGB, RGB2SH
from lab4d.nnutils.pose import CameraConst, CameraExplicit
from lab4d.utils.quat_transform import (
    quaternion_translation_to_se3,
    quaternion_mul,
    quaternion_apply,
    quaternion_to_matrix,
    quaternion_conjugate,
    matrix_to_quaternion,
)
from lab4d.utils.geom_utils import rot_angle
from lab4d.utils.vis_utils import get_colormap

colormap = get_colormap()


def knn_cuda(pts, k):
    pts = o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(pts))
    nns = o3c.nns.NearestNeighborSearch(pts)
    nns.knn_index()

    # Single query point.
    indices, distances = nns.knn_search(pts, k)
    indices = torch.utils.dlpack.from_dlpack(indices.to_dlpack())
    distances = torch.utils.dlpack.from_dlpack(distances.to_dlpack())
    return distances, indices


def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


def normalize_quaternion(q):
    return torch.nn.functional.normalize(q, 2, -1)


def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    def helper(step):
        if lr_init == lr_final:
            # constant lr, ignore other params
            return lr_init
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper


def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty


def strip_symmetric(sym):
    return strip_lowerdiag(sym)


def gaussian_3d_coeff(xyzs, covs):
    # xyzs: [N, 3]
    # covs: [N, 6]
    x, y, z = xyzs[:, 0], xyzs[:, 1], xyzs[:, 2]
    a, b, c, d, e, f = (
        covs[:, 0],
        covs[:, 1],
        covs[:, 2],
        covs[:, 3],
        covs[:, 4],
        covs[:, 5],
    )

    # eps must be small enough !!!
    inv_det = 1 / (
        a * d * f + 2 * e * c * b - e**2 * a - c**2 * d - b**2 * f + 1e-24
    )
    inv_a = (d * f - e**2) * inv_det
    inv_b = (e * c - b * f) * inv_det
    inv_c = (e * b - c * d) * inv_det
    inv_d = (a * f - c**2) * inv_det
    inv_e = (b * c - e * a) * inv_det
    inv_f = (a * d - b**2) * inv_det

    power = (
        -0.5 * (x**2 * inv_a + y**2 * inv_d + z**2 * inv_f)
        - x * y * inv_b
        - x * z * inv_c
        - y * z * inv_e
    )

    power[power > 0] = -1e10  # abnormal values... make weights 0

    return torch.exp(power)


def build_rotation(r):
    norm = torch.sqrt(
        r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3]
    )

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device="cuda")

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]

    L = R @ L
    return L


class BasicPointCloud(NamedTuple):
    points: np.array
    colors: np.array
    normals: np.array


class GaussianModel(nn.Module):
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = normalize_quaternion

    def __init__(self, sh_degree: int):
        super().__init__()
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.setup_functions()

    def get_extrinsics(self, frameid):
        quat, trans = self.camera_mlp.get_vals(frameid)
        w2c = quaternion_translation_to_se3(quat, trans)
        return w2c

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    def get_rotation(self, frameid=None):
        rotation = self.rotation_activation(self._rotation)
        if frameid is None:
            if hasattr(self, "_trajectory"):
                delta_rotation = self.rotation_activation(self._trajectory[:, 0, :4])
                return quaternion_mul(rotation, delta_rotation)
            else:
                return rotation
        delta_rotation = self.rotation_activation(self._trajectory[:, frameid, :4])
        return quaternion_mul(delta_rotation, rotation)  # w2c @ delta @ rest gaussian

    def get_xyz(self, frameid=None):
        if frameid is None:
            if hasattr(self, "_trajectory"):
                return self._xyz + self._trajectory[:, 0, 4:]
            else:
                return self._xyz
        return self._xyz + self._trajectory[:, frameid, 4:]

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_num_pts(self):
        return self._xyz.shape[0]

    @torch.no_grad()
    def update_geometry_aux(self):
        # xyz = self.get_xyz().detach().cpu().numpy()
        # # xyz = []
        # # for i in range(self.trajectory.shape[1]):
        # #     xyz.append(self.get_xyz(i).detach().cpu().numpy())
        # # xyz = np.concatenate(xyz, axis=0)

        # # f_dc = (
        # #     self._features_dc.detach()
        # #     .transpose(1, 2)
        # #     .flatten(start_dim=1)
        # #     .contiguous()
        # #     .cpu()
        # #     .numpy()
        # # )
        # # f_rest = (
        # #     self._features_rest.detach()
        # #     .transpose(1, 2)
        # #     .flatten(start_dim=1)
        # #     .contiguous()
        # #     .cpu()
        # #     .numpy()
        # # )
        # self.proxy_geometry = trimesh.Trimesh(vertices=xyz)

        # add bone center / joints
        self.proxy_geometry = self.create_mesh_visualization()

    def export_geometry_aux(self, path):
        self.proxy_geometry.export("%s-proxy.obj" % (path))

    @torch.no_grad()
    def create_mesh_visualization(self, frameid=None):
        meshes = []
        sph = trimesh.creation.uv_sphere(radius=1, count=[4, 4])
        centers = self.get_xyz(frameid).cpu()
        orientations = self.get_rotation(frameid).cpu()
        for k, gauss in enumerate(self.get_scaling.cpu().numpy()):
            ellips = sph.copy()
            ellips.vertices *= gauss[None]
            ellips.visual.vertex_colors = colormap[k % len(colormap)]
            articulation = quaternion_translation_to_se3(orientations[k], centers[k])
            ellips.apply_transform(articulation.numpy())
            meshes.append(ellips)
        meshes = trimesh.util.concatenate(meshes)
        return meshes

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(
            self.get_scaling, scaling_modifier, self._rotation
        )

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd: BasicPointCloud):
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float())

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(
            distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()),
            0.0000001,
        )

        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        self.init_gaussians(fused_point_cloud, fused_color, scales)

    def init_gaussians(self, pts, shs=None, scales=None):
        if not torch.is_tensor(pts):
            pts = torch.tensor(pts, dtype=torch.float)
        self._xyz = nn.Parameter(pts)
        if shs is None:
            shs = torch.rand((pts.shape[0], 3), dtype=torch.float)
        features = torch.zeros(
            (shs.shape[0], 3, (self.max_sh_degree + 1) ** 2),
            dtype=torch.float,
        )
        features[:, :3, 0] = shs
        features[:, 3:, 1:] = 0.0

        self._features_dc = nn.Parameter(
            features[:, :, 0:1].transpose(1, 2).contiguous()
        )
        self._features_rest = nn.Parameter(
            features[:, :, 1:].transpose(1, 2).contiguous()
        )

        if scales is None:
            scales = torch.zeros((pts.shape[0], 3))
            scales[:] = np.log(np.sqrt(0.002))
        self._scaling = nn.Parameter(scales)

        rots = torch.zeros((pts.shape[0], 4))
        rots[:, 0] = 1
        self._rotation = nn.Parameter(rots)

        opacities = self.inverse_opacity_activation(
            0.5 * torch.ones((pts.shape[0], 1), dtype=torch.float)
        )
        self._opacity = nn.Parameter(opacities)

    @property
    def get_num_frames(self):
        if hasattr(self, "_trajectory"):
            return self._trajectory.shape[1]
        else:
            return 0

    def init_trajectory(self, total_frames):
        trajectory = torch.zeros(self.get_num_pts, total_frames, 7)  # quat, trans
        trajectory[:, :, 0] = 1.0
        self._trajectory = nn.Parameter(trajectory)

    def construct_extrinsics(self, config, data_info):
        """Construct camera extrinsics module"""
        tracklet_id = 1
        rtmat = data_info["rtmat"][tracklet_id]
        frame_info = data_info["frame_info"]
        if config["extrinsics_type"] == "const":
            self.camera_mlp = CameraConst(rtmat, frame_info)
        elif config["extrinsics_type"] == "explicit":
            self.camera_mlp = CameraExplicit(rtmat, frame_info=frame_info)
        else:
            raise NotImplementedError

    def construct_stat_vars(self):
        self.xyz_gradient_accum = torch.zeros((self.get_num_pts, 1), device="cuda")
        self.denom = torch.zeros((self.get_num_pts, 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_num_pts), device="cuda")

    def update_learning_rate(self, iteration):
        """Learning rate scheduling per step"""
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group["lr"] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ["x", "y", "z", "nx", "ny", "nz"]
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append("f_dc_{}".format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append("f_rest_{}".format(i))
        l.append("opacity")
        for i in range(self._scaling.shape[1]):
            l.append("scale_{}".format(i))
        for i in range(self._rotation.shape[1]):
            l.append("rot_{}".format(i))
        return l

    def reset_opacity(self):
        if self.optimizer is None:
            return
        opacities_new = self.inverse_opacity_activation(
            torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01)
        )
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def update_point_stats(self, prune_mask, clone_mask):
        dev = prune_mask.device
        valid_mask = ~prune_mask
        clone_mask = torch.logical_and(valid_mask, clone_mask)
        valid_mask = torch.cat(
            (valid_mask, torch.ones(clone_mask.sum(), device=dev).bool())
        )

        # first clone
        self.xyz_gradient_accum = torch.cat(
            [self.xyz_gradient_accum, self.xyz_gradient_accum[clone_mask]], 0
        )
        self.denom = torch.cat([self.denom, self.denom[clone_mask]], 0)
        self.max_radii2D = torch.cat(
            [self.max_radii2D, self.max_radii2D[clone_mask]], 0
        )

        # then prune
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_mask]
        self.denom = self.denom[valid_mask]
        self.max_radii2D = self.max_radii2D[valid_mask]

        # reset
        self.xyz_gradient_accum[:] = 0
        self.denom[:] = 0
        self.max_radii2D[:] = 0

    def densify_and_prune(
        self, max_grad=1e-4, min_opacity=0.1, max_scale=0.1, min_grad=1e-6
    ):
        grads = self.xyz_gradient_accum / (self.denom + 1e-6)
        grads = grads[..., 0]
        print("min grad: ", torch.min(grads))
        print("max grad: ", torch.max(grads))

        # Clone if grad is high
        clone_mask = grads > max_grad

        # Prune if opacity is low, scale if high, or grad is low
        selected_trans_pts = (self.get_opacity < min_opacity).squeeze()
        selected_big_pts = self.get_scaling.max(dim=1).values > max_scale
        selected_out_pts = torch.logical_and(grads < min_grad, self.denom[..., 0] > 0)
        print("selected_out_pts: ", selected_out_pts.sum())
        print("max scale: ", torch.max(self.get_scaling.max(dim=1).values))
        prune_mask = torch.logical_or(selected_trans_pts, selected_big_pts)
        prune_mask = torch.logical_or(prune_mask, selected_out_pts)
        return clone_mask, prune_mask

    def add_xyz_grad_stats(self, viewspace_point_grad, update_filter):
        """
        viewspace_point_grad: [N, 3]
        update_filter: [N], bool
        """
        self.xyz_gradient_accum[update_filter] += torch.norm(
            viewspace_point_grad[update_filter, :2], dim=-1, keepdim=True
        )
        self.denom[update_filter] += 1

    @staticmethod
    def transform(center, rotations, w2c):
        """
        center: [N, 3]
        rotations: [N, 4]
        w2c: [4, 4]
        """
        center = center @ w2c[:3, :3].T + w2c[:3, 3][None]
        rmat = w2c[:3, :3][None].repeat(len(center), 1, 1)
        # gaussian space to world then to camera
        rotations = quaternion_mul(matrix_to_quaternion(rmat), rotations)  # wxyz
        return center, rotations

    @torch.no_grad()
    def get_aabb(self):
        xyz = self.get_xyz()
        aabb = (
            xyz.min(dim=0).values,
            xyz.max(dim=0).values,
        )
        aabb = torch.stack(aabb, dim=0).cpu().numpy()
        return aabb

    def get_arap_loss(self):
        """
        Compute arap loss with annealing.
        Begining: 100pts, 100nn
        End: 10000pts, 2nn
        num_pts * ratio_knn * num_pts = 10k
        num_pts = sq(10k / ratio_knn)
        """
        ratio_knn = 0.1
        num_pts = int(np.sqrt(4e5 / ratio_knn))  # 2k pts
        num_knn = max(int(ratio_knn * num_pts), 2)  # get 1-nn
        rand_frameid = np.random.randint(self.get_num_frames - 1)
        rand_ptsid = np.random.permutation(self.get_num_pts)[:num_pts]
        pts0 = self.get_xyz(rand_frameid)[rand_ptsid]
        pts1 = self.get_xyz(rand_frameid + 1)[rand_ptsid]  # N,3
        rot0 = self.get_rotation(rand_frameid)[rand_ptsid]
        rot1 = self.get_rotation(rand_frameid + 1)[rand_ptsid]

        # dist(t,t+1)
        sq_dist, neighbor_indices = knn_cuda(pts0, num_knn)

        # N, K, 3/4
        offset0 = pts0[neighbor_indices] - pts0[:, None]  # in camera 0
        offset1 = pts1[neighbor_indices] - pts1[:, None]  # in camera 1
        c1_to_c0 = quaternion_mul(rot0, quaternion_conjugate(rot1))  # N, 4
        c1_to_c0 = c1_to_c0[:, None].repeat(1, num_knn, 1)
        offset1_0 = quaternion_apply(c1_to_c0, offset1)
        arap_loss_trans = torch.norm(offset0 - offset1_0, 2, -1)
        # neighbor_weight = (-2000 * torch.tensor(neighbor_sq_dist, device="cuda")).exp()
        # arap_loss_trans = arap_loss_trans * neighbor_weight

        # # relrot
        # rot0inv = quaternion_conjugate(rot0)[:, None].repeat(1, num_knn, 1)
        # rot1inv = quaternion_conjugate(rot1)[:, None].repeat(1, num_knn, 1)
        # omega0 = quaternion_mul(rot0inv, rot0[neighbor_indices])  # in gauss space
        # omega1 = quaternion_mul(rot1inv, rot1[neighbor_indices])  # in gauss space
        # relrot = quaternion_mul(omega0, quaternion_conjugate(omega1))
        # arap_loss_rot = quaternion_to_matrix(relrot)
        # arap_loss_rot = rot_angle(arap_loss_rot)
        # # arap_loss_rot = arap_loss_rot * neighbor_weight

        # arap_loss_trans = arap_loss_trans.mean() + arap_loss_rot.mean()
        arap_loss_trans = arap_loss_trans.mean()
        return arap_loss_trans

    def get_least_deform_loss(self):
        if hasattr(self, "_trajectory"):
            # least deform loss
            least_trans_loss = torch.norm(self._trajectory[..., 4:], 2, -1).mean()
            least_rot_loss = quaternion_to_matrix(self._trajectory[..., :4])
            least_rot_loss = rot_angle(least_rot_loss).mean()
            least_deform_loss = least_trans_loss + least_rot_loss
            return least_deform_loss
        else:
            return torch.tensor(0.0, device="cuda")

    def get_least_action_loss(self):
        if hasattr(self, "_trajectory"):
            # least action loss
            x_traj = self._trajectory[..., 4:]
            v_traj = x_traj[:, 1:] - x_traj[:, :-1]
            a_traj = v_traj[:, 1:] - v_traj[:, :-1]
            least_trans_loss = torch.norm(a_traj, 2, -1).mean()

            theta_traj = self.rotation_activation(self._trajectory[..., :4])
            omega_traj = quaternion_mul(
                theta_traj[:, 1:].clone(),
                quaternion_conjugate(theta_traj[:, :-1]).clone(),
            )
            alpha_traj = quaternion_mul(
                omega_traj[:, 1:].clone(),
                quaternion_conjugate(omega_traj[:, :-1]).clone(),
            )
            least_rot_loss = quaternion_to_matrix(alpha_traj)
            least_rot_loss = rot_angle(least_rot_loss).mean()
            least_action_loss = least_trans_loss + 0.01 * least_rot_loss
            return least_action_loss
        else:
            return torch.tensor(0.0, device="cuda")


def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 1 / tanHalfFovX
    P[1, 1] = 1 / tanHalfFovY
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def getProjectionMatrix_K(znear, zfar, Kmat):
    if torch.is_tensor(Kmat):
        P = torch.zeros(4, 4)
    else:
        P = np.zeros((4, 4))

    z_sign = 1.0

    P[:2, :3] = Kmat[:2, :3]
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P
