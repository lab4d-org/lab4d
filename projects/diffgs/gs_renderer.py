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
from torch.nn import functional as F

from simple_knn._C import distCUDA2

sys.path.insert(0, os.getcwd())
from projects.diffgs.sh_utils import eval_sh, SH2RGB, RGB2SH
from projects.predictor.predictor import CameraPredictor, TrajPredictor
from lab4d.nnutils.base import CondMLP, ScaleLayer
from lab4d.nnutils.embedding import PosEmbedding, TimeEmbedding
from lab4d.nnutils.pose import CameraConst, CameraExplicit
from lab4d.utils.quat_transform import (
    quaternion_translation_to_se3,
    quaternion_mul,
    quaternion_apply,
    quaternion_to_matrix,
    quaternion_conjugate,
    matrix_to_quaternion,
    dual_quaternion_to_quaternion_translation,
    quaternion_translation_apply,
    quaternion_conjugate,
)
from lab4d.utils.geom_utils import rot_angle, extend_aabb
from lab4d.utils.vis_utils import get_colormap, draw_cams, mesh_cat

colormap = get_colormap()

def gs_transform(center, rotations, w2c):
    """
    center: [N, bs, 3]
    rotations: [N, bs, 4]
    w2c: [bs, 4, 4]
    """
    if not torch.is_tensor(w2c):
        w2c = torch.tensor(w2c, dtype=torch.float, device=center.device)
    rmat = w2c[None, :, :3, :3] # 1, bs, 3, 3
    tmat = w2c[None, :, :3, 3:] # 1, bs, 3, 1
    npts = center.shape[0]

    # gaussian space to world then to camera
    center = (rmat @ center[..., None] + tmat)[..., 0]
    rquat = matrix_to_quaternion(rmat) # 1, bs, 4
    rquat = rquat.repeat(npts, 1, 1)
    rotations = quaternion_mul(rquat, rotations)  # wxyz
    return center, rotations

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
    """
    each node stores transformation to parent node
    only leaf nodes store the actual points
    """
    def is_leaf(self):
        """
        parent_list: [N]
        index: int
        
        return: bool
        whether the index is a leaf node
        """
        return self.index not in self.parent_list

    def get_children_idx(self):
        """
        return: [N]
        children indices of the current node
        """
        children = []
        for i, parent in enumerate(self.parent_list):
            if parent == self.index:
                children.append(i)
        return children
    
    def get_all_children(self):
        """
        get all children incuding self
        """
        children = [(self, self.mode)]
        for gaussians in self.gaussians:
            children += gaussians.get_all_children()
        return children
        

    def __init__(self, sh_degree: int, config: dict, data_info: dict, 
                 parent_list: list, index: int, mode_list: list, lab4d_model, lab4d_meshes):
        """
        only store pts in the leaf node
        for non-leaf nodes: recursively call and aggregate the pts in the children
        """
        super().__init__()
        self.config = config
        self.parent_list = parent_list
        self.index = index
        self.mode = mode_list[index]
        self.is_inc_mode = False
        self.ratio_knn = 1.0
        self.setup_functions()
        self.gaussians = torch.nn.ModuleList()


        # load lab4d model
        self.lab4d_model = lab4d_model

        if self.is_leaf():
            self.max_sh_degree = sh_degree
            self._xyz = torch.empty(0)
            self._features_dc = torch.empty(0)
            self._features_rest = torch.empty(0)
            self._scaling = torch.empty(0)
            self._rotation = torch.empty(0)
            self._opacity = torch.empty(0)
            self.xyz_gradient_accum = torch.empty(0)
            self.denom = torch.empty(0)
            self.xyz_vis_accum = torch.empty(0)
            self.vis_denom = torch.empty(0)

            # load geometry
            num_pts = config["num_pts"]
            if self.mode == "bg":
                num_pts *= 10
            
            # initialize with lab4d
            mesh = lab4d_meshes[self.mode]
            pts, _, colors = trimesh.sample.sample_surface(mesh, num_pts, sample_color=True)
            scale_field = self.lab4d_model.fields.field_params[self.mode]
            self.register_buffer("scale_field", scale_field.logscale.exp())
            pcd = BasicPointCloud(
                pts / self.scale_field.detach().cpu().numpy(),
                colors[:, :3] / 255,
                np.zeros((pts.shape[0], 3)),
            )
            self.initialize(input=pcd)

            # # DEBUG
            # mesh = trimesh.load("tmp/0.obj")
            # pcd = BasicPointCloud(
            #     mesh.vertices,
            #     mesh.visual.vertex_colors[:, :3] / 255,
            #     np.zeros((mesh.vertices.shape[0], 3)),
            # )
            # self.initialize(input=pcd)

            # initialize temporal part: (dx,dy,dz)t
            if config["fg_motion"] == "explicit" and not self.mode=="bg":
                total_frames = data_info["frame_info"]["frame_offset_raw"][-1]
                if config["use_timesync"]:
                    num_vids = len(data_info["frame_info"]["frame_offset"]) - 1
                    total_frames = total_frames // num_vids
                self.init_trajectory(total_frames)

            self.frame_offset_raw = data_info["frame_info"]["frame_offset_raw"]
            self.use_timesync = config["use_timesync"]
                    
            # shadow field
            num_freq_xyz = 6
            num_freq_t = 6
            num_inst = len(data_info["frame_info"]["frame_offset"]) - 1
            self.pos_embedding = PosEmbedding(3, num_freq_xyz)
            self.time_embedding = TimeEmbedding(num_freq_t, data_info["frame_info"])
            self.shadow_field = CondMLP(num_inst=num_inst, 
                                        D=1,
                                        W=256,             
                                        in_channels=self.pos_embedding.out_channels+ self.time_embedding.out_channels,
                                        out_channels=1)

        else:
            for idx in self.get_children_idx():
                gaussians = GaussianModel(sh_degree, config, data_info, parent_list, idx, mode_list, lab4d_model, lab4d_meshes)
                self.gaussians.append(gaussians)

        if self.parent_list[self.index] == -1:
            # aux stats
            self.construct_stat_vars()

        # extrinsics
        self.construct_extrinsics(config, data_info)

        # init proxy
        self.update_geometry_aux()

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

    def initialize(self, input=None, num_pts=5000, radius=0.5):
        # load checkpoint
        if input is None:
            # init from random point cloud
            phis = np.random.random((num_pts,)) * 2 * np.pi
            costheta = np.random.random((num_pts,)) * 2 - 1
            thetas = np.arccos(costheta)
            mu = np.random.random((num_pts,))
            radius_rand = radius * np.cbrt(mu)
            x = radius_rand * np.sin(thetas) * np.cos(phis)
            y = radius_rand * np.sin(thetas) * np.sin(phis)
            z = radius_rand * np.cos(thetas)
            xyz = np.stack((x, y, z), axis=1)
            scales = np.log(np.ones_like(xyz) * radius * 0.1)
            self.init_gaussians(xyz, scales=scales)
        elif isinstance(input, BasicPointCloud):
            # load from a provided pcd
            self.create_from_pcd(input)
        else:
            raise NotImplementedError

    def init_camera_mlp(self):
        if isinstance(self.gs_camera_mlp, CameraPredictor):
            self.gs_camera_mlp.init_weights()
        for gaussians in self.gaussians:
            gaussians.init_camera_mlp()

    def frame_id_to_sub(self, frame_id, inst_id):
        """Convert frame id to frame id relative to the video.
        This forces all videos to share the same pose embedding.

        Args:
            frame_id: (M,) Frame id
            inst_id: (M,) Instance id
        Returns:
            frameid_sub: (M,) Frame id relative to the video
        """
        frame_offset_raw = torch.tensor(self.frame_offset_raw, device=frame_id.device)
        frameid_sub = frame_id - frame_offset_raw[inst_id]
        return frameid_sub

    def get_extrinsics(self, frameid=None, return_qt=False):
        """
        transormation from current node to parent node
        """
        if frameid is not None and not torch.is_tensor(frameid):
            dev = self.parameters().__next__().device
            frameid = torch.tensor(frameid, dtype=torch.long, device=dev)
        quat, trans = self.gs_camera_mlp.get_vals(frameid)
        if return_qt:
            return (quat, trans)
        else:
            w2c = quaternion_translation_to_se3(quat, trans)
            return w2c

    @property
    def get_scaling(self):
        if self.is_leaf():
            return self.scaling_activation(self._scaling)
        else:
            rt_tensor = []
            for gaussians in self.gaussians:
                rt_tensor.append(gaussians.get_scaling)
            return torch.cat(rt_tensor, dim=0)

    def get_rotation(self, frameid=None):
        if self.is_leaf():
            rot = self.rotation_activation(self._rotation).clone()
            # return N, 4
            if frameid is None:
                return rot
            
            # return N, T, 4
            npts = rot.shape[0]
            shape = frameid.shape
            frameid = frameid.reshape(-1)
            if self.mode=="bg":
                rot = rot[:, None].repeat(1, len(frameid), 1)
            else:
                delta_rot = []
                if torch.is_tensor(frameid):
                    frameid = frameid.cpu().numpy()
                for key in frameid:
                    delta_rot.append(self.rotation_activation(self.trajectory_cache[key][:,:4]))
                delta_rot = torch.stack(delta_rot, dim=1) # N,T,4
                rot = quaternion_mul(delta_rot, rot[:, None].repeat(1,delta_rot.shape[1],1))  # w2c @ delta @ rest gaussian
            rot = rot.view((npts,) + shape + (4,))
            return rot
        else:
            rt_tensor = []
            for gaussians in self.gaussians:
                if frameid is None:
                    rot = gaussians.get_rotation(frameid)
                else:
                    # apply rotation of camera
                    shape = frameid.shape
                    frameid_reshape = frameid.reshape(-1)
                    rot = gaussians.get_rotation(frameid_reshape)
                    quat = gaussians.get_extrinsics(frameid_reshape, return_qt=True)[0]
                    # N, T, 4
                    quat = quat[None].repeat(rot.shape[0], 1, 1)
                    rot = quaternion_mul(quat, rot)
                    rot = rot.view(rot.shape[:1] + shape + rot.shape[-1:])
                rt_tensor.append(rot)
            return torch.cat(rt_tensor, dim=0)

    def get_xyz(self, frameid=None):
        if self.is_leaf(): 
            # return N,3
            xyz_t = self._xyz.clone()
            if frameid is None:
                return xyz_t
            
            # return N, T, 3
            npts = self._xyz.shape[0]
            shape = frameid.shape
            frameid = frameid.reshape(-1)
            if self.mode=="bg":
                 xyz_t = xyz_t[:,None].repeat(1,len(frameid),1)
            else:
                # to prop grad to motion
                traj_pred = []
                if torch.is_tensor(frameid):
                    frameid = frameid.cpu().numpy()
                for key in frameid:
                    traj_pred.append(self.trajectory_cache[key][:,4:])
                traj_pred = torch.stack(traj_pred, dim=1) 
                # N,xyz,3
                xyz_t = xyz_t[:, None] + traj_pred
            xyz_t = xyz_t.view((npts,) + shape + (3,))
            return xyz_t
        else:
            rt_tensor = []
            for gaussians in self.gaussians:
                if frameid is None:
                    xyz = gaussians.get_xyz(frameid)
                else:
                    shape = frameid.shape
                    frameid_reshape = frameid.reshape(-1)
                    xyz = gaussians.get_xyz(frameid_reshape)
                    quat, trans = gaussians.get_extrinsics(frameid_reshape, return_qt=True)
                    quat = quat[None].repeat(xyz.shape[0], 1,1)
                    trans = trans[None].repeat(xyz.shape[0], 1,1)
                    # N, T, 3
                    xyz = quaternion_translation_apply(quat, trans, xyz)
                    xyz = xyz.view(xyz.shape[:1] + shape + xyz.shape[-1:])
                rt_tensor.append(xyz)
            return torch.cat(rt_tensor, dim=0)

    def get_features(self, frameid=None):
        if self.is_leaf():
            # N, k, 3
            features_dc = self._features_dc
            features_rest = self._features_rest
            shs = torch.cat((features_dc, features_rest), dim=1)

            if frameid is None:
                return shs
            else:
                shape = frameid.shape
                frameid = frameid.reshape(-1)
                shadow_pred = []
                for key in frameid.cpu().numpy():
                    shadow_pred.append(self.shadow_cache[key])
                shadow_pred = torch.stack(shadow_pred, dim=1) # N,T,1
                # N,T,1,1
                rgb = SH2RGB(shs)
                rgb_t = (rgb[:, None] * shadow_pred[...,None])
                rgb_t = rgb_t.clamp(0,1)
                shs_t = RGB2SH(rgb_t)
                return shs_t.reshape(shs_t.shape[:1] + shape + shs_t.shape[-2:])
        else:
            rt_tensor = []
            for gaussians in self.gaussians:
                rt_tensor.append(gaussians.get_features(frameid))
            return torch.cat(rt_tensor, dim=0)

    @property
    def get_opacity(self):
        if self.is_leaf():
            return self.opacity_activation(self._opacity)
        else:
            rt_tensor = []
            for gaussians in self.gaussians:
                rt_tensor.append(gaussians.get_opacity)
            return torch.cat(rt_tensor, dim=0)

    @property
    def get_num_pts(self):
        if self.is_leaf():
            return self._xyz.shape[0]
        else:
            return sum([gaussians.get_num_pts for gaussians in self.gaussians])

    @torch.no_grad()
    def update_geometry_aux(self):
        # add bone center / joints
        dev = self.parameters().__next__().device
        frameid = torch.tensor([0], device=dev)[0]
        self.update_trajectory(frameid)
        self.proxy_geometry = self.create_mesh_visualization(frameid=frameid,all_pts=False)

    @torch.no_grad()
    def export_geometry_aux(self, path):
        mesh_geo = self.proxy_geometry
        rtmat = self.get_extrinsics().cpu().numpy()
        # evenly pick max 200 cameras
        if rtmat.shape[0] > 200:
            idx = np.linspace(0, rtmat.shape[0] - 1, 200).astype(np.int32)
            rtmat = rtmat[idx]
        mesh_cam = draw_cams(rtmat)
        mesh = mesh_cat(mesh_geo, mesh_cam)

        mesh.export("%s-proxy.obj" % (path))

    @torch.no_grad()
    def create_mesh_visualization(self, frameid=None, all_pts=True, opacity_th=0.1, vis_portion=0.1):
        if self.is_leaf():
            meshes = []
            sph = trimesh.creation.uv_sphere(radius=1, count=[4, 4])
            centers = self.get_xyz(frameid).cpu()
            orientations = self.get_rotation(frameid).cpu()
            scalings = self.get_scaling.cpu().numpy()
            opacity = self.get_opacity.cpu().numpy()[...,0]
            vis_ratio = self.get_vis_ratio.cpu().numpy()
            visibility_th = np.quantile(vis_ratio, vis_portion) # use points with highest 95% vis

            # select those with high opacity
            select_idx = np.logical_and(opacity > opacity_th, vis_ratio > visibility_th)
            scalings_sel = scalings[select_idx]
            centers_sel = centers[select_idx]
            orientations_sel = orientations[select_idx]

            # subsample if too many gaussians
            if not all_pts:
                max_pts = 1000
                if len(scalings_sel) > max_pts:
                    rand_idx = np.random.permutation(scalings_sel.shape[0])[:max_pts]
                    scalings_sel = scalings_sel[rand_idx]
                    centers_sel = centers_sel[rand_idx]
                    orientations_sel = orientations_sel[rand_idx]

            for k, gauss in enumerate(scalings_sel):
                ellips = sph.copy()
                ellips.vertices *= gauss[None]
                ellips.visual.vertex_colors = colormap[k % len(colormap)]
                articulation = quaternion_translation_to_se3(orientations_sel[k], centers_sel[k])
                ellips.apply_transform(articulation.numpy())
                meshes.append(ellips)
            if len(meshes) == 0:
                meshes = sph.copy()
                # -1,1 to aabb
                aabb = torch.stack([centers.min(0)[0], centers.max(0)[0]], 0).cpu().numpy()
                meshes.vertices = (meshes.vertices + 1) / 2
                meshes.vertices = meshes.vertices * (aabb[1:] - aabb[:1]) + aabb[:1]
            else:
                meshes = trimesh.util.concatenate(meshes)
            return meshes
        else:
            meshes = []
            for gaussians in self.gaussians:
                mesh = gaussians.create_mesh_visualization(frameid, all_pts)
                if frameid is not None:
                    se3 = gaussians.get_extrinsics(frameid)
                    mesh.apply_transform(se3.cpu().numpy())
                meshes.append(mesh)
            return trimesh.util.concatenate(meshes)


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
            shs = torch.zeros_like(pts)
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
        elif not torch.is_tensor(scales):
            scales = torch.tensor(scales, dtype=torch.float)
        self._scaling = nn.Parameter(scales)

        rots = torch.zeros((pts.shape[0], 4))
        rots[:, 0] = 1
        self._rotation = nn.Parameter(rots)

        opacities = self.inverse_opacity_activation(
            0.9 * torch.ones((pts.shape[0], 1), dtype=torch.float)
        )
        self._opacity = nn.Parameter(opacities)

    # @property
    # def get_num_frames(self):
    #     if hasattr(self, "_trajectory"):
    #         return self._trajectory.shape[1]
    #     elif hasattr(self, "camera_mlp"):
    #         return self.gs_camera_mlp.quat.shape[0]
    #     else:
    #         return 0

    def construct_extrinsics(self, config, data_info):
        """Construct camera extrinsics module"""
        # update init camera if lab4d ckpt exists
        if self.mode=="":
            rtmat = np.eye(4)
            rtmat = np.repeat(rtmat[None], len(data_info["rtmat"][0]), axis=0)
        else:
            if config["lab4d_path"]!="":
                cams = self.lab4d_model.get_cameras()
                rtmat = cams[self.mode]
            else:
                if self.mode=="fg": 
                    trajectory_id = 1
                elif self.mode=="bg":
                    trajectory_id = 0
                else:
                    raise NotImplementedError
                rtmat = data_info["rtmat"][trajectory_id]

        frame_info = data_info["frame_info"]
        if config["extrinsics_type"] == "const":
            self.gs_camera_mlp = CameraConst(rtmat, frame_info)
        elif config["extrinsics_type"] == "explicit":
            if not config["use_init_cam"]:
                rtmat[:, :3, :3] = np.eye(3)
            self.gs_camera_mlp = CameraExplicit(rtmat, frame_info=frame_info)
        elif config["extrinsics_type"] == "image":
            if config["fg_motion"] == "image":
                self.gs_camera_mlp = TrajPredictor(
                    rtmat,
                    self._xyz.clone(),
                    self._trajectory[..., 4:].clone(),
                    data_info,
                )
            else:
                self.gs_camera_mlp = CameraPredictor(rtmat, data_info)
        else:
            raise NotImplementedError

    def construct_stat_vars(self):
        self.xyz_gradient_accum = torch.zeros((self.get_num_pts, 1), device="cuda")
        self.denom = torch.zeros((self.get_num_pts, 1), device="cuda")
        self.xyz_vis_accum = torch.zeros((self.get_num_pts, 1), device="cuda")
        self.vis_denom = torch.zeros((self.get_num_pts, 1), device="cuda")


    def update_vis_stats(self, pts_dict):
        xy_1 = pts_dict["xy_1"]
        bs = xy_1.shape[1]
        is_visible = (xy_1.abs() < 1).sum(-1) == 2
        visible_count = is_visible.sum(1, keepdims=True)

        self.vis_denom += bs
        self.xyz_vis_accum += visible_count


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
        self.xyz_vis_accum = torch.cat(
            [self.xyz_vis_accum, self.xyz_vis_accum[clone_mask]], 0
        )
        self.vis_denom = torch.cat([self.vis_denom, self.vis_denom[clone_mask]], 0)

        # then prune
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_mask]
        self.denom = self.denom[valid_mask]
        self.xyz_vis_accum = self.xyz_vis_accum[valid_mask]
        self.vis_denom = self.vis_denom[valid_mask]

        # reset
        self.xyz_gradient_accum[:] = 0
        self.denom[:] = 0
        self.xyz_vis_accum[:] = 0
        self.vis_denom[:] = 0

    def add_xyz_grad_stats(self, viewspace_point_grad, update_filter):
        """
        viewspace_point_grad: [N, 3]
        update_filter: [N], bool
        """
        self.xyz_gradient_accum[update_filter] += torch.norm(
            viewspace_point_grad[update_filter, :2], dim=-1, keepdim=True
        )
        self.denom[update_filter] += 1

    def add_xyz_vis_stats(self, xyz_vis, update_filter):
        """
        xyz_vis: [N, 1]
        update_filter: [N], bool
        """
        self.xyz_vis_accum[update_filter] += xyz_vis
        self.vis_denom[update_filter] += 1

    @property
    def get_vis_ratio(self):
        return (self.xyz_vis_accum / self.vis_denom)[...,0]

    def reset_gaussian_scale(self, scale_val=0.04):
        # reset the scale of large gaussians (for accuracy of flow rendering)
        sel_idx = (self._scaling.data > np.log(scale_val)).sum(-1)>0
        self._scaling.data[sel_idx] = np.log(scale_val)
        print("reset %d gaussian scale" % (sel_idx.sum()))

    def randomize_gaussian_center(self, portion=0.0, opacity_th=0.05, vis_portion = 0.05, aabb_ratio=0.5):
        dev = self.parameters().__next__().device
        num_pts = int(self.get_num_pts * portion)
        rand_ptsid = np.random.permutation(self.get_num_pts)[:num_pts]

        transparent_ptsid = (self.get_opacity[...,0] < opacity_th).cpu().numpy()
        transparent_ptsid = np.where(transparent_ptsid)[0]

        vis_ratio = self.get_vis_ratio.cpu().numpy()
        visibility_th = np.quantile(vis_ratio, vis_portion) # remove points with lowest 10% vis
        invis_ptsid = (vis_ratio < visibility_th)
        invis_ptsid = np.where(invis_ptsid)[0]
        sel_ptsid = np.concatenate([rand_ptsid, transparent_ptsid, invis_ptsid],0)

        # reset values
        rand_xyz = torch.rand_like(self._xyz.data[sel_ptsid])
        aabb = torch.tensor(self.get_aabb(), device=dev, dtype=torch.float32)
        aabb = extend_aabb(aabb, -aabb_ratio/2)
        rand_xyz = rand_xyz * (aabb[1:] - aabb[:1]) + aabb[:1]
        self._xyz.data[sel_ptsid] = rand_xyz

        # # randomize around proxy geometry
        # valid_pts = self.proxy_geometry.vertices
        # rand_xyz = valid_pts[np.random.randint(0, len(valid_pts), len(sel_ptsid))]
        # self._xyz.data[sel_ptsid] = torch.tensor(rand_xyz, device=dev, dtype=torch.float32)

        self._scaling.data[sel_ptsid] = np.log(0.04)
        self._opacity.data[sel_ptsid] = self.inverse_opacity_activation(torch.tensor([0.9], dtype=torch.float))


    @torch.no_grad()
    def get_aabb(self):
        # xyz = self.get_xyz()
        # aabb = (
        #     xyz.min(dim=0).values,
        #     xyz.max(dim=0).values,
        # )
        # aabb = torch.stack(aabb, dim=0).cpu().numpy()
        xyz = self.proxy_geometry.vertices
        aabb = (xyz.min(0), xyz.max(0))
        aabb = np.stack(aabb, 0)
        return aabb

    def randomize_frameid(self, frameid, frameid_2):
        if frameid is None:
            frameid = np.random.randint(self.get_num_frames - 1)
        elif torch.is_tensor(frameid):
            frameid = frameid.cpu().numpy()
        if frameid_2 is None:
            if self.is_inc_mode:
                if frameid == 0:
                    frameid_2 = frameid
                else:
                    frameid_2 = np.random.randint(frameid)  # 0-frameid-1
            else:
                frameid_2 = np.random.randint(self.get_num_frames)
        elif torch.is_tensor(frameid_2):
            frameid_2 = frameid_2.cpu().numpy()
        return frameid, frameid_2

    def get_arap_loss(self, frameid=None, frameid_2=None):
        """
        Compute arap loss with annealing.
        Begining: 100pts, 100nn
        End: 10000pts, 2nn
        num_pts * ratio_knn * num_pts = 10k
        num_pts = sq(10k / ratio_knn)
        """
        ratio_knn = self.ratio_knn
        num_pts = np.sqrt(4e6 / ratio_knn)
        if np.isinf(num_pts):
            num_pts = self.get_num_pts
        else:
            num_pts = min(self.get_num_pts, int(num_pts))  # max all pts
        num_knn = min(num_pts, max(int(ratio_knn * num_pts), 2))  # minimum 1-nn

        rand_ptsid = np.random.permutation(self.get_num_pts)[:num_pts]
        frameid, frameid_2 = self.randomize_frameid(frameid, frameid_2)
        pts0 = self.get_xyz(frameid)[rand_ptsid]
        pts1 = self.get_xyz(frameid_2)[rand_ptsid]  # N,3
        rot0 = self.get_rotation(frameid)[rand_ptsid]
        rot1 = self.get_rotation(frameid_2)[rand_ptsid]

        # dist(t,t+1)
        # print("pts: %d, knn: %d, ratio: %f" % (pts0.shape[0], num_knn, ratio_knn))
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

    def get_least_deform_loss(self, frameid=None, frameid_2=None):
        if hasattr(self, "_trajectory"):
            frameid, frameid_2 = self.randomize_frameid(frameid, frameid_2)
            # least deform loss
            xyz = self.get_xyz(frameid)
            xyz_2 = self.get_xyz(frameid_2)
            least_trans_loss = torch.norm((xyz - xyz_2), 2, -1).mean()

            rot = self.get_rotation(frameid)
            rot_2 = self.get_rotation(frameid_2)
            least_rot_loss = quaternion_to_matrix(
                quaternion_mul(rot, quaternion_conjugate(rot_2))
            )
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
            least_trans_loss = torch.norm(a_traj, 2, -1).max(-1)[0]

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
            least_rot_loss = rot_angle(least_rot_loss).max(-1)[0]
            least_action_loss = least_trans_loss + 0.01 * least_rot_loss
            return least_action_loss
        else:
            return torch.tensor(0.0, device="cuda")

    def set_future_time_params(self, last_opt_frameid):
        """
        set trajecories after last frame to be the same as last frame
        """
        # range is 0 to N-2
        if last_opt_frameid >= self.get_num_frames - 1 or last_opt_frameid < 0:
            return
        if hasattr(self, "_trajectory"):
            print("set future time params to id: ", last_opt_frameid)
            # TODO: set future params for each sequence
            last_pt = self._trajectory[:, last_opt_frameid : last_opt_frameid + 1, :]
            self._trajectory.data[:, last_opt_frameid + 1 :, :] = last_pt.data

        if hasattr(self, "camera_mlp") and isinstance(self.gs_camera_mlp, CameraExplicit):
            print("set future time params to id: ", last_opt_frameid)
            # set future params for each sequence
            frame_offset = self.gs_camera_mlp.frame_info["frame_offset"]
            for i in range(len(frame_offset) - 1):
                start_frame = frame_offset[i]
                end_frame = frame_offset[i + 1]
                last_opt_frameid_abs = last_opt_frameid + start_frame
                last_quat = self.gs_camera_mlp.quat[last_opt_frameid_abs]
                last_trans = self.gs_camera_mlp.trans[last_opt_frameid_abs]
                self.gs_camera_mlp.quat.data[last_opt_frameid_abs + 1 : end_frame] = (
                    last_quat.data
                )
                self.gs_camera_mlp.trans.data[last_opt_frameid_abs + 1 : end_frame] = (
                    last_trans.data
                )



    def get_lab4d_xyz_t(self, inst_id, frameid=None):
        dev_xyz = self._xyz.device
        dev_lab4d = self.lab4d_model.device
        field = self.lab4d_model.fields.field_params["fg"]
        samples_dict = {}
        (
            samples_dict["t_articulation"],
            samples_dict["rest_articulation"],
        ) = field.warp.articulation.get_vals_and_mean(frame_id=frameid)
        xyz = self._xyz.to(dev_lab4d)
        xyz = xyz[None, None].repeat(len(frameid), 1, 1, 1) * self.scale_field
        xyz_t_gt = field.warp(xyz, None, inst_id, samples_dict=samples_dict)
        xyz_t_gt = xyz_t_gt[:, 0] / self.scale_field  # bs, N, 3
        xyz_t_gt = xyz_t_gt.permute(1, 0, 2)  # N, bs, 3
        xyz_t_gt = xyz_t_gt.to(dev_xyz)
        return xyz_t_gt
    
    @torch.no_grad()
    def init_trajectory(self, total_frames):
        if self.mode=="fg":
            trajectory = torch.zeros(self.get_num_pts, total_frames, 7)  # quat, trans
            trajectory[:, :, 0] = 1.0

            # update init traj if lab4d ckpt exists
            if len(self.config["lab4d_path"]) != 0:
                dev = "cuda"
                frame_offsets = self.lab4d_model.data_info["frame_info"]["frame_offset"]
                for inst_id in range(0, len(frame_offsets)-1):
                    inst_id = torch.tensor([inst_id], device=dev)
                    frameid = torch.arange(frame_offsets[inst_id], frame_offsets[inst_id+1], device=dev)

                    chunk_size = 64
                    xyz_t = torch.zeros((self.get_num_pts, len(frameid), 3), device="cpu")
                    for i in range(0, len(frameid), chunk_size):
                        chunk_frameid = frameid[i : i + chunk_size]
                        xyz_t[:, i : i + chunk_size] = self.get_lab4d_xyz_t(inst_id, chunk_frameid).cpu()         
                    trajectory[:, frameid, 4:] = xyz_t - self._xyz[:, None]

            self._trajectory = nn.Parameter(trajectory)

    def update_trajectory(self, frameid):
        if self.is_leaf():
            field = self.lab4d_model.fields.field_params[self.mode]
            scale_fg = self.scale_field.detach()

            # shadow
            frameid = frameid.reshape(-1)
            if hasattr(field.camera_mlp, "time_embedding"):
                inst_id = field.camera_mlp.time_embedding.raw_fid_to_vid[frameid]
            else:
                inst_id = field.camera_mlp.time_info.raw_fid_to_vid[frameid]
            xyz_repeated = self._xyz[None].repeat(len(frameid), 1, 1)
            xyz_repeated_in = xyz_repeated[:,None] * scale_fg

            chunk_size = 32
            shadow_pred = []
            for idx in range(0, len(frameid), chunk_size):
                frameid_chunk = frameid[idx : idx + chunk_size]
                inst_id_chunk = inst_id[idx : idx + chunk_size]
                xyz_in_chunk = xyz_repeated_in[idx : idx + chunk_size]
                xyz_embed_chunk = self.pos_embedding(xyz_in_chunk)
                t_embed_chunk = self.time_embedding(frameid_chunk)
                t_embed_chunk = t_embed_chunk.view(-1, 1,1, t_embed_chunk.shape[-1])
                t_embed_chunk = t_embed_chunk.expand(xyz_in_chunk.shape[:-1] + (-1,))
                embed_chunk = torch.cat([xyz_embed_chunk, t_embed_chunk], dim=-1)
                shadow_pred_chunk = self.shadow_field(embed_chunk, inst_id_chunk)[:,0] # T, N, 1
                shadow_pred.append(shadow_pred_chunk)
            shadow_pred = torch.cat(shadow_pred, 0)

            # process
            shadow_pred = F.sigmoid(shadow_pred) * 2 # 0-2

            self.shadow_cache = {}
            for it, key in enumerate(frameid.cpu().numpy()):
                self.shadow_cache[key] = shadow_pred[it]

            # motion
            frameid_abs = frameid.clone()
            if self.use_timesync:
                frameid = self.frame_id_to_sub(frameid, inst_id)
                inst_id = inst_id.clone() * 0  # assume all videos capture the same instance

            if self.mode=="bg": return
            if isinstance(self.gs_camera_mlp, TrajPredictor):
                raise NotImplementedError
                # # image-based motion
                # _, _, motion = self.gs_camera_mlp.get_vals(frameid, xyz=self._xyz)  # N, bs, 3
                # self._trajectory.data[:, frameid, 4:] = motion
                # #TODO add delta quat
            elif self.config["fg_motion"] == "explicit":
                motion = self._trajectory[:, frameid, 4:]
                quat_delta = self._trajectory[:, frameid, :4]
            elif self.config["fg_motion"] == "rigid":
                frameid = frameid.reshape(-1)
                motion = torch.zeros(self.get_num_pts, len(frameid), 3, device="cuda")
                quat_delta = torch.zeros(self.get_num_pts, len(frameid), 4, device="cuda")
                quat_delta[:, :, 0] = 1.0
            else:
                # lab4d model (fourier basis motion)
                chunk_size = 64
                xyz_t, dq_r, dq_t = [], [], []
                for idx in range(0, len(frameid), chunk_size):
                    frameid_chunk = frameid[idx : idx + chunk_size]
                    xyz_in_chunk = xyz_repeated_in[idx : idx + chunk_size]
                    inst_id_chunk = inst_id[idx : idx + chunk_size]
                    xyz_t_chunk, warp_dict = field.warp(xyz_in_chunk, frameid_chunk, inst_id_chunk, return_aux=True)
                    xyz_t.append(xyz_t_chunk)
                    dq_r.append(warp_dict["dual_quat"][0])
                    dq_t.append(warp_dict["dual_quat"][1])
                xyz_t = torch.cat(xyz_t, 0)
                dq_r = torch.cat(dq_r, 0)
                dq_t = torch.cat(dq_t, 0)
                dual_quat = (dq_r, dq_t)

                xyz_t = xyz_t[:, 0] / scale_fg
                motion = (xyz_t - xyz_repeated).transpose(0, 1).contiguous()

                # rotataion and translation of each gaussian
                quat, _ = dual_quaternion_to_quaternion_translation(dual_quat)
                quat = quat.transpose(0, 1).contiguous()
                rot = self.rotation_activation(self._rotation)
                quat_delta = quaternion_mul(quat, quaternion_conjugate(rot)[:,None].repeat(1, quat.shape[1], 1))
            
            trajectory_pred = torch.cat((quat_delta, motion), dim=-1)
            self.trajectory_cache = {}
            for it, key in enumerate(frameid_abs.cpu().numpy()):
                self.trajectory_cache[key] = trajectory_pred[:, it]
        else:
            for gaussians in self.gaussians:
                gaussians.update_trajectory(frameid)

    # def get_lab4d_loss(self, frameid):
    #     dev = self._xyz.device
    #     frameid = frameid.view(-1)
    #     inst_id = 0
    #     inst_id = torch.tensor([inst_id], device=dev)

    #     # predicted pose
    #     w2c = self.get_extrinsics(frameid)
    #     xyz_t = self.get_xyz(frameid)  # N, bs, 3
    #     xyz = self._xyz
    #     # motion = self._trajectory[:, frameid, 4:]

    #     # pseudo ground-truth
    #     with torch.no_grad():
    #         # motion_gt = self.gs_camera_mlp.init_vals[2].to(dev)
    #         # motion_gt = motion_gt[:, frameid]
    #         # xyz_gt = self.gs_camera_mlp.init_vals[1].to(dev).detach()

    #         frame_offsets = self.lab4d_model.data_info["frame_info"]["frame_offset"]
    #         frameid = frameid + frame_offsets[inst_id]
    #         w2c_gt = self.lab4d_model.get_cameras(frameid)["fg"]
    #         w2c_gt = w2c_gt.view(*frameid.shape, 4, 4)

    #         xyz_t_gt = self.get_lab4d_xyz_t(inst_id, frameid)

    #     # loss for explicit params
    #     loss_rot = rot_angle(w2c_gt[..., :3, :3] @ w2c[..., :3, :3].permute(0, 2, 1))
    #     loss_trans = torch.norm(w2c_gt[..., :3, 3] - w2c[..., :3, 3], 2, -1)
    #     loss_xyz = torch.norm(xyz_t_gt - xyz_t, 2, -1)
    #     loss = loss_rot.mean() * 0.1 + loss_trans.mean() + loss_xyz.mean()

    #     # # loss for image basis
    #     # loss_root = F.mse_loss(w2c, w2c_gt)
    #     # loss_traj = F.mse_loss(motion, motion_gt)
    #     # loss = (loss_root + loss_traj) / 2

    #     # from geomloss import SamplesLoss

    #     # samploss = SamplesLoss(loss="sinkhorn", p=2, blur=0.002)
    #     # loss = loss + samploss(xyz, xyz_gt).mean()

    #     # sys.path.insert(
    #     #     0,
    #     #     "%s/../ppr/eval/third_party/ChamferDistancePytorch/"
    #     #     % os.path.join(os.path.dirname(__file__)),
    #     # )
    #     # from chamfer3D.dist_chamfer_3D import chamfer_3DDist

    #     # chamLoss = chamfer_3DDist()
    #     # loss_xyz_fw, loss_xyz_bw, _, _ = chamLoss(xyz_gt[None], xyz[None])
    #     # loss_xyz = (loss_xyz_fw.mean() + loss_xyz_bw.mean()) / 2 * 1000
    #     # print("loss_xyz: ", loss_xyz)
    #     # loss = loss + loss_xyz

    #     return loss