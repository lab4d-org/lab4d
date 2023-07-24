import torch
import torch.nn as nn
from lab4d.nnutils.nerf import NeRF

import trimesh
from pysdf import SDF

from lab4d.utils.quat_transform import quaternion_translation_to_se3
from lab4d.utils.geom_utils import get_near_far
from lab4d.nnutils.base import MixMLP, MultiMLP, CondMLP
from lab4d.nnutils.visibility import VisField


class BGNeRF(NeRF):
    """A static neural radiance field with an MLP backbone. Specialized to background."""

    # def __init__(self, data_info, field_arch=CondMLP, D=5, W=128, **kwargs):
    def __init__(self, data_info, field_arch=MixMLP, D=1, W=64, **kwargs):
        super(BGNeRF, self).__init__(
            data_info, field_arch=field_arch, D=D, W=W, **kwargs
        )
        # TODO: update per-scene beta
        # TODO: update per-scene scale

    def init_proxy(self, geom_paths, init_scale):
        """Initialize the geometry from a mesh

        Args:
            geom_path (Listy(str)): Initial shape mesh
            init_scale (float): Geometry scale factor
        """
        meshes = []
        for geom_path in geom_paths:
            mesh = trimesh.load(geom_path)
            mesh.vertices = mesh.vertices * init_scale
            meshes.append(mesh)
        self.proxy_geometry = meshes

    def get_proxy_geometry(self):
        """Get proxy geometry

        Returns:
            proxy_geometry (Trimesh): Proxy geometry
        """
        return self.proxy_geometry[0]

    def init_aabb(self):
        """Initialize axis-aligned bounding box"""
        self.register_buffer("aabb", torch.zeros(len(self.proxy_geometry), 2, 3))
        self.update_aabb(beta=0)

    def get_init_sdf_fn(self):
        """Initialize signed distance function from mesh geometry

        Returns:
            sdf_fn_torch (Function): Signed distance function
        """

        def sdf_fn_torch_sphere(pts):
            radius = 0.1
            # l2 distance to a unit sphere
            dis = (pts).pow(2).sum(-1, keepdim=True)
            sdf = torch.sqrt(dis) - radius  # negative inside, postive outside
            return sdf

        return sdf_fn_torch_sphere

    def update_proxy(self):
        """Extract proxy geometry using marching cubes"""
        for inst_id in range(self.num_inst):
            mesh = self.extract_canonical_mesh(level=0.005, inst_id=inst_id)
            if len(mesh.vertices) > 0:
                self.proxy_geometry[inst_id] = mesh

    def get_aabb(self, inst_id=None):
        """Get axis-aligned bounding box
        Args:
            inst_id: (N,) Instance id
        Returns:
            aabb: (2,3) Axis-aligned bounding box if inst_id is None, (N,2,3) otherwise
        """
        if inst_id is None:
            return self.aabb.mean(0)
        return self.aabb[inst_id]

    def update_aabb(self, beta=0.9):
        """Update axis-aligned bounding box by interpolating with the current
        proxy geometry's bounds

        Args:
            beta (float): Interpolation factor between previous/current values
        """
        device = self.aabb.device
        for inst_id in range(self.num_inst):
            bounds = self.proxy_geometry[inst_id].bounds
            if bounds is not None:
                aabb = torch.tensor(bounds, dtype=torch.float32, device=device)
                self.aabb[inst_id] = self.aabb[inst_id] * beta + aabb * (1 - beta)

    def update_near_far(self, beta=0.9):
        """Update near-far bounds by interpolating with the current near-far bounds

        Args:
            beta (float): Interpolation factor between previous/current values
        """
        device = next(self.parameters()).device
        with torch.no_grad():
            quat, trans = self.camera_mlp.get_vals()  # (B, 4, 4)
            rtmat = quaternion_translation_to_se3(quat, trans)

        frame_id_all = list(range(self.num_frames))
        frame_offset = self.frame_offset
        near_far_all = []
        for inst_id in range(self.num_inst):
            verts = self.proxy_geometry[inst_id].vertices
            frame_id = frame_id_all[frame_offset[inst_id] : frame_offset[inst_id + 1]]
            proxy_pts = torch.tensor(verts, dtype=torch.float32, device=device)
            near_far = get_near_far(proxy_pts, rtmat[frame_id]).to(device)
            near_far_all.append(
                self.near_far[frame_id].data * beta + near_far * (1 - beta)
            )
        self.near_far.data = torch.cat(near_far_all, 0)

    def get_near_far(self, frame_id, field2cam):
        device = next(self.parameters()).device
        frame_id_all = list(range(self.num_frames))
        frame_offset = self.frame_offset
        field2cam_mat = quaternion_translation_to_se3(field2cam[0], field2cam[1])

        near_far_all = []
        for inst_id in range(self.num_inst):
            frame_id_sel = frame_id_all[
                frame_offset[inst_id] : frame_offset[inst_id + 1]
            ]
            # find the overlap of frame_id and frame_id_sel
            id_sel = [i for i, x in enumerate(frame_id) if x in frame_id_sel]
            if len(id_sel) == 0:
                continue
            corners = trimesh.bounds.corners(self.proxy_geometry[inst_id].bounds)
            corners = torch.tensor(corners, dtype=torch.float32, device=device)
            near_far = get_near_far(corners, field2cam_mat[id_sel], tol_fac=1.5)
            near_far_all.append(near_far)
        near_far = torch.cat(near_far_all, 0)
        return near_far
