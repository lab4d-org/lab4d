# import torch
# import pdb
# from mmcv.ops import Voxelization

# voxel_holder = Voxelization(
#     max_num_points=10, voxel_size=0.1, point_cloud_range=[-20, -20, -20, 10, 10, 10]
# )
# pdb.set_trace()
# voxel_holder(torch.rand(1000, 3) * 20 - 10)

import trimesh
import open3d as o3d
import numpy as np
import pdb
import time
import viser
import viser.transforms as tf


class VoxelGrid:
    def __init__(self, mesh, res=0.1, ego_box=None):
        self.data, self.origin = self.trimesh_to_voxel(mesh, res, ego_box)
        self.res = res
        self.mesh = mesh

        # mesh.export("tmp/before.obj")
        # self.to_boxes().export("tmp/after.obj")

    @staticmethod
    def trimesh_to_voxel(mesh, voxel_size, ego_box=None):
        if ego_box is not None:
            # cut around the box
            box = trimesh.creation.box(extents=[ego_box, ego_box, ego_box])
            mesh = mesh.slice_plane(box.facets_origin, -box.facets_normal)

        # print("voxelization")
        mesh_o3d = o3d.geometry.TriangleMesh()
        mesh_o3d.vertices = o3d.utility.Vector3dVector(mesh.vertices)
        mesh_o3d.triangles = o3d.utility.Vector3iVector(mesh.faces)
        voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(
            mesh_o3d, voxel_size=voxel_size
        )
        # o3d.visualization.draw_geometries([voxel_grid])
        voxels = voxel_grid.get_voxels()  # returns list of voxels
        if len(voxels) == 0:
            indices = np.zeros((0, 3))
            colors = np.zeros((0, 3))
            origin = np.zeros(3)
        else:
            indices = np.stack(list(vx.grid_index for vx in voxels))
            colors = np.stack(list(vx.color for vx in voxels))
            origin = mesh.bounding_box.bounds[0]

        if ego_box is not None:
            tensor_size = 2 * int(np.ceil(ego_box / voxel_size))
            tensor = np.zeros((tensor_size, tensor_size, tensor_size))
        else:
            tensor = np.zeros(indices.max(0) + 1)
        for idx in indices:
            tensor[idx[0], idx[1], idx[2]] = 1

        return tensor, origin

    def to_boxes(self, mode="occupancy"):
        if mode == "occupancy":
            data = self.data
            color = np.array([0.0, 1.0, 0.0, 1.0])
        elif mode == "root_visitation":
            data = self.root_visitation
            color = np.array([1.0, 0.0, 0.0, 1.0])
        elif mode == "cam_visitation":
            data = self.cam_visitation
            color = np.array([0.0, 0.0, 1.0, 1.0])
        else:
            raise ValueError("mode not recognized")
        if data.sum() == 0:
            return trimesh.Trimesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3)))
        boxes = []
        grid_size = data.shape
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                for k in range(grid_size[2]):
                    if data[i, j, k] == 0:
                        continue
                    box = trimesh.creation.box(
                        extents=[self.res, self.res, self.res],
                        transform=trimesh.transformations.translation_matrix(
                            [i * self.res, j * self.res, k * self.res]
                        ),
                    )
                    colors = np.tile(color * data[i, j, k], (len(box.vertices), 1))
                    box.visual.vertex_colors = colors
                    boxes.append(box)
        boxes = trimesh.util.concatenate(boxes)
        boxes.apply_translation(self.origin)
        return boxes

    def run_viser(self):
        # visualizations
        server = viser.ViserServer()
        # Setup root frame
        server.add_frame(
            "/frames",
            wxyz=tf.SO3.exp(np.array([-np.pi / 2, 0.0, 0.0])).wxyz,
            position=(0, 0, 0),
            show_axes=False,
        )

        server.add_mesh_trimesh(
            name="/frames/environment",
            mesh=self.mesh,
        )

        boxes = self.to_boxes()
        server.add_mesh_simple(
            name="/frames/boxes",
            vertices=boxes.vertices,
            vertex_colors=boxes.visual.vertex_colors[:, :3],
            faces=boxes.faces,
            color=None,
            opacity=0.8,
        )

        if hasattr(self, "root_visitation"):
            boxes = self.to_boxes(mode="root_visitation")
            server.add_mesh_simple(
                name="/frames/root_visitation",
                vertices=boxes.vertices,
                vertex_colors=boxes.visual.vertex_colors[:, :3],
                faces=boxes.faces,
                color=None,
                opacity=0.8,
            )

        if hasattr(self, "cam_visitation"):
            boxes = self.to_boxes(mode="cam_visitation")
            server.add_mesh_simple(
                name="/frames/cam_visitation",
                vertices=boxes.vertices,
                vertex_colors=boxes.visual.vertex_colors[:, :3],
                faces=boxes.faces,
                color=None,
                opacity=0.8,
            )

        while True:
            time.sleep(10.0)

    def count_root_visitation(self, trajectory, splat_radius=0.2):
        trajectory[..., 1] += 0.2  # center to foot
        self.root_visitation = self.splat_trajectory_counts(trajectory, splat_radius)
        self.root_visitation = self.root_visitation / self.root_visitation.max()

    def count_cam_visitation(self, trajectory, splat_radius=0.2):
        self.cam_visitation = self.splat_trajectory_counts(trajectory, splat_radius)
        self.cam_visitation = self.cam_visitation / self.cam_visitation.max()

    def splat_trajectory_counts(self, trajectory, splat_radius=0.3):
        """
        splat the trajectory counts into the voxel grid
        trajectory: T,3
        grid_size: 3
        """
        res = self.res
        grid_size = self.data.shape
        counts = np.zeros_like(self.data)

        # for i in range(trajectory.shape[0]):
        #     # find voxel indices within splat_radius to trajectory[i]
        #     idx, idy, idz = np.floor((trajectory[i] - self.origin) / res).astype(int)
        #     if (
        #         idx < 0
        #         or idx >= grid_size[0]
        #         or idy < 0
        #         or idy >= grid_size[1]
        #         or idz < 0
        #         or idz >= grid_size[2]
        #     ):
        #         continue
        #     counts[idx, idy, idz] += 1

        # # TODO replace with torch convolution
        # import torch
        # import torch.functional as F

        # counts = torch.tensor(counts)
        # gaussian_kernel = torch.tensor(
        #     [[[1, 2, 1], [2, 4, 2], [1, 2, 1]]], dtype=torch.float32
        # )
        # gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
        # counts = F.conv3d(
        #     counts.unsqueeze(0).unsqueeze(0), gaussian_kernel.unsqueeze(0).unsqueeze(0)
        # )
        # counts = counts.squeeze(0).squeeze(0).numpy()

        for i in range(trajectory.shape[0]):
            # find voxel indices within splat_radius to trajectory[i]
            idx, idy, idz = np.floor((trajectory[i] - self.origin) / res).astype(int)
            # splat
            idx_range = int(splat_radius / res)
            for dx in range(-idx_range, idx_range + 1):
                for dy in range(-idx_range, idx_range + 1):
                    for dz in range(-idx_range, idx_range + 1):
                        dis = np.sqrt(dx**2 + dy**2 + dz**2) * res
                        if dis > splat_radius:
                            continue
                        if (
                            idx + dx < 0
                            or idx + dx >= grid_size[0]
                            or idy + dy < 0
                            or idy + dy >= grid_size[1]
                            or idz + dz < 0
                            or idz + dz >= grid_size[2]
                        ):
                            continue
                        counts[idx + dx, idy + dy, idz + dz] += 1
        return counts

    def readout_voxel(self, pts, mode="occupancy"):
        """
        readout the voxel values at the points
        pts: N,3
        """
        if mode == "occupancy":
            data = self.data
        elif mode == "root_visitation":
            data = self.root_visitation
        elif mode == "cam_visitation":
            data = self.cam_visitation

        # TODO: bilinear interpolation
        res = self.res
        grid_size = self.data.shape
        index = ((pts - self.origin) / res).astype(int)
        for dim in range(3):
            index[..., dim] = np.clip(index[..., dim], 0, grid_size[dim] - 1)
        # index: ...,3
        # data: H,W,D
        value = data[index[..., 0], index[..., 1], index[..., 2]]
        import cv2

        cv2.imwrite(
            "tmp/nn_mat.png",
            self.data[index[..., 0], index[..., 1], index[..., 2]] * 255,
        )
        return value


def __main__():
    mesh_path = "../vid2sim/logdir/home-2023-11-bg-adapt3/export_0001/bg-mesh.obj"
    mesh = trimesh.load(mesh_path)

    # get a coordinate frame

    voxel_grid = VoxelGrid(mesh)
    voxel_grid.run_viser()
