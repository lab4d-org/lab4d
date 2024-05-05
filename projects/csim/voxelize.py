# import torch
# import pdb
# from mmcv.ops import Voxelization

# voxel_holder = Voxelization(
#     max_num_points=10, voxel_size=0.1, point_cloud_range=[-20, -20, -20, 10, 10, 10]
# )
# pdb.set_trace()
# voxel_holder(torch.rand(1000, 3) * 20 - 10)

import torch
import sys, os
from torch.functional import F
import trimesh
import open3d as o3d
import numpy as np
import pdb
import time
import viser
import glob
import viser.transforms as tf
import pickle as pkl
import scipy

sys.path.insert(0, os.getcwd())
try:
    from lab4d.utils.mesh_loader import MeshLoader
    from lab4d.config import load_flags_from_file
    from lab4d.engine.trainer import Trainer
except:
    pass


class BGField:
    def __init__(self):
        # TODO delete this hack
        # load the model with good pca
        logdir = "logdir/home-2023-11-bg-adapt1/"
        opts = load_flags_from_file("%s/opts.log" % logdir)
        opts["load_suffix"] = "latest"
        opts["logroot"] = "logdir"
        _, data_info, _ = Trainer.construct_test_model(opts, return_refs=False)
        pca_fn = data_info["apply_pca_fn"]

        # load flags from file with absl
        # logdir = "logdir/home-2023-11-bg-adapt1/"  # old one
        logdir = "logdir/home-2023-curated3-compose-ft/"  # dino, but bg is too similar
        # logdir = "logdir/home-2023-curated3-compose-ft-old/"  # cse feature
        opts = load_flags_from_file("%s/opts.log" % logdir)
        opts["load_suffix"] = "latest"
        opts["logroot"] = "logdir"
        # opts["grid_size"] = 256
        opts["grid_size"] = 128
        opts["level"] = 0
        opts["vis_thresh"] = -20
        # opts["vis_thresh"] = -10
        opts["extend_aabb"] = False

        model, data_info, _ = Trainer.construct_test_model(opts, return_refs=False)
        bg_field = model.fields.field_params["bg"]
        self.bg_field = bg_field

        self.bg_meshes = {}
        self.voxel_grids = {}
        for dirpath in sorted(glob.glob("%s/export_*" % logdir)):
            inst_id = int(dirpath.split("_")[-1].split("export_")[-1])
            bgmesh_path = "%s/bg-mesh.obj" % dirpath
            # try to load mesh from dir
            # if False:
            if os.path.exists(bgmesh_path):
                print("loading mesh from %s" % bgmesh_path)
                bg_mesh = trimesh.load_mesh(bgmesh_path)
            else:
                print("extracting mesh from %s" % dirpath)
                opts["inst_id"] = inst_id
                bg_mesh = self.extract_bg_mesh(model, opts, pca_fn)
                bg_mesh.export(bgmesh_path)

            # try to load voxel from dir
            bgvoxel_path = "%s/bg-voxel.pkl" % dirpath
            # if False:
            if os.path.exists(bgvoxel_path):
                print("loading voxel from %s" % bgvoxel_path)
                bg_voxel = pkl.load(open(bgvoxel_path, "rb"))
            else:
                print("extracting voxel from %s" % dirpath)
                bg_voxel = VoxelGrid(bg_mesh)
                pkl.dump(bg_voxel, open(bgvoxel_path, "wb"))

            # save to dict
            self.bg_meshes[inst_id] = bg_mesh
            self.voxel_grids[inst_id] = bg_voxel

            # TODO remove this once we have different meshes for bg
            break

        self.voxel_grid = self.voxel_grids[0]
        self.bg_mesh = self.bg_meshes[0]

        # # TODO add obstacle
        # box = trimesh.creation.box((1, 1, 1))
        # box.apply_translation([0.3, 1, -3])
        # self.bg_mesh = trimesh.util.concatenate([self.bg_mesh, box])

        # self.root_trajs, self.cam_trajs = get_trajs_from_log("%s/export_*" % logdir)
        self.root_trajs, self.cam_trajs = get_trajs_from_log(
            "logdir/home-2023-curated3-compose-ft/export_*"
        )
        # voxel_grid.run_viser()
        self.voxel_grid.count_root_visitation(self.root_trajs[:, :3, 3])
        self.voxel_grid.count_cam_visitation(self.cam_trajs[:, :3, 3])

    def compute_feat(self, x):
        return self.bg_field.compute_feat(x)["feature"]

    def get_bg_mesh(self):
        return self.bg_mesh

    @torch.no_grad()
    def extract_bg_mesh(self, model, opts, pca_fn):
        bg_field = model.fields.field_params["bg"]
        meshes_rest = model.fields.extract_canonical_meshes(
            grid_size=opts["grid_size"],
            level=opts["level"],
            inst_id=opts["inst_id"],
            vis_thresh=opts["vis_thresh"],
            use_extend_aabb=opts["extend_aabb"],
        )

        # # rgb
        # color = bg_field.extract_canonical_color(meshes_rest["bg"])
        # meshes_rest["bg"].visual.vertex_colors = color * 255

        # # get dino feature
        # bg_feature = bg_field.extract_canonical_feature(meshes_rest["bg"], None)
        # # visualize the feature
        # # pca_fn = data_info["apply_pca_fn"]
        # # from lab4d.utils.numpy_utils import pca_numpy

        # # pdb.set_trace()
        # # pca_fn = pca_numpy(bg_feature, n_components=3)
        # bg_feature_vis = pca_fn(bg_feature, normalize=True)
        # meshes_rest["bg"].visual.vertex_colors = bg_feature_vis * 255
        # # self.bg_feature = bg_feature

        scale_bg = bg_field.logscale.exp().cpu().numpy()
        meshes_rest["bg"].apply_scale(1.0 / scale_bg)
        return meshes_rest["bg"]
        # # TODO add obstacle
        # box = trimesh.creation.box((1, 1, 1))
        # box.apply_translation([0.3, 1, -3])
        # self.bg_mesh = trimesh.util.concatenate([self.bg_mesh, box])


def get_trajs_from_log(dirpath):
    # camera trajs
    # get root trajectory
    root_trajs = []
    cam_trajs = []
    # testdirs = sorted(glob.glob("%s/export_*" % args.logdir))
    # testdirs = sorted(glob.glob("logdir/home-2023-11-compose-ft/export_*"))
    testdirs = sorted(glob.glob(dirpath))
    # testdirs = sorted(glob.glob("%s/export_*" % logdir))
    for it, loader_path in enumerate(testdirs):
        if "export_0000" in loader_path:
            continue
        root_loader = MeshLoader(loader_path, compose_mode="compose")
        # load root poses
        root_traj = root_loader.query_camtraj(data_class="fg")
        root_trajs.append(root_traj)

        # load cam poses
        cam_traj = root_loader.query_camtraj(data_class="bg")
        cam_trajs.append(cam_traj)
        print("loaded %d frames from %s" % (len(root_loader), loader_path))
    root_trajs = np.linalg.inv(np.concatenate(root_trajs))
    cam_trajs = np.linalg.inv(np.concatenate(cam_trajs))  # T1+...+TN,4,4

    return root_trajs, cam_trajs


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

    def get_data(self, mode="occupancy", opaticy=1.0):
        if mode == "occupancy":
            data = self.data
            color = np.array([0.0, 1.0, 0.0, opaticy])
        elif mode == "root_visitation":
            data = self.root_visitation
            color = np.array([1.0, 0.0, 0.0, opaticy])
        elif mode == "cam_visitation":
            data = self.cam_visitation
            color = np.array([0.0, 0.0, 1.0, opaticy])
        elif mode == "root_visitation_gradient":
            data = np.linalg.norm(self.root_visitation_gradient, 2, axis=-1)
            color = np.array([1.0, 1.0, 0.0, opaticy])
        elif mode == "root_visitation_edt":
            data = self.root_visitation_edt
            color = np.array([1.0, 1.0, 1.0, opaticy])
        elif mode == "root_visitation_edt_gradient":
            data = np.linalg.norm(self.root_visitation_edt_gradient, 2, axis=-1)
            color = np.array([1.0, 1.0, 1.0, opaticy])
        else:
            raise ValueError("mode not recognized")
        return data, color

    def to_boxes(self, mode="occupancy", opaticy=1.0):
        data, color = self.get_data(mode, opaticy)
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
                            [
                                (i + 0.5) * self.res,
                                (j + 0.5) * self.res,
                                (k + 0.5) * self.res,
                            ]
                        ),
                    )
                    colors = np.tile(color * data[i, j, k], (len(box.vertices), 1))
                    box.visual.vertex_colors = colors
                    boxes.append(box)
        boxes = trimesh.util.concatenate(boxes)
        boxes.apply_translation(self.origin)
        return boxes

    def to_pts(self):
        data, _ = self.get_data()
        grid_size = data.shape
        pts = []
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                for k in range(grid_size[2]):
                    pts.append([i * self.res, j * self.res, k * self.res])
        return np.array(pts) + self.origin

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

        if hasattr(self, "root_visitation_gradient"):
            boxes = self.to_boxes(mode="root_visitation_gradient")
            server.add_mesh_simple(
                name="/frames/root_visitation_gradient",
                vertices=boxes.vertices,
                vertex_colors=boxes.visual.vertex_colors[:, :3],
                faces=boxes.faces,
                color=None,
                opacity=0.8,
            )

        while True:
            time.sleep(10.0)

    @staticmethod
    def spatial_gradient(volume, eps=1):
        """
        x: H,W,D
        gradient: H,W,D,3
        """
        gradient_x = np.gradient(volume, axis=0)  # Gradient along width (x-axis)
        gradient_y = np.gradient(volume, axis=1)  # Gradient along height (y-axis)
        gradient_z = np.gradient(volume, axis=2)  # Gradient along depth (z-axis)
        gradient = np.stack([gradient_x, gradient_y, gradient_z], axis=-1)
        return gradient

    def compute_penetration_gradients(self, pts):
        """
        compute the penetration gradients at the points
        pts: N,3
        """
        # readout the voxel values at the points
        # gradients = self.readout_voxel(pts, mode="root_visitation_gradient")
        gradients = self.readout_voxel(pts, mode="root_visitation_edt_gradient")
        loss = self.readout_voxel(pts, mode="root_visitation")
        return loss, gradients

    def count_root_visitation(self, trajectory, splat_radius=0.2):
        # trajectory[..., 1] += 0.2  # center to foot
        self.root_visitation = self.splat_trajectory_counts(trajectory, splat_radius)
        self.root_visitation = self.root_visitation / self.root_visitation.max()
        smoothed_root_visitation = (self.root_visitation > 0).astype(float)
        # # use a gaussian kernel
        # smoothed_root_visitation = scipy.ndimage.gaussian_filter(
        #     smoothed_root_visitation, sigma=1
        # )
        # self.root_visitation_gradient = self.spatial_gradient(smoothed_root_visitation)

        self.root_visitation_edt = scipy.ndimage.distance_transform_edt(
            1 - smoothed_root_visitation
        )
        self.root_visitation_edt = 1 - (
            self.root_visitation_edt / self.root_visitation_edt.max()
        )
        self.root_visitation_edt_gradient = self.spatial_gradient(
            self.root_visitation_edt
        )

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
            # # splat
            # idx_range = int(splat_radius / res)
            # for dx in range(-idx_range, idx_range + 1):
            #     for dy in range(-idx_range, idx_range + 1):
            #         for dz in range(-idx_range, idx_range + 1):
            #             dis = np.sqrt(dx**2 + dy**2 + dz**2) * res
            #             if dis > splat_radius:
            #                 continue
            #             if (
            #                 idx + dx < 0
            #                 or idx + dx >= grid_size[0]
            #                 or idy + dy < 0
            #                 or idy + dy >= grid_size[1]
            #                 or idz + dz < 0
            #                 or idz + dz >= grid_size[2]
            #             ):
            #                 continue
            #             counts[idx + dx, idy + dy, idz + dz] += 1
            if (
                idx < 0
                or idx >= grid_size[0]
                or idy < 0
                or idy >= grid_size[1]
                or idz < 0
                or idz >= grid_size[2]
            ):
                continue
            counts[idx, idy, idz] += 1
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
        elif mode == "root_visitation_gradient":
            data = self.root_visitation_gradient
        elif mode == "root_visitation_edt":
            data = self.root_visitation_edt
        elif mode == "root_visitation_edt_gradient":
            data = self.root_visitation_edt_gradient

        if len(data.shape) == 3:
            data = data[None]  # 1,H,W,D
            value = readout_voxel_fn(data, pts, self.res, self.origin)
            value = value[0]  # N
        elif len(data.shape) == 4:
            data = data.transpose((3, 0, 1, 2))  # C,H,W,D
            value = readout_voxel_fn(data, pts, self.res, self.origin)
            value = value.T
        return value

    def readout_in_world(self, feature_vol, x_ego, ego_to_world):
        """
        x_ego: ...,KL
        ego_to_world: ...,L
        feat: ..., K, F
        """
        if isinstance(ego_to_world, tuple):
            ego_to_world_angle = ego_to_world[1]
            ego_to_world = ego_to_world[0]
        res = self.res
        origin = self.origin
        Ldim = ego_to_world.shape[-1]
        x_world = x_ego.view(x_ego.shape[:-1] + (-1, Ldim))
        if "ego_to_world_angle" in locals():
            x_world = (ego_to_world_angle[..., None, :, :] @ x_world[..., None])[..., 0]
        x_world = x_world + ego_to_world[..., None, :]
        feat = readout_features(feature_vol, x_world, res, origin)
        feat = feat.reshape(x_ego.shape[:-1] + (-1,))
        return feat

    def sample_from_voxel(self, num_samples, mode="occupancy"):
        """
        sample points from the voxel grid
        """
        if mode == "occupancy":
            data = self.data
        elif mode == "root_visitation":
            data = self.root_visitation
        elif mode == "cam_visitation":
            data = self.cam_visitation
        elif mode == "root_visitation_gradient":
            data = self.root_visitation_gradient
        else:
            raise ValueError("mode not recognized")
        # data: H,W,D
        # reshape data to 1d, normalize and sample
        grid_size = data.shape
        data = data.reshape(-1)
        data = data / data.sum()
        indices = np.random.choice(len(data), num_samples, p=data)
        indices = np.unravel_index(indices, grid_size)
        pts = np.stack([indices[0], indices[1], indices[2]], axis=-1)
        pts = pts * self.res + self.origin
        return pts


def readout_features(feature_vol, x_world, res, origin):
    """
    x_world: ...,3
    """
    # 3D convs then query B1HWD => B3HWD
    queried_feature = readout_voxel_fn(feature_vol, x_world.view(-1, 3), res, origin)
    queried_feature = queried_feature.T

    queried_feature = queried_feature.reshape(x_world.shape[:-1] + (-1,))
    return queried_feature


def readout_voxel_fn(data, pts, res, origin):
    """
    data: C,H,W,D
    pts: N,3
    """
    if not torch.is_tensor(data):
        data = torch.tensor(data, dtype=torch.float32)
        pts = torch.tensor(pts, dtype=torch.float32, device=data.device)
        is_numpy = True
    else:
        is_numpy = False

    device = data.device
    origin = torch.tensor(origin, dtype=torch.float32, device=device)

    grid_size = data.shape[1:]
    coord = (pts - origin) / res

    # coord ...,3
    # data: H,W,D

    # # NN interp
    # index = coord.astype(int)
    # for dim in range(3):
    #     index[..., dim] = np.clip(index[..., dim], 0, grid_size[dim] - 1)
    # value = data[index[..., 0], index[..., 1], index[..., 2]]

    # bilinear interpolation
    # data: N,C,D,H,W, N=1
    # coord: N,D,H,W,3, N=1, D=1, H=1
    # normalize to -1,1
    coord = coord / torch.tensor(grid_size, device=device) * 2 - 1
    # xyz to zyx
    coord = coord[..., [2, 1, 0]]
    value = F.grid_sample(data[None], coord[None, None, None], mode="bilinear")
    # value NCDHW
    value = value[0, :, 0, 0]

    if is_numpy:
        value = value.cpu().numpy()

    # import cv2

    # cv2.imwrite(
    #     "tmp/nn_mat.png",
    #     self.data[index[..., 0], index[..., 1], index[..., 2]] * 255,
    # )
    return value


if __name__ == "__main__":
    bg_field = BGField()
    bg_field.voxel_grid.run_viser()
    # mesh_path = "../vid2sim/logdir/home-2023-11-bg-adapt3/export_0001/bg-mesh.obj"
    # mesh = trimesh.load(mesh_path)

    # # get a coordinate frame

    # voxel_grid = VoxelGrid(mesh)

    # pts_score = voxel_grid.readout_voxel(mesh.vertices)
    # trimesh.Trimesh(mesh.vertices[pts_score > 0]).export("tmp/pts_score.obj")

    # voxel_grid.run_viser()
