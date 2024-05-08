import sys, os
import pdb
import numpy as np
import cv2
import torch
import trimesh
import tqdm
from scipy.spatial.transform import Rotation as R

cwd = os.getcwd()
if cwd not in sys.path:
    sys.path.insert(0, cwd)
from lab4d.engine.trainer import Trainer
from lab4d.utils.mesh_loader import MeshLoader
from lab4d.utils.vis_utils import draw_cams, get_pts_traj, get_colormap
from lab4d.utils.quat_transform import (
    dual_quaternion_to_quaternion_translation,
    dual_quaternion_to_se3,
)


class ArticulationLoader(MeshLoader):
    @torch.no_grad()
    def __init__(self, opts):
        self.raw_size = (1024, 1024)
        self.mode = "bone"
        self.compose_mode = "compose"
        self.opts = opts

        # get rest mesh
        model, data_info, ref_dict = Trainer.construct_test_model(opts)
        meshes_rest = model.fields.extract_canonical_meshes(
            grid_size=opts["grid_size"],
            level=opts["level"],
            inst_id=opts["inst_id"],
            vis_thresh=opts["vis_thresh"],
            use_extend_aabb=opts["extend_aabb"],
        )
        scale_fg = model.fields.field_params["fg"].logscale.exp().cpu().numpy()
        scale_bg = model.fields.field_params["bg"].logscale.exp().cpu().numpy()

        meshes_rest["bg"].apply_scale(1.0 / scale_bg)

        field = model.fields.field_params["fg"]
        samples_dict = {}
        samples_dict["rest_articulation"] = field.warp.articulation.get_mean_vals()

        self.field = field
        self.samples_dict = samples_dict
        self.meshes_rest = meshes_rest
        self.scale_fg = scale_fg

    @torch.no_grad()
    def load_files(self, sample):
        """
        Args:
            sample: np.array, shape (n_frames, 6 + N_JOINTS)
        """
        num_frames = len(sample)
        fake_frameid = torch.tensor(
            [self.field.frame_offset[self.opts["inst_id"]]],
            dtype=torch.long,
            device="cuda",
        )
        fake_frameid = fake_frameid.repeat(num_frames)
        inst_id = torch.tensor([self.opts["inst_id"]], dtype=torch.long, device="cuda")
        inst_id = inst_id.repeat(num_frames)
        xyz = torch.tensor(
            self.meshes_rest["fg"].vertices, dtype=torch.float32, device="cuda"
        )  # before scaling
        field = self.field
        samples_dict = self.samples_dict
        meshes_rest = self.meshes_rest
        scale_fg = self.scale_fg

        # world space keypoints
        root_to_world_list = np.eye(4).reshape(1, 4, 4).repeat(len(sample), axis=0)
        root_to_world_list[:, :3, 3] = sample[:, 3:6]
        root_to_world_list[:, :3, :3] = R.from_rotvec(sample[:, :3]).as_matrix()
        world_to_root_list = np.linalg.inv(root_to_world_list)

        mesh_roottraj = draw_cams(world_to_root_list, radius_base=0.005)
        mesh_roottraj.visual.vertex_colors = mesh_roottraj.visual.vertex_colors
        meshes_rest["bg"].visual.vertex_colors = meshes_rest["bg"].visual.vertex_colors

        # joint angles to articulations
        so3 = torch.tensor(sample[:, 6:], dtype=torch.float32, device="cuda")
        so3 = so3.view(num_frames, -1, 3)
        t_articulation = field.warp.articulation.get_vals(
            frame_id=fake_frameid, return_so3=False, override_so3=so3
        )
        scaled_t_articulation = dual_quaternion_to_se3(t_articulation)
        scaled_t_articulation = scaled_t_articulation.cpu().numpy()
        scaled_t_articulation[:, :, :3, 3] /= scale_fg

        # world space keypoints
        _, kps = dual_quaternion_to_quaternion_translation(t_articulation)
        kps = kps.cpu().numpy() / scale_fg
        kps = kps @ root_to_world_list[:, :3, :3].transpose(0, 2, 1)
        kps = kps + root_to_world_list[:, :3, 3][:, None]

        # deform mesh
        samples_dict["t_articulation"] = t_articulation
        xyz_t = field.warp(
            xyz[None, None].repeat(num_frames, 1, 1, 1),
            None,
            inst_id,
            samples_dict=samples_dict,
        )[:, 0]

        self.extr_dict = {}
        self.mesh_dict = {}
        self.bone_dict = {}
        self.t_articulation_dict = {}
        self.ghost_dict = {}
        self.scene_dict = {}
        self.pts_traj_dict = {}
        self.kps_dict = {}
        self.root_to_world = {}
        for frame_idx, fr_sample in tqdm.tqdm(enumerate(sample)):
            root_to_world = np.linalg.inv(world_to_root_list[frame_idx])
            self.root_to_world[frame_idx] = root_to_world

            # # bone mesh
            # mesh_bone = field.warp.skinning_model.draw_gaussian(
            #     (t_articulation[0][frame_idx], t_articulation[1][frame_idx]),
            #     field.warp.articulation.edges,
            # )
            # mesh_bone.apply_scale(1.0 / scale_fg)
            # mesh_bone.apply_transform(root_to_world)
            # self.bone_dict[frame_idx] = mesh_bone

            # mesh
            mesh = trimesh.Trimesh(
                vertices=xyz_t[frame_idx].cpu().numpy(),
                faces=meshes_rest["fg"].faces,
                process=False,
            )
            mesh.apply_scale(1.0 / scale_fg)
            mesh.apply_transform(root_to_world)

            # # TODO assign color based on segment id
            # # 0-64, 64-64+160*1, 64+160*1-64+160*2
            # # bucket = np.asarray([i * 160 for i in range(10)])
            # bucket = np.asarray([64 + i * 160 for i in range(10)])
            # bucket_id = np.digitize(frame_idx, bucket)

            # colormap = get_colormap()[bucket_id]
            # color = mesh.visual.vertex_colors
            # color[:, :3] = colormap
            # mesh.visual.vertex_colors = color
            self.mesh_dict[frame_idx] = mesh

            self.t_articulation_dict[frame_idx] = scaled_t_articulation[frame_idx]

            if self.compose_mode == "compose":
                if "mesh_bg" not in locals():
                    mesh_bg = trimesh.util.concatenate(
                        [meshes_rest["bg"], mesh_roottraj]
                    )
                self.scene_dict[frame_idx] = mesh_bg

            self.kps_dict[frame_idx] = kps[frame_idx]

            # pts traj
            kps_all = np.asarray(list(self.kps_dict.values())[: frame_idx + 1])
            self.pts_traj_dict[frame_idx] = get_pts_traj(kps_all, frame_idx)

            # extrinsics: observer
            # self.extr_dict[frame_idx] = self.get_following_camera(frame_idx)
            # self.extr_dict[frame_idx] = self.get_selfie_camera(frame_idx)
            # self.extr_dict[frame_idx] = self.get_identity_camera()
            self.extr_dict[frame_idx] = self.get_body_camera(frame_idx)

        # intrinsics
        self.intrinsics = np.zeros((len(self.extr_dict), 4))
        self.intrinsics[:, 0] = 1
        self.intrinsics[:, 1] = 1
        self.intrinsics[:, 2] = self.raw_size[1] / 2
        self.intrinsics[:, 3] = self.raw_size[0] / 2

        # compute aabb
        # self.aabb_min = np.min(sample[:, 3:6], axis=0)
        # self.aabb_max = np.max(sample[:, 3:6], axis=0)
        bounds = self.meshes_rest["bg"].bounds
        self.aabb_min = bounds[0, [0, 2]]
        self.aabb_max = bounds[1, [0, 2]]

    @torch.no_grad()
    def load_files_simple(self, sample):
        """
        Args:
            sample: np.array, shape (n_frames, 6 + N_JOINTS)
        """
        num_frames = len(sample)
        field = self.field

        fake_frameid = torch.tensor(
            [field.frame_offset[self.opts["inst_id"]]],
            dtype=torch.long,
            device="cuda",
        )
        fake_frameid = fake_frameid.repeat(num_frames)
        scale_fg = self.scale_fg

        # joint angles to articulations
        so3 = torch.tensor(sample[:, 6:], dtype=torch.float32, device="cuda")
        so3 = so3.view(num_frames, -1, 3)
        t_articulation = field.warp.articulation.get_vals(
            frame_id=fake_frameid, return_so3=False, override_so3=so3
        )
        _, kps = dual_quaternion_to_quaternion_translation(t_articulation)
        kps = kps[0].cpu().numpy() / scale_fg

        # world space keypoints
        root_to_world_list = np.eye(4).reshape(1, 4, 4).repeat(len(sample), axis=0)
        root_to_world_list[:, :3, 3] = sample[:, 3:6]
        root_to_world_list[:, :3, :3] = R.from_rotvec(sample[:, :3]).as_matrix()

        kps = kps @ root_to_world_list[:, :3, :3].transpose(0, 2, 1)
        kps = kps + root_to_world_list[:, :3, 3][:, None]
        return kps

    def get_max_extend_abs(self):
        return max(max(np.abs(self.aabb_max)), max(np.abs(self.aabb_min)))

    def __len__(self):
        return len(self.extr_dict)
