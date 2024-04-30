# Copyright (c) 2023 Gengshan Yang, Carnegie Mellon University.
import json
import glob
import numpy as np
import pdb
import os
import cv2
import argparse
import trimesh
import tqdm
import configparser
import matplotlib

from lab4d.utils.vis_utils import draw_cams, get_pts_traj
from lab4d.utils.quat_transform import dual_quaternion_to_se3


class MeshLoader:
    def __init__(self, testdir, mode="", compose_mode=""):
        # io
        camera_info = json.load(open("%s/camera.json" % (testdir), "r"))
        intrinsics = np.asarray(camera_info["intrinsics"], dtype=np.float32)
        raw_size = camera_info["raw_size"]  # h,w
        if os.path.exists("%s/fg/motion.json" % (testdir)):
            primary_dir = "%s/fg" % testdir
            secondary_dir = "%s/bg" % testdir
        else:
            primary_dir = "%s/bg" % testdir
            secondary_dir = "%s/fg" % testdir  # never use fg for secondary
        path_list = sorted([i for i in glob.glob("%s/mesh/*.obj" % (primary_dir))])
        if len(path_list) == 0:
            print("Warning: no mesh found that matches %s*" % (primary_dir))
            # raise ValueError

        # check render mode
        if mode != "":
            pass
        elif len(glob.glob("%s/bone/*" % primary_dir)) > 0:
            mode = "bone"
        else:
            mode = "shape"

        if compose_mode != "":
            pass
        elif len(glob.glob("%s/mesh/*" % secondary_dir)) > 0:
            compose_mode = "compose"
        else:
            compose_mode = "primary"

        # get cam dict
        field2cam_fg_dict = json.load(open("%s/motion.json" % (primary_dir), "r"))
        if "t_articulation" in field2cam_fg_dict:
            self.t_articulation = field2cam_fg_dict["t_articulation"]
        field2cam_fg_dict = field2cam_fg_dict["field2cam"]
        if compose_mode == "compose":
            field2cam_bg_dict = json.load(open("%s/motion.json" % (secondary_dir), "r"))
            field2cam_bg_dict = field2cam_bg_dict["field2cam"]

            field2world_path = "%s/bg/field2world.json" % (testdir)
            if os.path.exists(field2world_path):
                field2world = np.asarray(json.load(open(field2world_path, "r")))
            else:
                field2world = np.eye(4)
            world2field = np.linalg.inv(field2world)

        self.mode = mode
        self.compose_mode = compose_mode
        self.testdir = testdir
        self.intrinsics = intrinsics
        self.raw_size = raw_size
        self.path_list = path_list
        self.field2cam_fg_dict = field2cam_fg_dict
        if compose_mode == "compose":
            self.field2cam_bg_dict = field2cam_bg_dict
            self.field2world = field2world
            self.world2field = world2field
        else:
            self.field2cam_bg_dict = None
            self.field2world = None
            self.world2field = None

    def __len__(self):
        return len(self.field2cam_fg_dict)

    def load_files(self, ghosting=False):
        mode = self.mode
        compose_mode = self.compose_mode
        path_list = self.path_list
        field2cam_fg_dict = self.field2cam_fg_dict
        field2cam_bg_dict = self.field2cam_bg_dict
        field2world = self.field2world
        world2field = self.world2field

        mesh_dict = {}
        extr_dict = {}
        bone_dict = {}
        scene_dict = {}
        ghost_dict = {}
        kps_dict = {}
        pts_traj_dict = {}
        root_to_world = {}
        t_articulation_dict = {}
        aabb_min = np.asarray([np.inf, np.inf])
        aabb_max = np.asarray([-np.inf, -np.inf])
        for counter in range(self.__len__()):
            frame_idx = int(list(self.field2cam_fg_dict.keys())[counter])
            fid_str = str(frame_idx)
            if counter > 0 and len(path_list) == 1:
                pass
            else:
                mesh_path = path_list[counter]
                mesh = trimesh.load(mesh_path, process=False)
                if mesh.visual.kind == "vertex":
                    mesh.visual.vertex_colors = (
                        mesh.visual.vertex_colors
                    )  # visual.kind = 'vertex'

            field2cam_fg = np.asarray(field2cam_fg_dict[fid_str])

            # post-modify the scale of the fg
            # mesh.vertices = mesh.vertices / 2
            # field2cam_fg[:3, 3] = field2cam_fg[:3, 3] / 2

            mesh_dict[frame_idx] = mesh
            extr_dict[frame_idx] = field2cam_fg

            if mode == "bone":
                # load bone
                bone_path = mesh_path.replace("mesh", "bone")
                bone = trimesh.load(bone_path, process=False)
                bone.visual.vertex_colors = bone.visual.vertex_colors
                bone_dict[frame_idx] = bone

            if compose_mode == "compose":
                scene_path = mesh_path.replace("fg/mesh", "bg/mesh")
                if counter > 0 and os.path.exists(scene_path) == False:
                    pass
                else:
                    # load scene
                    scene = trimesh.load(scene_path, process=False)
                    scene.visual.vertex_colors = scene.visual.vertex_colors

                # align bg floor with xz plane
                scene_t = scene.copy()
                scene_t.vertices = (
                    scene_t.vertices @ field2world[:3, :3].T + field2world[:3, 3]
                )
                field2cam_bg = field2cam_bg_dict[fid_str] @ world2field
                field2cam_bg_dict[fid_str] = field2cam_bg

                scene_dict[frame_idx] = scene_t
                # use scene camera
                extr_dict[frame_idx] = field2cam_bg_dict[fid_str]
                # transform to scene
                object_to_scene = (
                    np.linalg.inv(field2cam_bg_dict[fid_str]) @ field2cam_fg
                )
                mesh_dict[frame_idx].apply_transform(object_to_scene)

                t_articulation = np.asarray(self.t_articulation[fid_str])
                if mode == "bone":
                    bone_dict[frame_idx].apply_transform(object_to_scene)

                    # world space keypoints
                    kps = t_articulation[:, :3, 3]
                    kps = kps @ object_to_scene[:3, :3].T + object_to_scene[:3, 3]
                    kps_dict[frame_idx] = kps

                    # pts traj
                    kps_all = np.asarray(list(kps_dict.values()))
                    pts_traj_dict[frame_idx] = get_pts_traj(kps_all, counter)
                root_to_world[frame_idx] = object_to_scene
                t_articulation_dict[frame_idx] = t_articulation

                if ghosting:
                    total_ghost = 10
                    ghost_skip = len(path_list) // total_ghost
                    if "ghost_list" in locals():
                        if counter % ghost_skip == 0:
                            mesh_ghost = mesh_dict[frame_idx].copy()
                            mesh_ghost.visual.vertex_colors[:, 3] = 102
                            ghost_list.append(mesh_ghost)
                    else:
                        ghost_list = [mesh_dict[frame_idx]]
                    ghost_dict[frame_idx] = [mesh.copy() for mesh in ghost_list]

            # update aabb # x,z coords
            if compose_mode == "compose":
                bounds = scene_dict[frame_idx].bounds
            else:
                bounds = mesh_dict[frame_idx].bounds
            aabb_min = np.minimum(aabb_min, bounds[0, [0, 2]])
            aabb_max = np.maximum(aabb_max, bounds[1, [0, 2]])

        self.mesh_dict = mesh_dict
        self.extr_dict = extr_dict
        self.bone_dict = bone_dict
        self.scene_dict = scene_dict
        self.ghost_dict = ghost_dict
        self.pts_traj_dict = pts_traj_dict
        self.root_to_world = root_to_world
        self.t_articulation_dict = t_articulation_dict
        self.aabb_min = aabb_min
        self.aabb_max = aabb_max

    def query_frame(self, frame_idx):
        input_dict = {}
        input_dict["shape"] = self.mesh_dict[frame_idx]
        if self.mode == "bone":
            input_dict["bone"] = self.bone_dict[frame_idx]
            # make shape transparent and gray
            input_dict["shape"].visual.vertex_colors[:3] = 102
            input_dict["shape"].visual.vertex_colors[3:] = 128
        if self.compose_mode == "compose":
            scene_mesh = self.scene_dict[frame_idx]
            # scene_mesh.visual.vertex_colors[:, :3] = np.asarray([[224, 224, 54]])
            # XYZ color
            xyz = scene_mesh.vertices
            xyz = (xyz - xyz.min(0)) / (xyz.max(0) - xyz.min(0))
            scene_mesh.visual.vertex_colors[:, :3] = xyz * 255
            input_dict["scene"] = scene_mesh
        if len(self.ghost_dict) > 0:
            ghost_mesh = trimesh.util.concatenate(self.ghost_dict[frame_idx])
            input_dict["ghost"] = ghost_mesh

        # pts trajectory: TxNx2x3
        if len(self.pts_traj_dict.keys()) > 0:
            pts_traj, pts_color = self.pts_traj_dict[frame_idx]
            input_dict["pts_traj"], input_dict["pts_color"] = pts_traj, pts_color

        # add camera
        cam_mesh = trimesh.creation.axis(axis_length=0.2)
        cam_mesh.apply_transform(np.linalg.inv(self.extr_dict[frame_idx]))
        input_dict["camera"] = cam_mesh
        return input_dict

    def print_info(self):
        print(
            "[mode=%s, compose=%s] rendering %d meshes from %s"
            % (self.mode, self.compose_mode, len(self), self.testdir)
        )

    def query_canonical_mesh(self, inst_id, data_class="bg"):
        path = self.testdir + "/../export_%04d/%s-mesh.obj" % (inst_id, data_class)
        if os.path.exists(path):
            mesh = trimesh.load(path, process=False)
            if data_class == "bg":
                field2world_path = (
                    self.testdir + "/../export_%04d/bg/field2world.json" % inst_id
                )
                if os.path.exists(field2world_path):
                    field2world = np.asarray(json.load(open(field2world_path, "r")))
                    mesh.vertices = (
                        mesh.vertices @ field2world[:3, :3].T + field2world[:3, 3]
                    )
        else:
            mesh = trimesh.Trimesh()
        return mesh

    def query_camtraj_mesh(self, data_class="bg"):
        world2cam_bg = np.asarray(list(self.field2cam_bg_dict.values()))
        field2cam_fg = np.asarray(list(self.field2cam_fg_dict.values()))
        if data_class == "bg":
            mesh = draw_cams(world2cam_bg, color="cool", radius_base=0.005)
        elif data_class == "fg":
            world2field_fg = np.linalg.inv(field2cam_fg) @ world2cam_bg
            mesh = draw_cams(world2field_fg, color="hot", radius_base=0.005)
        else:
            mesh = trimesh.Trimesh()
        # path = self.testdir + "/../export_%04d/%s-camtraj.obj" % (inst_id, data_class)
        # if os.path.exists(path):
        #     mesh = trimesh.load(path, process=False)
        #     if data_class == "bg":
        #         field2world_path = (
        #             self.testdir + "/../export_%04d/bg/field2world.json" % inst_id
        #         )
        #         if os.path.exists(field2world_path):
        #             field2world = np.asarray(json.load(open(field2world_path, "r")))
        #             mesh.vertices = (
        #                 mesh.vertices @ field2world[:3, :3].T + field2world[:3, 3]
        #             )
        # else:
        #     mesh = trimesh.Trimesh()
        return mesh

    def query_camtraj(self, data_class="bg"):
        world2cam_bg = np.asarray(list(self.field2cam_bg_dict.values()))
        field2cam_fg = np.asarray(list(self.field2cam_fg_dict.values()))
        if data_class == "bg":
            cam = world2cam_bg
        elif data_class == "fg":
            cam = np.linalg.inv(field2cam_fg) @ world2cam_bg
        else:
            raise ValueError
        return cam

    def find_seqname(self):
        testdir = self.testdir
        parts = [part for part in testdir.split("/") if part]
        logdir = "/".join(parts[:2])
        logdir = os.path.join(logdir, "opts.log")
        with open(logdir, "r") as file:
            for line in file:
                if "--seqname" in line:
                    seqname = line.split("--seqname=")[1].strip()
                    break
        if "seqname" not in locals():
            raise ValueError("Could not find seqname in opts.log")
        inst_id = int(parts[2].split("_")[-1])
        print("seqname: %s, inst_id: %d" % (seqname, inst_id))
        return seqname, inst_id

    def load_rgb(self, downsample_factor):
        seqname, inst_id = self.find_seqname()
        config = configparser.RawConfigParser()
        config.read("database/configs/%s.config" % seqname)
        img_dir = config.get("data_%d" % inst_id, "img_path")
        print("Loading images from %s" % img_dir)
        rgb_list = [
            cv2.imread(path) for path in sorted(glob.glob("%s/*.jpg" % img_dir))
        ]
        # downsample to around 320x240: 1080/16=67.5, 1920/16=120
        rgb_list = [
            rgb[::downsample_factor, ::downsample_factor, ::-1] for rgb in rgb_list
        ]
        return rgb_list

    def get_following_camera(self, frame_idx, dist=2):
        root_to_world = self.root_to_world[frame_idx]
        root_to_observer = np.eye(4)
        root_to_observer[:3, 3] = np.array([0, 0, dist])
        root_to_observer[:3, :3] = cv2.Rodrigues(np.array([0, np.pi, 0]))[0]
        world_to_observer = root_to_observer @ np.linalg.inv(root_to_world)
        return world_to_observer

    def get_selfie_camera(self, frame_idx, dist=1):
        root_to_world = self.root_to_world[frame_idx]
        root_to_observer = np.eye(4)
        root_to_observer[:3, 3] = np.array([0, 0, dist])
        world_to_observer = root_to_observer @ np.linalg.inv(root_to_world)
        return world_to_observer

    def get_body_camera(self, frame_idx, part_name="head"):
        if part_name == "head":
            part_idx = 3
            delta_mat = np.eye(4)
            delta_mat[:3, :3] = (
                cv2.Rodrigues(np.array([-np.pi / 4, 0, 0]))[0]
                @ cv2.Rodrigues(np.array([0, 0, np.pi]))[0]
            )

        observer_to_root = self.t_articulation_dict[frame_idx][part_idx]
        root_to_world = self.root_to_world[frame_idx]
        observer_to_world = root_to_world @ observer_to_root
        world_to_observer = delta_mat @ np.linalg.inv(observer_to_world)
        return world_to_observer

    def get_identity_camera(self):
        return np.eye(4)
