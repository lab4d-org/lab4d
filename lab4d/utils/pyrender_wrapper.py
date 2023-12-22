# Copyright (c) 2023 Gengshan Yang, Carnegie Mellon University.

import os
import numpy as np
import cv2
import pdb
import pyrender
import trimesh
from pyrender import (
    IntrinsicsCamera,
    Mesh,
    Node,
    Scene,
    OffscreenRenderer,
    MetallicRoughnessMaterial,
)

from lab4d.utils.cam_utils import depth_to_xyz

os.environ["PYOPENGL_PLATFORM"] = "egl"


class PyRenderWrapper:
    def __init__(self, image_size=(1024, 1024)) -> None:
        # renderer
        render_size = max(image_size)
        self.r = OffscreenRenderer(render_size, render_size)
        self.intrinsics = IntrinsicsCamera(
            render_size, render_size, render_size / 2, render_size / 2
        )
        self.image_size = image_size
        self.render_size = render_size
        # light
        self.light_pose = np.eye(4)
        self.set_light_topdown()
        self.direc_l = pyrender.DirectionalLight(color=np.ones(3), intensity=5.0)
        self.ambient_light = 0.1 * np.asarray([1.0, 1.0, 1.0, 1.0])
        self.material = MetallicRoughnessMaterial(
            roughnessFactor=0.75, metallicFactor=0.75, alphaMode="BLEND"
        )
        self.init_camera()

    def init_camera(self):
        # cv to gl coords
        self.flip_pose = -np.eye(4)
        self.flip_pose[0, 0] = 1
        self.flip_pose[-1, -1] = 1
        self.set_camera(np.eye(4))

    def set_ambient_light(self):
        self.direc_l = pyrender.DirectionalLight(color=np.ones(3), intensity=0.0)
        self.ambient_light = 1.0 * np.asarray([1.0, 1.0, 1.0, 1.0])
        self.material = None

    def set_camera_bev(self, depth, gl=False):
        # object to camera transforms
        if gl:
            rot = cv2.Rodrigues(np.asarray([-np.pi / 2, 0, 0]))[0]
        else:
            rot = cv2.Rodrigues(np.asarray([np.pi / 2, 0, 0]))[0]
        scene_to_cam = np.eye(4)
        scene_to_cam[:3, :3] = rot
        scene_to_cam[2, 3] = depth
        self.scene_to_cam = self.flip_pose @ scene_to_cam

    def set_camera_frontal(self, depth, gl=False, delta=0.0):
        # object to camera transforms
        if gl:
            rot = cv2.Rodrigues(np.asarray([np.pi + np.pi / 180, delta, 0]))[0]
        else:
            rot = cv2.Rodrigues(np.asarray([np.pi / 180, delta, 0]))[0]
        scene_to_cam = np.eye(4)
        scene_to_cam[:3, :3] = rot
        scene_to_cam[2, 3] = depth
        self.scene_to_cam = self.flip_pose @ scene_to_cam

    def set_camera(self, scene_to_cam):
        # object to camera transforms
        self.scene_to_cam = self.flip_pose @ scene_to_cam

    def set_light_topdown(self, gl=False):
        # top down light, slightly closer to the camera
        if gl:
            rot = cv2.Rodrigues(np.asarray([-np.pi / 2, 0, 0]))[0]
        else:
            rot = cv2.Rodrigues(np.asarray([np.pi / 2, 0, 0]))[0]
        self.light_pose[:3, :3] = rot

    def align_light_to_camera(self):
        self.light_pose = np.linalg.inv(self.scene_to_cam)

    def set_intrinsics(self, intrinsics):
        """
        Args:
            intrinsics: (4,) fx,fy,px,py
        """
        self.intrinsics = IntrinsicsCamera(
            intrinsics[0], intrinsics[1], intrinsics[2], intrinsics[3]
        )

    def get_intrinsics(self):
        return np.asarray(
            [
                self.intrinsics.fx,
                self.intrinsics.fy,
                self.intrinsics.cx,
                self.intrinsics.cy,
            ]
        )

    def get_cam_to_scene(self):
        cam_to_scene = np.eye(4)
        cam_to_scene[:3, :3] = self.scene_to_cam[:3, :3].T
        cam_to_scene[:3, 3] = -self.scene_to_cam[:3, :3].T @ self.scene_to_cam[:3, 3]
        return cam_to_scene

    def render(self, input_dict, crop_to_size=True, return_xyz=False):
        """
        Args:
            input_dict: Dict of trimesh objects. Keys: shape, bone
            "shape": trimesh object
            "bone": trimesh object
        Returns:
            color: (H,W,3)
            depth: (H,W)
        """
        scene = Scene(ambient_light=self.ambient_light)

        # add shape / camera
        if "bone" in input_dict:
            # add bone
            mesh_pyrender = Mesh.from_trimesh(input_dict["bone"], smooth=False)
            mesh_pyrender.primitives[0].material = self.material
            scene.add_node(Node(mesh=mesh_pyrender))
        # else:
        #     # make shape gray
        #     input_dict["shape"].visual.vertex_colors[:, :3] = 102

        if "scene" in input_dict:
            # add scene
            mesh_pyrender = Mesh.from_trimesh(input_dict["scene"], smooth=False)
            mesh_pyrender.primitives[0].material = self.material
            scene.add_node(Node(mesh=mesh_pyrender))

        # shape
        if "shape" not in input_dict:
            # use cached shape to save time
            if hasattr(self, "mesh_pyrender"):
                mesh_pyrender = self.mesh_pyrender
            else:
                raise ValueError("shape not in input_dict")
        else:
            # use new shape
            if isinstance(input_dict["shape"], trimesh.points.PointCloud):
                mesh_pyrender = Mesh.from_points(input_dict["shape"].vertices)
            elif isinstance(input_dict["shape"], trimesh.base.Trimesh):
                mesh_pyrender = Mesh.from_trimesh(input_dict["shape"], smooth=False)
            else:
                raise ValueError("shape type not compatible")
            self.mesh_pyrender = mesh_pyrender

        # change material
        if self.material is not None:
            mesh_pyrender.primitives[0].material = self.material
        scene.add_node(Node(mesh=mesh_pyrender))
        if "ghost" in input_dict:
            mesh_pyrender = Mesh.from_trimesh(input_dict["ghost"], smooth=False)
            mesh_pyrender.primitives[0].material = self.material
            scene.add_node(Node(mesh=mesh_pyrender))

        # camera
        scene.add(self.intrinsics, pose=self.get_cam_to_scene())

        # light
        scene.add(self.direc_l, pose=self.light_pose)

        # render
        if "ghost" in input_dict:
            flags = 0
        else:
            flags = pyrender.RenderFlags.SHADOWS_DIRECTIONAL
        color, depth = self.r.render(scene, flags=flags)

        if crop_to_size:
            color = color[: self.image_size[0], : self.image_size[1]]

        if return_xyz:
            xyz = self.depth_to_xyz(depth)
            if crop_to_size:
                xyz = xyz[: self.image_size[0], : self.image_size[1]]
            return color, xyz
        else:
            if crop_to_size:
                depth = depth[: self.image_size[0], : self.image_size[1]]
            return color, depth

    def get_xy_homogeneous(self):
        W = self.render_size
        H = self.render_size
        xy = np.stack(np.meshgrid(np.arange(W), np.arange(H)), axis=-1)
        xy = xy.reshape((-1, 2))
        xy_homo = np.hstack((xy, np.ones((xy.shape[0], 1))))
        return xy_homo

    def depth_to_xyz(self, depth):
        if not hasattr(self, "xy_homo"):
            self.xy_homo = self.get_xy_homogeneous()
        xyz = depth_to_xyz(depth, self.get_intrinsics(), xy_homo=self.xy_homo)
        return xyz

    def delete(self):
        self.r.delete()
