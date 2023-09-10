import os
import numpy as np
import cv2
import pdb
import pyrender
from pyrender import (
    IntrinsicsCamera,
    Mesh,
    Node,
    Scene,
    OffscreenRenderer,
    MetallicRoughnessMaterial,
)

os.environ["PYOPENGL_PLATFORM"] = "egl"


class PyRenderWrapper:
    def __init__(self, image_size=(1024, 1024)) -> None:
        # renderer
        self.image_size = image_size
        render_size = max(image_size)
        self.r = OffscreenRenderer(render_size, render_size)
        self.intrinsics = IntrinsicsCamera(
            render_size, render_size, render_size / 2, render_size / 2
        )
        # light
        self.light_pose = np.eye(4)
        self.set_light_topdown()
        self.direc_l = pyrender.DirectionalLight(color=np.ones(3), intensity=5.0)
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

    def set_camera(self, scene_to_cam):
        # object to camera transforms
        self.scene_to_cam = self.flip_pose @ scene_to_cam

    def set_light_topdown(self, gl=False):
        # top down light, slightly closer to the camera
        if gl:
            rot = cv2.Rodrigues(np.asarray([-np.pi / 3, 0, 0]))[0]
        else:
            rot = cv2.Rodrigues(np.asarray([np.pi / 3, 0, 0]))[0]
        self.light_pose[:3, :3] = rot

    def set_intrinsics(self, intrinsics):
        """
        Args:
            intrinsics: (4,) fx,fy,px,py
        """
        self.intrinsics = IntrinsicsCamera(
            intrinsics[0], intrinsics[1], intrinsics[2], intrinsics[3]
        )

    def get_cam_to_scene(self):
        cam_to_scene = np.eye(4)
        cam_to_scene[:3, :3] = self.scene_to_cam[:3, :3].T
        cam_to_scene[:3, 3] = -self.scene_to_cam[:3, :3].T @ self.scene_to_cam[:3, 3]
        return cam_to_scene

    def render(self, input_dict):
        """
        Args:
            input_dict: Dict of trimesh objects. Keys: shape, bone
        Returns:
            color: (H,W,3)
            depth: (H,W)
        """
        scene = Scene(ambient_light=0.1 * np.asarray([1.0, 1.0, 1.0, 1.0]))

        # add shape / camera
        if "bone" in input_dict:
            # add bone
            mesh_pyrender = Mesh.from_trimesh(input_dict["bone"], smooth=False)
            mesh_pyrender.primitives[0].material = self.material
            scene.add_node(Node(mesh=mesh_pyrender))

            # make shape transparent and gray
            input_dict["shape"].visual.vertex_colors[:] = 102
        # else:
        #     # make shape gray
        #     input_dict["shape"].visual.vertex_colors[:, :3] = 102

        if "scene" in input_dict:
            # add scene
            mesh_pyrender = Mesh.from_trimesh(input_dict["scene"], smooth=False)
            mesh_pyrender.primitives[0].material = self.material
            scene.add_node(Node(mesh=mesh_pyrender))

        mesh_pyrender = Mesh.from_trimesh(input_dict["shape"], smooth=False)
        mesh_pyrender.primitives[0].material = self.material
        scene.add_node(Node(mesh=mesh_pyrender))

        # camera
        scene.add(self.intrinsics, pose=self.get_cam_to_scene())

        # light
        scene.add(self.direc_l, pose=self.light_pose)

        # render
        color, depth = self.r.render(
            scene,
            flags=pyrender.RenderFlags.SHADOWS_DIRECTIONAL
            # | pyrender.RenderFlags.SKIP_CULL_FACES,
        )
        color = color[: self.image_size[0], : self.image_size[1]]
        depth = depth[: self.image_size[0], : self.image_size[1]]
        return color, depth

    def delete(self):
        self.r.delete()
