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
        self.direc_l = pyrender.DirectionalLight(color=np.ones(3), intensity=5.0)
        self.light_pose = np.asarray(
            [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=float
        )
        self.material = MetallicRoughnessMaterial(
            roughnessFactor=0.75, metallicFactor=0.75
        )
        self.init_camera()

    def init_camera(self):
        # cv to gl coords
        self.cam_pose = -np.eye(4)
        self.cam_pose[0, 0] = 1
        self.cam_pose[-1, -1] = 1
        self.scene_to_cam = np.eye(4)

    def set_camera_bev(self, depth):
        # object to camera transforms
        rot = cv2.Rodrigues(np.asarray([np.pi / 2, 0, 0]))[0]
        self.scene_to_cam[:3, :3] = rot
        self.scene_to_cam[2, 3] = depth

    def set_camera(self, scene_to_cam):
        # object to camera transforms
        self.scene_to_cam = scene_to_cam

    def set_intrinsics(self, intrinsics):
        """
        Args:
            intrinsics: (4,) fx,fy,px,py
        """
        self.intrinsics = IntrinsicsCamera(
            intrinsics[0], intrinsics[1], intrinsics[2], intrinsics[3]
        )

    def render(self, mesh_obj, force_gray=False):
        scene = Scene(ambient_light=0.1 * np.asarray([1.0, 1.0, 1.0, 1.0]))

        # add object / camera
        mesh_obj.apply_transform(self.scene_to_cam)
        if force_gray:
            mesh_obj.visual.vertex_colors = np.ones_like(mesh_obj.visual.vertex_colors)
            mesh_obj.visual.vertex_colors[:, :3] = 102
        mesh_pyrender = Mesh.from_trimesh(mesh_obj)
        mesh_pyrender.primitives[0].material = self.material
        scene.add_node(Node(mesh=mesh_pyrender))

        # camera
        scene.add(self.intrinsics, pose=self.cam_pose)

        # light
        scene.add(self.direc_l, pose=self.light_pose)

        # render
        color, depth = self.r.render(
            scene,
            flags=pyrender.RenderFlags.SHADOWS_DIRECTIONAL
            | pyrender.RenderFlags.SKIP_CULL_FACES,
        )
        color = color[: self.image_size[0], : self.image_size[1]]
        depth = depth[: self.image_size[0], : self.image_size[1]]
        return color, depth
