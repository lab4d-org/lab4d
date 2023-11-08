import numpy as np
import os, sys
import glob
import json
import numpy as np
import pdb
import trimesh
import cv2
import tqdm
import torch
import time

sys.path.insert(0, os.getcwd())
from lab4d.utils.vis_utils import draw_cams
from lab4d.utils.pyrender_wrapper import PyRenderWrapper
from lab4d.utils.cam_utils import xyz_to_canonical
from projects.csim.ace.polycam_to_ace import json_to_camera
from lab4d.utils.geom_utils import K2inv


class PolyCamRender:
    def __init__(self, poly_path, image_size=(2048, 2048)):
        meta_path = "%s/mesh_info.json" % poly_path
        pose_path = "%s/keyframes/cameras/" % poly_path
        # mesh_path = "%s/raw.glb" % poly_path
        # mesh = list(trimesh.load(mesh_path).geometry.values())[0]
        mesh_path = "%s/raw.ply" % poly_path
        mesh = trimesh.load(mesh_path)
        mesh_info = json.load(open(meta_path))
        scene_unrect = np.linalg.inv(
            np.asarray(mesh_info["alignmentTransform"]).reshape((4, 4)).T
        )

        # gl to cv
        gl_to_cv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        mesh.apply_transform(gl_to_cv)
        scene_unrect = scene_unrect @ gl_to_cv
        self.scene_unrect = scene_unrect
        self.mesh = mesh

        self.input_dict = {"shape": [self.mesh]}
        self.aabb = mesh.bounding_box.bounds

        self.renderer = PyRenderWrapper(image_size=image_size)
        # from (x-down, y-right, z-inward) to (x-right, y-down, z-forward)
        self.transformation_matrix = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])

        self.read_cams(pose_path)

    def __len__(self):
        return len(self.extrinsics)

    def read_cams(self, pose_path):
        camera_paths = sorted(glob.glob(pose_path + "*.json"))
        self.extrinsics = []
        self.intrinsics = []
        for camera_path in camera_paths:
            # copy to poses
            # both ace and polycam saves view-to-world poses
            json_data = json.load(open(camera_path))
            extrinsics, intrinsics = json_to_camera(json_data)
            # view to world
            extrinsics[:3, :3] = extrinsics[:3, :3] @ self.transformation_matrix
            # to correct for the vertical flip
            intrinsics = intrinsics[[1, 0, 3, 2]]
            # world to view
            extrinsics = np.linalg.inv(extrinsics)
            extrinsics = extrinsics @ self.scene_unrect
            self.extrinsics.append(extrinsics)
            self.intrinsics.append(intrinsics)
        self.extrinsics = np.stack(self.extrinsics, axis=0)
        self.intrinsics = np.stack(self.intrinsics, axis=0)

    def render(
        self, idx, intrinsics=None, extrinsics=None, crop_to_size=True, return_xyz=False
    ):
        if intrinsics is None:
            intrinsics = self.intrinsics[idx]
        if extrinsics is None:
            extrinsics = self.extrinsics[idx]

        self.renderer.set_camera(extrinsics)
        self.renderer.set_intrinsics(intrinsics)

        self.renderer.align_light_to_camera()
        if hasattr(self.renderer, "mesh_pyrender"):
            color, xyz = self.renderer.render(
                {}, crop_to_size=crop_to_size, return_xyz=return_xyz
            )
        else:
            color, xyz = self.renderer.render(
                self.input_dict, crop_to_size=crop_to_size, return_xyz=return_xyz
            )
        return color, xyz


if __name__ == "__main__":
    poly_name = "Oct5at10-49AM-poly"
    # poly_name = "Oct25at8-48PM-poly"

    outdir = "projects/csim/zero123_data/home/%s" % poly_name
    os.makedirs(outdir, exist_ok=True)
    poly_path = "database/polycam/%s" % poly_name
    polycam_loader = PolyCamRender(poly_path, image_size=(1024, 768))
    polycam_loader.renderer.set_ambient_light()
    for i in tqdm.tqdm(range(len(polycam_loader))):
        color, xyz = polycam_loader.render(i)
        extrinsics = polycam_loader.extrinsics[i]
        xyz = xyz_to_canonical(xyz, extrinsics)
        # normlize to aabb
        xyz = (xyz - polycam_loader.aabb[0]) / (
            polycam_loader.aabb[1] - polycam_loader.aabb[0]
        )
        cv2.imwrite("%s/%03d.jpg" % (outdir, i), color[..., ::-1])
        np.save("%s/%03d.npy" % (outdir, i), extrinsics)

        # TODO: render normal etc.
        # cv2.imwrite("tmp/xyz.jpg", xyz[..., ::-1] * 255)
        # trimesh.Trimesh(xyz.reshape(-1, 3)).export("tmp/0.obj")
        # print("rendered to tmp/test.jpg")
        # print("done")
    print("rendered to %s" % outdir)
