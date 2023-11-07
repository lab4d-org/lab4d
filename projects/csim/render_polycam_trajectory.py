import numpy as np
import os, sys
import glob
import json
import numpy as np
import tqdm
import pdb
import trimesh
import cv2

sys.path.insert(0, os.getcwd())
from projects.csim.render_polycam import PolyCamRender, depth_to_canonical
from lab4d.utils.mesh_loader import MeshLoader
from lab4d.utils.io import save_vid


if __name__ == "__main__":
    poly_path = "database/polycam/Oct5at10-49AM-poly"
    # logdir = "Oct5at10-49AM-poly-bg-feat"
    logdir = "home-cat-pikachu-6-adapt-ft"
    # logdir = "home-cat-pikachu-0-bg6"
    vid_id = 1
    img_scale = 1

    # read extrinsics trajectory
    lab4d_loader = MeshLoader("logdir/%s/export_%04d/" % (logdir, vid_id))
    lab4d_loader.load_files()
    bg_scale = lab4d_loader.bg_scale

    num_frames = len(lab4d_loader.extr_dict)

    # image_size = (1024, 768)
    image_size = lab4d_loader.raw_size
    image_size = (int(image_size[0] * img_scale), int(image_size[1] * img_scale))
    poly_loader = PolyCamRender(poly_path, image_size=image_size)
    intrinsics = poly_loader.intrinsics[0] * img_scale

    aabb = poly_loader.aabb
    poly_loader.renderer.set_ambient_light()

    # # TODO fix it
    # lab4d_loader_agent = MeshLoader("logdir/cat-pikachu-0-ppr//export_0000/")
    # lab4d_loader_agent.load_files()
    # cat_to_camera = np.asarray(lab4d_loader_agent.field2cam_fg_dict)
    # cat_to_camera[:, :3, 3] *= 0.1
    # camera_to_cat = np.linalg.inv(cat_to_camera)
    # rectify = np.eye(4)  #
    # # from x:right,
    # rectify[0, 0] = -1
    # rectify[2, 2] = -1
    # rectify[1, 3] = 0.15
    # rectify[2, 0] = -0.2
    # camera_to_cat = rectify[None] @ camera_to_cat

    rgb_frames = []
    xyz_frames = []
    for i in tqdm.tqdm(range(num_frames)):
        extrinsics = lab4d_loader.extr_dict[i]
        extrinsics[:3, 3] *= bg_scale

        # # extrinsics_home_to_cat (A) = camera_to_cat x home_to_camera
        # extrinsics = camera_to_cat[i] @ extrinsics

        intrinsics = lab4d_loader.intrinsics[i] * img_scale
        rgb, depth = poly_loader.render(0, intrinsics=intrinsics, extrinsics=extrinsics)
        xyz = depth_to_canonical(depth, intrinsics, extrinsics)
        # normlize to aabb
        xyz = (xyz - aabb[0]) / (aabb[1] - aabb[0])

        rgb_frames.append(rgb)
        xyz_frames.append(xyz)
        # cv2.imwrite("tmp/test.jpg", rgb[..., ::-1])
        # cv2.imwrite("tmp/xyz.jpg", xyz[..., ::-1] * 255)
        # # trimesh.Trimesh(xyz.reshape(-1, 3)).export("tmp/0.obj")
        # print("rendered to tmp/test.jpg")
        # print("rendered to tmp/xyz.jpg")
    save_vid("tmp/rgb", rgb_frames, fps=10)
    save_vid("tmp/xyz", xyz_frames, fps=10)
    print("saved to tmp/rgb.mp4")
    print("saved to tmp/xyz.mp4")
