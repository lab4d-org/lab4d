# Copyright (c) 2023 Gengshan Yang, Carnegie Mellon University.
# python scripts/render_intermediate.py --testdir logdir/human-48-category-comp/
import sys, os
import pdb
import json
import glob
import numpy as np
import cv2
import argparse
import trimesh
import tqdm


cwd = os.getcwd()
if cwd not in sys.path:
    sys.path.insert(0, cwd)
from lab4d.utils.io import save_vid
from lab4d.utils.mesh_render_utils import PyRenderWrapper
from lab4d.utils.geom_utils import compute_rectification_se3

parser = argparse.ArgumentParser(description="script to render extraced meshes")
parser.add_argument("--testdir", default="", help="path to the directory with results")
parser.add_argument("--fps", default=30, type=int, help="fps of the video")
parser.add_argument("--mode", default="", type=str, help="{shape, bone}")
parser.add_argument("--compose_mode", default="", type=str, help="{object, scene}")
args = parser.parse_args()


def main():
    # io
    camera_info = json.load(open("%s/camera.json" % (args.testdir), "r"))
    intrinsics = np.asarray(camera_info["intrinsics"], dtype=np.float32)
    raw_size = camera_info["raw_size"]  # h,w
    if len(glob.glob("%s/fg/mesh/*.obj" % (args.testdir))) > 0:
        primary_dir = "%s/fg" % args.testdir
        secondary_dir = "%s/bg" % args.testdir
    else:
        primary_dir = "%s/bg" % args.testdir
        secondary_dir = "%s/fg" % args.testdir  # never use fg for secondary
    path_list = sorted([i for i in glob.glob("%s/mesh/*.obj" % (primary_dir))])
    if len(path_list) == 0:
        print("no mesh found that matches %s*" % (primary_dir))
        return

    # check render mode
    if args.mode != "":
        mode = args.mode
    elif len(glob.glob("%s/bone/*" % primary_dir)) > 0:
        mode = "bone"
    else:
        mode = "shape"

    if args.compose_mode != "":
        compose_mode = args.compose_mode
    elif len(glob.glob("%s/mesh/*" % secondary_dir)) > 0:
        compose_mode = "compose"
    else:
        compose_mode = "primary"
    print(
        "[mode=%s+%s] rendering %d meshes to %s"
        % (mode, compose_mode, len(path_list), args.testdir)
    )

    # get cam dict
    field2cam_fg_dict = json.load(open("%s/motion.json" % (primary_dir), "r"))
    field2cam_fg_dict = field2cam_fg_dict["field2cam"]
    if compose_mode == "compose":
        field2cam_bg_dict = json.load(open("%s/motion.json" % (secondary_dir), "r"))
        field2cam_bg_dict = np.asarray(field2cam_bg_dict["field2cam"])

    mesh_dict = {}
    extr_dict = {}
    bone_dict = {}
    scene_dict = {}
    for mesh_path in path_list:
        frame_idx = int(mesh_path.split("/")[-1].split(".")[0])
        mesh = trimesh.load(mesh_path, process=False)
        mesh.visual.vertex_colors = mesh.visual.vertex_colors  # visual.kind = 'vertex'
        mesh_dict[frame_idx] = mesh
        extr_dict[frame_idx] = field2cam_fg_dict[frame_idx]

        if mode == "bone":
            # load bone
            bone_path = mesh_path.replace("mesh", "bone")
            bone = trimesh.load(bone_path, process=False)
            bone.visual.vertex_colors = bone.visual.vertex_colors
            bone_dict[frame_idx] = bone

        if compose_mode == "compose":
            # load scene
            scene_path = mesh_path.replace("fg/mesh", "bg/mesh")
            scene = trimesh.load(scene_path, process=False)
            scene.visual.vertex_colors = scene.visual.vertex_colors

            # align bg floor with xz plane
            if "field2world" not in locals():
                field2world_path = "%s/bg/field2world.json" % (args.testdir)
                field2world = np.asarray(json.load(open(field2world_path, "r")))
                world2field = np.linalg.inv(field2world)
            scene.vertices = scene.vertices @ field2world[:3, :3].T + field2world[:3, 3]
            field2cam_bg_dict[frame_idx] = field2cam_bg_dict[frame_idx] @ world2field

            scene_dict[frame_idx] = scene
            # use scene camera
            extr_dict[frame_idx] = field2cam_bg_dict[frame_idx]
            # transform to scene
            object_to_scene = (
                np.linalg.inv(field2cam_bg_dict[frame_idx])
                @ field2cam_fg_dict[frame_idx]
            )
            mesh_dict[frame_idx].apply_transform(object_to_scene)
            if mode == "bone":
                bone_dict[frame_idx].apply_transform(object_to_scene)

    # render
    renderer = PyRenderWrapper(raw_size)
    frames = []
    for frame_idx, mesh_obj in tqdm.tqdm(mesh_dict.items()):
        # input dict
        input_dict = {}
        input_dict["shape"] = mesh_obj
        if mode == "bone":
            input_dict["bone"] = bone_dict[frame_idx]
        if compose_mode == "compose":
            input_dict["scene"] = scene_dict[frame_idx]
        if compose_mode == "primary":
            # set camera extrinsics
            renderer.set_camera(extr_dict[frame_idx])
            # set camera intrinsics
            renderer.set_intrinsics(intrinsics[frame_idx])
        else:
            # set camera extrinsics
            renderer.set_camera(extr_dict[frame_idx])
            # set camera intrinsics
            renderer.set_intrinsics(intrinsics[frame_idx])

            # # bev
            # renderer.set_camera_bev(8)

            # # frontal view
            # renderer.set_camera_frontal(25, delta=np.pi / 2)
            # # set camera intrinsics
            # fl = max(raw_size)
            # intr = np.asarray([fl * 4, fl * 4, raw_size[1] / 2, raw_size[0] / 4 * 3])
            # renderer.set_intrinsics(intr)
        renderer.align_light_to_camera()

        color = renderer.render(input_dict)[0]
        # add text
        color = color.astype(np.uint8)
        color = cv2.putText(
            color,
            "frame: %02d" % frame_idx,
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (256, 0, 0),
            2,
        )
        frames.append(color)

    save_vid(
        "%s/render" % args.testdir,
        frames,
        suffix=".mp4",
        upsample_frame=-1,
        fps=args.fps,
    )
    print("saved to %s/render.mp4" % args.testdir)


if __name__ == "__main__":
    main()
