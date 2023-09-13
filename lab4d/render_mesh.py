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

parser = argparse.ArgumentParser(description="script to render extraced meshes")
parser.add_argument("--testdir", default="", help="path to the directory with results")
parser.add_argument("--fps", default=30, type=int, help="fps of the video")
parser.add_argument("--mode", default="", type=str, help="{shape, bone}")
parser.add_argument("--scene_mode", default="", type=str, help="{object, scene}")
args = parser.parse_args()


def main():
    # io
    camera_info = json.load(open("%s/camera.json" % (args.testdir), "r"))
    intrinsics = np.asarray(camera_info["intrinsics"], dtype=np.float32)
    raw_size = camera_info["raw_size"]  # h,w
    path_list = sorted([i for i in glob.glob("%s/fg/mesh/*.obj" % (args.testdir))])
    if len(path_list) == 0:
        print("no mesh found that matches %s*" % (args.testdir))
        return

    # check render mode
    if args.mode != "":
        mode = args.mode
    elif os.path.exists("%s/fg/bone" % args.testdir):
        mode = "bone"
    else:
        mode = "shape"

    if args.scene_mode != "":
        scene_mode = args.scene_mode
    elif os.path.exists("%s/bg/mesh/" % args.testdir):
        scene_mode = "scene"
    else:
        scene_mode = "object"
    print(
        "[mode=%s+%s] rendering %d meshes to %s"
        % (mode, scene_mode, len(path_list), args.testdir)
    )

    # get cam dict
    field2cam_fg_dict = json.load(open("%s/fg/motion.json" % (args.testdir), "r"))
    field2cam_fg_dict = field2cam_fg_dict["field2cam"]
    if scene_mode == "scene":
        field2cam_bg_dict = json.load(open("%s/bg/motion.json" % (args.testdir), "r"))
        field2cam_bg_dict = field2cam_bg_dict["field2cam"]

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

        if scene_mode == "scene":
            # load scene
            scene_path = mesh_path.replace("fg/mesh", "bg/mesh")
            scene = trimesh.load(scene_path, process=False)
            scene.visual.vertex_colors = scene.visual.vertex_colors
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
        if scene_mode == "scene":
            input_dict["scene"] = scene_dict[frame_idx]
        # set camera extrinsics
        renderer.set_camera(extr_dict[frame_idx])
        # # bev
        # renderer.set_camera_bev(3)
        # set camera intrinsics
        renderer.set_intrinsics(intrinsics[frame_idx])
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
