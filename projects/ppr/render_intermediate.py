# Copyright (c) 2023 Gengshan Yang, Carnegie Mellon University.
# python scripts/render_intermediate.py --testdir logdir/human-48-category-comp/
import sys, os
import pdb

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

parser = argparse.ArgumentParser(description="script to render cameras over epochs")
parser.add_argument("--testdir", default="", help="path to test dir")
parser.add_argument(
    "--data_class", default="distilled", type=str, help="which data to render, {fg, bg}"
)
args = parser.parse_args()


def main():
    renderer = PyRenderWrapper()
    # io
    path_list = [i for i in glob.glob("%s/%s_*.obj" % (args.testdir, args.data_class))]
    if len(path_list) == 0:
        print("no mesh found in %s for %s" % (args.testdir, args.data_class))
        return
    path_list = sorted(
        path_list, key=lambda x: int(x.split("/")[-1].split("-")[-1][:-4])
    )
    outdir = "%s/renderings_trajs" % args.testdir
    os.makedirs(outdir, exist_ok=True)

    mesh_dict = {}
    aabb_min = np.asarray([np.inf, np.inf, np.inf])
    aabb_max = np.asarray([-np.inf, -np.inf, -np.inf])
    for mesh_path in path_list:
        batch_idx = int(mesh_path.split("/")[-1].split("-")[-1][:-4])
        mesh_obj = trimesh.load(mesh_path)
        mesh_dict[batch_idx] = mesh_obj

        # update aabb
        aabb_min = np.minimum(aabb_min, mesh_obj.bounds[0])
        aabb_max = np.maximum(aabb_max, mesh_obj.bounds[1])

    # set camera translation
    # renderer.set_camera_bev(depth=max(aabb_max - aabb_min) * 1.2, gl=True)
    aabb_range = max(aabb_max - aabb_min)
    scene_to_cam = np.eye(4)
    rot = cv2.Rodrigues(np.asarray([-np.pi * 8 / 9, 0, 0]))[0]
    scene_to_cam[:3, :3] = rot
    scene_to_cam[2, 3] = aabb_range
    renderer.set_camera(scene_to_cam)
    renderer.set_light_topdown(gl=True)

    # render
    frames = []
    for batch_idx, mesh_obj in tqdm.tqdm(mesh_dict.items()):
        # percentage = batch_idx / list(mesh_dict.keys())[-1]
        # scene_to_cam[0, 3] = aabb_range * (percentage - 0.5) * 0.5
        renderer.set_camera(scene_to_cam)
        input_dict = {"shape": mesh_obj}
        color = renderer.render(input_dict)[0]

        # render another view
        scene_to_cam_vp2 = scene_to_cam.copy()
        # rotate 90 degrees along y axis
        rot = cv2.Rodrigues(np.asarray([0, np.pi / 2, 0]))[0]
        scene_to_cam_vp2[:3, :3] = scene_to_cam_vp2[:3, :3] @ rot
        renderer.set_camera(scene_to_cam_vp2)
        color_vp2 = renderer.render(input_dict)[0]
        color = np.concatenate([color, color_vp2], axis=1)

        # add text
        color = color.astype(np.uint8)
        color = cv2.putText(
            color,
            "iteration: %04d" % batch_idx,
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (256, 0, 0),
            2,
        )

        frames.append(color)

    save_path = "%s/%s" % (outdir, args.data_class)
    vid_secs = 5  # 5s
    fps = len(frames) / vid_secs
    save_vid(save_path, frames, suffix=".mp4", upsample_frame=-1, fps=fps)
    print("saved to %s.mp4" % (save_path))


if __name__ == "__main__":
    main()
