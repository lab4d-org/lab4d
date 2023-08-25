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
args = parser.parse_args()


def main():
    # io
    camera_info = json.load(open("%s/camera.json" % (args.testdir), "r"))
    intrinsics = np.asarray(camera_info["intrinsics"], dtype=np.float32)
    raw_size = camera_info["raw_size"]  # h,w
    path_list = sorted([i for i in glob.glob("%s/mesh/*.obj" % (args.testdir))])
    if len(path_list) == 0:
        print("no mesh found that matches %s*" % (args.testdir))
        return
    print("rendering %d meshes to %s" % (len(path_list), args.testdir))

    mesh_dict = {}
    for mesh_path in path_list:
        frame_idx = int(mesh_path.split("/")[-1].split("-")[1].split(".")[0])
        mesh_dict[frame_idx] = trimesh.load(mesh_path)

    # render
    renderer = PyRenderWrapper(raw_size)
    frames = []
    for frame_idx, mesh_obj in tqdm.tqdm(mesh_dict.items()):
        # set camera translation
        renderer.set_intrinsics(intrinsics[frame_idx])
        color = renderer.render(mesh_obj)[0]
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

    save_vid("%s/render" % args.testdir, frames, suffix=".mp4", upsample_frame=-1)
    print("saved to %s/render.mp4" % args.testdir)


if __name__ == "__main__":
    main()