# Copyright (c) 2023 Gengshan Yang, Carnegie Mellon University.
# python lab4d/render_intermediate.py --testdir logdir/human-48-category-comp/
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
from lab4d.utils.pyrender_wrapper import PyRenderWrapper

parser = argparse.ArgumentParser(description="script to render cameras over epochs")
parser.add_argument("--testdir", default="", help="path to test dir")
parser.add_argument(
    "--data_class", default="fg", type=str, help="which data to render, {fg, bg}"
)
args = parser.parse_args()


def main():
    renderer = PyRenderWrapper()
    # io
    path_list = [
        i for i in glob.glob("%s/*-%s-proxy.obj" % (args.testdir, args.data_class))
    ]
    if len(path_list) == 0:
        print("no mesh found in %s for %s" % (args.testdir, args.data_class))
        return
    path_list = sorted(path_list, key=lambda x: int(x.split("/")[-1].split("-")[0]))
    outdir = "%s/renderings_proxy" % args.testdir
    os.makedirs(outdir, exist_ok=True)

    mesh_dict = {}
    aabb_min = np.asarray([np.inf, np.inf])
    aabb_max = np.asarray([-np.inf, -np.inf])
    for mesh_path in path_list:
        batch_idx = int(mesh_path.split("/")[-1].split("-")[0])
        mesh_obj = trimesh.load(mesh_path)
        mesh_dict[batch_idx] = mesh_obj

        # update aabb
        aabb_min = np.minimum(aabb_min, mesh_obj.bounds[0, [0, 2]])  # x,z coords
        aabb_max = np.maximum(aabb_max, mesh_obj.bounds[1, [0, 2]])

    # set camera translation
    max_half_wh = max(np.concatenate([np.abs(aabb_min), np.abs(aabb_max)]))
    renderer.set_camera_bev(depth=2.5 * max_half_wh)
    renderer.set_light_topdown()

    # render
    frames = []
    for batch_idx, mesh_obj in tqdm.tqdm(mesh_dict.items()):
        input_dict = {"shape": mesh_obj}
        color = renderer.render(input_dict)[0]
        # add text
        color = color.astype(np.uint8)
        color = cv2.putText(
            color,
            "batch: %02d" % batch_idx,
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (256, 0, 0),
            2,
        )
        frames.append(color)

    save_path = "%s/%s" % (outdir, args.data_class)
    save_vid(save_path, frames, suffix=".mp4", upsample_frame=-1)
    print("saved to %s.mp4" % (save_path))


if __name__ == "__main__":
    main()
