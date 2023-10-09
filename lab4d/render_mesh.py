# Copyright (c) 2023 Gengshan Yang, Carnegie Mellon University.
# python lab4d/render_mesh.py --testdir logdir//ama-bouncing-4v-ppr/export_0000/ --view bev --ghosting
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
from lab4d.utils.pyrender_wrapper import PyRenderWrapper
from lab4d.utils.mesh_loader import MeshLoader

parser = argparse.ArgumentParser(description="script to render extraced meshes")
parser.add_argument("--testdir", default="", help="path to the directory with results")
parser.add_argument("--fps", default=30, type=int, help="fps of the video")
parser.add_argument("--mode", default="", type=str, help="{shape, bone}")
parser.add_argument("--compose_mode", default="", type=str, help="{object, scene}")
parser.add_argument("--ghosting", action="store_true", help="ghosting")
parser.add_argument("--view", default="ref", type=str, help="{ref, bev, front}")
args = parser.parse_args()


def main():
    loader = MeshLoader(args.testdir, args.mode, args.compose_mode)
    loader.print_info()
    loader.load_files(ghosting=args.ghosting)

    # render
    raw_size = loader.raw_size
    renderer = PyRenderWrapper(raw_size)
    print("Rendering [%s]:" % args.view)
    frames = []
    for frame_idx, mesh_obj in tqdm.tqdm(loader.mesh_dict.items()):
        # input dict
        input_dict = loader.query_frame(frame_idx)

        if loader.compose_mode == "primary":
            # set camera extrinsics
            renderer.set_camera(loader.extr_dict[frame_idx])
            # set camera intrinsics
            renderer.set_intrinsics(loader.intrinsics[frame_idx])
        else:
            if args.view == "ref":
                # set camera extrinsics
                renderer.set_camera(loader.extr_dict[frame_idx])
                # set camera intrinsics
                renderer.set_intrinsics(loader.intrinsics[frame_idx])
            elif args.view == "bev":
                # bev
                renderer.set_camera_bev(depth=max(loader.aabb_max - loader.aabb_min))
                # set camera intrinsics
                fl = max(raw_size)
                intr = np.asarray([fl * 2, fl * 2, raw_size[1] / 2, raw_size[0] / 2])
                renderer.set_intrinsics(intr)
            elif args.view == "front":
                # frontal view
                renderer.set_camera_frontal(25, delta=0.0)
                # set camera intrinsics
                fl = max(raw_size)
                intr = np.asarray(
                    [fl * 4, fl * 4, raw_size[1] / 2, raw_size[0] / 4 * 3]
                )
                renderer.set_intrinsics(intr)
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

    save_path = "%s/render-%s-%s-%s" % (
        args.testdir,
        loader.mode,
        loader.compose_mode,
        args.view,
    )
    save_vid(
        save_path,
        frames,
        suffix=".mp4",
        upsample_frame=-1,
        fps=args.fps,
    )
    print("saved to %s.mp4" % save_path)


if __name__ == "__main__":
    main()
