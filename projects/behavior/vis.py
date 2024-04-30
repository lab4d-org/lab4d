# python projects/behavior/vis.py --gendir ../guided-motion-diffusion/save/unet_adazero_xl_x0_abs_proj10_fp16_clipwd_224_custom_ego/samples_000115000_seed10/ --logdir logdir-12-05/home-2023-11-11--11-51-53-compose/ --fps 3
import sys, os
import pdb
import glob
import numpy as np
import cv2
import argparse
import tqdm

cwd = os.getcwd()
if cwd not in sys.path:
    sys.path.insert(0, cwd)
from lab4d.utils.io import save_vid
from lab4d.utils.pyrender_wrapper import PyRenderWrapper
from lab4d.config import load_flags_from_file
from projects.behavior.articulation_loader import ArticulationLoader

parser = argparse.ArgumentParser(description="script to render extraced meshes")
parser.add_argument("--logdir", default="", help="path to the directory with logs")
parser.add_argument("--gendir", default="", help="path to the dir with generation")
parser.add_argument("--fps", default=10, type=int, help="fps of the video")
parser.add_argument("--mode", default="", type=str, help="{shape, bone}")
parser.add_argument("--compose_mode", default="", type=str, help="{object, scene}")
parser.add_argument("--ghosting", action="store_true", help="ghosting")
parser.add_argument("--view", default="bev", type=str, help="{ref, bev, front}")
args = parser.parse_args()


def main():
    # load flags from file with absl
    opts = load_flags_from_file("%s/opts.log" % args.logdir)
    opts["load_suffix"] = "latest"
    opts["logroot"] = "logdir"
    opts["inst_id"] = 1
    opts["grid_size"] = 128
    opts["level"] = 0
    opts["vis_thresh"] = -10
    opts["extend_aabb"] = False

    loader = ArticulationLoader(opts)
    frames = []
    for sample_idx, genpath in enumerate(
        sorted(glob.glob("%s/sample/*.npy" % args.gendir))
    ):
        sample = np.load(genpath)
        loader.load_files(sample)

        # render
        raw_size = loader.raw_size
        renderer = PyRenderWrapper(raw_size)
        print("Rendering [%s]:" % args.view)

        for frame_idx in tqdm.tqdm(range(len(loader))):
            # input dict
            input_dict = loader.query_frame(frame_idx)

            if args.view == "bev":
                # bev
                renderer.set_camera_bev(depth=loader.get_max_extend_abs())
            elif args.view == "body":
                world_to_observer = loader.get_body_camera(frame_idx)
                renderer.set_camera(world_to_observer)
            elif args.view == "selfie":
                world_to_observer = loader.get_selfie_camera(frame_idx)
                renderer.set_camera(world_to_observer)
            elif args.view == "following":
                world_to_observer = loader.get_following_camera(frame_idx)
                renderer.set_camera(world_to_observer)
            else:
                raise ValueError("Unknown view")

            # set camera intrinsics
            fl = max(raw_size)
            intr = np.asarray([fl, fl, raw_size[1] / 2, raw_size[0] / 2])
            renderer.set_intrinsics(intr)
            renderer.align_light_to_camera()

            color = renderer.render(input_dict)[0]
            # add text
            color = color.astype(np.uint8)
            # color = cv2.putText(
            #     color,
            #     "frame: %02d" % frame_idx,
            #     (30, 50),
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     2,
            #     (256, 0, 0),
            #     2,
            # )
            frames.append(color)
        renderer.delete()

        save_path = "%s/render-%s-%s-%s" % (
            args.gendir,
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
