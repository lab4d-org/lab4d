"""modified from https://github.com/nerfstudio-project/viser/blob/main/examples/07_record3d_visualizer.py
python projects/behavior/generated_traj_viewer.py --genpath ../guided-motion-diffusion/./save/unet_adazero_xl_x0_abs_proj10_fp16_clipwd_224_custom_cond_ego2/samples_000080000_seed10/sample_002/sample.npy --logdir logdir-12-05/home-2023-11-11--11-51-53-compose/ --fps 30
"""

import os, sys
import pdb
import glob
import time
from pathlib import Path
from typing import List
import argparse

import cv2
import numpy as np
import tyro
from tqdm.auto import tqdm

import viser
import viser.extras
import viser.transforms as tf


cwd = os.getcwd()
if cwd not in sys.path:
    sys.path.insert(0, cwd)
from lab4d.config import load_flags_from_file
from projects.behavior.articulation_loader import ArticulationLoader


parser = argparse.ArgumentParser(description="script to render extraced meshes")
parser.add_argument("--logdir", default="", help="path to the directory with logs")
parser.add_argument("--genpath", default="", help="path to the dir with generation")
parser.add_argument("--fps", default=30, type=int, help="fps of the video")
parser.add_argument("--mode", default="", type=str, help="{shape, bone}")
parser.add_argument("--compose_mode", default="", type=str, help="{object, scene}")
parser.add_argument("--ghosting", action="store_true", help="ghosting")
parser.add_argument("--view", default="ref", type=str, help="{ref, bev, front}")
parser.add_argument("--show_img", action="store_true", help="show image")
parser.add_argument("--port", default=8080, type=int, help="port")
args = parser.parse_args()


class MeshViewer:
    def __init__(self, share, args) -> None:
        server = viser.ViserServer(share=share, port=args.port)

        # load flags from file with absl
        opts = load_flags_from_file("%s/opts.log" % args.logdir)
        opts["load_suffix"] = "latest"
        opts["logroot"] = "logdir"
        opts["inst_id"] = 1
        opts["grid_size"] = 128
        opts["level"] = 0
        opts["vis_thresh"] = -10
        opts["extend_aabb"] = False

        print("Loading frames!")
        loader = ArticulationLoader(opts)
        sample = np.load(args.genpath)
        loader.load_files(sample)

        num_frames = len(loader)
        fps = args.fps

        # Add playback UI.
        with server.add_gui_folder("Playback"):
            gui_timestep = server.add_gui_slider(
                "Timestep",
                min=0,
                max=num_frames - 1,
                step=1,
                initial_value=0,
                disabled=True,
            )
            gui_next_frame = server.add_gui_button("Next Frame", disabled=True)
            gui_prev_frame = server.add_gui_button("Prev Frame", disabled=True)
            gui_playing = server.add_gui_checkbox("Playing", True)
            gui_framerate = server.add_gui_slider(
                "FPS", min=1, max=60, step=0.1, initial_value=fps
            )
            gui_framerate_options = server.add_gui_button_group(
                "FPS options", ("10", "20", "30", "60")
            )

        # Frame step buttons.
        @gui_next_frame.on_click
        def _(_) -> None:
            gui_timestep.value = (gui_timestep.value + 1) % num_frames

        @gui_prev_frame.on_click
        def _(_) -> None:
            gui_timestep.value = (gui_timestep.value - 1) % num_frames

        # Disable frame controls when we're playing.
        @gui_playing.on_update
        def _(_) -> None:
            gui_timestep.disabled = gui_playing.value
            gui_next_frame.disabled = gui_playing.value
            gui_prev_frame.disabled = gui_playing.value

        # Set the framerate when we click one of the options.
        @gui_framerate_options.on_click
        def _(_) -> None:
            gui_framerate.value = int(gui_framerate_options.value)

        self.prev_timestep = gui_timestep.value

        # Toggle frame visibility when the timestep slider changes.
        @gui_timestep.on_update
        def _(_) -> None:
            current_timestep = gui_timestep.value
            with server.atomic():
                self.frame_nodes[current_timestep].visible = True
                self.frame_nodes[self.prev_timestep].visible = False
            self.prev_timestep = current_timestep

        # Load in frames.
        server.add_frame(
            "/frames",
            wxyz=tf.SO3.exp(np.array([-np.pi / 2, 0.0, 0.0])).wxyz,
            position=(0, 0, 0),
            show_axes=False,
        )

        input_dict = loader.query_frame(0)
        if "scene" in input_dict:
            server.add_mesh_trimesh(name=f"/frames/scene", mesh=input_dict["scene"])

        self.num_frames = num_frames
        self.server = server
        self.loader = loader
        self.gui_timestep = gui_timestep
        self.gui_playing = gui_playing
        self.gui_framerate = gui_framerate
        self.frame_nodes: List[viser.FrameHandle] = []

    def run(self) -> None:
        # Start the server.
        for i in tqdm(range(self.num_frames)):
            # Add base frame.
            self.frame_nodes.append(
                self.server.add_frame(f"/frames/t{i}", show_axes=False)
            )

            input_dict = self.loader.query_frame(i)
            # for foreground
            self.server.add_mesh_trimesh(
                name=f"/frames/t{i}/shape", mesh=input_dict["shape"]
            )
            if "bone" in input_dict:
                self.server.add_mesh_trimesh(
                    name=f"/frames/t{i}/bone", mesh=input_dict["bone"]
                )

            # # Add some axes.
            # server.add_frame(
            #     f"/frames/t{i}/frustum/axes",
            #     axes_length=0.01,
            #     axes_radius=0.005,
            # )

        # Hide all but the current frame.
        for i, frame_node in enumerate(self.frame_nodes):
            frame_node.visible = i == self.gui_timestep.value

        # Playback update loop.
        self.prev_timestep = self.gui_timestep.value
        while True:
            if self.gui_playing.value:
                self.gui_timestep.value = (
                    self.gui_timestep.value + 1
                ) % self.num_frames

            time.sleep(1.0 / self.gui_framerate.value)


def main(
    share: bool = False,
) -> None:
    mesh_viewer = MeshViewer(share, args)
    mesh_viewer.run()


if __name__ == "__main__":
    # tyro.cli(main)
    main()
