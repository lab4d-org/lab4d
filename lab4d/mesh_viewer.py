"""modified from https://github.com/nerfstudio-project/viser/blob/main/examples/07_record3d_visualizer.py
python lab4d/mesh_viewer.py --testdir logdir//ama-bouncing-4v-ppr-exp/export_0000/
"""

import os, sys
import pdb
import time
from pathlib import Path
from typing import List
import argparse
import trimesh

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
from lab4d.utils.mesh_loader import MeshLoader


parser = argparse.ArgumentParser(description="script to render extraced meshes")
parser.add_argument("--testdir", default="", help="path to the directory with results")
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
        print("Loading frames!")
        loader = MeshLoader(args.testdir, args.mode, args.compose_mode)
        loader.print_info()
        loader.load_files(ghosting=args.ghosting)
        num_frames = len(loader)
        fps = args.fps

        # load images
        downsample_factor = 16
        rgb_list = loader.load_rgb(downsample_factor)

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
            # change the color
            input_dict["scene"].visual.vertex_colors = np.tile(
                [128, 234, 234, 255], [len(input_dict["scene"].vertices), 1]
            )
            server.add_mesh_trimesh(name=f"/frames/scene", mesh=input_dict["scene"])
        if len(loader.path_list) == 1:
            # for background-only
            server.add_mesh_trimesh(name=f"/frames/scene", mesh=input_dict["shape"])

        if "scene" in input_dict or len(loader.path_list) == 1:
            # add canonical geometry for background
            mesh_canonical = loader.query_canonical_mesh(inst_id=0)
            # change the color
            mesh_canonical.visual.vertex_colors = np.tile(
                [255, 189, 227, 255], [len(mesh_canonical.vertices), 1]
            )
            # # TODO: load polycam mesh, which is higher resolution
            # # mesh_canonical = trimesh.load("database/polycam/Oct31at1-13AM-poly/raw.ply")
            # mesh_canonical = trimesh.load("database/polycam/Oct5at10-49AM-poly/raw.ply")
            # mesh_canonical.vertices = mesh_canonical.vertices * np.asarray(
            #     [[1, -1, -1]]
            # )

            server.add_mesh_trimesh(name=f"/frames/canonical", mesh=mesh_canonical)

            # add camtraj geometry for background
            mesh_camtraj = loader.query_camtraj_mesh()
            server.add_mesh_trimesh(name=f"/frames/camtraj", mesh=mesh_camtraj)

            # add roottraj geometry for foreground
            mesh_roottraj = loader.query_camtraj_mesh(data_class="fg")
            server.add_mesh_trimesh(name=f"/frames/roottraj", mesh=mesh_roottraj)

        self.num_frames = num_frames
        self.server = server
        self.loader = loader
        self.rgb_list = rgb_list
        self.downsample_factor = downsample_factor
        self.gui_timestep = gui_timestep
        self.gui_playing = gui_playing
        self.gui_framerate = gui_framerate
        self.frame_nodes: List[viser.FrameHandle] = []

    def run(self) -> None:
        # Start the server.
        for i, (frame_idx,_) in enumerate(tqdm(self.loader.extr_dict.items())):
            # Add base frame.
            self.frame_nodes.append(
                self.server.add_frame(f"/frames/t{i}", show_axes=False)
            )

            input_dict = self.loader.query_frame(frame_idx)
            if len(self.loader.path_list) > 1:
                # for foreground
                self.server.add_mesh_trimesh(
                    name=f"/frames/t{i}/shape", mesh=input_dict["shape"]
                )
                if "bone" in input_dict:
                    self.server.add_mesh_trimesh(
                        name=f"/frames/t{i}/bone/", mesh=input_dict["bone"]
                    )

            # Place the frustum.
            rgb = self.rgb_list[i]
            extrinsics = np.linalg.inv(self.loader.extr_dict[frame_idx])
            intrinsics = self.loader.intrinsics[i] / self.downsample_factor
            fov = 2 * np.arctan2(rgb.shape[0] / 2, intrinsics[0])
            aspect = rgb.shape[1] / rgb.shape[0]
            self.server.add_camera_frustum(
                f"/frames/t{i}/frustum",
                fov=fov,
                aspect=aspect,
                scale=0.1,
                image=rgb if args.show_img else None,
                wxyz=tf.SO3.from_matrix(extrinsics[:3, :3]).wxyz,
                position=extrinsics[:3, 3],
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
