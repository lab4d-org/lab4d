"""modified from https://github.com/nerfstudio-project/viser/blob/main/examples/07_record3d_visualizer.py
python lab4d/mesh_viewer.py --testdir logdir//ama-bouncing-4v-ppr-exp/export_0000/
"""

import os, sys
import pdb
import time
from pathlib import Path
from typing import List
import argparse
import configparser

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
parser.add_argument("--seqname", default="", help="path to the directory of inputs")
parser.add_argument("--fps", default=30, type=int, help="fps of the video")
parser.add_argument("--mode", default="", type=str, help="{shape, bone}")
parser.add_argument("--compose_mode", default="", type=str, help="{object, scene}")
parser.add_argument("--ghosting", action="store_true", help="ghosting")
parser.add_argument("--view", default="ref", type=str, help="{ref, bev, front}")
args = parser.parse_args()


def find_seqname(testdir):
    parts = [part for part in testdir.split("/") if part]
    logdir = "/".join(parts[:2])
    logdir = os.path.join(logdir, "opts.log")
    with open(logdir, "r") as file:
        for line in file:
            if "--seqname" in line:
                seqname = line.split("--")[1].split("=")[1].strip()
                break
    if "seqname" not in locals():
        raise ValueError("Could not find seqname in opts.log")
    inst_id = int(parts[2].split("_")[-1])
    return seqname, inst_id


def main(
    share: bool = False,
) -> None:
    server = viser.ViserServer(share=share)

    downsample_factor = 4
    print("Loading frames!")
    loader = MeshLoader(args.testdir, args.mode, args.compose_mode)
    loader.print_info()
    loader.load_files(ghosting=args.ghosting)
    num_frames = len(loader)
    fps = args.fps

    # load images
    seqname, inst_id = find_seqname(args.testdir)
    config = configparser.RawConfigParser()
    config.read("database/configs/%s.config" % seqname)
    img_dir = config.get("data_%d" % inst_id, "img_path")
    print("Loading images from %s" % img_dir)
    rgb_list = [cv2.imread("%s/%05d.jpg" % (img_dir, i)) for i in range(num_frames)]
    rgb_list = [rgb[::downsample_factor, ::downsample_factor, ::-1] for rgb in rgb_list]

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

    prev_timestep = gui_timestep.value

    # Toggle frame visibility when the timestep slider changes.
    @gui_timestep.on_update
    def _(_) -> None:
        nonlocal prev_timestep
        current_timestep = gui_timestep.value
        with server.atomic():
            frame_nodes[current_timestep].visible = True
            frame_nodes[prev_timestep].visible = False
        prev_timestep = current_timestep

    # Load in frames.
    server.add_frame(
        "/frames",
        wxyz=tf.SO3.exp(np.array([-np.pi / 2, 0.0, 0.0])).wxyz,
        position=(0, 0, 0),
        show_axes=True,
    )
    frame_nodes: List[viser.FrameHandle] = []
    input_dict = loader.query_frame(0)
    if "scene" in input_dict:
        server.add_mesh_trimesh(name=f"/frames/scene", mesh=input_dict["scene"])
    if len(loader.path_list) == 1:
        # for background-only
        server.add_mesh_trimesh(name=f"/frames/scene", mesh=input_dict["shape"])

    if "scene" in input_dict or len(loader.path_list) == 1:
        mesh_canonical = loader.query_canonical_mesh(inst_id=0)
        # add canonical geometry for background
        server.add_mesh_trimesh(name=f"/frames/canonical", mesh=mesh_canonical)

    for i in tqdm(range(num_frames)):
        # Add base frame.
        frame_nodes.append(server.add_frame(f"/frames/t{i}", show_axes=False))

        input_dict = loader.query_frame(i)
        if len(loader.path_list) > 1:
            # for foreground
            server.add_mesh_trimesh(
                name=f"/frames/t{i}/shape", mesh=input_dict["shape"]
            )
        if "bone" in input_dict:
            server.add_mesh_trimesh(name=f"/frames/t{i}/bone", mesh=input_dict["bone"])

        # Place the frustum.
        rgb = rgb_list[i]
        extrinsics = np.linalg.inv(loader.extr_dict[i])
        intrinsics = loader.intrinsics[i] / downsample_factor
        fov = 2 * np.arctan2(rgb.shape[0] / 2, intrinsics[0])
        aspect = rgb.shape[1] / rgb.shape[0]
        server.add_camera_frustum(
            f"/frames/t{i}/frustum",
            fov=fov,
            aspect=aspect,
            scale=0.1,
            image=rgb,
            wxyz=tf.SO3.from_matrix(extrinsics[:3, :3]).wxyz,
            position=extrinsics[:3, 3],
        )

        # Add some axes.
        server.add_frame(
            f"/frames/t{i}/frustum/axes",
            axes_length=0.01,
            axes_radius=0.005,
        )

    # Hide all but the current frame.
    for i, frame_node in enumerate(frame_nodes):
        frame_node.visible = i == gui_timestep.value

    # Playback update loop.
    prev_timestep = gui_timestep.value
    while True:
        if gui_playing.value:
            gui_timestep.value = (gui_timestep.value + 1) % num_frames

        time.sleep(1.0 / gui_framerate.value)


if __name__ == "__main__":
    # tyro.cli(main)
    main()
