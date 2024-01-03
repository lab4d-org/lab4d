"""modified from https://github.com/nerfstudio-project/viser/blob/main/examples/07_record3d_visualizer.py
python lab4d/mesh_viewer.py --testdir logdir//ama-bouncing-4v-ppr-exp/export_0000/
"""

import os, sys
import pdb
import time
from typing import List
import argparse
import trimesh

import numpy as np
from tqdm.auto import tqdm
import glob

import viser
import viser.extras
import viser.transforms as tf

cwd = os.getcwd()
if cwd not in sys.path:
    sys.path.insert(0, cwd)
from lab4d.utils.mesh_loader import MeshLoader
from lab4d.utils.vis_utils import draw_cams, get_colormap

colormap = get_colormap()


parser = argparse.ArgumentParser(description="script to render extraced meshes")
parser.add_argument("--testdir", default="", help="path to the directory with results")
parser.add_argument("--fps", default=10, type=int, help="fps of the video")
parser.add_argument("--skip_frames", default=10, type=int, help="skip frames")
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

        # load canonical data
        canonical_dir = "logdir-12-05/home-2023-11-08--20-29-39-compose/export_0001/"
        testdirs = glob.glob("logdir-12-05/*-compose/export_0001/")
        loader = MeshLoader(canonical_dir, args.mode, args.compose_mode)
        loader.print_info()
        loader.load_files(ghosting=args.ghosting)

        # add fg root poses
        self.root_trajs = []
        self.cam_trajs = []
        self.meshes = []
        self.skeleton_trajs = []
        max_num_frames = 0
        for it, loader_path in enumerate(testdirs):
            color = np.concatenate([colormap[it], [255]])
            root_loader = MeshLoader(loader_path, args.mode, args.compose_mode)
            # # load bones, but this is slow
            # root_loader.load_files()
            # root_mesh = root_loader.query_camtraj_mesh(data_class="fg")
            # skeleton_traj = []
            # for i in range(len(root_loader)):
            #     input_dict = root_loader.query_frame(i)
            #     skel = input_dict["bone"]
            #     skel.visual.vertex_colors = np.tile(color, [len(skel.vertices), 1])
            #     skeleton_traj.append(skel)
            # self.skeleton_trajs.append(skeleton_traj)

            # load root poses
            root_traj = root_loader.query_camtraj(data_class="fg")
            self.root_trajs.append(root_traj)

            # load cam poses
            cam_traj = root_loader.query_camtraj(data_class="bg")
            self.cam_trajs.append(cam_traj)

            # setup mesh
            # mesh = trimesh.load("tmp/cat_face.obj")
            mesh = trimesh.load("tmp/spot_remeshed.ply")
            # mesh = trimesh.creation.uv_sphere(radius=0.12, count=[4, 4])
            mesh.visual.vertex_colors = np.tile(color, [len(mesh.vertices), 1])
            self.meshes.append(mesh)
            max_num_frames = max(max_num_frames, len(root_loader) // args.skip_frames)
            print("loaded %d frames from %s" % (len(root_loader), loader_path))

        # Add playback UI.
        fps = args.fps // args.skip_frames
        with server.add_gui_folder("Playback"):
            gui_timestep = server.add_gui_slider(
                "Timestep",
                min=0,
                max=max_num_frames - 1,
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
            gui_timestep.value = (gui_timestep.value + 1) % self.num_frames

        @gui_prev_frame.on_click
        def _(_) -> None:
            gui_timestep.value = (gui_timestep.value - 1) % self.num_frames

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
                for it in range(len(self.root_trajs)):
                    try:
                        self.frame_nodes_list[it][current_timestep].visible = True
                        self.frame_nodes_list[it][self.prev_timestep].visible = False
                    except:
                        pass
            self.prev_timestep = current_timestep

        # Setup root frame
        server.add_frame(
            "/frames",
            wxyz=tf.SO3.exp(np.array([-np.pi / 2, 0.0, 0.0])).wxyz,
            position=(0, 0, 0),
            show_axes=False,
        )

        # Add canonical geometry for background
        mesh_canonical = loader.query_canonical_mesh(inst_id=0)
        # change the color
        mesh_canonical.visual.vertex_colors = np.tile(
            [207, 207, 207, 128], [len(mesh_canonical.vertices), 1]
        )
        server.add_mesh_trimesh(name=f"/frames/canonical", mesh=mesh_canonical)

        self.num_frames = max_num_frames
        self.skip_frames = args.skip_frames
        self.server = server
        self.loader = loader
        self.gui_timestep = gui_timestep
        self.gui_playing = gui_playing
        self.gui_framerate = gui_framerate
        self.frame_nodes_list = []

        # Frame control
        for it, root_traj in enumerate(self.root_trajs):
            frame_nodes: List[viser.FrameHandle] = []
            self.frame_nodes_list.append(frame_nodes)

    def build_traj_visual(self, root_traj, it, color, prefix="root"):
        positions = np.linalg.inv(root_traj)[:, :3, 3]
        start_mesh = trimesh.creation.uv_sphere(radius=0.025, count=[32, 32])
        start_mesh.visual.vertex_colors = np.tile(color, [len(start_mesh.vertices), 1])
        end_mesh = trimesh.creation.box(extents=[0.05, 0.05, 0.05])
        end_mesh.visual.vertex_colors = np.tile(color, [len(end_mesh.vertices), 1])

        self.server.add_spline_catmull_rom(
            f"/frames/seq_{it}/{prefix}-traj",
            positions,
            tension=1.0,
            line_width=2.0,
            color=color,
            segments=len(positions) - 1,
        )
        # create start point
        self.server.add_mesh_trimesh(
            name=f"/frames/seq_{it}/{prefix}-start",
            mesh=start_mesh,
            scale=1.0,
            position=positions[0],
        )
        # create end point
        self.server.add_mesh_trimesh(
            name=f"/frames/seq_{it}/{prefix}-end",
            mesh=end_mesh,
            scale=1.0,
            position=positions[-1],
        )
        return positions, start_mesh, end_mesh

    def run(self) -> None:
        # Plot overall trajectory and start/end points.
        for it, root_traj in enumerate(self.root_trajs):
            self.build_traj_visual(root_traj, it, colormap[it], prefix="root")
            self.build_traj_visual(
                self.cam_trajs[it],
                it,
                colormap[it + len(self.root_trajs)],
                prefix="cam",
            )

        for i in tqdm(range(self.num_frames)):
            i = i * self.skip_frames
            # Add base frame.
            for it in range(len(self.root_trajs)):
                self.frame_nodes_list[it].append(
                    self.server.add_frame(f"/frames/seq_{it}/t{i}", show_axes=False)
                )

            for it, root_traj in enumerate(self.root_trajs):
                if i >= len(root_traj):
                    continue
                extrinsics = np.linalg.inv(root_traj[i])
                cam_extrinsics = np.linalg.inv(self.cam_trajs[it][i])
                # Show camera frustum.
                self.server.add_camera_frustum(
                    f"/frames/seq_{it}/t{i}/cam",
                    fov=np.pi / 2,
                    aspect=1.0,
                    scale=0.05,
                    color=colormap[it + len(self.root_trajs)],
                    wxyz=tf.SO3.from_matrix(cam_extrinsics[:3, :3]).wxyz,
                    position=cam_extrinsics[:3, 3],
                )

                # mesh = self.skeleton_trajs[it][i]
                mesh = self.meshes[it]
                # Show mesh
                self.server.add_mesh_trimesh(
                    name=f"/frames/seq_{it}/t{i}/root",
                    mesh=mesh,
                    scale=1.0,
                    wxyz=tf.SO3.from_matrix(extrinsics[:3, :3]).wxyz,
                    position=extrinsics[:3, 3],
                )

        # Hide all but the current frame.
        for frame_nodes in self.frame_nodes_list:
            for i, frame_node in enumerate(frame_nodes):
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
    main()
