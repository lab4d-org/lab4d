# Copyright (c) 2023 Gengshan Yang, Carnegie Mellon University.
"""
python projects/ppr/simulate.py --flagfile=logdir/cat-pikachu-0-ppr/opts.log --load_suffix latest --load_suffix_phys latest --inst_id 0
"""

import os, sys
import numpy as np
import cv2
import pdb
import json
from absl import app, flags


cwd = os.getcwd()
if cwd not in sys.path:
    sys.path.insert(0, cwd)

from lab4d.config import get_config
from lab4d.utils.io import make_save_dir, save_rendered

sys.path.insert(0, "%s/../" % os.path.join(os.path.dirname(__file__)))
from trainer import PPRTrainer
from trainer import PhysVisualizer


class ExportMeshFlags:
    flags.DEFINE_integer("inst_id", 0, "video/instance id")


def export_simulate_mesh(save_dir, data, tag):
    traj = data[tag]
    camera = data["camera"]
    save_dir = os.path.join(save_dir, tag)
    save_dir_fg = os.path.join(save_dir, "fg/mesh")
    save_dir_bg = os.path.join(save_dir, "bg/mesh")
    os.makedirs(save_dir_fg, exist_ok=True)
    os.makedirs(save_dir_bg, exist_ok=True)

    # save fg
    bg_motion = {"field2cam": []}
    fg_motion = {"field2cam": []}
    for frame_idx in range(len(traj)):
        # mesh
        mesh = traj[frame_idx]
        mesh.export(os.path.join(save_dir_fg, "%05d.obj" % frame_idx))

        # camera pose
        camera_pose = np.eye(4)
        camera_pose[:3] = camera[frame_idx][:3]
        bg_motion["field2cam"].append(camera_pose.tolist())
        fg_motion["field2cam"].append(camera_pose.tolist())

        # save bg mesh
        data["floor"].export(os.path.join(save_dir_bg, "%05d.obj" % frame_idx))

    # save to json
    with open(os.path.join(save_dir, "bg/motion.json"), "w") as file:
        json.dump(bg_motion, file)

    with open(os.path.join(save_dir, "fg/motion.json"), "w") as file:
        json.dump(fg_motion, file)

    # save field2world
    with open(os.path.join(save_dir, "bg/field2world.json"), "w") as file:
        field2world = np.eye(4)
        field2world[:3, :3] = cv2.Rodrigues(np.asarray([np.pi, 0, 0]))[0]
        json.dump(field2world.tolist(), file)

    # save intrinsics
    with open(os.path.join(save_dir, "camera.json"), "w") as file:
        camera_info = {}
        camera_info["raw_size"] = []
        camera_info["intrinsics"] = camera[:, 3].tolist()
        json.dump(camera_info, open("%s/camera.json" % (save_dir), "w"))


def simulate(opts):
    opts["urdf_template"] = opts["fg_motion"].split("-")[1].split("_")[0]
    (
        model,
        data_info,
        ref_dict,
        phys_model,
    ) = PPRTrainer.construct_test_model(opts)

    save_dir = make_save_dir(opts, sub_dir="simulate_%04d" % (opts["inst_id"]))
    phys_visualizer = PhysVisualizer(save_dir)

    # reset scale to avoid initial penetration
    data = PPRTrainer.simulate(phys_model, data_info, opts["inst_id"])
    fps = 1.0 / phys_model.frame_interval
    phys_visualizer.show("simulated_ref", data, fps=fps, view_mode="ref")
    phys_visualizer.show("simulated_bev", data, fps=fps, view_mode="bev")
    phys_visualizer.show("simulated_front", data, fps=fps, view_mode="front")

    data["floor"] = phys_visualizer.floor
    export_simulate_mesh(save_dir, data, tag="sim_traj")

    print("Results saved to %s" % (save_dir))
    return


def main(_):
    opts = get_config()
    simulate(opts)


if __name__ == "__main__":
    app.run(main)
