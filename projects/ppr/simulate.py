# Copyright (c) 2023 Gengshan Yang, Carnegie Mellon University.
"""
python projects/ppr/export.py --flagfile=logdir/cat-85-sub-sub-bob-pika-cate-b02/opts.log --load_suffix latest --inst_id 0
"""

import os, sys
import pdb
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
    print("Results saved to %s" % (save_dir))
    return


def main(_):
    opts = get_config()
    simulate(opts)


if __name__ == "__main__":
    app.run(main)
