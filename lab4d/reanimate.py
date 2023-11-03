# Copyright (c) 2023 Gengshan Yang, Carnegie Mellon University.
# python lab4d/reanimate.py --flagfile=logdir/human-48-dinov2-skel-e120/opts.log --load_suffix latest --motion_id 20 --inst_id 0

import json
import os
import sys
import pdb

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from absl import app, flags

cwd = os.getcwd()
if cwd not in sys.path:
    sys.path.insert(0, cwd)

from lab4d.config import get_config
from lab4d.render import construct_batch_from_opts, render
from lab4d.utils.profile_utils import torch_profile
from lab4d.utils.quat_transform import se3_to_quaternion_translation

cudnn.benchmark = True


class RenderFlags:
    """Flags for the renderer."""

    flags.DEFINE_integer("motion_id", 0, "motion id")


def construct_batch_from_opts_reanimate(opts, model, data_info):
    device = "cuda"
    motion_id = opts["motion_id"]
    frame_offset_raw = data_info["frame_info"]["frame_offset_raw"]
    opts["num_frames"] = frame_offset_raw[motion_id + 1] - frame_offset_raw[motion_id]
    batch, raw_size = construct_batch_from_opts(opts, model, data_info)
    batch["field2cam"] = {}

    if opts["field_type"] == "fg" or opts["field_type"] == "comp":
        # load motion data
        motion_path = "%s/%s-%s/export_%04d/fg/motion.json" % (
            opts["logroot"],
            opts["seqname"],
            opts["logname"],
            opts["motion_id"],
        )
        with open(motion_path, "r") as fp:
            motion_data = json.load(fp)
        t_articulation = np.asarray(motion_data["t_articulation"])
        field2cam = np.asarray(list(motion_data["field2cam"].values()))

        # joint angles
        joint_so3 = np.asarray(motion_data["joint_so3"])
        joint_so3 = torch.tensor(joint_so3, dtype=torch.float32, device=device)

        # root pose
        field2cam = torch.tensor(field2cam, dtype=torch.float32, device=device)
        field2cam = field2cam.reshape(-1, 4, 4)
        field2cam = se3_to_quaternion_translation(field2cam, tuple=False)

        batch["joint_so3"] = joint_so3
        batch["field2cam"]["fg"] = field2cam

    if opts["field_type"] == "bg" or opts["field_type"] == "comp":
        # add background
        motion_path = "%s/%s-%s/export_%04d/bg/motion.json" % (
            opts["logroot"],
            opts["seqname"],
            opts["logname"],
            opts["motion_id"],
        )
        with open(motion_path, "r") as fp:
            motion_data = json.load(fp)
        field2cam = np.asarray(list(motion_data["field2cam"].values()))
        field2cam = torch.tensor(field2cam, dtype=torch.float32, device=device)
        field2cam = field2cam.reshape(-1, 4, 4)
        field2cam = se3_to_quaternion_translation(field2cam, tuple=False)

        batch["field2cam"]["bg"] = field2cam

    # indicator for reanimation
    batch["motion_id"] = opts["motion_id"] * torch.ones_like(batch["dataid"])

    return batch, raw_size


def main(_):
    opts = get_config()
    render(opts, construct_batch_func=construct_batch_from_opts_reanimate)


if __name__ == "__main__":
    app.run(main)
