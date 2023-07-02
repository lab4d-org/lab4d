# Copyright (c) 2023 Gengshan Yang, Carnegie Mellon University.
"""
python lab4d/export.py --flagfile=logdir/cat-85-sub-sub-bob-pika-cate-b02/opts.log --load_suffix latest --inst_id 0
"""

import os, sys
import json
from typing import NamedTuple, Tuple

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import trimesh
from absl import app, flags

cwd = os.getcwd()
if cwd not in sys.path:
    sys.path.insert(0, cwd)

from lab4d.config import get_config
from lab4d.dataloader import data_utils
from lab4d.engine.trainer import Trainer
from lab4d.nnutils.warping import SkinningWarp
from lab4d.nnutils.pose import ArticulationSkelMLP
from lab4d.utils.io import make_save_dir, save_rendered
from lab4d.utils.quat_transform import (
    dual_quaternion_to_se3,
    quaternion_translation_to_se3,
)

cudnn.benchmark = True


class ExportMeshFlags:
    flags.DEFINE_integer("inst_id", 0, "video/instance id")
    flags.DEFINE_integer("grid_size", 128, "grid size of marching cubes")
    flags.DEFINE_integer("num_frames", -1, "number of frames to render")
    flags.DEFINE_float("level", 0.0, "level of marching cubes")


class MotionParamsExpl(NamedTuple):
    """
    explicit motion params for reanimation and transfer
    """

    field2cam: Tuple[torch.Tensor, torch.Tensor]  # (quaternion, translation)
    t_articulation: Tuple[
        torch.Tensor, torch.Tensor
    ]  # dual quaternion, applies to skinning
    so3: torch.Tensor  # so3, applies to skeleton
    mesh_t: trimesh.Trimesh  # mesh at time t


def extract_deformation(field, mesh_rest, inst_id, render_length):
    device = next(field.parameters()).device
    xyz = torch.tensor(mesh_rest.vertices, dtype=torch.float32, device=device)
    offset = field.frame_offset[inst_id]
    inst_id = torch.tensor([inst_id], dtype=torch.long, device=device)

    motion_tuples = {}
    for frame_id in range(render_length):
        frame_id_torch = torch.tensor(
            [offset + frame_id], dtype=torch.long, device=device
        )
        field2cam = field.camera_mlp.get_vals(frame_id_torch)

        samples_dict = {}
        if isinstance(field.warp, SkinningWarp):
            (
                samples_dict["t_articulation"],
                samples_dict["rest_articulation"],
            ) = field.warp.articulation.get_vals_and_mean(frame_id_torch)
            t_articulation = samples_dict["t_articulation"]

            if isinstance(field.warp.articulation, ArticulationSkelMLP):
                so3 = field.warp.articulation.get_vals(frame_id_torch, return_so3=True)
            else:
                so3 = None
        else:
            t_articulation = None
            so3 = None

        xyz_t = field.forward_warp(
            xyz[None, None],
            field2cam,
            frame_id_torch,
            inst_id,
            samples_dict=samples_dict,
        )
        xyz_t = xyz_t[0, 0]
        mesh_t = trimesh.Trimesh(vertices=xyz_t.cpu().numpy(), faces=mesh_rest.faces)

        field2cam[1][:] /= field.logscale.exp()  # to world scale
        motion_expl = MotionParamsExpl(
            field2cam=field2cam,
            t_articulation=t_articulation,
            so3=so3,
            mesh_t=mesh_t,
        )
        motion_tuples[frame_id] = motion_expl
    return motion_tuples


def save_motion_params(meshes_rest, motion_tuples, save_dir):
    for cate, mesh_rest in meshes_rest.items():
        mesh_rest.export("%s/%s.obj" % (save_dir, cate))
        motion_params = {"field2cam": [], "t_articulation": [], "joint_so3": []}
        for frame_id, motion_expl in motion_tuples[cate].items():
            # save mesh
            motion_expl.mesh_t.export("%s/%s-%05d.obj" % (save_dir, cate, frame_id))

            # save motion params
            field2cam = quaternion_translation_to_se3(
                motion_expl.field2cam[0], motion_expl.field2cam[1]
            )  # 1,4,4
            motion_params["field2cam"].append(field2cam.cpu().numpy()[0].tolist())

            if motion_expl.t_articulation is not None:
                t_articulation = dual_quaternion_to_se3(
                    motion_expl.t_articulation
                )  # 1,K,4,4
                motion_params["t_articulation"].append(
                    t_articulation.cpu().numpy()[0].tolist()
                )
            if motion_expl.so3 is not None:
                motion_params["joint_so3"].append(
                    motion_expl.so3.cpu().numpy()[0].tolist()
                )  # K,3

        with open("%s/%s-motion.json" % (save_dir, cate), "w") as fp:
            json.dump(motion_params, fp)


@torch.no_grad()
def extract_motion_params(model, opts, data_info):
    # get rest mesh
    meshes_rest = model.fields.extract_canonical_meshes(
        grid_size=opts["grid_size"],
        level=opts["level"],
        inst_id=opts["inst_id"],
        use_visibility=False,
    )

    # get length of the seq
    vid_length = data_utils.get_vid_length(opts["inst_id"], data_info)
    if opts["num_frames"] > 0:
        render_length = opts["num_frames"]
    else:
        render_length = vid_length

    # get deformation
    motion_tuples = {}
    for cate, field in model.fields.field_params.items():
        motion_tuples[cate] = extract_deformation(
            field, meshes_rest[cate], opts["inst_id"], render_length
        )
    return meshes_rest, motion_tuples


def export(opts):
    model, data_info, ref_dict = Trainer.construct_test_model(opts)
    save_dir = make_save_dir(opts, sub_dir="export_%04d" % (opts["inst_id"]))

    # save motion paramters
    meshes_rest, motion_tuples = extract_motion_params(model, opts, data_info)
    save_motion_params(meshes_rest, motion_tuples, save_dir)

    # save reference images
    raw_size = data_info["raw_size"][opts["inst_id"]]  # full range of pixels
    save_rendered(ref_dict, save_dir, raw_size, data_info["apply_pca_fn"])
    print("Saved to %s" % save_dir)


def main(_):
    opts = get_config()
    export(opts)


if __name__ == "__main__":
    app.run(main)
