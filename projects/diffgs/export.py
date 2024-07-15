# Copyright (c) 2023 Gengshan Yang, Carnegie Mellon University.
"""
python projects/ppr/export.py --flagfile=logdir/cat-85-sub-sub-bob-pika-cate-b02/opts.log --load_suffix latest --inst_id 0
"""
import pdb
import os, sys
from absl import app
import json
import numpy as np
import torch

cwd = os.getcwd()
if cwd not in sys.path:
    sys.path.insert(0, cwd)

from lab4d.export import get_config, MotionParamsExpl, save_motion_params
from lab4d.utils.io import make_save_dir, save_rendered
from lab4d.utils.quat_transform import quaternion_translation_to_se3
from projects.diffgs.trainer import GSplatTrainer as Trainer


def extract_deformation(field, frame_ids, use_timesync=False):
    field.update_trajectory(frame_ids)
    start_id = frame_ids[0]

    motion_tuples = {}
    for frame_id in frame_ids:
        print("Frame id:", frame_id)
        frame_id_sub = frame_id - start_id

        # get field2cam
        se3_mat = field.get_extrinsics(frame_id).cpu().numpy()

        if use_timesync:
            mesh_t = field.create_mesh_visualization(frame_id_sub)
        else:
            mesh_t = field.create_mesh_visualization(frame_id)

        t_articulation = None
        so3 = None
        mesh_bones_t = None

        motion_expl = MotionParamsExpl(
            field2cam=se3_mat,
            t_articulation=t_articulation,
            so3=so3,
            mesh_t=mesh_t,
            bone_t=mesh_bones_t,
        )

        motion_tuples[frame_id_sub] = motion_expl

    return motion_tuples


@torch.no_grad()
def extract_motion_params(model, opts):
    device = next(model.parameters()).device
    inst_id = torch.tensor([opts["inst_id"]], dtype=torch.long, device=device)
    # get corresponding frame ids
    frame_mapping = torch.tensor(model.frame_mapping, device=device)
    frame_offset = model.frame_offset
    frame_ids = frame_mapping[frame_offset[inst_id] : frame_offset[inst_id + 1]]
    print("Extracting motion parameters for inst id:", inst_id)
    print("Frame ids with the video:", frame_ids - frame_ids[0])

    # get rest mesh
    meshes_rest = {}
    motion_tuples = {}
    for field, cate in model.gaussians.get_all_children():
        if cate == "": continue
        field.update_geometry_aux(all_pts=True)
        meshes_rest[cate] = field.proxy_geometry
        motion_tuples[cate] = extract_deformation(field, frame_ids, use_timesync=opts["use_timesync"])
    return meshes_rest, motion_tuples


@torch.no_grad()
def export(opts, Trainer=Trainer):
    model, data_info, ref_dict = Trainer.construct_test_model(opts)
    save_dir = make_save_dir(opts, sub_dir="export_%04d" % (opts["inst_id"]))

    # save motion paramters
    meshes_rest, motion_tuples = extract_motion_params(model, opts)
    save_motion_params(meshes_rest, motion_tuples, save_dir)

    # same raw image size and intrinsics
    with torch.no_grad():
        intrinsics = model.intrinsics.get_intrinsics(opts["inst_id"])
        camera_info = {}
        camera_info["raw_size"] = data_info["raw_size"][opts["inst_id"]].tolist()
        camera_info["intrinsics"] = intrinsics.cpu().numpy().tolist()
        json.dump(camera_info, open("%s/camera.json" % (save_dir), "w"))

    # save reference images
    raw_size = data_info["raw_size"][opts["inst_id"]]  # full range of pixels
    save_rendered(ref_dict, save_dir, raw_size, data_info["apply_pca_fn"])
    print("Saved to %s" % save_dir)

    # mesh rendering
    cmd = "python lab4d/render_mesh.py --testdir %s" % (save_dir)
    print("Running: %s" % cmd)
    os.system(cmd)


def main(_):
    opts = get_config()
    export(opts, Trainer=Trainer)


if __name__ == "__main__":
    app.run(main)
