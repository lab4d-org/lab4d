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

from lab4d.export import get_config, MotionParamsExpl
from lab4d.utils.io import make_save_dir, save_rendered
from lab4d.utils.quat_transform import quaternion_translation_to_se3
from projects.diffgs.trainer import GSplatTrainer as Trainer


def extract_deformation(field, inst_id):
    device = next(field.parameters()).device
    # get corresponding frame ids
    frame_mapping = torch.tensor(field.frame_mapping, device=device)
    frame_offset = field.frame_offset
    frame_ids = frame_mapping[frame_offset[inst_id] : frame_offset[inst_id + 1]]
    start_id = frame_ids[0]
    print("Extracting motion parameters for inst id:", inst_id)
    print("Frame ids with the video:", frame_ids - start_id)

    inst_id = torch.tensor([inst_id], dtype=torch.long, device=device)

    motion_tuples = {}
    for frame_id in frame_ids:
        print("Frame id:", frame_id)
        frame_id_sub = frame_id - start_id

        # get field2cam
        se3_mat = field.gaussians.get_extrinsics(frame_id).cpu().numpy()

        if field.gaussians.use_timesync:
            mesh_t = field.gaussians.create_mesh_visualization(frame_id_sub)
        else:
            mesh_t = field.gaussians.create_mesh_visualization(frame_id)

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
    # get rest mesh
    model.gaussians.update_geometry_aux()
    meshes_rest = {"fg": model.gaussians.proxy_geometry}

    # get deformation
    motion_tuples = {"fg": extract_deformation(model, inst_id=opts["inst_id"])}
    return meshes_rest, motion_tuples


def save_motion_params(meshes_rest, motion_tuples, save_dir):
    for cate, mesh_rest in meshes_rest.items():
        motion_params = {"field2cam": {}, "t_articulation": {}, "joint_so3": {}}
        os.makedirs("%s/fg/mesh/" % save_dir, exist_ok=True)
        os.makedirs("%s/bg/mesh/" % save_dir, exist_ok=True)
        os.makedirs("%s/fg/bone/" % save_dir, exist_ok=True)
        for frame_id, motion_expl in motion_tuples[cate].items():
            frame_id = int(frame_id)
            # save motion params
            motion_params["field2cam"][frame_id] = motion_expl.field2cam.tolist()

            if motion_expl.t_articulation is not None:
                motion_params["t_articulation"][
                    frame_id
                ] = motion_expl.t_articulation.tolist()

            if motion_expl.so3 is not None:
                motion_params["joint_so3"][frame_id] = motion_expl.so3.tolist()  # K,3
            # save mesh
            if cate == "bg" and frame_id != 0:
                continue
            motion_expl.mesh_t.export(
                "%s/%s/mesh/%05d.obj" % (save_dir, cate, frame_id)
            )
            if motion_expl.bone_t is not None:
                motion_expl.bone_t.export(
                    "%s/%s/bone/%05d.obj" % (save_dir, cate, frame_id)
                )

        with open("%s/%s/motion.json" % (save_dir, cate), "w") as fp:
            json.dump(motion_params, fp)

        # save camera mesh
        mesh_rest.export("%s/%s-mesh.obj" % (save_dir, cate))


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
