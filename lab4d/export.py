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
from lab4d.utils.vis_utils import append_xz_plane, draw_cams

cudnn.benchmark = True


class ExportMeshFlags:
    flags.DEFINE_integer("inst_id", 0, "video/instance id")
    flags.DEFINE_integer("grid_size", 128, "grid size of marching cubes")
    flags.DEFINE_float(
        "level", 0.0, "contour value of marching cubes use to search for isosurfaces"
    )
    flags.DEFINE_float(
        "vis_thresh", 0.0, "visibility threshold to remove invisible pts, -inf to inf"
    )
    flags.DEFINE_boolean("extend_aabb", False, "use extended aabb for meshing (for bg)")


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
    bone_t: trimesh.Trimesh  # bone center at time t


def extract_deformation(field, mesh_rest, inst_id):
    device = next(field.parameters()).device
    # get corresponding frame ids
    frame_mapping = torch.tensor(field.frame_mapping, device=device)
    frame_offset = field.frame_offset
    frame_ids = frame_mapping[frame_offset[inst_id] : frame_offset[inst_id + 1]]
    start_id = frame_ids[0]
    print("Extracting motion parameters for inst id:", inst_id)
    print("Frame ids with the video:", frame_ids - start_id)

    xyz = torch.tensor(mesh_rest.vertices, dtype=torch.float32, device=device)
    inst_id = torch.tensor([inst_id], dtype=torch.long, device=device)

    motion_tuples = {}
    for frame_id in frame_ids:
        frame_id = frame_id[None]
        field2cam = field.camera_mlp.get_vals(frame_id)

        samples_dict = {}
        se3_mat = quaternion_translation_to_se3(field2cam[0], field2cam[1])[0]
        se3_mat = se3_mat.cpu().numpy()
        if hasattr(field, "warp") and isinstance(field.warp, SkinningWarp):
            (
                samples_dict["t_articulation"],
                samples_dict["rest_articulation"],
            ) = field.warp.articulation.get_vals_and_mean(frame_id)
            t_articulation = samples_dict["t_articulation"]

            if isinstance(field.warp.articulation, ArticulationSkelMLP):
                so3 = field.warp.articulation.get_vals(frame_id, return_so3=True)[0]
                so3 = so3.cpu().numpy()
            else:
                so3 = None

            # bones
            mesh_bones_t = field.warp.skinning_model.draw_gaussian(
                (
                    samples_dict["t_articulation"][0][0],
                    samples_dict["t_articulation"][1][0],
                ),
                field.warp.articulation.edges,
            )
            # 1,K,4,4
            t_articulation = dual_quaternion_to_se3(t_articulation)[0]
            t_articulation = t_articulation.cpu().numpy()
        else:
            t_articulation = None
            so3 = None
            mesh_bones_t = None

        if hasattr(field, "warp"):
            # warp mesh
            xyz_t = field.warp(
                xyz[None, None], frame_id, inst_id, samples_dict=samples_dict
            )[0, 0]
            mesh_t = trimesh.Trimesh(
                vertices=xyz_t.cpu().numpy(), faces=mesh_rest.faces, process=False
            )
        else:
            mesh_t = mesh_rest.copy()

        # # color
        # color = field.extract_canonical_color(mesh_t)
        # mesh_t.visual.vertex_colors[:,:3] = color * 255
        # XYZ color
        xyz_color = mesh_rest.vertices
        xyz_color = (xyz_color - xyz_color.min(0)) / (xyz_color.max(0) - xyz_color.min(0))
        mesh_t.visual.vertex_colors[:, :3] = xyz_color * 255

        motion_expl = MotionParamsExpl(
            field2cam=se3_mat,
            t_articulation=t_articulation,
            so3=so3,
            mesh_t=mesh_t,
            bone_t=mesh_bones_t,
        )
        frame_id_sub = frame_id[0] - start_id
        motion_tuples[frame_id_sub] = motion_expl

    if hasattr(field, "warp") and isinstance(field.warp, SkinningWarp):
        # modify rest mesh based on instance morphological changes on bones
        # idendity transformation of cameras
        field2cam_rot_idn = torch.zeros_like(field2cam[0])
        field2cam_rot_idn[..., 0] = 1.0
        field2cam_idn = (field2cam_rot_idn, torch.zeros_like(field2cam[1]))
        # bone stretching from rest to instance id
        samples_dict["t_articulation"] = field.warp.articulation.get_mean_vals(
            inst_id=inst_id
        )
        xyz_i = field.forward_warp(
            xyz[None, None],
            field2cam_idn,
            None,
            inst_id,
            samples_dict=samples_dict,
        )
        xyz_i = xyz_i[0, 0]
        mesh_rest = trimesh.Trimesh(vertices=xyz_i.cpu().numpy(), faces=mesh_rest.faces)

    return mesh_rest, motion_tuples


def rescale_motion_tuples(motion_tuples, field_scale):
    """
    rescale motion tuples to world scale
    """
    for frame_id, motion_tuple in motion_tuples.items():
        motion_tuple.field2cam[:3, 3] /= field_scale
        motion_tuple.mesh_t.apply_scale(1.0 / field_scale)
        if motion_tuple.bone_t is not None:
            motion_tuple.bone_t.apply_scale(1.0 / field_scale)
        if motion_tuple.t_articulation is not None:
            motion_tuple.t_articulation[:, :3, 3] /= field_scale
    return


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
        mesh_camtraj = draw_cams(list(motion_params["field2cam"].values()))
        mesh_camtraj.export("%s/%s-camtraj.obj" % (save_dir, cate))
        mesh_rest.export("%s/%s-mesh.obj" % (save_dir, cate))


@torch.no_grad()
def extract_motion_params(model, opts, data_info):
    # get rest mesh
    meshes_rest = model.fields.extract_canonical_meshes(
        grid_size=opts["grid_size"],
        level=opts["level"],
        inst_id=opts["inst_id"],
        vis_thresh=opts["vis_thresh"],
        use_extend_aabb=opts["extend_aabb"],
    )

    # get deformation
    motion_tuples = {}
    for cate, field in model.fields.field_params.items():
        meshes_rest[cate], motion_tuples[cate] = extract_deformation(
            field, meshes_rest[cate], opts["inst_id"]
        )
        # color
        color = field.extract_canonical_color(meshes_rest[cate])
        meshes_rest[cate].visual.vertex_colors[:,:3] = color * 255

    # scale
    if "bg" in model.fields.field_params.keys():
        bg_field = model.fields.field_params["bg"]
        bg_scale = bg_field.logscale.exp().cpu().numpy()
    if "fg" in model.fields.field_params.keys():
        fg_field = model.fields.field_params["fg"]
        fg_scale = fg_field.logscale.exp().cpu().numpy()

    if (
        "bg" in model.fields.field_params.keys()
        and model.fields.field_params["bg"].valid_field2world()
    ):
        # visualize ground plane
        field2world = (
            model.fields.field_params["bg"].get_field2world(opts["inst_id"]).cpu()
        )
        field2world[..., :3, 3] *= bg_scale
        meshes_rest["bg"] = append_xz_plane(
            meshes_rest["bg"], field2world.inverse(), scale=20 * bg_scale
        )

    if "fg" in model.fields.field_params.keys():
        meshes_rest["fg"] = meshes_rest["fg"].apply_scale(1.0 / fg_scale)
        rescale_motion_tuples(motion_tuples["fg"], fg_scale)
    if "bg" in model.fields.field_params.keys():
        meshes_rest["bg"] = meshes_rest["bg"].apply_scale(1.0 / bg_scale)
        rescale_motion_tuples(motion_tuples["bg"], bg_scale)
    return meshes_rest, motion_tuples


@torch.no_grad()
def export(opts, Trainer=Trainer):
    model, data_info, ref_dict = Trainer.construct_test_model(opts, return_refs=False)
    save_dir = make_save_dir(opts, sub_dir="export_%04d" % (opts["inst_id"]))

    # save motion paramters
    meshes_rest, motion_tuples = extract_motion_params(model, opts, data_info)
    save_motion_params(meshes_rest, motion_tuples, save_dir)

    # save scene to world transform
    if (
        "bg" in model.fields.field_params.keys()
        and model.fields.field_params["bg"].valid_field2world()
    ):
        field2world = model.fields.field_params["bg"].get_field2world(opts["inst_id"])
        field2world = field2world.cpu().numpy().tolist()
        json.dump(field2world, open("%s/bg/field2world.json" % (save_dir), "w"))

    # same raw image size and intrinsics
    with torch.no_grad():
        intrinsics = model.intrinsics.get_intrinsics(opts["inst_id"])
        camera_info = {}
        camera_info["raw_size"] = data_info["raw_size"][opts["inst_id"]].tolist()
        camera_info["intrinsics"] = intrinsics.cpu().numpy().tolist()
        if "bg" in model.fields.field_params.keys():
            bg_field = model.fields.field_params["bg"]
            bg_scale = bg_field.logscale.exp().cpu().numpy().tolist()
            camera_info["bg_scale"] = bg_scale
        json.dump(camera_info, open("%s/camera.json" % (save_dir), "w"))

    if ref_dict is not None:
        # save reference images
        raw_size = data_info["raw_size"][opts["inst_id"]]  # full range of pixels
        save_rendered(ref_dict, save_dir, raw_size, data_info["apply_pca_fn"])
        print("Saved to %s" % save_dir)

    # mesh rendering: ref | bev
    cmd = "python lab4d/render_mesh.py --mode shape --testdir %s" % (save_dir)
    print("Running: %s" % cmd)
    os.system(cmd)
    cmd = "python lab4d/render_mesh.py --view bev --testdir %s" % (save_dir)
    print("Running: %s" % cmd)
    os.system(cmd)


def main(_):
    opts = get_config()
    export(opts)


if __name__ == "__main__":
    app.run(main)
