# Copyright (c) 2023 Gengshan Yang, Carnegie Mellon University.
# python scripts/render.py --seqname --flagfile=logdir/cat-0t10-fg-bob-d0-long/opts.log --load_suffix latest

import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from absl import app, flags

cwd = os.getcwd()
if cwd not in sys.path:
    sys.path.insert(0, cwd)

from lab4d.config import get_config
from lab4d.dataloader import data_utils
from lab4d.engine.trainer import Trainer
from lab4d.utils.camera_utils import (
    construct_batch,
    get_fixed_cam,
    get_object_to_camera_matrix,
    get_orbit_camera,
    get_rotating_cam,
    create_field2cam,
    get_bev_cam,
)
from lab4d.utils.geom_utils import K2inv, K2mat, mat2K
from lab4d.utils.io import make_save_dir, save_rendered
from lab4d.utils.profile_utils import torch_profile

cudnn.benchmark = True


class RenderFlags:
    """Flags for the renderer."""

    flags.DEFINE_integer("inst_id", 0, "video/instance id")
    flags.DEFINE_integer("render_res", 128, "rendering resolution")
    flags.DEFINE_string(
        "viewpoint", "ref", "camera viewpoint, {ref,rot-elevation-degree,rot-60-0,...}"
    )
    flags.DEFINE_integer("freeze_id", -1, "freeze frame id to render, no freeze if -1")
    flags.DEFINE_integer("num_frames", -1, "number of frames to render")


def construct_batch_from_opts(opts, model, data_info):
    device = "cuda"
    # data info
    if "motion_id" in opts:
        video_id = opts["motion_id"]
    else:
        video_id = opts["inst_id"]
    # ref video size
    raw_size = data_info["raw_size"][video_id]  # full range of pixels
    # ref video length
    vid_length = data_utils.get_vid_length(video_id, data_info)
    if opts["num_frames"] > 0:
        render_length = opts["num_frames"]
    else:
        render_length = vid_length

    # whether to freeze a frame
    if opts["freeze_id"] >= 0 and opts["freeze_id"] < vid_length:
        frameid_sub = np.asarray([opts["freeze_id"]] * render_length)  # full video
    elif opts["freeze_id"] == -1:
        frameid_sub = np.linspace(0, render_length - 1, render_length).astype(np.int32)
    else:
        raise ValueError("frame id %d out of range" % opts["freeze_id"])

    # get cameras wrt each field
    with torch.no_grad():
        field2cam_fr = model.fields.get_cameras(inst_id=opts["inst_id"])
        intrinsics_fr = model.intrinsics.get_vals(
            frameid_sub + data_info["frame_info"]["frame_offset"][video_id]
        )
        aabb = model.fields.get_aabb()
    # convert to numpy
    for k, v in field2cam_fr.items():
        field2cam_fr[k] = v.cpu().numpy()
        aabb[k] = aabb[k].cpu().numpy()
    intrinsics_fr = intrinsics_fr.cpu().numpy()

    # construct batch from user input
    if opts["viewpoint"] == "ref":
        # rotate around viewpoint
        field2cam = None

        # camera_int = None
        crop2raw = np.zeros((len(frameid_sub), 4))
        crop2raw[:, 0] = raw_size[1] / opts["render_res"]
        crop2raw[:, 1] = raw_size[0] / opts["render_res"]
        camera_int = mat2K(K2inv(crop2raw) @ K2mat(intrinsics_fr))
        crop2raw = None
    elif opts["viewpoint"].startswith("rot"):
        # rotate around field, format: rot-evelvation-degree
        elev, max_angle = [int(val) for val in opts["viewpoint"].split("-")[1:]]

        # bg_to_cam
        obj_size = (aabb["fg"][1, :] - aabb["fg"][0, :]).max()
        cam_traj = get_rotating_cam(
            len(frameid_sub), distance=obj_size * 2.5, max_angle=max_angle
        )
        cam_elev = get_object_to_camera_matrix(elev, [1, 0, 0], 0)[None]
        cam_traj = cam_traj @ cam_elev
        field2cam = create_field2cam(cam_traj, field2cam_fr.keys())

        camera_int = np.zeros((len(frameid_sub), 4))

        # focal length = img height * distance / obj height
        camera_int[:, :2] = opts["render_res"] * 2 * 0.8  # zoom out a bit
        camera_int[:, 2:] = opts["render_res"] / 2
        raw_size = (640, 640)  # full range of pixels
        crop2raw = None
    elif opts["viewpoint"].startswith("bev"):
        elev = int(opts["viewpoint"].split("-")[1])
        # render bird's eye view
        if "bg" in field2cam_fr.keys():
            # get bev wrt first frame image
            # center_to_bev = centered_to_camt0 x centered_to_rotated x camt0_to_centered x bg_to_camt0
            center_to_bev = get_object_to_camera_matrix(elev, [1, 0, 0], 0)[None]
            camt0_to_center = np.eye(4)
            camt0_to_center[2, 3] = -field2cam_fr["bg"][0, 2, 3]
            camt0_to_bev = (
                np.linalg.inv(camt0_to_center) @ center_to_bev @ camt0_to_center
            )
            bg2bev = camt0_to_bev @ field2cam_fr["bg"][:1]
            # push cameras away
            bg2bev[..., 2, 3] *= 3
            field2cam = {"bg": np.tile(bg2bev, (vid_length, 1, 1))}
            if "fg" in field2cam_fr.keys():
                # if both fg and bg
                camt2bg = np.linalg.inv(field2cam_fr["bg"])
                fg2camt = field2cam_fr["fg"]
                field2cam["fg"] = field2cam["bg"] @ camt2bg @ fg2camt
        elif "fg" in field2cam_fr.keys():
            # if only fg
            field2cam = {"fg": get_bev_cam(field2cam_fr["fg"], elev=elev)}
        else:
            raise NotImplementedError

        camera_int = np.zeros((len(frameid_sub), 4))
        camera_int[:, :2] = opts["render_res"] * 2
        camera_int[:, 2:] = opts["render_res"] / 2
        raw_size = (640, 640)  # full range of pixels
        crop2raw = None
    else:
        raise ValueError("Unknown viewpoint type %s" % opts.viewpoint)

    batch = construct_batch(
        inst_id=opts["inst_id"],
        frameid_sub=frameid_sub,
        eval_res=opts["render_res"],
        field2cam=field2cam,
        camera_int=camera_int,
        crop2raw=crop2raw,
        device=device,
    )
    return batch, raw_size


@torch.no_grad()
def render_batch(model, batch):
    # render batch
    start_time = time.time()
    rendered = model.evaluate(batch, is_pair=False)
    print("rendering time: %.3f" % (time.time() - start_time))

    return rendered


def render(opts, construct_batch_func):
    # load model/data
    opts["logroot"] = sys.argv[1].split("=")[1].rsplit("/", 2)[0]
    model, data_info, ref_dict = Trainer.construct_test_model(opts)
    batch, raw_size = construct_batch_func(opts, model, data_info)
    save_dir = make_save_dir(opts, sub_dir="renderings_%04d" % (opts["inst_id"]))

    # render
    with torch_profile(save_dir, "profile", enabled=opts["profile"]):
        rendered = render_batch(model, batch)

    rendered.update(ref_dict)
    save_rendered(rendered, save_dir, raw_size, data_info["apply_pca_fn"])
    print("Saved to %s" % save_dir)


def main(_):
    opts = get_config()
    render(opts, construct_batch_func=construct_batch_from_opts)


if __name__ == "__main__":
    app.run(main)
