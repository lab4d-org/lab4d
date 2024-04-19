# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
import os, sys
import pdb

sys.path.insert(0, os.getcwd())

import numpy as np
import trimesh
import torch
import cv2
import pdb
from scipy.spatial.transform import Rotation as R

from lab4d.utils.geom_utils import (
    obj_to_cam,
    pinhole_projection,
    K2mat,
    # render_color,
    # render_flow,
)
from preprocess.scripts.write_config import write_config
from preprocess.scripts.crop import extract_crop

from projects.diffgs.render_utils import render_color, render_flow

sys.path.insert(0, "preprocess/third_party/vcnplus/")
from flowutils.flowlib import flow_to_image

# from ext_utils.util_flow import write_pfm
import soft_renderer as sr
import argparse

parser = argparse.ArgumentParser(description="render data")
parser.add_argument("--outdir", default="eagle-0000", help="output dir")
parser.add_argument("--model", default="eagle", help="model to render, {eagle, hands}")
parser.add_argument("--rot_axis", default="y", help="axis to rotate around")
parser.add_argument("--nframes", default=8, type=int, help="number of frames to render")
parser.add_argument(
    "--alpha", default=1.0, type=float, help="0-1, percentage of a full cycle"
)
parser.add_argument(
    "--init_a",
    default=0.25,
    type=float,
    help="0-1, percentage of a full cycle for initial pose",
)
parser.add_argument("--xspeed", default=0, type=float, help="times speed up")
parser.add_argument("--focal", default=2, type=float, help="focal length")
parser.add_argument("--d_obj", default=3, type=float, help="object depth")
parser.add_argument(
    "--can_rand", dest="can_rand", action="store_true", help="ranomize canonical space"
)
parser.add_argument("--img_size", default=512, type=int, help="image size")
parser.add_argument(
    "--render_flow", dest="render_flow", action="store_true", help="render flow"
)


if __name__ == "__main__":
    args = parser.parse_args()
    # io
    img_size = args.img_size
    d_obj = args.d_obj
    filedir = "database/processed/"
    datadir = "database/synthetic/"

    rot_rand = torch.Tensor(R.random().as_matrix()).cuda()

    overts_list = []
    for i in range(args.nframes):
        if args.model == "eagle":
            mesh = sr.Mesh.from_obj(
                "%s/eagle/Eagle-original_%06d.obj" % (datadir, int(i * args.xspeed)),
                load_texture=True,
                texture_res=5,
                texture_type="surface",
            )
        elif args.model == "hands":
            mesh = sr.Mesh.from_obj(
                "%s/hands/hands_%06d.obj" % (datadir, int(1 + i * args.xspeed)),
                load_texture=True,
                texture_res=100,
                texture_type="surface",
            )

        overts = mesh.vertices
        if i == 0:
            center = overts.mean(1)[:, None]
            scale = max((overts - center)[0].abs().max(0)[0])

        overts -= center
        overts *= 1.0 / float(scale)
        overts[:, :, 1] *= -1  # aligh with camera coordinate

        # random rot
        if args.can_rand:
            overts[0] = overts[0].matmul(rot_rand.T)

        overts_list.append(overts)
    colors = mesh.textures
    faces = mesh.faces

    os.makedirs(
        "%s/JPEGImages/Full-Resolution/%s/" % (filedir, args.outdir), exist_ok=True
    )
    os.makedirs(
        "%s/JPEGImagesRaw/Full-Resolution/%s/" % (filedir, args.outdir), exist_ok=True
    )
    os.makedirs(
        "%s/Annotations/Full-Resolution/%s/" % (filedir, args.outdir), exist_ok=True
    )
    os.makedirs(
        "%s/Cameras/Full-Resolution/%s/" % (filedir, args.outdir), exist_ok=True
    )
    os.makedirs("%s/Meshes/Full-Resolution/%s/" % (filedir, args.outdir), exist_ok=True)
    os.makedirs("%s/Depth/Full-Resolution/%s/" % (filedir, args.outdir), exist_ok=True)
    os.makedirs("%s/Normal/Full-Resolution/%s/" % (filedir, args.outdir), exist_ok=True)

    # soft renderer
    renderer = sr.SoftRenderer(
        image_size=img_size,
        sigma_val=1e-12,
        camera_mode="look_at",
        perspective=False,
        aggr_func_rgb="hard",
        light_mode="vertex",
        light_intensity_ambient=1.0,
        light_intensity_directionals=0.0,
    )
    # light_intensity_ambient=0.,light_intensity_directionals=1., light_directions=[-1.,-0.5,1.])

    rtks = []
    verts_ndc_list = []
    for i in range(0, args.nframes):
        verts = overts_list[i]

        # set cameras
        # rotx = np.random.rand()
        if args.rot_axis == "x":
            rotx = (args.init_a * +args.alpha * i / args.nframes) * np.pi * 2
        else:
            rotx = 0.0
        #    if i==0: rotx=0.
        if args.rot_axis == "y":
            roty = (args.init_a + args.alpha * i / args.nframes) * np.pi * 2
        else:
            roty = 0
        rotz = 0.0
        Rmat = cv2.Rodrigues(np.asarray([rotx, roty, rotz]))[0]
        Rmat = torch.Tensor(Rmat).cuda()
        # random rot
        if args.can_rand:
            Rmat = Rmat.matmul(rot_rand.T)
        Tmat = torch.Tensor([0, 0, d_obj]).cuda()
        K = torch.Tensor([args.focal, args.focal, 0, 0]).cuda()
        Kimg = torch.Tensor(
            [
                args.focal * img_size / 2.0,
                args.focal * img_size / 2.0,
                img_size / 2.0,
                img_size / 2.0,
            ]
        ).cuda()
        Kmat = K2mat(K)

        # use first frame cam as world-to-cam
        # add RTK: [R_3x3|T_3x1]
        #          [fx,fy,px,py], to the ndc space
        rtk = torch.eye(4).to(Rmat.device)
        rtk[:3, :3] = Rmat
        rtk[:3, 3] = Tmat

        if i == 0:
            rtk0 = rtk.clone()
        rtks.append(rtk0.cpu().numpy())
        # rtks.append(rtk.cpu().numpy())

        # obj-cam transform
        verts_view = obj_to_cam(verts, rtk[None])
        # mesh_cam = trimesh.Trimesh(
        #     vertices=verts_view[0].cpu().numpy(), faces=faces[0].cpu().numpy()
        # )
        # trimesh.repair.fix_inversion(mesh_cam)

        # pespective projection
        verts = pinhole_projection(Kmat[None], verts_view, keep_depth=True)
        verts_ndc_list.append(verts.clone())

        # render sil+rgb
        rendered = render_color(renderer, verts, faces, colors, texture_type="surface")
        rendered_img = rendered[0, :3].permute(1, 2, 0).cpu().numpy() * 255
        rendered_sil = rendered[0, -1].cpu().numpy() * 128
        bgcolor = 255 - rendered_img[rendered_sil.astype(bool)].mean(0)
        rendered_img[~rendered_sil.astype(bool)] = bgcolor[None]

        # render depth
        depth = (verts_view[..., 2:3]).repeat((1, 1, 3))
        rendered_depth = render_color(
            renderer, verts, faces, depth, texture_type="vertex"
        )
        rendered_depth = rendered_depth[0, 0].cpu().numpy()

        # zero noermal
        rendered_normal = np.zeros_like(rendered_img)

        cv2.imwrite(
            "%s/JPEGImages/Full-Resolution/%s/%05d.jpg" % (filedir, args.outdir, i),
            rendered_img[:, :, ::-1],
        )
        cv2.imwrite(
            "%s/JPEGImagesRaw/Full-Resolution/%s/%05d.jpg" % (filedir, args.outdir, i),
            rendered_img[:, :, ::-1],
        )
        np.save(
            "%s/Annotations/Full-Resolution/%s/%05d.npy" % (filedir, args.outdir, i),
            (rendered_sil > 0).astype(np.int8),
        )
        # mesh_cam.export(
        #     "%s/Meshes/Full-Resolution/%s/%05d.obj" % (filedir, args.outdir, i)
        # )
        np.save(
            "%s/Depth/Full-Resolution/%s/%05d.npy" % (filedir, args.outdir, i),
            rendered_depth,
        )
        np.save(
            "%s/Normal/Full-Resolution/%s/%05d.npy" % (filedir, args.outdir, i),
            rendered_normal,
        )
        print(
            "saved to %s/JPEGImages/Full-Resolution/%s/%05d.jpg"
            % (filedir, args.outdir, i)
        )

    rtks = np.asarray(rtks)
    np.save(
        "%s/Cameras/Full-Resolution/%s/01-canonical.npy" % (filedir, args.outdir), rtks
    )

    if args.render_flow:
        for dframe in [1, 2, 4, 8]:
            print("dframe: %d" % (dframe))
            flobw_outdir = "%s/FlowBW_%d/Full-Resolution/%s/" % (
                filedir,
                dframe,
                args.outdir,
            )
            flofw_outdir = "%s/FlowFW_%d/Full-Resolution/%s/" % (
                filedir,
                dframe,
                args.outdir,
            )
            os.makedirs(flofw_outdir, exist_ok=True)
            os.makedirs(flobw_outdir, exist_ok=True)
            # render flow
            for i in range(dframe, args.nframes):
                verts_ndc = verts_ndc_list[i - dframe]
                verts_ndc_n = verts_ndc_list[i]
                flow_fw = render_flow(renderer, verts_ndc, faces, verts_ndc_n)
                flow_bw = render_flow(renderer, verts_ndc_n, faces, verts_ndc)
                # to pixels
                flow_fw = flow_fw * (img_size - 1) / 2
                flow_bw = flow_bw * (img_size - 1) / 2
                flow_fw = flow_fw.cpu().numpy()[0].astype(np.float16)
                flow_bw = flow_bw.cpu().numpy()[0].astype(np.float16)

                np.save("%s/%05d.npy" % (flofw_outdir, i - dframe), flow_fw)
                np.save("%s/%05d.npy" % (flobw_outdir, i), flow_bw)
                cv2.imwrite(
                    "%s/visflo-%05d.jpg" % (flofw_outdir, i - dframe),
                    flow_to_image(flow_fw)[:, :, ::-1],
                )
                cv2.imwrite(
                    "%s/visflo-%05d.jpg" % (flobw_outdir, i),
                    flow_to_image(flow_bw)[:, :, ::-1],
                )

    vidname = args.outdir.rsplit("-", 1)[0]
    write_config(vidname)

    res = 256
    extract_crop(args.outdir, res, 0)
