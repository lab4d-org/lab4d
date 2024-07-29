# Copyright (c) 2023 Gengshan Yang, Carnegie Mellon University.
# python preprocess/scripts/canonical_registration.py 2023-04-03-18-02-32-cat-pikachu-5-0000 256 quad
# this assumes one component (object) per video
import glob
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R

sys.path.insert(
    0,
    "%s/../third_party" % os.path.join(os.path.dirname(__file__)),
)

sys.path.insert(
    0,
    "%s/../" % os.path.join(os.path.dirname(__file__)),
)

sys.path.insert(
    0,
    "%s/../../" % os.path.join(os.path.dirname(__file__)),
)

from libs.io import get_bbox, read_images_densepose, read_raw
from libs.torch_models import CanonicalRegistration, get_class
from libs.utils import robust_rot_align
from viewpoint.dp_viewpoint import ViewponitNet

from lab4d.utils.geom_utils import K2inv, K2mat, Kmatinv
from lab4d.utils.quat_transform import quaternion_translation_to_se3
from lab4d.utils.vis_utils import draw_cams


def save_resampled_feat(imglist, feats, dp2raws, prefix, crop_size):
    # load crop2raw
    crop2raw_path = imglist[0].replace("JPEGImages", "Annotations").rsplit("/", 1)[0]
    crop2raw_path = crop2raw_path + "/%s-%d-crop2raw.npy" % (prefix, crop_size)
    crop2raws = np.load(crop2raw_path)
    feats_expanded = []
    for it, impath in enumerate(imglist):
        feat = feats[it]
        feat_width = feat.shape[-1]
        crop2raw = crop2raws[it]
        # compute transform
        crop2dp = K2inv(dp2raws[it]) @ K2mat(crop2raw)
        # resample
        feat = torch.tensor(feat, dtype=torch.float32)[None]
        crop2dp = torch.tensor(crop2dp, dtype=torch.float32)
        xy_range = torch.linspace(0, crop_size, steps=feat_width, dtype=torch.float32)
        hxy = torch.cartesian_prod(xy_range, xy_range)
        hxy = torch.stack([hxy[:, 1], hxy[:, 0], torch.ones_like(hxy[:, 0])], -1)
        hxy = hxy @ crop2dp.T
        hxy = hxy[..., :2] / feat_width * 2 - 1
        feat = F.grid_sample(feat, hxy.view(-1, feat_width, feat_width, 2))
        feats_expanded.append(feat)
    feats = torch.cat(feats_expanded, dim=0).numpy()
    feats = np.transpose(feats, (0, 2, 3, 1))
    return feats


def canonical_registration(seqname, crop_size, obj_class, component_id=1, mode="opt"):
    """
    mode can be "max", "opt", "zero"
    max: most likely canonical pose, not implemented
    opt: optimize canonical pose
    zero: no canonical pose is used
    """
    # load rgb/depth
    imgdir = "database/processed/JPEGImages/Full-Resolution/%s" % seqname
    imglist = sorted(glob.glob("%s/*.jpg" % imgdir))
    save_path = imgdir.replace("JPEGImages", "Cameras")

    cams_view1 = np.load("%s/%02d.npy" % (save_path, component_id))
    if mode == "opt":
        # classifiy human or not
        if obj_class == "other":
            import json, pdb

            cam_path = (
                "database/processed/Cameras/Full-Resolution/%s/%02d-manual.json"
                % (
                    seqname,
                    component_id,
                )
            )
            with open(cam_path) as f:
                cams_canonical = json.load(f)
                cams_canonical = {
                    int(k): np.asarray(v) for k, v in cams_canonical.items()
                }
        else:
            if obj_class == "human":
                is_human = True
            elif obj_class == "quad":
                is_human = False
            else:
                raise ValueError("Unknown object class: %s" % obj_class)
            viewpoint_net = ViewponitNet(is_human=is_human)
            viewpoint_net.cuda()
            viewpoint_net.eval()

            # densepose inference
            rgbs, masks = read_images_densepose(imglist)
            with torch.no_grad():
                cams_canonical, feats, dp2raws = viewpoint_net.run_inference(
                    rgbs, masks
                )

            # save densepose features
            # resample features to the cropped image size
            feats_crop = save_resampled_feat(imglist, feats, dp2raws, "crop", crop_size)
            feats_full = save_resampled_feat(imglist, feats, dp2raws, "full", crop_size)
            save_path_dp = save_path.replace("Cameras", "Features")
            os.makedirs(save_path_dp, exist_ok=True)
            np.save(
                "%s/crop-cse-%02d.npy" % (save_path_dp, component_id),
                feats_crop.astype(np.float16),
            )
            np.save(
                "%s/full-cse-%02d.npy" % (save_path_dp, component_id),
                feats_full.astype(np.float16),
            )
            cams_canonical = {k: v for k, v in enumerate(cams_canonical)}

        # canonical registration (smoothes the camera poses)
        print("num cams annotated: %d" % len(cams_canonical.keys()))
        rgbpath_list = [imglist[i] for i in cams_canonical.keys()]
        cams_canonical_vals = np.stack(list(cams_canonical.values()), 0)
        draw_cams(cams_canonical_vals, rgbpath_list=rgbpath_list).export(
            "%s/cameras-%02d-canonical-prealign.obj" % (save_path, component_id)
        )
        registration = CanonicalRegistration(cams_canonical, cams_view1)
        registration.cuda()
        quat, trans = registration.optimize()
        cams_pred = quaternion_translation_to_se3(quat, trans).cpu().numpy()

    elif mode == "zero":
        cams_pred = cams_view1
    elif mode == "from_predictor":
        cams_canonical = np.load("tmp/predictor/extrinsics-%s.npy"%seqname)
        cams_canonical = {k: v for k, v in enumerate(cams_canonical)}
        registration = CanonicalRegistration(cams_canonical, cams_view1)
        registration.cuda()
        quat, trans = registration.optimize()
        cams_pred = quaternion_translation_to_se3(quat, trans).cpu().numpy()


    if component_id == 1:
        # fixed depth
        cams_pred[:, :3, 3] = 0

        # compute initial camera trans with 2d bbox
        # depth = focal * sqrt(surface_area / bbox_area) = focal / bbox_size
        # xytrn = depth * (pxy - crop_size/2) / focal
        # surface_area = 1
        misdet_arr = np.zeros(len(imglist), dtype=bool)
        for it, imgpath in enumerate(imglist):
            bbox = get_bbox(imgpath, component_id=component_id)
            if bbox is None:
                misdet_arr[it] = True
                continue
            shape = cv2.imread(imgpath).shape[:2]

            focal = max(shape)
            # depth = focal / np.sqrt(bbox[2] * bbox[3])
            # depth = min(depth, 10)  # depth might be too large for mis-detected frames
            # TODO load mean depth of the object
            raw_dict = read_raw(imgpath, 1, 256, False, with_flow=False)
            depth = np.median(raw_dict["depth"][raw_dict["mask"][..., 0]])

            center_bbox = bbox[:2] + bbox[2:] / 2
            center_img = np.array(shape[::-1]) / 2
            xytrn = depth * (center_bbox - center_img) / focal

            cams_pred[it, 2, 3] = depth
            cams_pred[it, :2, 3] = xytrn        
        cams_pred[:, :3, 3][misdet_arr] = cams_pred[:, :3, 3][~misdet_arr].mean(0)

    np.save("%s/%02d-canonical.npy" % (save_path, component_id), cams_pred)
    draw_cams(cams_pred, rgbpath_list=imglist).export(
        "%s/cameras-%02d-canonical.obj" % (save_path, component_id)
    )
    print("canonical registration (crop_size: %d) done: %s" % (crop_size, seqname))


if __name__ == "__main__":
    seqname = sys.argv[1]
    crop_size = int(sys.argv[2])
    obj_class = sys.argv[3]
    mode = sys.argv[4]

    canonical_registration(seqname, crop_size, obj_class, mode=mode)
