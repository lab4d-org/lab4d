"""python eval_4drecon.py --flagfile=logdir/cat-pikachu-2024-08-v2-compose-ft2/opts.log"""
"""python eval_4drecon.py --flagfile=logdir/2024-05-07--19-25-33-0-comp/opts.log"""
# remember to turn on n_depth=256
import sys, os
import numpy as np
import pdb
import torch
import json
import glob
from absl import app, flags
import cv2
from torch.nn import functional as F
import trimesh

sys.path.insert(0, os.getcwd())
from projects.csim.scripts import lpips_models

from lab4d.engine.trainer import Trainer
from lab4d.utils.geom_utils import K2inv, Kmatinv, se3_inv, rot_angle, K2mat, mat2K
from lab4d.utils.vis_utils import img2color, draw_cams
from lab4d.config import get_config
from lab4d.utils.camera_utils import construct_batch
from preprocess.scripts.depth import depth2pts

from kps_to_extrinsics import compute_relative_pose, pad_image

def im2tensor(image, imtype=np.uint8, cent=1., factor=1./2.):
    return torch.Tensor((image / factor - cent)
                        [:, :, :, np.newaxis].transpose((3, 2, 0, 1)))

def compute_depth_acc_at_10cm(dph_gt, dph, conf_gt, mask=None, dep_scale = 0.2):
    """
    from totalrecon
    """
    # INPUTS:
    # 1. dph_gt:            Ground truth depth image (scaled by "dep_scale"):                                       numpy array of shape = (H, W)
    # 2. dph:               Rendered depth image     (scaled by "dep_scale"):                                       numpy array of shape = (H, W)
    # 3. conf_gt:           Ground truth depth-confidence image:                                                    numpy array of shape = (H, W)
    # 4. mask:              Binary spatial mask over which to compute the metric:                                   numpy array of shape = (H, W)
    # 5. dep_scale:         Scale used to scale the ground truth depth during training
    #
    # RETURNS:
    # 1. depth accuracy at 0.1m:  Computes the number of test rays estimated with 0.1m of their ground truth
    depth_diff = (dph - dph_gt) / dep_scale                                             # depth difference in meters

    if mask is None:
        mask = np.ones_like(conf_gt)                                                    # shape = (H, W)

    # compute depth accuracy @ 0.1m over pixels that 1) have high confidence value (conf_gt > 1.5) and 2) a mask value of 1
    is_depth_accurate = (np.abs(depth_diff) < 0.1)
    depth_acc_at_10cm = np.mean(is_depth_accurate[(conf_gt > 1.5) & (mask == 1.)])

    return depth_acc_at_10cm, is_depth_accurate

def compute_lpips(rgb_gt, rgb, lpips_model, mask=None):
    # IMPORTANT!!! Both rgb_gt and rgb need to be in range [0, 1]
    # INPUTS:
    # 1. rgb_gt:            Ground truth image:                                         numpy array of shape = (H, W, 3)         
    # 2. rgb:               Rendered image:                                             numpy array of shape = (H, W, 3)
    # 3. lpips_model:       torch lpips_model (instantiate once in the main script, and feed as input to this method):      "lpips_model = lpips_models.PerceptualLoss(model='net-lin', net='alex', use_gpu=True,version=0.1)""
    # 4. mask:              Binary spatial mask over which to compute the metric:       numpy array of shape = (H, W)
    #
    # # OUTPUTS:
    # 1. lpips

    rgb_gt_0 = im2tensor(rgb_gt).cuda()                                                     # torch tensor of shape = (H, W, 3)
    rgb_0 = im2tensor(rgb).cuda()                                                           # torch tensor of shape = (H, W, 3)
    
    if mask is not None:
        mask_rgb = np.repeat(mask[..., np.newaxis], 3, axis=-1)                     # shape = (H, W, 3)
    else:
        mask_rgb = np.ones_like(rgb)                                                # shape = (H, W, 3)
    
    mask_0 = torch.Tensor(mask_rgb[:, :, :, np.newaxis].transpose((3, 2, 0, 1)))            # torch tensor of shape = (1, 3, H, W)
    lpips = lpips_model.forward(rgb_gt_0, rgb_0, mask_0).item()
    return lpips


def subsample(batch, skip_idx):
    for k, v in batch.items():
        if torch.is_tensor(v):
            batch[k] = v[::skip_idx] # every 10-th frame
        else:
            subsample(v, skip_idx)


def main(_):
    seqname0 = "2024-05-07--19-25-33-v0-0000"
    seqname1 = "2024-05-07--19-25-33-v1-0000"
    imgidx_0 = 0
    imgidx_1 = 0
    kp_path = "tmp/kps-cat-sync.npy"
    logdir=sys.argv[1].split("=")[1].split("/opts.log")[0]
    # skip_idx = 2000
    skip_idx = 100
    render_res = 256
    inst_id = 23
    # inst_id = 0

    logpath0_fg="%s/export_%04d/fg/motion.json" % (logdir, inst_id)
    logpath0_bg="%s/export_%04d/bg/motion.json" % (logdir, inst_id)
    # compute relative camera pose at each time instance
    imu_pose0 = np.load("database/processed/Cameras/Full-Resolution/%s/aligned-00.npy" % seqname0)
    imu_pose1 = np.load("database/processed/Cameras/Full-Resolution/%s/aligned-00.npy" % seqname1)
    fg_pose0 = np.asarray(list(json.load(open(logpath0_fg, "r"))["field2cam"].values())).astype(np.float32)
    bg_pose0 = np.asarray(list(json.load(open(logpath0_bg, "r"))["field2cam"].values())).astype(np.float32)
    min_len = min(imu_pose0.shape[0], imu_pose1.shape[0])
    imu_pose0 = imu_pose0[:min_len]
    imu_pose1 = imu_pose1[:min_len]
    fg_pose0 = fg_pose0[:min_len]
    bg_pose0 = bg_pose0[:min_len]

    R, T, kps, depth0, depth1, img0, img1, Kinv_0, Kinv_1  = compute_relative_pose(kp_path, seqname0, seqname1, imgidx_0, imgidx_1)

    # # camera reprojection
    # pts0 = depth2pts(depth0, Kmat=Kmatinv(Kinv_0))
    # pts1 = depth2pts(depth1, Kmat=Kmatinv(Kinv_1))
    # pts2 = (R @ pts0.T + T[:, None]).T
    # trimesh.Trimesh(pts0[::100]).export("tmp/0.obj")
    # trimesh.Trimesh(pts1[::100]).export("tmp/1.obj")
    # trimesh.Trimesh(pts2[::100]).export("tmp/2.obj")

    # # compute relative pose world-to-cam1
    # cam_rel_gt = np.eye(4)
    # cam_rel_gt[:3,:3] = R
    # cam_rel_gt[:3,3] = T
    # # cam1t-to-cam1 @ cam0t-to-cam1t @ cam0-to-cam0t @ world-to-cam0
    # cam_gt1_fg = (imu_pose1 @ se3_inv(imu_pose1[imgidx_1])) @ cam_rel_gt @ (bg_pose0[imgidx_0] @ se3_inv(bg_pose0)) @ fg_pose0
    # cam_gt1_bg = (imu_pose1 @ se3_inv(imu_pose1[imgidx_1])) @ cam_rel_gt @ (bg_pose0[imgidx_0] @ se3_inv(bg_pose0)) @ bg_pose0
    # draw_cams(cam_gt1_bg).export("tmp/0a.obj")
    # draw_cams(imu_pose1).export("tmp/0.obj")

    #TODO another way to get gt cam
    bg_pose0p = np.asarray(list(json.load(open("logdir-0701/2024-05-07--19-25-33-comp/export_0000/fg/motion.json", "r"))["field2cam"].values())).astype(np.float32)
    bg_pose1p = np.asarray(list(json.load(open("logdir-0701/2024-05-07--19-25-33-comp/export_0001/fg/motion.json", "r"))["field2cam"].values())).astype(np.float32)
    bg_pose0p = bg_pose0p[:min_len]
    bg_pose1p = bg_pose1p[:min_len]
    cam_gt1_fg = bg_pose1p @ se3_inv(bg_pose0p) @ fg_pose0
    cam_gt1_bg = bg_pose1p @ se3_inv(bg_pose0p) @ bg_pose0

    # load lab4d model, render depthlist 1
    opts = get_config()
    opts["render_res"] = render_res
    opts["inst_id"] = inst_id
    opts["load_suffix"] = "latest"
    model, _, _ = Trainer.construct_test_model(opts, return_refs=False, force_reload=False)
    device = model.device

    num_frames = len(cam_gt1_bg)
    frameid_sub = np.asarray(range(num_frames))
    raw_size = depth1.shape

    crop2raw = np.zeros(4)
    crop2raw[0] = raw_size[1] / opts["render_res"]
    crop2raw[1] = raw_size[0] / opts["render_res"]
    camera_int = mat2K(K2inv(crop2raw) @ Kmatinv(Kinv_1))
    camera_int = np.tile(camera_int[None], (num_frames,1))

    field2cam = {
        "fg": cam_gt1_fg,
        # "fg": fg_pose0,
        "bg": cam_gt1_bg,
        # "bg": bg_pose0,
        }

    batch = construct_batch(
        inst_id=opts["inst_id"],
        frameid_sub=frameid_sub,
        eval_res=opts["render_res"],
        field2cam=field2cam,
        camera_int=camera_int,
        crop2raw=None,
        device=device,
    )
    # two view
    # batch = construct_batch(
    #     inst_id=1,
    #     frameid_sub=frameid_sub,
    #     eval_res=opts["render_res"],
    #     field2cam=None,
    #     camera_int=camera_int,
    #     crop2raw=None,
    #     device=device,
    # )

    subsample(batch, skip_idx)
    with torch.no_grad():
        rendered = model.evaluate(batch, is_pair=False)
        depth_pred_all = F.interpolate(rendered["depth"].permute(0,3,1,2), raw_size, mode="bilinear").permute(0,2,3,1)
        rgb_pred_all = F.interpolate(rendered["rgb"].permute(0,3,1,2), raw_size, mode="bilinear").permute(0,2,3,1)

    # load depth ground-truth, compute depth error
    lpips_model = lpips_models.PerceptualLoss(model='net-lin', net='alex', use_gpu=True,version=0.1)
    depth_acc_list = []
    lpips_list = []
    depth_acc_bg_list = []
    lpips_bg_list = []
    depth_acc_fg_list = []
    lpips_fg_list = []
    for it, depth_path in enumerate(sorted(glob.glob("database/processed/Depth/Full-Resolution/%s/0*.npy" % seqname1))[::skip_idx]):
        print(depth_path)
        mask = np.load(depth_path.replace("Depth", "Annotations")) > 0
        depth_gt = cv2.resize(np.load(depth_path), raw_size[::-1])
        depth_pred = depth_pred_all.cpu().numpy()[it, ..., 0]

        depth_acc, depth_err = compute_depth_acc_at_10cm(depth_gt, depth_pred, np.ones_like(depth_gt) * 2, mask=None, dep_scale = 1)
        depth_acc_list.append(depth_acc)
        depth_acc_fg, _ = compute_depth_acc_at_10cm(depth_gt, depth_pred, np.ones_like(depth_gt) * 2, mask=mask, dep_scale = 1)
        depth_acc_fg_list.append(depth_acc_fg)
        depth_acc_bg, _ = compute_depth_acc_at_10cm(depth_gt, depth_pred, np.ones_like(depth_gt) * 2, mask=~mask, dep_scale = 1)
        depth_acc_bg_list.append(depth_acc_bg)

        rgb_gt = cv2.imread(depth_path.replace("Depth", "JPEGImages").replace(".npy", ".jpg"))[...,::-1]/255.
        rgb_pred = rgb_pred_all.cpu().numpy()[it]
        lpips = compute_lpips(rgb_gt, rgb_pred, lpips_model, mask=None)
        lpips_list.append(lpips)
        lpips_fg = compute_lpips(rgb_gt, rgb_pred, lpips_model, mask=mask)
        lpips_fg_list.append(lpips_fg)
        lpips_bg = compute_lpips(rgb_gt, rgb_pred, lpips_model, mask=~mask)
        lpips_bg_list.append(lpips_bg)

        depth_vis = img2color("depth", np.concatenate([depth_gt, depth_pred], axis=0)[...,None])
        # depth_vis = np.concatenate([depth_vis, np.tile(depth_err[...,None], (1,1,4))], 0)
        cv2.imwrite("tmp/%05d-depth.jpg"%it, depth_vis[...,::-1]*255)
        cv2.imwrite("tmp/%05d-rgb.jpg"%it, np.concatenate([rgb_gt, rgb_pred], axis=0)[...,::-1]*255)
    depth_acc_list = np.stack(depth_acc_list, 0)
    depth_acc_fg_list = np.stack(depth_acc_fg_list, 0)
    depth_acc_bg_list = np.stack(depth_acc_bg_list, 0)
    lpips_list = np.stack(lpips_list, 0)
    lpips_fg_list = np.stack(lpips_fg_list, 0)
    lpips_bg_list = np.stack(lpips_bg_list, 0)

    print("depth acc: %.3f" % depth_acc_list.mean())
    print("depth-fg acc: %.3f" % depth_acc_fg_list.mean())
    print("depth-bg acc: %.3f" % depth_acc_bg_list.mean())
    print("lpips: %.3f" % lpips_list.mean())
    print("lpips-fg: %.3f" % lpips_fg_list.mean())
    print("lpips-bg: %.3f" % lpips_bg_list.mean())

if __name__ == "__main__":
    app.run(main)