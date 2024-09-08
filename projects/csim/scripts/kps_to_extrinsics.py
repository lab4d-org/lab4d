import sys,os
import cv2
import random
import pdb
import numpy as np
import json
import torch
import trimesh
import configparser

sys.path.insert(0, os.getcwd())

from lab4d.utils.vis_utils import img2color, draw_cams
from lab4d.utils.geom_utils import K2inv, Kmatinv, se3_inv, rot_angle
from preprocess.libs.geometry import compute_procrustes_robust, compute_procrustes, compute_procrustes_median
from preprocess.scripts.depth import depth2pts



def random_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

def pad_image(img, target_width):
    _, width = img.shape[:2]
    delta_w = target_width - width
    color = [0, 0, 0]  # Color for padding, black in this case
    new_img = cv2.copyMakeBorder(img, 0, 0, 0, delta_w, cv2.BORDER_CONSTANT, value=color)
    return new_img

def compute_relative_pose(kp_path, seqname0, seqname1, imgidx_0, imgidx_1):
    imgpath_0 = "database/processed/JPEGImages/Full-Resolution/%s/%05d.jpg"%(seqname0, imgidx_0)
    imgpath_1 = "database/processed/JPEGImages/Full-Resolution/%s/%05d.jpg"%(seqname1, imgidx_1)
    # load ckpt to be evaluated
    kps = np.load(kp_path).astype(np.float32)
    print(kps.shape)
    print(kps)

    # read intrinsics from both cameras
    config0 = configparser.RawConfigParser()
    config0.read("database/configs/%s.config" % seqname0[:-5], encoding='utf-8')
    config1 = configparser.RawConfigParser()
    config1.read("database/configs/%s.config" % seqname1[:-5], encoding='utf-8')

    Kinv_0 = K2inv(np.asarray([float(i) for i in config0["data_0"]["ks"].split(" ")])).astype(np.float32)
    Kinv_1 = K2inv(np.asarray([float(i) for i in config1["data_0"]["ks"].split(" ")])).astype(np.float32)

    # read depth image
    img0 = cv2.imread(imgpath_0)
    img1 = cv2.imread(imgpath_1)
    # Get the dimensions of both images
    height0, width0 = img0.shape[:2]
    height1, width1 = img1.shape[:2]

    depthpath_0 = imgpath_0.replace("JPEGImages", "Depth").replace(".jpg", ".npy")
    depthpath_1 = imgpath_1.replace("JPEGImages", "Depth").replace(".jpg", ".npy")

    depth0 = cv2.resize(np.load(depthpath_0), (width0, height0)).astype(np.float32)
    depth1 = cv2.resize(np.load(depthpath_1), (width1, height1)).astype(np.float32)

    # save img visuals
    # Find the maximum width and height
    max_width = max(width0, width1)
    # Function to pad an image to the desired dimensions

    # Now concatenate the images vertically
    img = np.concatenate([pad_image(img0, max_width), pad_image(img1, max_width)], axis=0)
    depth = np.concatenate([pad_image(depth0, max_width), pad_image(depth1, max_width)], axis=0)

    h = img0.shape[0]

    for kp in kps:
        start_point = (int(kp[0]), int(kp[1]))
        end_point = (int(kp[2]), int(kp[3] + h))
        cv2.line(img, start_point, end_point, random_color(), 5)  # Blue color, thickness 5

    cv2.imwrite("tmp/0.jpg", img)
    cv2.imwrite("tmp/1.jpg", img2color("depth", depth[...,None])*255)
    print("saved to tmp/0.jpg")

    # query depth at pixel locations, + intrinsics => (8x3)
    # Interpolate using remap
    Zval0 = cv2.remap(depth0, kps[:, 0], kps[:, 1], interpolation=cv2.INTER_LINEAR)
    Zval1 = cv2.remap(depth1, kps[:, 2], kps[:, 3], interpolation=cv2.INTER_LINEAR)

    hxy0 = np.concatenate([kps[:,0:2], np.ones_like(kps[:,:1])],-1)
    hxy1 = np.concatenate([kps[:,2:4], np.ones_like(kps[:,:1])],-1)

    xyz0 = hxy0 @ Kinv_0.T * Zval0
    xyz1 = hxy1 @ Kinv_1.T * Zval1

    # do PnP to get extrinsics
    # R,T = compute_procrustes_robust(xyz0, xyz1, min_samples=4)
    # R,T,_ = compute_procrustes_median(xyz0, xyz1, pts_limit=4)
    R,T = compute_procrustes(xyz0, xyz1, pts_limit=4)
    trimesh.Trimesh(xyz0).export("tmp/0.obj")
    trimesh.Trimesh(xyz1).export("tmp/1.obj")
    xyz2 = (R @ xyz0.T + T[:, None]).T
    trimesh.Trimesh(xyz2).export("tmp/2.obj")
    # xyz2p = (R_pred @ xyz0.T + T_pred[:, None]).T
    # trimesh.Trimesh(xyz2p).export("tmp/2p.obj")

    return R, T, kps, depth0, depth1, img0, img1, Kinv_0, Kinv_1

if __name__ == "__main__":
    # # bunny results not good
    # imgidx_0 = 105
    # imgidx_1 = 66
    # imgpath_0 = "database/processed/JPEGImages/Full-Resolution/Feb14at5-55PM-poly-0000/%05d.jpg"%imgidx_0
    # imgpath_1 = "database/processed/JPEGImages/Full-Resolution/2024-02-14--17-51-46-0000/%05d.jpg"%imgidx_1
    # logdir="logdir-0606/bunny-2024-02-14-bg-adapt3-abs_betteranneal"
    # # logdir="logdir-0606/bunny-2024-02-14-bg-adapt3-abs_noanneal"
    # # logdir="logdir-0606/bunny-2024-02-14-bg-adapt3-abs_nolocalizer"
    # # logdir="logdir-0606/bunny-2024-02-14-bg-adapt3-abs_nofeature"
    # # logdir="logdir-0606/bunny-2024-02-14-bg-totalrecon"
    # logpath0="%s/export_0000/bg/motion.json" % logdir
    # logpath1="%s/export_0002/bg/motion.json" % logdir
    # kp_path = "tmp/kps-bunny.npy"

    # # cat results
    imgidx_0 = 0
    imgidx_1 = 73
    seqname0 = "Oct5at10-49AM-poly-0000"
    seqname1 = "2023-11-11--11-51-53-0000"

    # logdir="logdir-neurips-aba/cat-pikachu-2024-08-bg-totalrecon-v2" # multivideo totalrecon
    # logdir="logdir-neurips-aba/cat-pikachu-2024-08-bg-adapt3-abs_nolocalizer"
    # logdir="logdir/cat-pikachu-2024-08-bg-adapt3-abs_noanneal/"
    # logdir="logdir-neurips-aba/cat-pikachu-2024-08-bg-adapt3-abs_nofeature/"
    logdir="logdir-neurips-aba/cat-pikachu-2024-08-bg-adapt3/" # ours

    logpath0="%s/export_0000/bg/motion.json" % logdir
    logpath1="%s/export_0009/bg/motion.json" % logdir
    kp_path = "tmp/kps-cat.npy"

    cam_pred0 = np.asarray(list(json.load(open(logpath0, "r"))["field2cam"].values())).astype(np.float32)
    cam_pred1 = np.asarray(list(json.load(open(logpath1, "r"))["field2cam"].values())).astype(np.float32)
    cam_rel_pred = cam_pred1[imgidx_1] @ se3_inv(cam_pred0[imgidx_0]) # camera 0 to camera 1
    R_pred = cam_rel_pred[:3,:3]
    T_pred = cam_rel_pred[:3,3]


    R, T, kps, depth0, depth1, img0, img1, Kinv_0, Kinv_1 = compute_relative_pose(kp_path, seqname0, seqname1, imgidx_0, imgidx_1)

    # camera reprojection
    pts0 = depth2pts(depth0, Kmat=Kmatinv(Kinv_0))
    pts1 = depth2pts(depth1, Kmat=Kmatinv(Kinv_1))
    pts2 = (R @ pts0.T + T[:, None]).T
    pts2p = (R_pred @ pts0.T + T_pred[:, None]).T
    trimesh.Trimesh(pts0[::100]).export("tmp/0.obj")
    trimesh.Trimesh(pts1[::100]).export("tmp/1.obj")
    trimesh.Trimesh(pts2[::100]).export("tmp/2.obj")
    trimesh.Trimesh(pts2p[::100]).export("tmp/2p.obj")



    # eval
    # construct GT pose for vid1
    # load imu pose
    imu_pose = np.load("database/processed/Cameras/Full-Resolution/%s/aligned-00.npy" % seqname1)

    # align imu_pose to canonical coordinate
    # world'-to-camera1 @ cam1t-to-world' @ cam0t-to-cam1t @ world-to-cam0t
    cam_rel_gt = np.eye(4)
    cam_rel_gt[:3,:3] = R
    cam_rel_gt[:3,3] = T
    cam_gt1 = imu_pose @ se3_inv(imu_pose[imgidx_1]) @ cam_rel_gt @ cam_pred0[imgidx_0]

    draw_cams(cam_pred0).export("tmp/0.obj")
    draw_cams(cam_gt1).export("tmp/1.obj")

    # rot_err = rot_angle(torch.tensor(R@R_pred.T)).numpy() / np.pi * 180
    # trans_err = np.linalg.norm(T - T_pred, 2,-1)

    R_pred = cam_pred1[:,:3,:3]
    T_pred = cam_pred1[:,:3,3]
    R = cam_gt1[:,:3,:3]
    T = cam_gt1[:,:3,3]

    rot_err = rot_angle(torch.tensor(R@np.transpose(R_pred, (0,2,1)))).numpy().mean() / np.pi * 180
    trans_err = np.linalg.norm(T - T_pred, 2,-1).mean()
    # DCRE
    # xy_2 = pts2 @ Kmatinv(Kinv_0).T[...,:2]

    print("rotation error: %.2f" % rot_err) 
    print("translation error: %.2f" % trans_err) 