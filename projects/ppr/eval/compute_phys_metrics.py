# """
# WIP: compute physical metrics for the predicted keypoints
# python scripts/eval/eval_phys.py logdir/bgnerf-new-human_mod-ama-d-e120-b96-pft3/D_handstand5-kps.npy
# """
# from absl import flags, app
# import cv2
# import trimesh
# import json
# import glob
# import sys

# sys.path.insert(0, "")
# sys.path.insert(0, "third_party")
# import configparser
# import numpy as np
# import pdb
# import imageio
# import pyrender
# from scipy.spatial.transform import Rotation as R
# import torch

# from nnutils.geom_utils import vec_to_sim3, optimize_scale, fit_plane_contact
# from nnutils.geom_utils import (
#     extract_mesh,
#     zero_to_rest_bone,
#     zero_to_rest_dpose,
#     skinning,
#     lbs,
#     se3_vec2mat,
# )
# from nnutils.urdf_utils import articulate_robot, angles2cfg
# from utils.io import vis_kps, draw_cams
# from utils.io import save_vid, str_to_frame, save_bones
# from nnutils.train_utils import v2s_trainer
# from sklearn.metrics import f1_score

# kp_path = sys.argv[1]
# seqname = kp_path.split("/")[-1].rsplit("-", 1)[0]
# if "D_bouncing" in kp_path or "D_handstand" in kp_path or "T_samba" in kp_path:
#     seqname = seqname[:-1]
#     is_ama = True
# else:
#     is_ama = False  # assume cat
# frame_duration = 1 / 30.0


# def main(_):
#     # get numbers
#     kps = np.load(kp_path)
#     contact_labels = np.loadtxt("misc/gt_contact/%s-contact.txt" % seqname)
#     n_kp = kps.shape[1]
#     n_fr = min(kps.shape[0], contact_labels.shape[0])

#     contact_labels = contact_labels[:n_fr]
#     kps = kps[:n_fr]

#     pdb.set_trace()
#     if is_ama:
#         contact_pred = np.abs(kps[..., 1]) < 0.1
#     else:
#         contact_pred = np.abs(kps[..., 1]) < 0.2
#     f1 = f1_score(contact_labels.flatten(), contact_pred.flatten())

#     # TODO visualize
#     try:
#         kps_gt = np.load("misc/gt_contact/%s-kps.npy" % seqname)
#         kps_gt = kps_gt[:n_fr]
#         vis_kps(
#             np.transpose(kps_gt, [0, 2, 1]),
#             "tmp/kps_gt.obj",
#             binary_labels=contact_labels,
#         )
#     except:
#         pass
#     vis_kps(
#         np.transpose(kps, [0, 2, 1]),
#         "tmp/kps_pred.obj",
#         binary_labels=contact_pred == contact_labels,
#     )

#     ## jerk: this is not accurate due to finite difference
#     # kps_vel  = (kps[2:] - kps[:-2]) / (2*frame_duration)
#     # kps_acc  = (kps_vel[2:] - kps_vel[:-2]) / (2*frame_duration)
#     ##kps_acn = np.linalg.norm(kps_acc, 2,-1).mean()
#     # kps_jrk  = np.linalg.norm(kps_acc[1:] - kps_acc[:-1], 2,-1).mean()
#     #
#     # kps_vel_gt  = (kps_gt[2:] - kps_gt[:-2]) / (2*frame_duration)
#     # kps_acc_gt  = (kps_vel_gt[2:] - kps_vel_gt[:-2]) / (2*frame_duration)
#     ##kps_acn_gt = np.linalg.norm(kps_acc_gt, 2,-1).mean()
#     # kps_jrk_gt  = np.linalg.norm(kps_acc_gt[1:] - kps_acc_gt[:-1], 2,-1).mean()

#     # pdb.set_trace()
#     # acc_err = np.abs(kps_acn - kps_acn_gt).mean()

#     # skate
#     move_dis = np.linalg.norm(kps[1:] - kps[:-1], 2, -1)
#     move_dis_all = move_dis[np.logical_and(contact_labels[:-1], contact_labels[1:])]

#     state_5cm = (np.asarray(move_dis_all) > 0.05).mean()
#     state_ave = np.asarray(move_dis_all).mean()

#     # print('tv: %.1f'%(kps_jrk))
#     # print('tv-gt: %.1f'%(kps_jrk_gt))
#     print("f1: %.1f" % (f1 * 100))
#     print("sk-5cm: %.1f" % (state_5cm * 100))
#     print("sk-ave: %.1f" % (state_ave * 100))


# if __name__ == "__main__":
#     app.run(main)
