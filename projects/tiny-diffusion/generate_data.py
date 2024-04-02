# convert lab4d format to mdm format
import sys, os
import numpy as np
import torch
import glob
import pdb
import pickle as pkl
import trimesh
import json

seqnum = 26
in_path = "../vid2sim/logdir/home-2023-curated3-compose-ft/export_*"

type = "train"
# type = "test"
seq_len = 64
# seq_len = 80

test_seq = [9, 18, 20, 24]
# test_seq = list(range(1, seqnum + 1))[1:]
train_seq = [i for i in range(1, seqnum + 1) if i not in test_seq]
if type == "train":
    include_dir = train_seq
    stride = 1
elif type == "test":
    include_dir = test_seq
    stride = 10
else:
    raise ValueError("type should be train or test")

save_path = "database/motion/S%d-%s-L%d-S%d.pkl" % (seqnum, type, seq_len, stride)

# logdirs = sorted(glob.glob("../vid2sim/logdir-12-05/*compose/"))
# logdirs = sorted(glob.glob("../vid2sim/logdir/home-2023-11-compose-ft/export_*"))
logdirs = sorted(glob.glob(in_path))
_pose = []
_joints = []
_se3 = []
cam_se3 = []
total_length = 0
for dataid, logdir in enumerate(logdirs):
    if "export_0000" in logdir:
        continue
    seqid = int(logdir.split("/")[-1].split("_")[-1])
    if seqid not in include_dir:
        continue

    with open(logdir + "/fg/motion.json", "r") as f:
        data = json.load(f)
        pose = np.asarray(list(data["joint_so3"].values()))
        joints = np.asarray(list(data["t_articulation"].values()))[:, :, :3, 3]
        # object to cam
        field2cam = np.asarray(list(data["field2cam"].values()))

    with open(logdir + "/bg/motion.json", "r") as f:
        data = json.load(f)
        # world to cam
        field2cam_bg = np.asarray(list(data["field2cam"].values()))

        # world to obj =  inv( object to cam) @ world to cam
        world2obj = np.linalg.inv(field2cam) @ field2cam_bg

    print("length of %s is %d" % (logdir, len(pose)))
    total_length += len(pose)

    # cut it to length 64 seqs every 10 frame
    for i in range(0, len(pose), stride):
        if i + seq_len > len(pose):
            break
        pose_sub = pose[i : i + seq_len]
        joints_sub = joints[i : i + seq_len]
        se3_sub = world2obj[i : i + seq_len]
        cam_se3_sub = field2cam_bg[i : i + seq_len]

        _pose.append(pose_sub)
        _joints.append(joints_sub)
        _se3.append(se3_sub)
        cam_se3.append(cam_se3_sub)

print("total length:", total_length)
print("total seqs:", len(_pose))

# save to custom
custom_data = {}
custom_data["poses"] = _pose
custom_data["joints3D"] = _joints
custom_data["se3"] = _se3
custom_data["cam_se3"] = cam_se3

pkl.dump(custom_data, open(save_path, "wb"))
print("saved to", save_path)
