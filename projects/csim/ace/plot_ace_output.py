import os, sys
import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, os.getcwd())
from lab4d.utils.vis_utils import draw_cams

seqname = "cat-pikachu-0-0000"
path = "../ace/output/poses_%s_.txt" % seqname

with open(path, "r") as f:
    lines = f.readlines()

extrinsics_all = []
for line in lines:
    line = line.strip()
    # file_name rot_quaternion_w rot_quaternion_x rot_quaternion_y rot_quaternion_z translation_x translation_y translation_z rot_err_deg tr_err_m inlier_count
    quat = np.array([float(x) for x in line.split(" ")[1:5]])
    quat = quat / np.linalg.norm(quat)
    quat = np.array([quat[1], quat[2], quat[3], quat[0]])
    rot = R.from_quat(quat)

    trans = np.array([float(x) for x in line.split(" ")[5:8]])

    extrinsics = np.eye(4)
    extrinsics[:3, :3] = rot.as_matrix()
    extrinsics[:3, 3] = trans

    extrinsics = np.linalg.inv(extrinsics)

    extrinsics_all.append(extrinsics)

mesh = draw_cams(np.stack(extrinsics_all))
mesh.export("tmp/0.obj")
