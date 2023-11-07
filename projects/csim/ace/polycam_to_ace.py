import os, sys
import glob
import json
import numpy as np
import pdb
import cv2


sys.path.insert(0, os.getcwd())
from lab4d.utils.vis_utils import draw_cams


def json_to_camera(json_data):
    """
    {'t_02': 0.7978225946426392, 't_13': -0.10880815237760544, 'cy': 381.9624938964844, 'fy': 764.6701049804688, 't_22': 0.21414992213249207, 't_11': -0.001555086113512516, 't_00': 0.5438205599784851, 't_20': 0.1479204148054123, 'angular_velocity': 3.4028234663852886e+38, 't_03': 1.1343777179718018, 'center_depth': 1.4739747047424316, 't_23': 2.1962790489196777, 't_12': 0.5635768175125122, 't_01': 0.2602657079696655, 'timestamp': 262195552056, 'blur_score': 165.65264892578125, 't_21': -0.9655357599258423, 't_10': -0.8260622024536133, 'width': 1024, 'height': 768, 'manual_keyframe': False, 'cx': 516.1546630859375, 'fx': 764.6701049804688}
    """
    extrinsics = np.eye(4)
    extrinsics[0, 0] = json_data["t_00"]
    extrinsics[0, 1] = json_data["t_01"]
    extrinsics[0, 2] = json_data["t_02"]
    extrinsics[0, 3] = json_data["t_03"]
    extrinsics[1, 0] = json_data["t_10"]
    extrinsics[1, 1] = json_data["t_11"]
    extrinsics[1, 2] = json_data["t_12"]
    extrinsics[1, 3] = json_data["t_13"]
    extrinsics[2, 0] = json_data["t_20"]
    extrinsics[2, 1] = json_data["t_21"]
    extrinsics[2, 2] = json_data["t_22"]
    extrinsics[2, 3] = json_data["t_23"]

    intrinsics = np.zeros(4)
    intrinsics[0] = json_data["fx"]
    intrinsics[1] = json_data["fy"]
    intrinsics[2] = json_data["cx"]
    intrinsics[3] = json_data["cy"]
    return extrinsics, intrinsics


target_dir = "../ace/datasets/home"
source_dir = "database/polycam/Oct5at10-49AM-poly/keyframes"
os.makedirs("%s/train/rgb" % target_dir, exist_ok=True)
os.makedirs("%s/train/poses" % target_dir, exist_ok=True)
os.makedirs("%s/train/calibration" % target_dir, exist_ok=True)

# from (x-down, y-right, z-inward) to (x-right, y-down, z-forward)
transformation_matrix = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])

extrinsics_all = []
for imgpath in sorted(glob.glob("%s/images/*.jpg" % source_dir)):
    filename = imgpath.split("/")[-1].split(".")[0]
    # copy to target dir
    target_file = "%s/train/rgb/%s.color.png" % (target_dir, filename)
    image = cv2.imread(imgpath)
    image = np.transpose(image, [1, 0, 2])  # vertical to horizontal
    image = image[:, ::-1, :]  # flip horizontally
    cv2.imwrite(target_file, image)

    # copy to poses
    # both ace and polycam saves view-to-world poses
    camera_path = imgpath.replace("images", "cameras").replace(".jpg", ".json")
    json_data = json.load(open(camera_path))
    extrinsics, intrinsics = json_to_camera(json_data)
    extrinsics[:3, :3] = extrinsics[:3, :3] @ transformation_matrix

    target_file = "%s/train/poses/%s.pose.txt" % (target_dir, filename)
    np.savetxt(target_file, extrinsics)

    target_file = "%s/train/calibration/%s.calibration.txt" % (target_dir, filename)
    np.savetxt(target_file, intrinsics[0:1])

    extrinsics_all.append(extrinsics)

# extrinsics_all = np.stack(extrinsics_all)
extrinsics_all = np.linalg.inv(extrinsics_all)
mesh = draw_cams(extrinsics_all)
mesh.export("tmp/0.obj")
