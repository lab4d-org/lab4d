import cv2
import numpy as np
import glob
import pdb
import os, sys


sys.path.insert(0, os.getcwd())
from lab4d.utils.vis_utils import draw_cams

camera_path = "../ace/datasets/home/train/poses/*.txt"

pose = np.asarray([np.loadtxt(pose) for pose in sorted(glob.glob(camera_path))])
pose = np.linalg.inv(pose)
mesh = draw_cams(pose)
mesh.export("tmp/0.obj")
