import os, sys
import glob
import json
import numpy as np
import pdb
import cv2


sys.path.insert(0, os.getcwd())
from lab4d.utils.vis_utils import draw_cams

seqname = "cat-pikachu-0001"
target_dir = "../ace/datasets/%s/test" % seqname
source_dir = "database/processed/JPEGImages/Full-Resolution/%s" % seqname
os.makedirs("%s/rgb" % target_dir, exist_ok=True)
os.makedirs("%s/poses" % target_dir, exist_ok=True)
os.makedirs("%s/calibration" % target_dir, exist_ok=True)

fl = 1920
extrinsics = np.eye(4)
for imgpath in sorted(glob.glob("%s/*.jpg" % source_dir))[:10]:
    filename = imgpath.split("/")[-1].split(".")[0]
    # copy to target dir
    target_file = "%s/rgb/%s.color.png" % (target_dir, filename)
    image = cv2.imread(imgpath)
    cv2.imwrite(target_file, image)

    target_file = "%s/poses/%s.pose.txt" % (target_dir, filename)
    np.savetxt(target_file, extrinsics)

    target_file = "%s/calibration/%s.calibration.txt" % (target_dir, filename)
    np.savetxt(target_file, [fl])
