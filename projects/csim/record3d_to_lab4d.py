import os, sys
import glob
import json
import numpy as np
import pdb
import cv2
import configparser
import trimesh
import re
import tqdm


sys.path.insert(0, os.getcwd())
from lab4d.utils.vis_utils import draw_cams
from projects.csim.ace.polycam_to_ace import json_to_camera
from projects.csim.render_polycam import PolyCamRender

from preprocess.scripts.crop import extract_crop
from preprocess.third_party.omnivision.normal import extract_normal
from preprocess.scripts.extract_dinov2 import extract_dinov2
from preprocess.libs.io import run_bash_command
from preprocess.scripts.download import download_seq
from preprocess.scripts.camera_registration import camera_registration
from preprocess.scripts.canonical_registration import canonical_registration
from preprocess.scripts.crop import extract_crop
from preprocess.scripts.depth import extract_depth
from preprocess.scripts.extract_dinov2 import extract_dinov2
from preprocess.scripts.extract_frames import extract_frames
from preprocess.scripts.tsdf_fusion import tsdf_fusion
from preprocess.scripts.write_config import write_config
from preprocess.third_party.vcnplus.compute_flow import compute_flow
from preprocess.third_party.vcnplus.frame_filter import frame_filter
from preprocess.third_party.omnivision.normal import extract_normal
from preprocess.scripts.fake_data import create_fake_masks


def extract_number(filename):
    # This regex assumes that the number is composed of digits and is preceded by a non-digit character
    match = re.search(r"(\d+)", os.path.basename(filename))
    return int(match.group(1)) if match else None


vidname = "2023-11-03--20-46-57"
seqname = "%s-0000" % vidname
target_dir = "database/processed/"
source_dir = "database/record3d/%s/EXR_RGBD/" % vidname

meta_path = "%s/metadata.json"
meta = json.load(open(meta_path))
pdb.set_trace()

for idx, rgb_path in enumerate(
    tqdm.tqdm(sorted(glob.glob("%s/rgb/*.jpg" % source_dir), key=extract_number))
):
    rgb = cv2.imread(rgb_path)

    depth_path = rgb_path.replace("rgb", "depth").replace(".jpg", ".exr")
    depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)


# cv2.imwrite("tmp/0.png", depth * 200)
# print("written to tmp/0.png")
# print(depth.shape)
# cv2.imwrite("tmp/1.jpg", rgb)
# print("written to tmp/1.jpg")
# print(rgb.shape)
