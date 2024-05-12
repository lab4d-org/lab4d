import os, sys
import glob
import json
import numpy as np
import pdb
import cv2
import configparser
import trimesh
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

def polycam_to_lab4d_all(folder_path, target_dir="database/processed/"):
    for vidname in glob.glob(folder_path):
        print(vidname)

def polycam_to_lab4d(vidname, target_dir="database/processed/"):
    seqname = "%s-0000" % vidname
    source_dir = "database/polycam/%s/keyframes" % vidname

    for idx, imgpath in enumerate(
        tqdm.tqdm(sorted(glob.glob("%s/images/*.jpg" % source_dir)))
    ):
        filename = imgpath.split("/")[-1].split(".")[0]

        # image
        src = "%s/images/%s.jpg" % (source_dir, filename)
        trg1 = "%s/JPEGImagesRaw/Full-Resolution/%s/%05d.jpg" % (
            target_dir,
            seqname,
            idx,
        )
        trg2 = "%s/JPEGImages/Full-Resolution/%s/%05d.jpg" % (target_dir, seqname, idx)
        image = cv2.imread(src)
        image = np.transpose(image, [1, 0, 2])  # vertical to horizontal
        image = image[:, ::-1, :]  # flip horizontally
        os.makedirs(os.path.dirname(trg1), exist_ok=True)
        os.makedirs(os.path.dirname(trg2), exist_ok=True)
        cv2.imwrite(trg1, image)
        cv2.imwrite(trg2, image)

        # depth
        depth_path = imgpath.replace("images", "depth").replace(".jpg", ".png")
        depth = cv2.imread(depth_path, -1) / 1000
        depth = np.transpose(depth, [1, 0])  # vertical to horizontal
        depth = depth[:, ::-1]  # flip horizontally
        trg = "%s/Depth/Full-Resolution/%s/%05d.npy" % (target_dir, seqname, idx)
        os.makedirs(os.path.dirname(trg), exist_ok=True)
        np.save(trg, depth)

    # save cameras
    polycam_loader = PolyCamRender("%s/../" % source_dir)
    extrinsics_all = polycam_loader.extrinsics
    mesh = draw_cams(extrinsics_all)
    trg = "%s/Cameras/Full-Resolution/%s/00.npy" % (target_dir, seqname)
    os.makedirs(os.path.dirname(trg), exist_ok=True)
    np.save(trg, extrinsics_all)
    np.save(trg.replace("00.npy", "aligned-00.npy"), extrinsics_all)
    mesh.export("%s/Cameras/Full-Resolution/%s/cameras-00.obj" % (target_dir, seqname))

    # copy mesh
    export_path = "%s/Cameras/Full-Resolution/%s/mesh-00-centered.obj" % (
        target_dir,
        seqname,
    )
    polycam_loader.mesh.export(export_path)

    # run preprocessing
    write_config(vidname)

    # modify intrinsics
    config_path = "database/configs/%s.config" % vidname
    config = configparser.ConfigParser()
    config.read(config_path)
    intrinsics = polycam_loader.intrinsics[0]
    config["data_0"]["ks"] = " ".join([str(j) for j in intrinsics.flatten()])
    with open(config_path, "w") as configfile:
        config.write(configfile)

    create_fake_masks(seqname, target_dir)
    # flow
    for dframe in [1, 2, 4, 8]:
        compute_flow(seqname, target_dir, dframe)

    extract_normal(seqname)

    res = 256
    extract_crop(seqname, res, 1)
    extract_crop(seqname, res, 0)
    extract_dinov2(vidname, component_id=0, ndim=-1)


if __name__ == "__main__":
    # vidname = sys.argv[1]
    # vidname = "Oct25at8-48PM-poly"
    # vidname = "Oct5at10-49AM-poly"
    # vidname = "Oct31at1-13AM-poly"
    # vidname = "Feb14at5-55тАпPM-poly"
    # vidname = "Feb19at9-29 PM-poly"
    vidname = "Feb26at10-02PM-poly"
    polycam_to_lab4d(vidname)
