import os, sys

# os.environ["TRANSFORMERS_OFFLINE"] = "1"
# os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
# os.environ["OMP_NUM_THREADS"] = "1"  # OpenMP threads
# os.environ["MKL_NUM_THREADS"] = "1"  # For Intel MKL
# os.environ["NUMEXPR_NUM_THREADS"] = "1"  # For NumExpr
# os.system("rm -rf ~/.cache")
# os.system("ln -s /mnt/home/gengshany/.cache/ ~/.cache")
# os.system("rm -rf ~/.torch")
# os.system("ln -s /mnt/home/gengshany/.torch/ ~/.torch")

import glob
import json
import numpy as np
import pdb
import importlib
import cv2
import configparser
import trimesh
import re
import tqdm
from scipy.spatial.transform import Rotation as R
import OpenEXR


sys.path.insert(0, os.getcwd())
from lab4d.utils.vis_utils import draw_cams
from lab4d.utils.cam_utils import depth_to_xyz
from lab4d.utils.geom_utils import mat2K
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

track_anything_cli = importlib.import_module(
    "preprocess.third_party.Track-Anything.track_anything_cli"
)
track_anything_cli = track_anything_cli.track_anything_cli


def track_anything_lab4d(seqname, outdir, text_prompt):
    ## a duplicate of scripts/run_preprocess.py, try remove this
    input_folder = "%s/JPEGImages/Full-Resolution/%s" % (outdir, seqname)
    output_folder = "%s/Annotations/Full-Resolution/%s" % (outdir, seqname)
    track_anything_cli(input_folder, text_prompt, output_folder)


def extract_number(filename):
    # This regex assumes that the number is composed of digits and is preceded by a non-digit character
    match = re.search(r"(\d+)", os.path.basename(filename))
    return int(match.group(1)) if match else None


def record3d_to_lab4d(
    folder_path,
    vidname,
    flip=0,
    home_path="database/configs/Oct5at10-49AM-poly.config",
    prompt="cat",
):
    seqname = "%s-0000" % vidname
    target_dir = "database/processed/"
    source_dir = "%s/%s/EXR_RGBD/" % (folder_path, vidname)
    envname = home_path.split("/")[-1].split(".")[0]

    meta_path = "%s/metadata.json" % source_dir
    meta = json.load(open(meta_path))
    intrinsics = mat2K(np.asarray(meta["K"]).reshape(3, 3).T)
    if flip == 90 or flip == 270:
        intrinsics = intrinsics[[1, 0, 3, 2]]
    skip_ratio = int(meta["fps"] / 10)
    print("skipping every %d frames" % skip_ratio)

    list_imgs = sorted(glob.glob("%s/rgb/*.jpg" % source_dir), key=extract_number)[
        ::skip_ratio
    ]
    for idx, rgb_path in enumerate(tqdm.tqdm(list_imgs)):
        rgb = cv2.imread(rgb_path)

        depth_path = rgb_path.replace("rgb", "depth").replace(".jpg", ".exr")
        depth = OpenEXR.InputFile(depth_path)
        depth = np.frombuffer(depth.channel("R"), dtype=np.float16)
        depth = depth.reshape(256, 192).astype(np.float32)

        if flip == 90:
            rgb = np.transpose(rgb, [1, 0, 2])[::-1] # counter clockwise
            depth = np.transpose(depth, [1, 0])[::-1]
        elif flip == 270:
            rgb = np.transpose(rgb, [1, 0, 2]) # clockwise
            depth = np.transpose(depth, [1, 0])
        elif flip == 180:
            rgb = rgb[::-1]
            depth = depth[::-1]
        elif flip == 0:
            pass
        else:
            raise ValueError

        # save
        trg1 = "%s/JPEGImagesRaw/Full-Resolution/%s/%05d.jpg" % (
            target_dir,
            seqname,
            idx,
        )
        trg2 = "%s/JPEGImages/Full-Resolution/%s/%05d.jpg" % (target_dir, seqname, idx)
        os.makedirs(os.path.dirname(trg1), exist_ok=True)
        os.makedirs(os.path.dirname(trg2), exist_ok=True)
        cv2.imwrite(trg1, rgb)
        cv2.imwrite(trg2, rgb)

        # depth
        trg = "%s/Depth/Full-Resolution/%s/%05d.npy" % (target_dir, seqname, idx)
        os.makedirs(os.path.dirname(trg), exist_ok=True)
        np.save(trg, depth)

        # # sanity check
        # cv2.imwrite("tmp/0.png", depth * 200)
        # print("written to tmp/0.png")
        # print(depth.shape)
        # cv2.imwrite("tmp/1.jpg", rgb)
        # print("written to tmp/1.jpg")
        # print(rgb.shape)
        # xyz = depth_to_xyz(depth, intrinsics / 4)
        # trimesh.Trimesh(vertices=xyz.reshape(-1, 3)[::100]).export("tmp/0.obj")
        # pdb.set_trace()

    # save cameras
    poses = np.asarray(meta["poses"])[::skip_ratio]  # xyzw / xyz
    extrinsics = np.tile(np.eye(4)[None], (len(poses), 1, 1))
    extrinsics[:, :3, 3] = poses[:, 4:]
    extrinsics[:, :3, :3] = R.from_quat(poses[:, :4]).as_matrix()
    if flip == 90:
        # from (x-up, y-left, z-inward) to (x-right, y-down, z-forward)
        transformation_matrix = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]])
    elif flip == 270:
        # from (x-down, y-right, z-inward) to (x-right, y-down, z-forward)
        transformation_matrix = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
    elif flip == 180:
        # from (x-left, y-down, z-inward) to (x-right, y-down, z-forward)
        transformation_matrix = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
    elif flip == 0:
        # from (x-right, y-up, z-inward) to (x-right, y-down, z-forward)
        transformation_matrix = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    else:
        raise ValueError
    extrinsics[:, :3, :3] = extrinsics[:, :3, :3] @ transformation_matrix[None]
    extrinsics = np.linalg.inv(extrinsics)
    gl_to_cv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    # scene_unrect = np.linalg.inv(extrinsics[:1]) @ gl_to_cv[None]
    # extrinsics = extrinsics @ scene_unrect
    extrinsics = extrinsics @ gl_to_cv

    mesh = draw_cams(extrinsics)
    trg = "%s/Cameras/Full-Resolution/%s/00.npy" % (target_dir, seqname)
    os.makedirs(os.path.dirname(trg), exist_ok=True)
    mesh.export("%s/Cameras/Full-Resolution/%s/cameras-00.obj" % (target_dir, seqname))
    np.save(trg, extrinsics)
    np.save(trg.replace("00.npy", "aligned-00.npy"), extrinsics)

    # run preprocessing
    write_config(vidname)

    # modify intrinsics
    config_path = "database/configs/%s.config" % vidname
    config = configparser.ConfigParser()
    config.read(config_path)
    config["data_0"]["ks"] = " ".join([str(j) for j in intrinsics.flatten()])
    with open(config_path, "w") as configfile:
        config.write(configfile)

    # combine with home data and save to a new file
    config_home = configparser.ConfigParser()
    config_home.read(home_path)
    config_home["data_1"] = config["data_0"]
    with open("database/configs/home-%s.config" % vidname, "w") as configfile:
        config_home.write(configfile)

    track_anything_lab4d(seqname, target_dir, prompt)
    # flow
    for dframe in [1, 2, 4, 8]:
        compute_flow(seqname, target_dir, dframe)
    extract_normal(seqname)

    res = 256
    extract_crop(seqname, res, 1)
    extract_crop(seqname, res, 0)
    extract_dinov2(vidname, component_id=0)
    extract_dinov2(vidname, component_id=1)

    camera_registration(seqname, 1)
    canonical_registration(seqname, 256, "quad")

    # run camera inference on the scene
    cmd = f"python projects/predictor/inference.py --flagfile=logdir/predictor-{envname}-ft4/opts.log --load_suffix latest --image_dir database/processed/JPEGImages/Full-Resolution/{seqname}/"
    os.system(cmd)
    os.system(f"python projects/csim/transform_bg_cams.py {seqname}")


if __name__ == "__main__":
    vidname = sys.argv[1]

    # vidname = "2024-05-07--19-25-33-v0"
    # home_path = "database/configs/Oct5at10-49AM-poly.config"
    # flip = True
    # prompt = "cat"
    # record3d_to_lab4d(vidname, home_path=home_path, flip=flip, prompt=prompt)

    # vidname = "2024-05-07--19-25-33-v1"
    # record3d_to_lab4d(vidname, home_path=home_path, flip=flip, prompt=prompt)

    # flip = False
    # home_path = "database/configs/Feb14at5-55тАпPM-poly.config"
    # prompt = "bunny"
    # # vidname = "2024-02-14--17-50-52"
    # vidname = "2024-02-14--17-51-46"

    # flip=True
    # home_path = "database/configs/Feb26at10-02 PM-poly.config"
    # prompt = "cat"
    # vidname = "2024-02-26--22-00-31"
    # record3d_to_lab4d(vidname, home_path=home_path, flip=flip, prompt=prompt)

    # home_path = "database/configs/Feb19at9-47 PM-poly.config"
    # folder_path = "database/record3d/dog"
    # flip = 270
    # prompt = "dog"
    # record3d_to_lab4d(folder_path, vidname, home_path=home_path, flip=flip, prompt=prompt)

    home_path = "database/configs/Oct5at10-49AM-poly.config"
    folder_path = "database/record3d/human"
    flip = 0
    prompt = "human"

    # home_path = "database/configs/May15at5-23PM-poly.config"
    # folder_path = "database/record3d/hand-controller"
    # flip = 90
    # prompt = "hand"
    record3d_to_lab4d(folder_path, vidname, home_path=home_path, flip=flip, prompt=prompt)


    # for vidname in sorted(glob.glob("%s/*" % folder_path)):
    #     vidname = vidname.split("/")[-1]
    #     record3d_to_lab4d(folder_path, vidname, home_path=home_path, flip=flip, prompt=prompt)