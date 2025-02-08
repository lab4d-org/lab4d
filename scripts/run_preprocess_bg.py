# python scripts/run_preprocess.py shiba-haru "0,1,2,3,4,5,6,7"
import configparser
import glob
import os
import pdb
import numpy as np
import cv2
import struct
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lab4d.utils.gpu_utils import gpu_map
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
from preprocess.scripts.fake_data import create_fake_masks
from preprocess.third_party.vcnplus.compute_flow import compute_flow
from preprocess.third_party.vcnplus.frame_filter import frame_filter
from preprocess.third_party.omnivision.normal import extract_normal


def remove_exist_dir(seqname, outdir):
    run_bash_command(f"rm -rf {outdir}/JPEGImages/Full-Resolution/{seqname}")
    run_bash_command(f"rm -rf {outdir}/Cameras/Full-Resolution/{seqname}")
    run_bash_command(f"rm -rf {outdir}/Features/Full-Resolution/{seqname}")
    run_bash_command(f"rm -rf {outdir}/Depth/Full-Resolution/{seqname}")
    run_bash_command(f"rm -rf {outdir}/Flow*/Full-Resolution/{seqname}")
    run_bash_command(f"rm -rf {outdir}/Annotations/Full-Resolution/{seqname}")
    run_bash_command(f"rm -rf {outdir}/Normal/Full-Resolution/{seqname}")


def run_extract_frames(seqname, outdir, infile, use_filter_frames, fps):
    # extract frames
    imgpath = f"{outdir}/JPEGImagesRaw/Full-Resolution/{seqname}"
    run_bash_command(f"rm -rf {imgpath}")
    os.makedirs(imgpath, exist_ok=True)
    extract_frames(infile, imgpath, desired_fps=fps)

    # remove existing dirs for preprocessing
    remove_exist_dir(seqname, outdir)

    # filter frames without motion: frame id is the time stamp
    if use_filter_frames:
        frame_filter(seqname, outdir)
    else:
        outpath = f"{outdir}/JPEGImages/Full-Resolution/{seqname}"
        run_bash_command(f"rm -rf {outpath}")
        os.makedirs(outpath, exist_ok=True)
        run_bash_command(f"cp {imgpath}/* {outpath}/")


def run_extract_priors(seqname, outdir):
    print("extracting priors: ", seqname)
    # flow
    for dframe in [1, 2, 4, 8]:
        compute_flow(seqname, outdir, dframe)

    # depth
    extract_depth(seqname)
    extract_normal(seqname)

    # TODO create fake masks
    create_fake_masks(seqname, outdir)

    # crop around object and process flow
    extract_crop(seqname, 256, 1)

    # compute bg/fg cameras
    camera_registration(seqname, 0)
    tsdf_fusion(seqname, 0)
    canonical_registration(seqname, 256, "other")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} <vidname> <gpulist>")
        print(f"  Example: python {sys.argv[0]} cat-pikachu-0 '0,1,2,3,4,5,6,7'")
        exit()
    vidname = sys.argv[1]
    gpulist = [int(n) for n in sys.argv[2].split(",")]

    use_filter_frames = False
    fps = 10

    outdir = "database/processed/"
    viddir = "database/raw/%s" % vidname
    print("using gpus: ", gpulist)
    os.makedirs("tmp", exist_ok=True)

    # download the videos
    download_seq(vidname)

    # set up parallel extraction
    frame_args = []
    for counter, infile in enumerate(sorted(glob.glob("%s/*" % viddir))):
        seqname = "%s-%04d" % (vidname, counter)
        frame_args.append((seqname, outdir, infile, use_filter_frames, fps))

    # extract frames and filter frames without motion: frame id is the time stamp
    gpu_map(run_extract_frames, frame_args, gpus=gpulist)

    # write config
    write_config(vidname)

    # read config
    config = configparser.RawConfigParser()
    config.read("database/configs/%s.config" % vidname)
    prior_args = []
    for vidid in range(len(config.sections()) - 1):
        seqname = config.get("data_%d" % vidid, "img_path").strip("/").split("/")[-1]
        prior_args.append((seqname, outdir))

    # extract flow/depth/camera/etc
    gpu_map(run_extract_priors, prior_args, gpus=gpulist)

    # extract dinov2 features
    extract_dinov2(vidname, 256, component_id=0, gpulist=gpulist)
