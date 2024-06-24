# python scripts/run_preprocess.py shiba-haru animal quad "0,1,2,3,4,5,6,7"
import configparser
import glob
import importlib
import os
import pdb
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
from preprocess.third_party.vcnplus.compute_flow import compute_flow
from preprocess.third_party.vcnplus.frame_filter import frame_filter
from preprocess.third_party.omnivision.normal import extract_normal

track_anything_module = importlib.import_module(
    "preprocess.third_party.Track-Anything.app"
)
track_anything_gui = track_anything_module.track_anything_interface
track_anything_cli = importlib.import_module(
    "preprocess.third_party.Track-Anything.track_anything_cli"
)
track_anything_cli = track_anything_cli.track_anything_cli


def track_anything_lab4d(seqname, outdir, text_prompt):
    input_folder = "%s/JPEGImages/Full-Resolution/%s" % (outdir, seqname)
    output_folder = "%s/Annotations/Full-Resolution/%s" % (outdir, seqname)
    track_anything_cli(input_folder, text_prompt, output_folder)


def remove_exist_dir(seqname, outdir):
    run_bash_command(f"rm -rf {outdir}/JPEGImages/Full-Resolution/{seqname}")
    run_bash_command(f"rm -rf {outdir}/Annotations/Full-Resolution/{seqname}")
    run_bash_command(f"rm -rf {outdir}/Cameras/Full-Resolution/{seqname}")
    run_bash_command(f"rm -rf {outdir}/Features/Full-Resolution/{seqname}")
    run_bash_command(f"rm -rf {outdir}/Depth/Full-Resolution/{seqname}")
    run_bash_command(f"rm -rf {outdir}/Flow*/Full-Resolution/{seqname}")


def run_extract_frames(seqname, outdir, infile, use_filter_frames):
    # extract frames
    imgpath = f"{outdir}/JPEGImagesRaw/Full-Resolution/{seqname}"
    run_bash_command(f"rm -rf {imgpath}")
    os.makedirs(imgpath, exist_ok=True)
    extract_frames(infile, imgpath)

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


def run_extract_priors(seqname, outdir, obj_class_cam):
    print("extracting priors: ", seqname)
    # flow
    for dframe in [1, 2, 4, 8]:
        compute_flow(seqname, outdir, dframe)

    # depth
    extract_depth(seqname)
    extract_normal(seqname)

    # crop around object and process flow
    extract_crop(seqname, 256, 0)
    extract_crop(seqname, 256, 1)

    # compute bg/fg cameras
    camera_registration(seqname, 0)
    camera_registration(seqname, 1)
    tsdf_fusion(seqname, 0)
    canonical_registration(seqname, 256, obj_class_cam)


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print(
            f"Usage: python {sys.argv[0]} <vidname> <text_prompt_seg> <obj_class_cam> <gpulist>"
        )
        print(
            f"  Example: python {sys.argv[0]} cat-pikachu-0 cat quad '0,1,2,3,4,5,6,7'"
        )
        exit()
    vidname = sys.argv[1]
    text_prompt_seg = sys.argv[2]
    obj_class_cam = sys.argv[3]
    assert obj_class_cam in ["human", "quad", "other"]
    gpulist = [int(n) for n in sys.argv[4].split(",")]

    # True: manually annotate object masks | False: use detect object based on text prompt
    use_manual_segment = True if text_prompt_seg == "other" else False
    # True: manually annotate camera for key frames
    use_manual_cameras = True if obj_class_cam == "other" else False
    # True: filter frame based on motion magnitude | False: use all frames
    use_filter_frames = False

    outdir = "database/processed/"
    viddir = "database/raw/%s" % vidname
    print("using gpus: ", gpulist)
    os.makedirs("tmp", exist_ok=True)

    # check if the directory with the video already exists
    viddir_path = os.path.join("database", "raw", vidname)
    if not os.path.exists(viddir_path):
        # download the videos only if the directory does not exist
        download_seq(vidname)

    # set up parallel extraction
    frame_args = []
    for counter, infile in enumerate(sorted(glob.glob("%s/*" % viddir))):
        seqname = "%s-%04d" % (vidname, counter)
        frame_args.append((seqname, outdir, infile, use_filter_frames))

    # extract frames and filter frames without motion: frame id is the time stamp
    gpu_map(run_extract_frames, frame_args, gpus=gpulist)

    # write config
    write_config(vidname)

    # read config
    config = configparser.RawConfigParser()
    config.read("database/configs/%s.config" % vidname)
    seg_args = []
    prior_args = []
    for vidid in range(len(config.sections()) - 1):
        seqname = config.get("data_%d" % vidid, "img_path").strip("/").split("/")[-1]
        seg_args.append((seqname, outdir, text_prompt_seg))
        prior_args.append((seqname, outdir, obj_class_cam))

    # let the user specify the segmentation mask
    if use_manual_segment:
        track_anything_gui(vidname)
    else:
        gpu_map(track_anything_lab4d, seg_args, gpus=gpulist)

    # Manually adjust camera positions
    if use_manual_cameras:
        from preprocess.scripts.manual_cameras import manual_camera_interface

        mesh_path = "database/mesh-templates/cat-pikachu-remeshed.obj"
        manual_camera_interface(vidname, mesh_path)

    # extract flow/depth/camera/etc
    gpu_map(run_extract_priors, prior_args, gpus=gpulist)

    # extract dinov2 features
    extract_dinov2(vidname, 256, gpulist=gpulist)
