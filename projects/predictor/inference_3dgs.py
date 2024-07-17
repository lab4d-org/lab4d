import os, sys
from absl import app, flags
import pdb
import cv2
import glob
import numpy as np

sys.path.append(os.getcwd())

from lab4d.config import get_config
from lab4d.utils.io import save_vid
from lab4d.utils.vis_utils import draw_cams
from preprocess.libs.io import read_raw
from projects.predictor.predictor import Predictor
from projects.predictor.trainer import PredTrainer
from projects.predictor.dataloader.dataset import convert_to_batch
from projects.csim.transform_bg_cams import transform_bg_cams


class InferenceFlags:
    """Flags for the renderer."""

    flags.DEFINE_string(
        "image_dir",
        "database/processed/JPEGImages/Full-Resolution/cat-pikachu-0010/",
        "image directory that contains the images to make predictions on",
    )
    pass


def run_inference(opts):
    # instantiate network
    model = Predictor(opts)
    logname = "%s-%s" % (opts["seqname"], opts["logname"])
    load_path = "%s/%s/ckpt_%s.pth" % (
        opts["logroot"],
        logname,
        opts["load_suffix"],
    )
    _ = PredTrainer.load_checkpoint(load_path, model)
    model.cuda()
    model.eval()
    # load from sim
    batch = model.data_generator1.generate_batch(100)
    rgb_input = batch["img"].permute(0,2,3,1).cpu().numpy() * 255
    depth_input = batch["xyz"][...,2].cpu().numpy()

    # # load input image
    # rgb_input = []
    # depth_input = []
    # for path in sorted(glob.glob(os.path.join(opts["image_dir"], "*.jpg"))):
    #     data_dict = read_raw(path, 0, crop_size=256, use_full=False, with_flow=False, crop_mode="median")
    #     img = data_dict["img"] * data_dict["mask"][...,:1].astype(float) * 255
    #     depth = data_dict["depth"]
    #     rgb_input.append(img)
    #     depth_input.append(depth)
    # rgb_input = np.stack(rgb_input, 0)
    # depth_input = np.stack(depth_input, 0)
    # batch = convert_to_batch(rgb_input, None, depth_input)

    # predict pose and visualize
    re_rgb, extrinsics, uncertainty = model.predict_batch(batch)

    # resize rerendered images
    dsize = rgb_input.shape[1:3][::-1]
    osize = re_rgb.shape[1:3][::-1]
    dsize = ((dsize[1] * osize[0]) // osize[1], dsize[1])
    re_rgb_resized = []
    for i, img in enumerate(re_rgb):
        re_rgb_resized.append(cv2.resize(img, dsize=dsize) * 255)
    re_rgb_resized = np.stack(re_rgb_resized, 0)
    depth_input_vis = np.tile(depth_input[..., None], (1, 1, 1, 3)) * 50
    out_frames = np.concatenate([rgb_input, depth_input_vis, re_rgb_resized], 2)
    seqname = opts["image_dir"].strip("/").split("/")[-1]
    os.makedirs("tmp/predictor", exist_ok=True)
    save_vid("tmp/predictor/input_rendered-%s" % seqname, out_frames)
    print("saved to tmp/predictor/input_rendered-%s.mp4" % seqname)

    # save cameras
    trg_path = "database/processed/Cameras/Full-Resolution/%s/" % seqname
    extrinsics = extrinsics.cpu().numpy()
    # mesh = draw_cams(extrinsics)
    # mesh.export("%s/cameras-00.obj" % trg_path)
    # print("cameras vis exported to %s/cameras-00.obj" % trg_path)
    # np.save("%s/00.npy" % trg_path, extrinsics)
    # np.save("%s/aligned-00.npy" % trg_path, extrinsics)
    np.save("tmp/predictor/extrinsics-%s.npy" % seqname, extrinsics)
    np.save("tmp/predictor/errors-%s.npy" % seqname, uncertainty)
    # transform_bg_cams(seqname, src_dir="tmp/predictor/")


def main(_):
    opts = get_config()
    run_inference(opts)


if __name__ == "__main__":
    app.run(main)
