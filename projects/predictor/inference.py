import os, sys
from absl import app, flags
import pdb
import cv2
import glob
import numpy as np

sys.path.append(os.getcwd())

from lab4d.config import get_config
from lab4d.utils.io import save_vid
from projects.predictor.predictor import Predictor
from projects.predictor.trainer import PredTrainer


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
    # load a new image
    rgb_input, extr = model.data_generator.sample(np.asarray(range(0, 200, 10)))
    rgb_input_new = []
    for path in sorted(
        glob.glob(os.path.join(opts["image_dir"], "*.jpg"))
        # glob.glob(
        # )
        # glob.glob("database/polycam/Oct25at8-48PM-poly/keyframes/images/*.jpg")
        # glob.glob("database/polycam/Oct5at10-49AM-poly/keyframes/images/*.jpg")
        # glob.glob("database/polycam/Oct31at1-13AM-poly/keyframes/images/*.jpg")
    ):
        img = cv2.imread(path)[..., ::-1]

        # remove foreground
        sil_path = path.replace("JPEGImages", "Annotations").replace(".jpg", ".npy")
        if os.path.exists(sil_path):
            sil = np.load(sil_path)
            # find the bbox
            indices = np.where(sil > 0)
            if len(indices[0]) > 0:
                xid = indices[1]
                yid = indices[0]
                ul = (xid.min(), yid.min())
                br = (xid.max(), yid.max())
                # set bg sil
                sil = (sil == 0).astype(np.uint8)
                sil[ul[1] : br[1], ul[0] : br[0]] = 0
                sil = sil[..., None]
                img = img * sil + img[sil[..., 0] > 0].mean() * (1 - sil)

        if "poly" in path:
            # flip the image
            img = np.transpose(img, (1, 0, 2))[:, ::-1]

        # resize
        img = cv2.resize(img, dsize=None, fx=0.25, fy=0.25)
        rgb_input_new.append(img)
    rgb_input_new = np.stack(rgb_input_new, 0)
    rgb_input = rgb_input_new

    batch = model.data_generator.convert_to_batch(rgb_input)
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
    out_frames = np.concatenate([rgb_input, re_rgb_resized], 2)
    save_vid("tmp/input_rendered", out_frames)
    print("saved to tmp/input_rendered.mp4")


def main(_):
    opts = get_config()
    run_inference(opts)


if __name__ == "__main__":
    app.run(main)
