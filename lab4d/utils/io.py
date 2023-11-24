# Copyright (c) 2023 Gengshan Yang, Carnegie Mellon University.
import glob
import os

import cv2
import einops
import imageio
import numpy as np

from lab4d.utils.vis_utils import img2color, make_image_grid


def make_save_dir(opts, sub_dir="renderings"):
    """Create a subdirectory to save outputs

    Args:
        opts (Dict): Command-line options
        sub_dir (str): Subdirectory to create
    Returns:
        save_dir (str): Output directory
    """
    logname = "%s-%s" % (opts["seqname"], opts["logname"])
    save_dir = "%s/%s/%s/" % (opts["logroot"], logname, sub_dir)
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


def resize_to_nearest_multiple(image, multiple=16):
    height, width = image.shape[:2]
    new_height = int(np.ceil(height / multiple) * multiple)
    new_width = int(np.ceil(width / multiple) * multiple)
    return cv2.resize(image, (new_width, new_height))


def save_vid(
    outpath,
    frames,
    suffix=".mp4",
    upsample_frame=0,
    fps=10,
    target_size=None,
    max_pixels=4e6,
):
    """Save frames to video

    Args:
        outpath (str): Output directory
        frames: (N, H, W, x) Frames to output
        suffix (str): File type to save (".mp4" or ".gif")
        upsample_frame (int): Target number of frames
        fps (int): Target frames per second
        target_size: If provided, (H, W) target size of frames
    """
    # convert to 150 frames
    if upsample_frame < 1:
        upsample_frame = len(frames)
    frame_150 = []
    for i in range(int(upsample_frame)):
        fid = int(i / upsample_frame * len(frames))
        frame = frames[fid]
        if frame.max() <= 1:
            frame = frame * 255
        frame = frame.astype(np.uint8)
        if target_size is not None:
            frame = cv2.resize(frame, target_size[::-1])
        elif frame.shape[0] * frame.shape[1] > max_pixels:
            fxy = np.sqrt(max_pixels / (frame.shape[0] * frame.shape[1]))
            frame = cv2.resize(frame, None, fx=fxy, fy=fxy)
        if suffix == ".gif":
            h, w = frame.shape[:2]
            fxy = np.sqrt(4e4 / (h * w))
            frame = cv2.resize(frame, None, fx=fxy, fy=fxy)

        # resize to make divisible by marco block size = 16
        h, w = frame.shape[:2]
        h = int(np.ceil(h / 16) * 16)
        w = int(np.ceil(w / 16) * 16)
        frame = cv2.resize(frame, (w, h))

        frame_150.append(frame)

    # to make divisible by 16
    frame_150_resized = [resize_to_nearest_multiple(frame) for frame in frame_150]
    imageio.mimsave("%s%s" % (outpath, suffix), frame_150_resized, fps=fps)


def save_rendered(rendered, save_dir, raw_size, pca_fn):
    """Save rendered outputs

    Args:
        rendered (Dict): Maps arbitrary keys to outputs of shape (N, H, W, x)
        save_dir (str): Output directory
        raw_size: (2,) Target height and width
        pca_fn (Function): Function to apply PCA on feature outputs
    """
    # save rendered images
    for k, v in rendered.items():
        n, h, w = v.shape[:3]
        img_grid = make_image_grid(v)
        img_grid = img2color(k, img_grid, pca_fn=pca_fn)
        img_grid = (img_grid * 255).astype(np.uint8)
        # cv2.imwrite("%s/%s.jpg" % (save_dir, k), img_grid[:, :, ::-1])

        # save video
        frames = einops.rearrange(img_grid, "(m h) (n w) c -> (m n) h w c", h=h, w=w)
        frames = frames[:n]
        save_vid(
            "%s/%s" % (save_dir, k),
            frames,
            fps=10,
            target_size=(raw_size[0], raw_size[1]),
        )
