# Copyright (c) 2023 Gengshan Yang, Carnegie Mellon University.
# python preprocess/scripts/extract_frames.py database/raw/cat-1015/10415567.mp4 tmp/
import sys

import imageio
import numpy as np


def extract_frames(in_path, out_path, desired_fps=30):
    print("extracting frames: ", in_path)
    # Open the video file
    reader = imageio.get_reader(in_path)
    original_fps = reader.get_meta_data()["fps"]
    # If a desired frame rate is higher than original
    if original_fps < desired_fps:
        desired_fps = original_fps

    # If a desired frame rate is given, calculate the frame skip rate
    skip_rate = 1
    if desired_fps:
        skip_rate = int(original_fps / desired_fps)

    # Find the first non-black frame
    for i, im in enumerate(reader):
        if np.any(im > 0):
            start_frame = i
            break

    # Write the video starting from the first non-black frame, considering the desired frame rate
    count = 0
    for i, im in enumerate(reader):
        if i >= start_frame and i % skip_rate == 0:
            imageio.imsave("%s/%05d.jpg" % (out_path, count), im)
            count += 1


if __name__ == "__main__":
    in_path = sys.argv[1]
    out_path = sys.argv[2]
    extract_frames(in_path, out_path)
