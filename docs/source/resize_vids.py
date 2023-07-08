# python source/resize_vids.py
import os
import numpy as np
import imageio
from PIL import Image

src_dir = "source/_static/media"
dst_dir = "source/_static/media_resized/"
max_dim = 640 * 640
video_exts = [".mp4", ".avi", ".mov", ".flv", ".mkv", ".wmv"]

# check for destination directory and create if it doesn't exist
if not os.path.exists(dst_dir):
    os.makedirs(dst_dir)

# iterate over video files in source directory
for filename in os.listdir(src_dir):
    # check if file is a video, ignoring the case of the extension
    if any(filename.lower().endswith(ext) for ext in video_exts):
        # add other conditions if there are other video formats
        src_filepath = os.path.join(src_dir, filename)
        dst_filepath = os.path.splitext(filename)[0] + ".mp4"
        dst_filepath = os.path.join(dst_dir, dst_filepath)

        reader = imageio.get_reader(src_filepath)
        fps = reader.get_meta_data()["fps"]

        # obtain video dimensions
        first_frame = reader.get_data(0)
        orig_height, orig_width = first_frame.shape[:2]

        # check if resolution is greater than 640x640
        if orig_height * orig_width > max_dim:
            print("Resizing video: " + filename)
            # resize maintaining aspect ratio
            ratio = np.sqrt(max_dim / (orig_height * orig_width))
            new_width = int(orig_width * ratio)
            new_height = int(orig_height * ratio)

            writer = imageio.get_writer(dst_filepath, fps=fps)

            # iterate over frames in the video
            for i, frame in enumerate(reader):
                frame = Image.fromarray(frame)
                frame = frame.resize((new_width, new_height), Image.ANTIALIAS)
                writer.append_data(np.array(frame))

            writer.close()
        else:
            # copy video to destination directory
            print("Copying video: " + filename)
            os.system("cp " + src_filepath + " " + dst_filepath)

print("Video resizing is complete!")
