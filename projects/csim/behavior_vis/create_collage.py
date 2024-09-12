import os, sys
import cv2
import glob
import numpy as np
import pdb
from einops import rearrange, reduce, repeat
import configparser
import moviepy

# data_path = "database/processed/JPEGImages/Full-Resolution/2023-1"
# dir_list = sorted(glob.glob(data_path + "*"))[:6]
# vidname = "home-2023-curated3"
# results_dir = "logdir//home-2023-curated3-compose-ft/"
vidname = sys.argv[1]
results_dir = sys.argv[2]

# read config
config = configparser.RawConfigParser()
config.read("database/configs/%s.config" % vidname)
dir_list = []
for vidid in range(len(config.sections()) - 1):
    seqname = config.get("data_%d" % vidid, "img_path").strip("/").split("/")[-1]
    dirpath = "database/processed/JPEGImages/Full-Resolution/%s" % seqname
    dir_list.append(dirpath)
dir_list = dir_list[1:7]

down_ratio = 3
vid_list = []
for inst_id, dirpath in enumerate(dir_list):
    img_list = []
    counter = 0
    # TODO: find the reconstruction of the video
    vidpath = "%s/export_%04d/render-bone-compose-ref.mp4" % (
        results_dir,
        inst_id + 1,
    )
    vid = cv2.VideoCapture(vidpath)

    for path in sorted(glob.glob(dirpath + "/*.jpg")):
        # TODO: find the reconstruction of the video
        vidpath = "%s/export_%04d/render-bone-compose-ref.mp4" % (
            results_dir,
            inst_id + 1,
        )

        if counter > 2:
            break
        # sample every 50 frames
        frameid = int(path.split("/")[-1].split(".")[0].split("_")[-1])
        if frameid % 80 == 0:
            img = cv2.imread(path)
            img = cv2.resize(
                img, (img.shape[1] // down_ratio, img.shape[0] // down_ratio)
            )
            img = img[20:]
            img_list.append(img)
            vid.set(cv2.CAP_PROP_POS_FRAMES, frameid - 1)
            ret, frame = vid.read()
            frame = cv2.resize(
                frame, (frame.shape[1] // down_ratio, frame.shape[0] // down_ratio)
            )
            frame = frame[20:]
            img_list.append(frame)
            counter += 1

    img_list = np.asarray(img_list)  # T,H,W,C
    vid_list.append(img_list)
vid_list = np.asarray(vid_list)  # V,T,H,W,C
vid_list = rearrange(vid_list, "v t h w c -> t v h w c")

# fine the best cols and rows
cols = vid_list.shape[0]
rows = vid_list.shape[1]

# Create a collage from the images
# add a white line of with 10 between the images
collage = 255 * np.ones(
    (
        rows * img_list[0].shape[0] + (rows + 1) * 10,
        cols * img_list[0].shape[1] + (cols + 1) * 10,
        3,
    ),
    dtype=np.uint8,
)
for i in range(rows):
    for j in range(cols):
        img = vid_list[j, i]
        collage[
            i * img.shape[0] + (i + 1) * 10 : (i + 1) * img.shape[0] + (i + 1) * 10,
            j * img.shape[1] + (j + 1) * 10 : (j + 1) * img.shape[1] + (j + 1) * 10,
            :,
        ] = img
# Save the collage
cv2.imwrite("collage.jpg", collage)
print("Collage saved as collage.jpg")
