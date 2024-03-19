import os
import pdb
import dominate
from dominate.tags import *
import glob
import sys
from showdata import generate_html_table


class model_viewer(dominate.tags.html_tag):
    pass


input_dir = "logdir/home-2023-curated3-compose-ft/"
outname = "index"
vidlist = sorted(glob.glob("%s/*/*ref.mp4" % (input_dir)))

configpath = "database/configs/home-2023-curated3.config"
import configparser

config = configparser.ConfigParser()
config.read(configpath)

data = []
for it, vidpath in enumerate(vidlist):
    it = int(vidpath.split("/")[-2].split("_")[-1])
    if it == 0:
        continue
    print(vidpath)
    bev_vidpath = vidpath.replace("ref.mp4", "bev.mp4").replace("shape", "bone")
    vidpath_down = vidpath.replace("ref.mp4", "ref-down.mp4")
    bev_vidpath_down = bev_vidpath.replace("bev.mp4", "bev-down.mp4")
    outpath = vidpath.replace("ref.mp4", "concat.mp4")
    raw_path = vidpath.replace("ref.mp4", "raw.mp4")
    raw_path_down = vidpath.replace("ref.mp4", "raw-down.mp4")
    ref_concat = vidpath.replace("ref.mp4", "ref-concat.mp4")
    # read the two videos and resize height to 640
    os.system(
        "ffmpeg -y -i %s -vf \"scale='if(gt(a,1),-2,320)':'if(gt(a,1),320,-2)'\" %s"
        % (vidpath, vidpath_down)
    )

    os.system(
        "ffmpeg -y -i %s -vf \"scale='if(gt(a,1),-2,640)':'if(gt(a,1),640,-2)',"
        'crop=480:640:180:0" %s' % (bev_vidpath, bev_vidpath_down)
    )

    img_path = config.get("data_%d" % it, "img_path") + "/%05d.jpg"
    os.system(
        'ffmpeg -y -framerate 10 -i ../../code/vid2sim/%s -vf "scale=iw/4:ih/4" %s'
        % (img_path, raw_path)
    )
    os.system(
        "ffmpeg -y -i %s -vf \"scale='if(gt(a,1),-2,320)':'if(gt(a,1),320,-2)'\" %s"
        % (raw_path, raw_path_down)
    )

    os.system(
        "ffmpeg -y -i %s -i %s -filter_complex vstack %s"
        % (raw_path_down, vidpath_down, ref_concat)
    )
    os.system(
        "ffmpeg -y -i %s -i %s -filter_complex hstack %s"
        % (ref_concat, bev_vidpath_down, outpath)
    )

    frame = {}
    frame["Sequence ID"] = it
    frame["Reconstruction (Left: reference view | Right: bird's eye view)"] = (
        outpath.split("/", 2)[-1]
    )

    data.append(frame)
os.chdir(input_dir)
generate_html_table(data, output_path="%s.html" % outname, image_height="320px")
