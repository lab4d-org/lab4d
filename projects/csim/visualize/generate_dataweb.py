import os
import pdb
import dominate
from dominate.tags import *
import glob
import sys
import configparser
import subprocess
from showdata import generate_html_table

def concatenate_and_speedup_videos(video_paths, output_path, speed_factor):
    # Temporary file names
    concat_list_filename = 'concat_list.txt'
    temp_output_path = 'temp_output.mp4'
    
    # Create a temporary text file to list all videos
    with open(concat_list_filename, 'w') as f:
        for path in video_paths:
            f.write(f"file '{path}'\n")
    
    # Step 1: Concatenate videos
    subprocess.run([
        'ffmpeg',
        '-y',
        '-f', 'concat',
        '-safe', '0',
        '-i', concat_list_filename,
        '-c', 'copy',
        '-an',  # This option skips including any audio stream in the output
        temp_output_path
    ])
    
    # Step 2: Speed up the concatenated video by the specified factor, without considering audio
    subprocess.run([
        'ffmpeg',
        '-y', 
        '-i', temp_output_path,
        '-filter:v', f"setpts=PTS/{speed_factor}",  # Only apply the speed-up to the video
        '-an',  # Again, ensure no audio is included
        output_path
    ])
    
    # Clean up temporary files
    os.remove(concat_list_filename)
    os.remove(temp_output_path)

class model_viewer(dominate.tags.html_tag):
    pass

if __name__ == "__main__":
    # input_dir = "logdir/home-2023-curated3-compose-ft/"
    # configpath = "database/configs/home-2023-curated3.config"
    #input_dir = "logdir/home-2024-02-26-compose-ft/"
    #configpath = "database/configs/home-2024-02-26.config"
    #input_dir = "logdir/home-2024-02-14--17-compose-ft/"
    #configpath = "database/configs/home-2024-02-14--17.config"
    input_dir = sys.argv[1]
    configpath = sys.argv[2]
    outname = "index"
    vidlist = sorted(glob.glob("%s/*/render-*-ref.mp4" % (input_dir)))

    config = configparser.ConfigParser()
    config.read(configpath)

    data = []
    concat_paths =[]
    for it, vidpath in enumerate(vidlist):
        it = int(vidpath.split("/")[-2].split("_")[-1])
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
            "ffmpeg -y -i %s -vf \"scale=-2:320\" %s"
            % (vidpath, vidpath_down)
        )

        os.system(
            "ffmpeg -y -i %s -vf \"scale=-2:640\" %s" % (bev_vidpath, bev_vidpath_down)
        )

        img_path = config.get("data_%d" % it, "img_path") + "/%05d.jpg"
        os.system(
            'ffmpeg -y -framerate 10 -i ../../code/vid2sim/%s -vf "scale=iw/4:ih/4" %s'
            % (img_path, raw_path)
        )
        os.system(
            "ffmpeg -y -i %s -vf \"scale=-2:320\" %s"
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
        concat_paths.append(outpath)
    previous_dir = os.getcwd()
    os.chdir(input_dir)
    generate_html_table(data, output_path="%s.html" % outname, image_height="320px")
    os.chdir(previous_dir)

    # fast forward video
    concat_path = "%s/fast_forward.mp4"%input_dir
    concatenate_and_speedup_videos(concat_paths, concat_path, 10)