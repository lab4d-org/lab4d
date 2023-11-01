# python gallery.py --vid_dir database/raw/cat-pikachu/ --filter_height 1080
from moviepy.editor import VideoFileClip, clips_array, vfx
from moviepy.editor import ImageClip, concatenate_videoclips
import glob
import argparse
import os
import math
import cv2
import tqdm
import pdb
import subprocess

parser = argparse.ArgumentParser(description="Combine videos into a grid gallery")
parser.add_argument(
    "--vid_dir", default="database/raw/cat-pikachu-170", help="path to test dir"
)
parser.add_argument(
    "--filter_height", type=int, default=1920, help="filter videos with height"
)
parser.add_argument(
    "--max_duration", type=int, default=10, help="max duration of each video"
)
parser.add_argument(
    "--divide", type=int, default=8, help="divide the video resolution by this number"
)
args = parser.parse_args()


def get_video_rotation(video_path):
    cmd = [
        "ffprobe",
        "-v",
        "0",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream_side_data=rotation",
        "-of",
        "default=nw=1:nk=1",
        video_path,
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        # Convert the output to integer (rotation angle)
        return int(result.stdout)
    except ValueError:
        # Return 0 if no rotation data found
        return 0


def rotate_frame(frame, angle):
    if angle == -90:
        return cv2.transpose(cv2.flip(frame, 0))
    else:
        return frame


def get_hw(path):
    rotation_angle = get_video_rotation(path)
    cap = cv2.VideoCapture(path)
    ret, frame = cap.read()
    frame = rotate_frame(frame, rotation_angle)
    cap.release()
    return frame.shape[:2]


def ensure_seconds(clip, duration=10):
    """Ensure the video clip is exactly x seconds."""

    if clip.duration > duration:
        # Trim the clip if it's longer than x seconds
        clip = clip.subclip(0, duration)
    if clip.duration < duration:
        # Pad the clip with the last frame if it's shorter than x seconds
        padding_duration = duration - clip.duration

        frame_time = 1 / clip.fps  # Duration of a single frame
        last_frame_time = clip.duration - 3 * frame_time  # Timestamp of the last frame

        last_frame = clip.get_frame(last_frame_time)
        padding_clip = ImageClip(last_frame, duration=padding_duration)

        clip = concatenate_videoclips([clip, padding_clip])

    return clip


def main():
    save_path = os.path.join("tmp", "collage.mp4")

    # Get a list of all videos in the specified directory
    video_files = sorted(glob.glob(os.path.join(args.vid_dir, "*.MOV")))

    # Filter out videos with a height of 1080 pixels
    height_1080_videos = []
    sizes = []
    for clip in tqdm.tqdm(video_files):
        size = get_hw(clip)
        if size[0] == args.filter_height:
            height_1080_videos.append(VideoFileClip(clip))
            sizes.append(size)

    # Calculate closest square number for grid arrangement
    num_videos = len(height_1080_videos)
    side_length = int(math.sqrt(num_videos))

    # Reduce the number of videos to the closest square number
    height_1080_videos = height_1080_videos[: side_length**2]

    # Downscale by 8 and place in the grid
    grid = []
    for i in range(side_length):
        row = []
        for j in range(side_length):
            idx = i * side_length + j
            clip = height_1080_videos[idx]
            clip = ensure_seconds(clip, duration=args.max_duration)
            size = sizes[idx]
            new_width = size[1] // args.divide
            new_height = size[0] // args.divide
            clip_resized = clip.resize((new_width, new_height))
            row.append(clip_resized)
        grid.append(row)

    final_clip = clips_array(grid)
    final_clip.write_videofile(save_path, codec="libx264", audio_codec="aac", fps=10)


if __name__ == "__main__":
    main()
