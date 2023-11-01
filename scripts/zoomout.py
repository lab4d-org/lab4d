# python scripts/zoomout.py tmp/collage.mp4 3
from moviepy.editor import VideoFileClip, clips_array, vfx
from moviepy.editor import ImageClip, concatenate_videoclips
import os
import sys
import math
import numpy as np
import cv2


def zoom_out_effect(input_path, output_path, divide):
    # Load the video clip
    clip = VideoFileClip(input_path)
    duration = clip.duration

    # Define a function for the zoom effect
    def zoom(get_frame, t):
        """
        Function to zoom the frame.
        """
        frame = get_frame(t)

        # Calculate zoom factor linearly increasing with time
        # It goes from 1/7 to 1 throughout the duration
        t = np.clip(2 * t - 0.5 * duration, 0, duration)
        factor = 1 / divide + t * (1 - 1 / divide) / duration

        # Center of the video
        center_x, center_y = frame.shape[1] / 2, frame.shape[0] / 2

        # Size of the zoomed in section
        w_zoom, h_zoom = frame.shape[1] * factor, frame.shape[0] * factor

        # Get the coordinates of the zoomed in section
        x1, y1 = int(center_x - w_zoom / 2), int(center_y - h_zoom / 2)
        x2, y2 = int(center_x + w_zoom / 2), int(center_y + h_zoom / 2)

        # Crop and resize to achieve zoom effect
        zoomed_frame = frame[y1:y2, x1:x2]
        return cv2.resize(zoomed_frame, (frame.shape[1], frame.shape[0]))

    # Apply the zoom effect
    zoom_clip = clip.fl(zoom)

    # Set the duration and output the result
    zoom_clip.set_duration(duration).write_videofile(
        output_path, codec="libx264", audio_codec="aac"
    )


# Example usage
input_video_path = sys.argv[1]
divide = int(sys.argv[2])
output_video_path = input_video_path.replace(".mp4", "_zoomout.mp4")
zoom_out_effect(input_video_path, output_video_path, divide)
