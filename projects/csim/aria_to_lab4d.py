import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from tqdm import tqdm
import pdb
import cv2
import os
import glob

from projectaria_tools.core import data_provider, calibration
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
from projectaria_tools.core.stream_id import RecordableTypeId, StreamId

from lab4d.utils.io import save_vid
from preprocess.libs.io import run_bash_command

def image_config_example(config):
    print(f"device_type {config.device_type}")
    print(f"device_version {config.device_version}")
    print(f"device_serial {config.device_serial}")
    print(f"sensor_serial {config.sensor_serial}")
    print(f"nominal_rate_hz {config.nominal_rate_hz}")
    print(f"image_width {config.image_width}")
    print(f"image_height {config.image_height}")
    print(f"pixel_format {config.pixel_format}")


def convert_vrs_file(vrsfile, vidname):
    print(f"Creating data provider from {vrsfile}")
    provider = data_provider.create_vrs_data_provider(vrsfile)
    if not provider:
        print("Invalid vrs data provider")

    sensor_name = "camera-rgb"
    sensor_stream_id = provider.get_stream_id_from_label(sensor_name)
    config = provider.get_image_configuration(sensor_stream_id)
    image_config_example(config)

    # get all image data by index
    frames = []
    num_data = provider.get_num_data(sensor_stream_id)
    for index in range(0, num_data):
        image_data = provider.get_image_data_by_index(sensor_stream_id, index)
        print(
            f"Get image: {index} with timestamp {image_data[1].capture_timestamp_ns}"
        )

        # input: retrieve image as a numpy array
        image_array = image_data[0].to_numpy_array()
        # input: retrieve image distortion
        device_calib = provider.get_device_calibration()
        src_calib = device_calib.get_camera_calib(sensor_name)

        # create output calibration: a linear model of image size 1024 and focal length 500
        # Invisible pixels are shown as black.
        dst_calib = calibration.get_linear_camera_calibration(1024, 1024, 500, sensor_name)

        # distort image
        rectified_array = calibration.distort_by_calibration(image_array, dst_calib, src_calib)
        rectified_array = np.transpose(rectified_array[::-1], (1, 0, 2))
        # cv2.imwrite("undistorted_image.png", rectified_array[...,::-1])
        # cv2.imwrite("tmp/distorted_image.png", image_array[...,::-1])
        frames.append(rectified_array)
    print(dst_calib.projection_params())

    seqname = vrsfile.split("/")[-1].split(".")[0]
    os.makedirs("database/raw/%s/"%(vidname), exist_ok=True)
    save_vid("database/raw/%s/%s"%(vidname, seqname), frames, fps=10)
    print("saved to database/raw/%s/%s"%(vidname, seqname))

    # run is: ffmpeg -i tmp/undistorted.mp4 -vcodec libx265 -crf 28 output.mp4
    # run_bash_command("ffmpeg -i tmp/undistorted.mp4 -vcodec libx264 tmp/output.mp4")


if __name__ == "__main__":

    import sys
    vrsfile_folder = sys.argv[1]
    vidname = sys.argv[2]
    assert len(sys.argv[2]) > 0

    for vrsfile in sorted(glob.glob("%s/*.vrs" % vrsfile_folder)):
        convert_vrs_file(vrsfile, vidname)
    