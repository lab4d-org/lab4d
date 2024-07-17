import numpy as np
import os, sys
import glob
import json
import numpy as np
import pdb
import trimesh
import cv2
import tqdm

sys.path.insert(0, os.getcwd())
from projects.csim.render_polycam import PolyCamRender
from lab4d.utils.io import save_vid


def cartesian_to_spherical(xcoord, ycoord, zcoord):
    """
    xyz are normlized
    """
    xy = xcoord**2 + ycoord**2
    elevation = np.arctan2(np.sqrt(xy), zcoord)
    azimuth = np.arctan2(ycoord, xcoord)
    return elevation, azimuth

def sample_extrinsics_outside_in(
    elevation_limit=np.pi / 6,
    azimuth_limit=np.pi / 6,
    roll_limit=np.pi / 6,
):
    elevation = np.random.uniform(-elevation_limit, elevation_limit)
    azimuth = np.random.uniform(-azimuth_limit, azimuth_limit)
    roll = np.random.uniform(-roll_limit, roll_limit)

    # convert to rotation matrix
    Rotx = np.eye(4)
    Rotx[:3, :3] = cv2.Rodrigues(np.array([elevation, 0, 0]))[0]
    Roty = np.eye(4)
    Roty[:3, :3] = cv2.Rodrigues(np.array([0, azimuth, 0]))[0]
    Rotz = np.eye(4)
    Rotz[:3, :3] = cv2.Rodrigues(np.array([0, 0, roll]))[0]
    d_extrinsics = Rotx @ Roty @ Rotz
    return d_extrinsics

def sample_extrinsics(
    extrinsics_base,
    elevation_limit=np.pi / 4,
    azimuth_limit=np.pi,
    roll_limit=np.pi / 6,
    trans_std=0.25,
    aabb=None,
):
    elevation = np.random.uniform(-elevation_limit, elevation_limit)
    azimuth = np.random.uniform(-azimuth_limit, azimuth_limit)
    roll = np.random.uniform(-roll_limit, roll_limit)
    trans = np.linalg.inv(extrinsics_base)[:3, 3]

    # random translation
    if aabb is not None and np.random.binomial(1, 0.5) == 1:
        trans = np.random.uniform(aabb[0], aabb[1], size=3)
    else:
        # resample translation
        delta_trans = np.random.randn(3) * trans_std  # std = 0.25m
        trans = trans + delta_trans

    extrinsics_base = np.eye(4)
    extrinsics_base[:3, 3] = -trans

    # convert to rotation matrix
    Rotx = np.eye(4)
    Rotx[:3, :3] = cv2.Rodrigues(np.array([elevation, 0, 0]))[0]
    Roty = np.eye(4)
    Roty[:3, :3] = cv2.Rodrigues(np.array([0, azimuth, 0]))[0]
    Rotz = np.eye(4)
    Rotz[:3, :3] = cv2.Rodrigues(np.array([0, 0, roll]))[0]
    extrinsics = Rotx @ Roty @ Rotz @ extrinsics_base

    return extrinsics


if __name__ == "__main__":
    # poly_name = "Oct5at10-49AM-poly"
    # poly_name = "Oct25at8-48PM-poly"
    poly_name = "Oct31at1-13AM-poly"

    poly_path = "database/polycam/%s" % poly_name
    polycam_loader = PolyCamRender(poly_path, image_size=(1024, 768))
    polycam_loader.renderer.set_ambient_light()
    for frame_idx in range(len(polycam_loader)):
        outdir = "projects/csim/zero123_data/home_rand/%s-%03d" % (
            poly_name,
            frame_idx,
        )
        os.makedirs(outdir, exist_ok=True)
        extrinsics_base = polycam_loader.extrinsics[frame_idx]
        frames = []
        for rot_idx in tqdm.tqdm(range(12), desc="rotating frame %d" % frame_idx):
            # random xyz direction from -1,1
            extrinsics = sample_extrinsics(extrinsics_base)
            color, depth = polycam_loader.render(frame_idx, extrinsics=extrinsics)
            cv2.imwrite("%s/%03d.jpg" % (outdir, rot_idx), color[..., ::-1])
            # cv2.imwrite("%s/%03d_depth.jpg" % (outdir, rot_idx), depth * 50)
            np.save("%s/%03d.npy" % (outdir, rot_idx), extrinsics)
            frames.append(color)
        print("\nrendered to %s/render.mp4\n" % outdir)
        save_vid("%s/render" % outdir, frames, suffix=".mp4", upsample_frame=-1)
        break
