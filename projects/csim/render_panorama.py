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

if __name__ == "__main__":
    # poly_name = "Oct5at10-49AM-poly"
    # poly_name = "Oct25at8-48PM-poly"
    poly_name = "Oct31at1-13AM-poly"

    poly_path = "database/polycam/%s" % poly_name
    polycam_loader = PolyCamRender(poly_path, image_size=(1024, 768))
    polycam_loader.renderer.set_ambient_light()
    for frame_idx in tqdm.tqdm(range(len(polycam_loader))):
        outdir = "projects/csim/zero123_data/home_panorama/%s-%03d" % (
            poly_name,
            frame_idx,
        )
        os.makedirs(outdir, exist_ok=True)
        extrinsics_base = polycam_loader.extrinsics[frame_idx]
        # rectify extrinsics
        extrinsics_base_rot = extrinsics_base[:3, :3]
        axis_angle = cv2.Rodrigues(extrinsics_base_rot)[0]
        axis_angle[0] = 0
        axis_angle[2] = 0
        rect_T = np.eye(4)
        rect_T[:3, :3] = cv2.Rodrigues(axis_angle)[0] @ extrinsics_base_rot.T
        unrect_T = np.eye(4)
        unrect_T[:3, :3] = rect_T[:3, :3].T
        for rot_idx in tqdm.tqdm(range(12), desc="rotating frame %d" % frame_idx):
            rotation = np.eye(4)
            rotation[:3, :3] = cv2.Rodrigues(
                np.array([0, np.deg2rad(30 * rot_idx), 0])
            )[0]
            extrinsics = unrect_T @ rotation @ rect_T @ extrinsics_base
            color, depth = polycam_loader.render(frame_idx, extrinsics=extrinsics)

            cv2.imwrite("%s/%03d.jpg" % (outdir, rot_idx), color[..., ::-1])
            cv2.imwrite("%s/%03d_depth.jpg" % (outdir, rot_idx), depth * 50)
            np.save("%s/%03d.npy" % (outdir, rot_idx), extrinsics)
        print("rendered to %s" % outdir)
        break
