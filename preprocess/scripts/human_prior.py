import configparser
import cv2
import glob
import pdb
import math
import numpy as np
import trimesh
import os
import sys
import tqdm
import torch
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.getcwd())
from lab4d.utils.vis_utils import draw_cams
from lab4d.utils.io import save_vid

# Import HMR2.0
from preprocess.third_party.hmr2.hmr2.configs import CACHE_DIR_4DHUMANS
from preprocess.third_party.hmr2.hmr2.datasets.utils import (
    expand_to_aspect_ratio, generate_image_patch_cv2, convert_cvimg_to_tensor
)
from preprocess.third_party.hmr2.hmr2.utils.renderer import Renderer, cam_crop_to_full
from preprocess.third_party.hmr2.hmr2.models import load_hmr2, download_models, DEFAULT_CHECKPOINT

def extract_joint_angles_hmr2(vidname):
    """Extract SMPL joint angles using HMR2.0"""
    # Load models
    download_models(CACHE_DIR_4DHUMANS)
    hmr2_model, hmr2_model_cfg = load_hmr2(DEFAULT_CHECKPOINT)
    hmr2_model = hmr2_model.cuda()

    # Read dataset focal length
    dataset_config = configparser.RawConfigParser()
    dataset_config.read(f"database/configs/{vidname}.config")

    for vidid in range(len(dataset_config.sections()) - 1):
        seqname = dataset_config.get("data_%d" % vidid, "img_path").strip("/").split("/")[-1]
        dataset_intrinsics = [float(x) for x in dataset_config.get("data_%d" % vidid, "ks").split(" ")]
        dataset_focal = math.sqrt(dataset_intrinsics[0] * dataset_intrinsics[1])
        extract_hmr_oneseq(seqname, dataset_focal, hmr2_model, hmr2_model_cfg)

def annotate_keypoints(img, kps):
    """
    Annotates keypoints on the given image.
    Args:
    img (numpy.ndarray): The input image (HxWx3).
    kps (numpy.ndarray): The keypoints (44x2), each row should be (x, y).
    Returns:
    numpy.ndarray: The annotated image.
    """
    # Ensure keypoints are integers, as they represent pixel indices
    kps = kps.astype(int)
    # Create a copy of the image to draw on
    annotated_image = img.copy()
    # Define the color and size of the keypoints
    color = (0, 255, 0)  # Green color
    radius = 3  # Radius of the circle
    # Draw each keypoint
    for (x, y) in kps:
        cv2.circle(annotated_image, (x, y), radius, color, -1)  # -1 fills the circle
    return annotated_image

def extract_hmr_oneseq(seqname, dataset_focal, hmr2_model, hmr2_model_cfg):
    device = next(hmr2_model.parameters()).device
    hmr2_size = hmr2_model_cfg.MODEL.IMAGE_SIZE  # must use hmr2's default
        
    # RGB impaths
    out_root = f"database/processed/Cameras/Full-Resolution/{seqname}"
    os.makedirs(out_root, exist_ok=True)
    out_pose = f"database/processed/Keypoint3D/Full-Resolution/{seqname}"
    os.makedirs(out_pose, exist_ok=True)

    imgdir = f"database/processed/JPEGImages/Full-Resolution/{seqname}"
    imglist = sorted(glob.glob(f"{imgdir}/*.jpg"))

    camlist_fg = []
    joint_angles_list = []
    hmr_vertices = []
    kps_2d = []
    for impath in tqdm.tqdm(imglist, desc=f"hmr2_smpl {seqname}"):
        maskpath = impath.replace("JPEGImages", "Annotations").replace(".jpg", ".npy")

        img_cv2 = cv2.imread(impath)  # H, W, C
        if img_cv2 is None:
            continue
        img_H, img_W, img_C = img_cv2.shape

        # mask = cv2.imread(maskpath, cv2.IMREAD_GRAYSCALE)  # H, W
        # y_idxs, x_idxs = np.nonzero(mask >= 64)  # -1, 2
        # img_mask = np.where(mask[..., None] >= 64, img_cv2, 0)
        mask = np.load(maskpath)
        if (mask > 0).sum() == 0:
            mask[:] = 1
            is_invalid = True
        else:
            is_invalid = False
        y_idxs, x_idxs = np.nonzero(mask > 0)  # -1, 2
        img_mask = np.where(mask[..., None] >= 0, img_cv2, 0)
        x0 = np.min(x_idxs)
        x1 = np.max(x_idxs)
        y0 = np.min(y_idxs)
        y1 = np.max(y_idxs)

        personid = 0
        center_x = (x0 + x1) / 2.0
        center_y = (y0 + y1) / 2.0
        scale = np.array([(x1 - x0) / 200.0, (y1 - y0) / 200.0])
        

        # Step 1: Generate HMR2.0 input image, and run HMR2.0 to estimate SMPL joint angles
        bbox_shape = hmr2_model_cfg.MODEL.get("BBOX_SHAPE", None)
        bbox_size_hmr2 = expand_to_aspect_ratio(scale * 200., target_aspect_ratio=bbox_shape).max()

        img_hmr2_patch, transform = generate_image_patch_cv2(
            img_mask, center_x, center_y,
            bbox_size_hmr2, bbox_size_hmr2, hmr2_size, hmr2_size, False, 1.0, 0,
            border_mode=cv2.BORDER_CONSTANT
        )  # 256, 256, 3
        img_hmr2_patch = img_hmr2_patch[:, :, ::-1]  # 256, 256, 3
        img_hmr2 = np.moveaxis(img_hmr2_patch, -1, 0)[None]  # 1, 3, 256, 256
        img_hmr2 = torch.tensor(img_hmr2.copy(), dtype=torch.uint8, device=device).to(torch.float32)
        for n_c in range(min(img_C, 3)):
            img_mean = 255.0 * hmr2_model_cfg.MODEL.IMAGE_MEAN[n_c]
            img_std = 255.0 * hmr2_model_cfg.MODEL.IMAGE_STD[n_c]
            img_hmr2[:, n_c, :, :] = (img_hmr2[:, n_c, :, :] - img_mean) / img_std

        hmr2_batch = {
            "img": img_hmr2,  # 1, 3, 256, 256
            "personid": torch.tensor([personid], dtype=torch.int64, device=device),  # 1,
            "box_center": torch.tensor([center_x, center_y], dtype=torch.float32, device=device)[None],  # 1, 2
            "box_size": torch.tensor([bbox_size_hmr2], dtype=torch.float32, device=device),  # 1,
            "img_size": torch.tensor([img_W, img_H], dtype=torch.float32, device=device)[None],  # 1, 2
        }
        with torch.no_grad():
            hmr2_out = hmr2_model(hmr2_batch)

        # Compute camera translation in full-image coordinates
        pred_cam_t_full = cam_crop_to_full(
            hmr2_out["pred_cam"], hmr2_batch["box_center"], hmr2_batch["box_size"],
            hmr2_batch["img_size"], dataset_focal
        ).detach().cpu().numpy()  # 3,

        # Joint angles: From predicted SMPL fit
        joint_angles_matrix = hmr2_out["pred_smpl_params"]["body_pose"][0].detach().cpu().numpy()  # J, 3, 3
        joint_angles = np.stack([
            Rotation.from_matrix(joint).as_rotvec() for joint in joint_angles_matrix
        ], axis=0)  # J, 3
        joint_angles_list.append(joint_angles)

        # Foreground: Object-to-camera transform
        o2c_fg = np.eye(4)  # 4, 4
        o2c_fg[:3, :3] = hmr2_out["pred_smpl_params"]["global_orient"][0, 0].cpu().numpy()  # 3, 3
        o2c_fg[:3, 3] = pred_cam_t_full  # 3,
        o2c_fg[:, (1, 2)] *= -1  # OpenGL => OpenCV camera coordinates (+x: right, -y: up, +z: forward)
        camlist_fg.append(o2c_fg)

        hmr_vertices.append(hmr2_out["pred_vertices"][0].detach().cpu().numpy())
        kp = hmr2_out["pred_keypoints_2d"][0].detach().cpu().numpy()*2 # -1,1
        kp = (kp * 128 + 128 - transform[:, 2]) / transform[0,0]
        kps_2d.append(kp) # 44,2

    camlist_fg = np.stack(camlist_fg, axis=0)  # N, 4, 4
    joint_angles_list = np.stack(joint_angles_list, axis=0)  # N, J, 3

    # Render SMPL using HMR2 renderer
    cam_view = Renderer(hmr2_model_cfg, faces=hmr2_model.smpl.faces).render_rgba_multiple(
        hmr_vertices,
        cam_t=camlist_fg[:, :3, 3],
        render_res=[img_W, img_H],
        mesh_base_color=(0.651, 0.741, 0.859),
        scene_bg_color=(1, 1, 1),
        focal_length=dataset_focal,
    )

    vis_imgs = []
    for it, impath in enumerate(tqdm.tqdm(imglist, desc=f"saving {seqname}")):
        imgidx = int(impath.split("/")[-1].split(".")[0])
        img_cv2 = cv2.imread(impath)  # H, W, C

        input_img = img_cv2.astype(np.float32)[:, :, ::-1] / 255.0
        input_img_overlay = np.where(
            cam_view[it][:, :, 3:], (cam_view[it][:, :, :3] + input_img[:, :, :3]) / 2, input_img[:, :, :3]
        )
        input_img_overlay *= 255
        # annotate kps
        input_img_overlay = annotate_keypoints(input_img_overlay, kps_2d[it])
        cv2.imwrite(f"{out_pose}/{imgidx:05}.jpg", input_img_overlay[:, :, ::-1])
        vis_imgs.append(input_img_overlay)

        # # Render predicted and template meshes
        # trimesh.Trimesh(
        #     vertices=hmr2_out["pred_vertices"][0].detach().cpu().numpy(),
        #     faces=hmr2_model.smpl.faces.astype(np.int64),
        # ).export("smpl_pred.obj")
        # trimesh.Trimesh(
        #     vertices=hmr2_model.smpl.v_template.detach().cpu().numpy(),
        #     faces=hmr2_model.smpl.faces.astype(np.int64),
        # ).export("smpl_template.obj")
        # trimesh.Trimesh(
        #     vertices=hmr2_model.smpl.v_template.detach().cpu().numpy() * np.array([1, -1, -1]),
        #     faces=hmr2_model.smpl.faces.astype(np.int64),
        # ).export("smpl_flip.obj")

    save_vid(f"{out_pose}/vis", vis_imgs)

    # Compute rolling average over 5 frames to make foreground cameras more robust
    camlist_fg_rot_padded = np.pad(camlist_fg[:, :3, :3], ((2, 2), (0, 0), (0, 0)), mode="edge")  # N+4, 3, 3
    camlist_fg[:, :3, :3] = np.stack([
        Rotation.from_matrix(camlist_fg_rot_padded[i : i + 5]).mean().as_matrix()
        for i in range(camlist_fg.shape[0])
    ], axis=0)

    camlist_fg_tra_padded = np.pad(camlist_fg[:, :3, 3], ((2, 2), (0, 0)), mode="edge")  # N+4, 3
    camlist_fg[:, :3, 3] = np.stack([
        camlist_fg_tra_padded[i : i + 5].mean(axis=0)
        for i in range(camlist_fg.shape[0])
    ], axis=0)

    # Compute rolling average over 5 frames to make joint angles more robust
    joint_angles_padded = np.pad(joint_angles_list, ((2, 2), (0, 0), (0, 0)), mode="edge")  # N+4, J, 3
    joint_angles_list = np.stack([
        np.stack([
            Rotation.from_rotvec(joint_angles_padded[i : i + 5, j]).mean().as_rotvec()
            for j in range(joint_angles_padded.shape[1])
        ], axis=0)
        for i in range(joint_angles_list.shape[0])
    ], axis=0)  # N, J, 3

    # Save predicted cameras again, to overwrite cameras from tsdf fusion
    np.save(f"{out_root}/01-canonical.npy", camlist_fg)
    np.save(f"{out_root}/hmr.npy", joint_angles_list)

    # Visualize cameras
    draw_cams(camlist_fg, rgbpath_list=imglist).export(f"{out_root}/cameras-01-hmr.obj")


if __name__ == "__main__":
    # vidname = "2024-05-11--01-06-03"
    # extract_joint_angles_hmr2(vidname)
    # # vidname = "2024-05-11--01-08-29"
    # # extract_joint_angles_hmr2(vidname)
    # vidname = "2024-05-11--01-10-54"
    # extract_joint_angles_hmr2(vidname)
    # extract_joint_angles_hmr2("2024-05-15--19-51-31")
    # extract_joint_angles_hmr2("2024-05-15--19-56-12")
    extract_joint_angles_hmr2("HearSay_Camera_test_e_4444_pga_64_g_512_al_0_down")