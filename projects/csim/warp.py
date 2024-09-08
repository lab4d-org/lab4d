import cv2
import numpy as np


def warp_homography(img, R, intrinsics, boarder_mode=cv2.BORDER_CONSTANT):
    """
    Warp an image using a homography derived from a rotation and intrinsics.

    Parameters:
    - img: The input image
    - R: Rotation matrix (3x3)
    - intrinsics: Camera intrinsics matrix (3x3)

    Returns:
    - warped_img: The warped image
    """
    # Construct the homography H = K * [R|t]
    # Since we are only using rotation here, the translation t is assumed to be zero.
    H = intrinsics @ R @ np.linalg.inv(intrinsics)

    # Get the image dimensions
    h, w = img.shape[:2]

    # Perform the warp
    warped_img = cv2.warpPerspective(img, H, (w, h), borderMode=boarder_mode)

    return warped_img
