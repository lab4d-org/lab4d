import numpy as np
import cv2
import os
import glob


def create_fake_masks(seqname, outdir):
    anno_dir = f"{outdir}/Annotations/Full-Resolution/{seqname}"
    os.makedirs(anno_dir, exist_ok=True)
    ref_list = sorted(glob.glob(f"{outdir}/JPEGImages/Full-Resolution/{seqname}/*"))
    shape = cv2.imread(ref_list[0]).shape[:2]
    mask = -1 * np.ones(shape).astype(np.int8)
    for ref in ref_list:
        img_ext = ref.split("/")[-1].split(".")[0]
        save_path = "%s/%s.npy" % (anno_dir, img_ext)
        np.save(save_path, mask)
