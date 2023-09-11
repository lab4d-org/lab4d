# Copyright (c) 2023 Gengshan Yang, Carnegie Mellon University.
# python preprocess/third_party/omnivision/normal.py cat-pikachu-0-0000
import pdb
import glob
import os
import sys

import numpy as np
import torch
import cv2

sys.path.insert(
    0,
    "%s/../../" % os.path.join(os.path.dirname(__file__)),
)

from libs.utils import resize_to_target

sys.path.insert(
    0,
    "%s/" % os.path.join(os.path.dirname(__file__)),
)
from modules.midas.dpt_depth import DPTDepthModel


def load_model():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    pretrained_weights_path = (
        "./preprocess/third_party/omnivision/omnidata_dpt_normal_v2_cleaned.ckpt"
    )
    model = DPTDepthModel(backbone="vitb_rn50_384", num_channels=3)  # DPT Hybrid
    checkpoint = torch.load(pretrained_weights_path)
    if "state_dict" in checkpoint:
        state_dict = {}
        for k, v in checkpoint["state_dict"].items():
            state_dict[k[6:]] = v
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(device)
    return model


@torch.no_grad()
def predict_normal(model, img):
    # resize
    testres = np.sqrt(2e5 / (img.shape[0] * img.shape[1]))
    maxh = img.shape[0] * testres
    maxw = img.shape[1] * testres
    max_h = int(maxh // 64 * 64)
    max_w = int(maxw // 64 * 64)
    if max_h < maxh:
        max_h += 64
    if max_w < maxw:
        max_w += 64

    input_size = img.shape
    img = cv2.resize(img, (max_w, max_h))
    img = np.transpose(img, [2, 0, 1])[None]
    img_tensor = torch.Tensor(img / 255.0).cuda()
    if img_tensor.shape[1] == 1:
        img_tensor = img_tensor.repeat_interleave(3, 1)

    output = model(img_tensor).clamp(min=0, max=1)
    normal = output[0].permute(1, 2, 0).cpu().numpy()
    normal = cv2.resize(
        normal, (input_size[1], input_size[0]), interpolation=cv2.INTER_LINEAR
    )

    return normal


def extract_normal(seqname):
    image_dir = "database/processed/JPEGImages/Full-Resolution/%s/" % seqname
    output_dir = image_dir.replace("JPEGImages", "Normal")
    os.makedirs(output_dir, exist_ok=True)

    model = load_model()

    for img_path in sorted(glob.glob(f"{image_dir}/*.jpg")):
        # print(img_path)
        img = cv2.imread(img_path)[..., ::-1]
        normal = predict_normal(model, img)
        normal = resize_to_target(normal, is_flow=False)
        normal = 2 * normal - 1  # [-1, 1]
        normal = normal / (1e-6 + np.linalg.norm(normal, 2, -1)[..., None])

        out_path = f"{output_dir}/{os.path.basename(img_path).replace('.jpg', '.npy')}"
        np.save(out_path, normal.astype(np.float16))
        vis_path = f"{output_dir}/vis-{os.path.basename(img_path)}"
        normal = (normal + 1) / 2
        cv2.imwrite(vis_path, normal[..., ::-1] * 255)

    print("surface normal saved to %s" % output_dir)


if __name__ == "__main__":
    seqname = sys.argv[1]

    extract_normal(seqname)
