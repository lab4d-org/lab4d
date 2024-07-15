import pdb
import logging
import os
import time

import numpy as np
import torch
from PIL import Image

from tsr.system import TSR
from tsr.utils import save_video

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)

import sys
path = sys.argv[1]
image_pathes = [path]

output_dir = "tmp/TripoSR_output/"
os.makedirs(output_dir, exist_ok=True)

device = "cuda"

model = TSR.from_pretrained(
    "stabilityai/TripoSR",
    config_name="config.yaml",
    weight_name="model.ckpt",
)
model.renderer.set_chunk_size(8192)
model.to(device)

images = []

def add_margin(pil_img, top, right, bottom, left, color):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result

for i, image_path in enumerate(image_pathes):
    image = np.array(Image.open(image_path).convert("RGB")).astype(float)
    mask_path = image_path.replace("JPEGImages", "Annotations").replace(".jpg", ".npy")
    mask = (np.load(mask_path, allow_pickle=True)[:, :, None] > 0).astype(float)
    image = (image * mask + (1 - mask) * 128).clip(0,255)
    image = Image.fromarray(image.astype(np.uint8))

    # # center crop
    # w, h = image.size
    # if w > h:
    #     image = image.crop(((w - h) // 2, 0, (w + h) // 2, h))
    # elif h > w:
    #     image = image.crop(((h - w) // 2, 0, (h + w) // 2, w))
    
    w, h = image.size
    # pad to square to both sides
    if w > h:
        image = add_margin(image, (w - h) // 2, 0, (w - h) // 2, 0, (128, 128, 128))
    elif h > w:
        image = add_margin(image, 0, (h - w) // 2, 0, (h - w) // 2, (128, 128, 128))

    if not os.path.exists(os.path.join(output_dir, str(i))):
        os.makedirs(os.path.join(output_dir, str(i)))
    image.save(os.path.join(output_dir, str(i), f"input.png"))
    images.append(image)

for i, image in enumerate(images):
    logging.info(f"Running image {i + 1}/{len(images)} ...")

    with torch.no_grad():
        scene_codes = model([image], device=device)

    render_images = model.render(scene_codes, n_views=5, return_type="pil")
    for ri, render_image in enumerate(render_images[0]):
        render_image.save(os.path.join(output_dir, str(i), f"render_{ri:03d}.png"))
    save_video(
        render_images[0], os.path.join(output_dir, str(i), f"render.mp4"), fps=30
    )

    meshes = model.extract_mesh(scene_codes, True, resolution=128)

    out_mesh_path = os.path.join(output_dir, str(i), f"mesh.obj")
    meshes[0].export(out_mesh_path)
