import pdb
import os
import torch
import cv2
import requests
from PIL import Image
import numpy as np
from torchvision.utils import make_grid, save_image
from diffusers import DiffusionPipeline  # only tested on diffusers[torch]==0.19.3, may have conflicts with newer versions of diffusers

def load_wonder3d_pipeline():

    pipeline = DiffusionPipeline.from_pretrained(
    'flamehaze1115/wonder3d-v1.0', # or use local checkpoint './ckpts'
    custom_pipeline='flamehaze1115/wonder3d-pipeline',
    torch_dtype=torch.float16
    )

    # enable xformers
    pipeline.unet.enable_xformers_memory_efficient_attention()

    if torch.cuda.is_available():
        pipeline.to('cuda:0')
    return pipeline

pipeline = load_wonder3d_pipeline()

# Download an example image.
# cond = Image.open(requests.get("https://d.skis.ltd/nrp/sample-data/lysol.png", stream=True).raw)
seqname = "cat-pikachu-0-0000"
image_dir = "database/processed/JPEGImages/Full-Resolution/%s/" % seqname
mask_dir = image_dir.replace("JPEGImages", "Annotations")
output_dir = image_dir.replace("JPEGImages", "Multiview")
os.makedirs(output_dir, exist_ok=True)

cond = np.load("%s/crop-256.npy"%image_dir)
mask = np.load("%s/crop-256.npy"%mask_dir)
cond = cond * mask[...,:1]

for i in range(len(cond)):
    cond_sub = (cond[i]*255).astype(np.uint8)

    # The object should be located in the center and resized to 80% of image height.
    cond_sub = Image.fromarray(cond_sub)

    # Run the pipeline!
    pdb.set_trace()
    images = pipeline(cond_sub, num_inference_steps=20, output_type='pt', guidance_scale=1.0).images
    images[images < 0.05] = 0
    images[images >= 0.05] = 1

    result = make_grid(images, nrow=6, ncol=2, padding=0, value_range=(0, 1))

    save_image(result, 'tmp/result_%04d.png'%i)
    print("Saved to tmp/result_%04d.png"%i)