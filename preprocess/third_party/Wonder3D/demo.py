import pdb
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
cond = np.load("database/processed/JPEGImages/Full-Resolution/cat-pikachu-0-0000/crop-256.npy")
mask = np.load("database/processed/Annotations/Full-Resolution/cat-pikachu-0-0000/crop-256.npy")

cond = cond * mask[...,:1]
cond = (cond[0]*255).astype(np.uint8)

# The object should be located in the center and resized to 80% of image height.
cond = Image.fromarray(cond)

# Run the pipeline!
images = pipeline(cond, num_inference_steps=20, output_type='pt', guidance_scale=1.0).images

result = make_grid(images, nrow=6, ncol=2, padding=0, value_range=(0, 1))

save_image(result, 'tmp/result.png')
print("Saved to tmp/result.png")