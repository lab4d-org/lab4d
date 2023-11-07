import cv2
import numpy as np
from PIL import Image
import torch
from diffusers import StableDiffusionInpaintPipeline

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting",
    torch_dtype=torch.float16,
)
pipe.to("cuda")
# image and mask_image should be PIL images.
# The mask structure is white for inpainting and black for keeping as is
image = Image.open(
    "/home/gengshay/code/lab4d/projects/csim/zero123_data/home_panorama/Oct5at10-49AM-poly-227/011.jpg"
)
mask_image = (np.sum(np.abs(np.asarray(image)), -1) == 1).astype(np.uint8)
image = pipe(prompt="", image=image, mask_image=[mask_image]).images[0]
image.save("./yellow_cat_on_park_bench.png")
