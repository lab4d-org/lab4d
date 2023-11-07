import PIL
import requests
import torch
from io import BytesIO
import numpy as np
import pdb
import cv2
import numpy as np
import os, sys

from diffusers import StableDiffusionInpaintPipeline
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image


sys.path.insert(0, os.getcwd())
from projects.csim.warp import warp_homography


sys.path.insert(0, os.getcwd() + "/../Perspective-and-Equirectangular/")
import lib.multi_Perspec2Equirec as m_P2E


def repaint2(image, mask_image):
    pipe = AutoPipelineForInpainting.from_pretrained(
        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        torch_dtype=torch.float16,
        variant="fp16",
    ).to("cuda")

    # mask_image = mask_image * 255

    # img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
    # mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"
    # image = load_image(img_url).resize((1024, 1024))
    # mask_image = load_image(mask_url).resize((1024, 1024))
    # mask_image = np.zeros_like(np.asarray(mask_image))
    # mask_image = PIL.Image.fromarray(mask_image)

    # prompt = "a tiger sitting on a park bench"
    prompt = ""
    generator = torch.Generator(device="cuda").manual_seed(0)
    image = pipe(
        prompt=prompt,
        image=image,
        mask_image=mask_image,
        guidance_scale=8.0,
        num_inference_steps=20,  # steps between 15 and 30 work well for us
        strength=0.99,  # make sure to use `strength` below 1.0
        generator=generator,
    ).images[0]
    image = np.array(image)
    return image


def repaint1(init_image, mask_image):
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16
    )
    pipe = pipe.to("cuda")
    prompt = ""
    image = pipe(prompt=prompt, image=init_image, mask_image=mask_image).images[0]
    image = np.array(image)
    return image


def get_mask_holes(init_image):
    mask_image = np.sum(np.abs(255 - np.asarray(init_image)), -1) == 0
    # dilate
    mask_image = cv2.dilate(mask_image.astype(np.uint8), np.ones((10, 10), np.uint8))
    mask_image = PIL.Image.fromarray(mask_image * 255).resize((512, 512))
    return mask_image


img_path = "database/processed/JPEGImages/Full-Resolution/cat-pikachu-0000/00049.jpg"

img = cv2.imread(img_path)

# # Define a rotation matrix (example: rotate around the Z-axis)
# R = cv2.Rodrigues(np.array([0, -np.pi / 6, 0]))[0]
# # Define the camera intrinsics (example values)
# intrinsics = np.array(
#     [[1920, 0, img.shape[1] / 2], [0, 1920, img.shape[0] / 2], [0, 0, 1]]
# )
# # Warp the image using the homography
# warped_img = warp_homography(img, R, intrinsics)
# mask_image = 1 - warp_homography(np.ones_like(img[..., 0]), R, intrinsics)


equ = m_P2E.Perspective([img_path], [[90, 0, 0]])
warped_img = equ.GetEquirec(img.shape[0], img.shape[0] * 4)
mask_image = (np.sum(warped_img, -1) == 0).astype(np.uint8)

# warped_img = img

# fill in noise
noise = np.random.rand(*warped_img.shape) * 255
warped_img[mask_image > 0] = noise[mask_image > 0]

# resize
init_image = warped_img[..., ::-1] / 255.0
size = init_image.shape[:2][::-1]
init_image = cv2.resize(init_image, (256, 256))
mask_image = cv2.resize(mask_image, (256, 256))

# inpaint
image1 = repaint1(init_image, mask_image)
image = repaint2(image1 / 255.0, mask_image)

image = cv2.resize(image, size)
image1 = cv2.resize(image1, size)
mask_image = cv2.resize(mask_image, size)

# Save or display the warped image
cv2.imwrite("tmp/test1.png", warped_img)
cv2.imwrite("tmp/test2.png", image1[..., ::-1])
cv2.imwrite("tmp/test3.png", image[..., ::-1])
cv2.imwrite("tmp/test4.png", mask_image * 255)
