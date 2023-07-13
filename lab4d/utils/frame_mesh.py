import trimesh
import numpy as np
from PIL import Image


def gen_frame_mesh(image_path, z_displacement=0.04):
    image = Image.open(image_path)
    width, height = image.size

    image_res = 0.025
    image = image.resize((int(width * image_res), int(height * image_res)))
    width, height = image.size

    numPixels = (width + 1) * (height + 1)

    points = np.stack(
        (
            np.tile(np.arange(width + 1), height + 1),
            np.repeat(np.arange(height + 1), width + 1),
            np.zeros(numPixels),
        )
    ).T
    points = points.astype(int)

    indices = np.setdiff1d(
        np.arange((width + 1) * height),
        np.arange(width, (width + 1) * height, width + 1),
    )
    # upper left triangle
    ult = np.stack((indices + width + 1, indices + 1, indices))
    # lower right triangle
    lrt = np.stack((indices + width + 1, indices + width + 2, indices + 1))

    faces = np.zeros((len(indices) * 2, 3)).astype(int)
    faces[0::2] = ult.T
    faces[1::2] = lrt.T

    mesh_scale = 0.005
    points = points.astype(float)
    points[:, 0:2] -= np.mean(points[:, 0:2], axis=0, keepdims=True)
    points[:, 0:2] *= mesh_scale
    points[:, 2] = -z_displacement * np.ones(numPixels)
    mesh = trimesh.Trimesh(vertices=points, faces=faces)

    imgArr = np.asarray(image)

    pixels = np.stack(
        (np.tile(np.arange(width), height), np.repeat(np.arange(height), width))
    ).T
    pixels = pixels.astype(int)

    colors = np.repeat(imgArr[pixels[:, 1], pixels[:, 0]], 2, axis=0) / 255

    mesh.visual.face_colors = colors

    return mesh
