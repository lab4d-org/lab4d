# Copyright (c) 2023 Gengshan Yang, Alex Lyons, Carnegie Mellon University.
import os
import sys

import cv2
import numpy as np
import torch
import trimesh
from matplotlib.pyplot import cm

sys.path.insert(
    0,
    "%s/../../preprocess/third_party/vcnplus" % os.path.join(os.path.dirname(__file__)),
)

from flowutils.flowlib import flow_to_image, point_vec


def to_cpu(img):
    if type(img) is tuple:
        img = (to_cpu(img[0]), to_cpu(img[1]))
    else:
        if torch.is_tensor(img):
            img = img.cpu().numpy()
    return img

def img2color(tag, img, pca_fn=None):
    """Convert depth/flow/normal/feature/xyz/vis images to RGB

    Args:
        tag (str): Image type
        img: (H,W,x) Image, could be depth, flow, normal, feature, xyz, or vis
        pca_fn (Function): Function that applies PCA to output features
    Returns:
        img: (H,W,3) Image converted to RGB
    """
    img = to_cpu(img)

    if "depth" in tag:
        img = quantile_normalize(img)
        img = cm.get_cmap("plasma_r")(img[..., 0])[...,:3]

    if "flow" in tag:
        if type(img) is tuple:
            rgb, flow = img
            rgb = (rgb * 255).astype(np.uint8)
            flow = np.concatenate([flow, np.zeros_like(flow[..., 0:1])], -1)
            img = point_vec(rgb, flow, skip=10)[..., ::-1] / 255.0
        else:
            img = flow_to_image(img)

    if "normal" in tag:
        img = (img + 1) / 2

    if "feature" in tag:
        img = pca_fn(img, normalize=True)

    if "xyz" in tag:
        img = minmax_normalize(img, dim=(-2,-3))

    if "vis2d" in tag:
        img = minmax_normalize(img)

    if "xy_reproj" in tag:
        img = minmax_normalize(img)

    if "mask" in tag or "alpha" in tag or "vis2d" in tag:
        img = np.tile(img, (1,)*len(img.shape[:-1]) + (3,))
    return img


def mesh_cat(mesh_a, mesh_b):
    """Concatenate two meshes together

    Args:
        mesh_a (Trimesh): First mesh
        mesh_b (Trimesh): Second mesh
    Returns:
        mesh (Trimesh): Concatenated mesh
    """
    mesh = trimesh.util.concatenate(mesh_a, mesh_b)
    if hasattr(mesh_a.visual, "vertex_colors") and hasattr(
        mesh_b.visual, "vertex_colors"
    ):
        colors_a = mesh_a.visual.vertex_colors
        colors_b = mesh_b.visual.vertex_colors
        mesh.visual.vertex_colors = np.vstack((colors_a, colors_b))
    return mesh


def get_camera_mesh(radius, color=[255, 255, 255, 255]):
    vertices = (
        np.array(
            [
                [1.0, 1.0, 1],  # Vertex 0
                [-1.0, 1.0, 1],  # Vertex 1
                [-1.0, -1.0, 1],  # Vertex 2
                [1.0, -1.0, 1],  # Vertex 3
                [0.0, 0.0, 0.0],  # Vertex 4 (Focal point)
            ]
        )
        * radius
        * 2
    )

    faces = np.array(
        [
            [0, 1, 2],  # Triangle for half of the square base
            [0, 2, 3],  # Triangle for the other half of the square base
            [1, 0, 4],  # Triangle face connecting Vertex 0, 1 to Focal point
            [2, 1, 4],  # Triangle face connecting Vertex 1, 2 to Focal point
            [3, 2, 4],  # Triangle face connecting Vertex 2, 3 to Focal point
            [0, 3, 4],  # Triangle face connecting Vertex 3, 0 to Focal point
        ]
    )
    # vertices, faces = trimesh.remesh.subdivide(vertices, faces)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh.visual.vertex_colors = np.tile(color, [len(vertices), 1])
    return mesh


def get_user_mesh(scale=0.2, color = [255,0,0,255]):
    mesh = trimesh.load("database/mesh-templates/user.obj")
    mesh.vertices = mesh.vertices * scale
    mesh.vertices[:,1] = mesh.vertices[:,1] * -1
    mesh.visual.vertex_colors = 255
    mesh.visual.vertex_colors = np.tile(color, [len(mesh.vertices), 1])
    return mesh

def draw_cams(
    all_cam,
    color="cool",
    axis=True,
    color_list=None,
    rgbpath_list=None,
    radius_base=0.02,
):
    """Draw cameras as meshes

    Args:
        all_cam: (M,4,4) Camera SE(3) transforms to draw
        color (str): Matplotlib colormap to use
        axis (bool): If True, draw camera xyz axes
        color_list (np.array or None): List of colors to draw cameras with
        rgbpath_list: (List(str) or None): List of image path corresponding to the cameras
    Returns:
        mesh_cam (Trimesh): Mesh of cameras
    """
    # scale: the scene bound\
    if isinstance(color, str):
        cmap_raw = cm.get_cmap(color)
        cmap = lambda x: np.concatenate([cmap_raw(x)[:3], [x]])
    else:
        cmap = lambda x: color
    all_cam = np.asarray(all_cam)
    trans_norm = np.linalg.norm(all_cam[:, :3, 3], 2, -1)
    valid_cams = trans_norm > 0
    trans_max = np.median(trans_norm[valid_cams])
    if np.isnan(trans_max):
        trans_max = 0.1
    scale = trans_max
    traj_len = len(all_cam)
    cam_list = []
    if color_list is None:
        color_list = np.asarray(range(traj_len)) / float(traj_len)
    for j in range(traj_len):
        cam_rot = all_cam[j][:3, :3].T
        cam_tran = -cam_rot.dot(all_cam[j][:3, 3:])[:, 0]

        radius = radius_base * scale

        if axis:
            axis = trimesh_coarse_axis(
                origin_size=radius,
                origin_color=cmap(color_list[j]),
                axis_radius=radius * 0.1,
                axis_length=radius * 5,
            )
            cam = axis
        else:
            cam = get_camera_mesh(radius, cmap(color_list[j]))

        cam.vertices = cam.vertices.dot(cam_rot.T) + cam_tran
        cam_list.append(cam)

        if rgbpath_list is not None:
            image_mesh = image_to_mesh(rgbpath_list[j], radius)
            image_mesh.vertices = image_mesh.vertices.dot(cam_rot.T) + cam_tran
            cam_list.append(image_mesh)

    mesh_cam = trimesh.util.concatenate(cam_list)
    return mesh_cam


def trimesh_coarse_axis(
    origin_size=0.04,
    transform=None,
    origin_color=None,
    axis_radius=None,
    axis_length=None,
):
    """Return an XYZ axis marker as a Trimesh, which represents position and
    orientation. If you set the origin size the other parameters will be set
    relative to it. Adapted from https://github.com/mikedh/trimesh

    Args:
        transform (np.array or None): (4,4) Camera SE(3) transform
        origin_size (float): Radius of sphere that represents the origin
        origin_color: (3,) Color of the origin (float or int, uint8)
        axis_radius (float): Radius of cylinder that represents x, y, z axis
        axis_length (float): Length of cylinder that represents x, y, z axis
    Returns:
        marker (Trimesh): Mesh geometry of axis indicators
    """
    # the size of the ball representing the origin
    origin_size = float(origin_size)

    # set the transform and use origin-relative
    # sized for other parameters if not specified
    if transform is None:
        transform = np.eye(4)
    if origin_color is None:
        origin_color = [255, 255, 255, 255]
    if axis_radius is None:
        axis_radius = origin_size / 5.0
    if axis_length is None:
        axis_length = origin_size * 10.0

    # generate a ball for the origin
    axis_origin = trimesh.creation.uv_sphere(radius=origin_size, count=[4, 4])
    axis_origin.apply_transform(transform)

    # apply color to the origin ball
    axis_origin.visual.face_colors = origin_color

    # create the cylinder for the z-axis
    translation = trimesh.transformations.translation_matrix([0, 0, axis_length / 2])
    z_axis = trimesh.creation.cylinder(
        radius=axis_radius,
        height=axis_length,
        transform=transform.dot(translation),
        sections=3,
    )
    # XYZ->RGB, Z is blue
    z_axis.visual.face_colors = [0, 0, 255]

    # create the cylinder for the y-axis
    translation = trimesh.transformations.translation_matrix([0, 0, axis_length / 2])
    rotation = trimesh.transformations.rotation_matrix(np.radians(-90), [1, 0, 0])
    y_axis = trimesh.creation.cylinder(
        radius=axis_radius,
        height=axis_length,
        transform=transform.dot(rotation).dot(translation),
        sections=3,
    )
    # XYZ->RGB, Y is green
    y_axis.visual.face_colors = [0, 255, 0]

    # create the cylinder for the x-axis
    translation = trimesh.transformations.translation_matrix([0, 0, axis_length / 2])
    rotation = trimesh.transformations.rotation_matrix(np.radians(90), [0, 1, 0])
    x_axis = trimesh.creation.cylinder(
        radius=axis_radius,
        height=axis_length,
        transform=transform.dot(rotation).dot(translation),
        sections=3,
    )
    # XYZ->RGB, X is red
    x_axis.visual.face_colors = [255, 0, 0]

    # append the sphere and three cylinders
    marker = trimesh.util.concatenate([axis_origin, x_axis, y_axis, z_axis])
    return marker


def make_image_grid(img):
    """Reshape a batch of images into a grid of images

    Args:
        img (M,H,W,x): Batch of images
    Returns:
        collage (H_out, W_out, x): Image collage
    """
    bs, h, w, c = img.shape
    col = int(np.ceil(np.sqrt(bs)))
    row = col

    if not torch.is_tensor(img):
        img = torch.from_numpy(img)
    collage = torch.zeros(h * row, w * col, c, device=img.device)
    for i in range(row):
        for j in range(col):
            if i * col + j >= bs:
                break
            collage[i * h : (i + 1) * h, j * w : (j + 1) * w] = img[i * col + j]
    return collage


def minmax_normalize(data, dim=None):
    """Normalize a tensor or array within 0 to 1

    Args:
        data: (...,) Data to normalize
    Returns:
        normalized_data: (...,) Normalized data
    """
    if dim is None:
        normalized_data = (data - data.min()) / (data.max() - data.min() + 1e-6)
    else:
        normalized_data = (data - data.min(dim, keepdims=True)) / (
            data.max(dim, keepdims=True) - data.min(dim, keepdims=True) + 1e-6
        )
    return normalized_data


def quantile_normalize(data, dim=None, quantile=0.95):
    """Normalize a tensor or array within 0 to 1

    Args:
        data: (...,) Data to normalize
    Returns:
        normalized_data: (...,) Normalized data
    """
    if dim is None:
        mind = np.quantile(data, 1-quantile)
        maxd = np.quantile(data, quantile)
    else:
        mind = np.quantile(data, 1-quantile, dim, keepdims=True)
        maxd = np.quantile(data, quantile, dim, keepdims=True)
    normalized_data = (data - mind) / (maxd - mind+ 1e-6)
    normalized_data = np.clip(normalized_data, 0, 1)
    return normalized_data

def get_colormap(num_colors=-1, repeat=1):
    """Colormap for visualizing bones

    Args:
        num_colors (int): Number of colors to return
        repeat (int): Number of times to tile the colormap
    Returns:
        colors (np.array): (N,3) Bone colors
    """
    colors = np.asarray(
        [
            [155, 122, 157],
            [45, 245, 50],
            [71, 25, 64],
            [231, 176, 35],
            [125, 249, 245],
            [32, 75, 253],
            [241, 31, 111],
            [218, 71, 252],
            [248, 220, 197],
            [34, 194, 198],
            [108, 178, 96],
            [33, 101, 119],
            [125, 100, 26],
            [209, 235, 102],
            [116, 105, 241],
            [100, 50, 147],
            [193, 159, 222],
            [95, 254, 138],
            [197, 130, 75],
            [144, 31, 211],
            [46, 150, 26],
            [242, 90, 174],
            [179, 41, 38],
            [118, 204, 174],
            [145, 209, 38],
            [188, 74, 125],
            [95, 158, 210],
            [237, 152, 130],
            [53, 151, 157],
            [69, 86, 193],
            [60, 204, 122],
            [251, 77, 58],
            [174, 248, 170],
            [28, 81, 36],
            [252, 134, 243],
            [62, 254, 193],
            [68, 209, 254],
            [44, 25, 184],
            [131, 58, 80],
            [188, 251, 27],
            [156, 25, 132],
            [248, 36, 225],
            [95, 130, 63],
            [222, 204, 244],
            [185, 186, 134],
            [160, 146, 44],
            [244, 196, 89],
            [39, 60, 87],
            [134, 239, 87],
            [25, 166, 97],
            [79, 36, 229],
            [45, 130, 216],
            [177, 90, 200],
            [86, 218, 30],
            [97, 115, 165],
            [159, 104, 99],
            [168, 220, 219],
            [134, 76, 180],
            [31, 238, 157],
            [79, 140, 253],
            [124, 23, 27],
            [245, 234, 46],
            [188, 30, 174],
            [253, 246, 148],
            [228, 94, 92],
        ]
    )
    if num_colors > len(colors):
        raise ValueError("num_colors must be less than {}".format(len(colors)))
    if num_colors > 0:
        colors = colors[:num_colors]
    if repeat > 1:
        colors = np.tile(colors[:, None], (1, repeat, 1))
        colors = np.reshape(colors, (-1, 3))
    return colors


def image_to_mesh(image_path, z_displacement=0.04, mesh_scale=0.0005, mesh_res=5e3):
    """Convert image to mesh

    Args:
        img_path (str): Path to image
        z_displacement (float): Z displacement of mesh
        mesh_scale (float): Scale of mesh
        max_res (int): Maximum resolution of mesh
    Returns:
        mesh (Trimesh): Mesh
    """
    image = cv2.imread(image_path)[:, :, ::-1]
    inp_h, inp_w, _ = image.shape
    res_fac = np.sqrt(mesh_res / (inp_h * inp_w))
    height = int(np.ceil(inp_h * res_fac))
    width = int(np.ceil(inp_w * res_fac))
    image = cv2.resize(image, (width, height))

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

    points = points.astype(float)
    points[:, 0:2] -= np.mean(points[:, 0:2], axis=0, keepdims=True)
    points[:, 0:2] *= mesh_scale
    points[:, 2] = -z_displacement * np.ones(numPixels)

    pixels = np.stack(
        (np.tile(np.arange(width), height), np.repeat(np.arange(height), width))
    ).T
    pixels = pixels.astype(int)
    colors = np.repeat(image[pixels[:, 1], pixels[:, 0]], 2, axis=0) / 255

    mesh = trimesh.Trimesh(vertices=points, faces=faces, face_colors=colors)
    return mesh


def create_plane(size, offset):
    """
    Create a plane mesh spaning x,z axis
    """
    vertices = np.array(
        [
            [-0.5, 0, -0.5],  # vertex 0
            [0.5, 0, -0.5],  # vertex 1
            [0.5, 0, 0.5],  # vertex 2
            [-0.5, 0, 0.5],  # vertex 3
        ]
    )
    vertices = vertices * size + np.asarray(offset)

    faces = np.array(
        [
            [0, 2, 1],  # triangle 0
            [2, 0, 3],  # triangle 1
        ]
    )
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    return mesh


def create_floor_mesh(scale=20, gl=True):
    # create scene
    floor1 = create_plane(scale, [0, 0, 0])
    floor1.visual.vertex_colors[:, 0] = 10
    floor1.visual.vertex_colors[:, 1] = 255
    floor1.visual.vertex_colors[:, 2] = 102
    floor1.visual.vertex_colors[:, 3] = 102

    floor2 = create_plane(scale / 4, [0, scale * 0.001, 0])
    floor2.visual.vertex_colors[:, 0] = 10
    floor2.visual.vertex_colors[:, 1] = 102
    floor2.visual.vertex_colors[:, 2] = 255
    floor2.visual.vertex_colors[:, 3] = 102

    floor = trimesh.util.concatenate([floor1, floor2])
    if not gl:
        floor.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]))
    return floor


# visualze all meshes
def visualize_trajectory(all_verts, tag):
    meshes_concat = []
    for it, verts_pred in enumerate(all_verts[::30]):
        verts_pred = verts_pred.cpu().numpy()
        verts_pred[:, 0] += 1 * it
        meshes_concat.append([trimesh.Trimesh(vertices=verts_pred)])
    trimesh.util.concatenate(meshes_concat).export("tmp/%s.obj" % tag)


def append_xz_plane(mesh, world_to_cam, scale=5, gl=True):
    mesh.visual.vertex_colors = mesh.visual.vertex_colors  # visual.kind = 'vertex'
    # scale = np.abs(mesh.vertices).max() * 2
    plane = create_floor_mesh(scale=scale, gl=gl)
    plane.apply_transform(world_to_cam)
    return trimesh.util.concatenate([mesh, plane])


def get_pts_traj(kps, frame_idx, traj_len=100, cmap=cm.get_cmap("cool")):
    """
    Args:
        kps: (T, K, 3)

    Returns:
        pts_traj: (T', K, 2, 3)
        pts_color: (T', K, 2, 4)
    """
    _, traj_num, _ = kps.shape
    pts_traj = np.zeros((traj_len, traj_num, 2, 3))
    pts_color = np.zeros((traj_len, traj_num, 2, 4))
    i = frame_idx
    for j in range(traj_len):
        if i - j - 1 < 0:
            continue
        pts_traj[j, :, 0] = kps[i - j - 1]
        pts_traj[j, :, 1] = kps[i - j]
        pts_color[j, :, 0] = cmap(float(i - j - 1) / traj_len)
        pts_color[j, :, 1] = cmap(float(i - j) / traj_len)
    return (pts_traj, pts_color)
