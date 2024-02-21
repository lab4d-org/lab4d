import math
import cv2
import os, sys
import pdb
import matplotlib.pyplot as plt
import numpy as np
import torch
from celluloid import Camera
from tqdm.auto import tqdm
import trimesh

import ddpm
from arch import TemporalUnet, TimestepEmbedder
from ddpm import get_data, get_lab4d_data, BGField

sys.path.insert(0, os.getcwd())
from projects.csim.voxelize import VoxelGrid, readout_voxel_fn


### forward and reverse process animation
num_timesteps = 50
noise_scheduler = ddpm.NoiseScheduler(num_timesteps=num_timesteps).cuda()

# forward
# x0, y = get_data()
x0, past, cam, x0_to_world, root_world_all = get_lab4d_data()
x0 = x0.cuda()
past = past.cuda()
cam = cam.cuda()
x0_to_world = x0_to_world.cuda()
sample_idx = 200
x0_sampled = x0[sample_idx : sample_idx + 1]
past_sampled = past[sample_idx : sample_idx + 1]
cam_sampled = cam[sample_idx : sample_idx + 1]
x0_to_world_sampled = x0_to_world[sample_idx : sample_idx + 1]

# scene data
bg_field = BGField()
occupancy = bg_field.voxel_grid.data[None]
occupancy = torch.tensor(occupancy, dtype=torch.float32, device="cuda")

forward_samples = []
forward_samples.append(x0.cpu().numpy())
for t in range(len(noise_scheduler)):
    timesteps = torch.tensor(np.repeat(t, len(x0)), dtype=torch.long, device="cuda")
    noise = torch.randn_like(x0, device="cuda")
    sample = noise_scheduler.add_noise(x0, noise, timesteps)
    forward_samples.append(sample.cpu().numpy())

# reverse
model = ddpm.MLP()
path = "projects/tiny-diffusion/exps/base/model.pth"
model.load_state_dict(torch.load(path), strict=True)
model = model.cuda()
model.eval()

# plot the gradiant field
xmin, ymin, zmin = x0.min(0)[0].cpu().numpy()
xmax, ymax, zmax = x0.max(0)[0].cpu().numpy()
xzmin = min(xmin, zmin) - 1
xzmax = max(xmax, zmax) + 1
x = np.linspace(xzmin, xzmax, 30)
z = np.linspace(xzmin, xzmax, 30)
y = np.linspace(ymin, ymax, 5)
xyz = np.stack(np.meshgrid(x, y, z), axis=-1).reshape(-1, 3)
xyz_cuda = torch.tensor(xyz, dtype=torch.float32, device="cuda")

eval_batch_size = 1024
sample = torch.randn(eval_batch_size, 3, device="cuda")
timesteps = list(range(num_timesteps))[::-1]
reverse_samples = []
reverse_grad = []
reverse_samples.append(sample.cpu().numpy())
for i, t in enumerate(tqdm(timesteps)):
    t_grid = torch.tensor(np.repeat(t, len(xyz)), dtype=torch.long, device="cuda")
    t = torch.tensor(np.repeat(t, len(sample)), dtype=torch.long, device="cuda")

    past_grid = past_sampled.repeat(len(xyz), 1)
    past = past_sampled.repeat(sample.shape[0], 1)

    cam_grid = cam_sampled.repeat(len(xyz), 1)
    cam = cam_sampled.repeat(sample.shape[0], 1)

    x0_to_world_grid = x0_to_world_sampled.repeat(len(xyz), 1)
    x0_to_world = x0_to_world_sampled.repeat(sample.shape[0], 1)

    with torch.no_grad():
        # 3D convs then query, # B1HWD => B3HWD
        feat = model.extract_env(
            occupancy,
            sample + x0_to_world,
            bg_field.voxel_grid.res,
            bg_field.voxel_grid.origin,
        )
        feat_grid = model.extract_env(
            occupancy,
            xyz_cuda + x0_to_world_grid,
            bg_field.voxel_grid.res,
            bg_field.voxel_grid.origin,
        )

        # feat = bg_field.compute_feat(sample + x0_to_world)
        # feat_grid = bg_field.compute_feat(xyz_cuda + x0_to_world_grid)
        grad = model(
            sample,
            t[:, None] / num_timesteps,
            past,
            cam,
            feat,
            drop_cam=False,
            drop_past=False,
        )
        xy_grad = model(
            xyz_cuda,
            t_grid[:, None] / num_timesteps,
            past_grid,
            cam_grid,
            feat_grid,
            drop_cam=False,
            drop_past=False,
        )
        # xy_grad = voxel_grid.readout_voxel(
        #     (xyz_cuda + x0_to_world_grid).cpu().numpy(), mode="root_visitation_gradient"
        # )
        # xy_grad = torch.tensor(xy_grad, dtype=torch.float32, device="cuda")
        # xy_grad = -10 * xy_grad
    sample = noise_scheduler.step(grad, t[0], sample)
    reverse_samples.append(sample.cpu().numpy())
    reverse_grad.append(xy_grad.cpu().numpy())

# mesh rendering
from lab4d.utils.pyrender_wrapper import PyRenderWrapper
from lab4d.utils.io import save_vid

raw_size = [512, 512]
renderer = PyRenderWrapper(raw_size)
renderer.set_camera_bev(depth=(xzmax - xzmin) * 2)
# set camera intrinsics
fl = max(raw_size)
intr = np.asarray([fl * 2, fl * 2, raw_size[1] / 2, raw_size[0] / 2])
renderer.set_intrinsics(intr)
renderer.align_light_to_camera()


def put_text(img, text, pos, color=(255, 0, 0)):
    img = img.astype(np.uint8)
    img = cv2.putText(
        img,
        text,
        pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
    )
    return img


# rotate the sample
def render_rotate_sample(shape, n_frames=10):
    sample_raw = shape.vertices
    frames = []
    for i in range(n_frames):
        progress = -0.25 + 0.5 * i / n_frames
        rot = cv2.Rodrigues(np.array([progress * math.pi, 0, 0]))[0]
        sample = np.dot(sample_raw, rot.T)
        shape = trimesh.PointCloud(sample, colors=shape.visual.vertex_colors)
        input_dict = {"shape": shape}
        color = renderer.render(input_dict)[0]
        frames.append(color)
    return frames


# forward
frames = []
colors = np.ones((len(forward_samples[0]), 4))
colors[:, 1:3] = 0
shape_clean = trimesh.PointCloud(forward_samples[0], colors=colors)
frames += render_rotate_sample(shape_clean)

for i, sample in enumerate(forward_samples):
    shape = trimesh.PointCloud(sample, colors=colors)
    input_dict = {"shape": shape}
    color = renderer.render(input_dict)[0]
    color = put_text(color, f"step {i: 4} / {num_timesteps}", (10, 30))
    color = put_text(color, "Forward process", (10, 60))
    frames.append(color)

# reverse
colors = np.ones((len(reverse_samples[0]), 4))
colors[:, 0:2] = 0

# get past location as a sphere
past_shape = trimesh.creation.uv_sphere(radius=0.03)
past_shape = past_shape.apply_translation(past[0].cpu().numpy())
past_colors = np.array([[0, 1, 0, 1]]).repeat(len(past_shape.vertices), axis=0)

# get bg mesh
bg_pts = bg_field.bg_mesh.vertices - x0_to_world[0].cpu().numpy()
# bg_pts = (
#     voxel_grid.to_boxes(mode="root_visitation").vertices - x0_to_world[0].cpu().numpy()
# )
bg_colors = np.ones((len(bg_pts), 4)) * 0.6


for i, sample in enumerate(reverse_samples):
    # sample = np.concatenate([sample, past_shape.vertices], axis=0)
    # sample_colors = np.concatenate([colors, past_colors], axis=0)

    sample = np.concatenate([sample, past_shape.vertices, bg_pts], axis=0)
    sample_colors = np.concatenate([colors, past_colors, bg_colors], axis=0)

    shape = trimesh.PointCloud(sample, colors=sample_colors)
    input_dict = {"shape": shape}
    color = renderer.render(input_dict)[0]
    color = put_text(
        color, f"step {i: 4} / {num_timesteps}", (10, 30), color=(0, 0, 255)
    )
    color = put_text(color, "Reverse process", (10, 60), color=(0, 0, 255))
    frames.append(color)


def concatenate_points(pointcloud1, pointcloud2):
    pts = np.concatenate([pointcloud1.vertices, pointcloud2.vertices], axis=0)
    colors = np.concatenate(
        [pointcloud1.visual.vertex_colors, pointcloud2.visual.vertex_colors], axis=0
    )
    return trimesh.PointCloud(pts, colors=colors)


# concat two shapes
shape = concatenate_points(shape_clean, shape)
frames += render_rotate_sample(shape)

shape.export("tmp/0.obj")
save_vid("projects/tiny-diffusion/static/rendering", frames)
print("Animation saved to projects/tiny-diffusion/static/rendering.mp4")
print("Mesh saved to tmp/0.obj")


# 2D plot
fig, ax = plt.subplots()
camera = Camera(fig)

# forward
for i, sample in enumerate(forward_samples):
    plt.scatter(sample[:, 0], sample[:, 2], alpha=0.5, s=15, color="blue")
    plt.scatter(
        sample[sample_idx, 0], sample[sample_idx, 2], alpha=0.5, s=30, color="red"
    )
    ax.text(0.0, 0.95, f"step {i: 4} / {num_timesteps}", transform=ax.transAxes)
    ax.text(0.0, 1.01, "Forward process", transform=ax.transAxes, size=15)
    plt.xlim(xzmin, xzmax)
    plt.ylim(xzmin, xzmax)
    plt.axis("off")
    camera.snap()

# reverse
for i, sample in enumerate(reverse_samples):
    plt.scatter(sample[:, 0], sample[:, 2], alpha=0.5, s=15, color="blue")
    plt.scatter(past[0, 0].cpu(), past[0, 2].cpu(), s=100, color="green", marker="x")
    plt.scatter(cam[0, 0].cpu(), cam[0, 2].cpu(), s=100, color="red", marker="o")
    plt.scatter(
        x0_sampled[0, 0].cpu(), x0_sampled[0, 2].cpu(), s=100, color="black", marker="x"
    )
    if i < len(reverse_samples) - 1:
        grad = reverse_grad[i]
        grad = grad.reshape(y.shape + (-1, 3)).mean(0)  # aver over height
        xyz_sliced = xyz.reshape(y.shape + (-1, 3))[0]  # find a slice of grid
        plt.quiver(
            xyz_sliced[:, 0],
            xyz_sliced[:, 2],
            -grad[:, 0],
            -grad[:, 2],
            angles="xy",
            scale_units="xy",
            scale=10,
            color="red",
        )
    ax.text(0.0, 0.95, f"step {i: 4} / {num_timesteps}", transform=ax.transAxes)
    ax.text(0.0, 1.01, "Reverse process", transform=ax.transAxes, size=15)
    plt.xlim(xzmin, xzmax)
    plt.ylim(xzmin, xzmax)
    plt.axis("off")
    camera.snap()

animation = camera.animate(blit=True, interval=35)
animation.save("projects/tiny-diffusion/static/animation.mp4")
print("Animation saved to projects/tiny-diffusion/static/animation.mp4")
