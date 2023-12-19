import sys, os
import pdb
import numpy as np
import trimesh
import glob
import time
import viser
import viser.transforms as tf
import cv2

sys.path.insert(0, os.getcwd())
from lab4d.utils.mesh_loader import MeshLoader
from lab4d.utils.vis_utils import draw_cams
from lab4d.utils.io import save_vid


def count_points_in_voxels(pts, grid_size, voxel_size):
    # Create an empty array to count points in each voxel
    voxel_counts = np.zeros(grid_size)

    # Iterate over each point
    for point in pts:
        # Calculate the voxel index for each point
        voxel_index = (point / voxel_size).astype(int)

        # Ensure the point is within the grid bounds
        if (voxel_index < grid_size).all() and (voxel_index >= 0).all():
            voxel_counts[tuple(voxel_index)] += 1

    return voxel_counts


bg_path = "logdir-12-05/home-2023-11-08--20-29-39-compose/export_0000/bg-mesh.obj"
testdirs = glob.glob("logdir-12-05/*-compose/export_0001/")
res = 0.5  # 10cm


mesh = trimesh.load(bg_path)

root_trajs = []
cam_trajs = []
for it, loader_path in enumerate(testdirs):
    root_loader = MeshLoader(loader_path)
    # load root poses
    root_traj = root_loader.query_camtraj(data_class="fg")
    root_trajs.append(root_traj)

    # load cam poses
    cam_traj = root_loader.query_camtraj(data_class="bg")
    cam_trajs.append(cam_traj)
    print("loaded %d frames from %s" % (len(root_loader), loader_path))

server = viser.ViserServer()
# Setup root frame
server.add_frame(
    "/frames",
    wxyz=tf.SO3.exp(np.array([-np.pi / 2, 0.0, 0.0])).wxyz,
    position=(0, 0, 0),
    show_axes=False,
)

server.add_mesh_trimesh(
    name="/frames/environment",
    mesh=mesh,
)

# add box mesh
aabb = mesh.bounding_box.bounds
grid_size = np.ceil((aabb[1] - aabb[0]) / res).astype(int)

# create boxes
root_centers = np.linalg.inv(np.concatenate(root_trajs, 0))[..., :3, 3]
cam_centers = np.linalg.inv(np.concatenate(cam_trajs, 0))[..., :3, 3]
occupancy = count_points_in_voxels(mesh.vertices - aabb[0], grid_size, res)
occupancy = occupancy / (1e-6 + occupancy.max())

preference_root = count_points_in_voxels(root_centers - aabb[0], grid_size, res)
preference_root = preference_root / (1e-6 + preference_root.max())
preference_root = np.clip(preference_root, 0, 1)

preference_cam = count_points_in_voxels(cam_centers - aabb[0], grid_size, res)
preference_cam = preference_cam / (1e-6 + preference_cam.max())
preference_cam = np.clip(preference_cam, 0, 1)

boxes = []
for i in range(grid_size[0]):
    for j in range(grid_size[1]):
        for k in range(grid_size[2]):
            if preference_root[i, j, k] == 0:
                continue
            box = trimesh.creation.box(
                extents=[res, res, res],
                transform=trimesh.transformations.translation_matrix(
                    [i * res, j * res, k * res]
                ),
            )
            box.visual.vertex_colors = np.tile(
                np.array([1.0, 0.0, 0.0, 1.0]) * preference_root[i, j, k],
                (len(box.vertices), 1),
            )
            boxes.append(box)
boxes = trimesh.util.concatenate(boxes)
boxes.vertices += aabb[0]

server.add_mesh_simple(
    name="/frames/boxes",
    vertices=boxes.vertices,
    vertex_colors=boxes.visual.vertex_colors[:, :3],
    faces=boxes.faces,
    color=None,
    opacity=0.8,
)

from mdp import (
    best_policy,
    value_iteration,
    print_table,
    GridMDP,
    visulize_value_update,
    visulize_reward,
    sample_trajectory,
    plot_trajectory,
)

# sample trajectory from preference function
reward_grid = preference_root.sum(1).tolist()  # xz is left
mdp_env = GridMDP(reward_grid, terminals=[])
pi = best_policy(
    mdp_env,
    value_iteration(mdp_env, 0.001),
)

# frames = visulize_value_update(mdp_env, 100)
# save_vid("tmp/vid", frames)
# print("saved to tmp/vid.mp4")


# path = sample_trajectory(reward_grid, pi, start_position=(17, 6))
path = sample_trajectory(reward_grid, pi, start_position=(7, 10))

# visulzie reward, policy and trajectory
img = visulize_reward(reward_grid, pi)
cv2.imwrite("tmp/reward.jpg", img[..., ::-1])
img = plot_trajectory(path)
cv2.imwrite("tmp/path.jpg", img[..., ::-1])
print("saved to tmp/path.jpg")

while True:
    time.sleep(10.0)
