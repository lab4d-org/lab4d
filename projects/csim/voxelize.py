# import torch
# import pdb
# from mmcv.ops import Voxelization

# voxel_holder = Voxelization(
#     max_num_points=10, voxel_size=0.1, point_cloud_range=[-20, -20, -20, 10, 10, 10]
# )
# pdb.set_trace()
# voxel_holder(torch.rand(1000, 3) * 20 - 10)

import trimesh
import open3d as o3d
import numpy as np
import pdb
import time
import viser
import viser.transforms as tf

voxel_size = 0.1
mesh_path = "../vid2sim/logdir/home-2023-11-bg-adapt3/export_0001/bg-mesh.obj"

mesh_o3d = o3d.io.read_triangle_mesh(mesh_path)

mesh = trimesh.load(mesh_path)
aabb = mesh.bounding_box.bounds
grid_size = np.ceil((aabb[1] - aabb[0]) / voxel_size).astype(int)


print("voxelization")
voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(
    mesh_o3d, voxel_size=voxel_size
)
voxels = voxel_grid.get_voxels()  # returns list of voxels
indices = np.stack(list(vx.grid_index for vx in voxels))
colors = np.stack(list(vx.color for vx in voxels))
tensor = np.zeros(indices.max(0) + 1)
for idx in indices:
    tensor[idx[0], idx[1], idx[2]] = 1

# o3d.visualization.draw_geometries([voxel_grid])
# visualizations
boxes = []
for i in range(grid_size[0]):
    for j in range(grid_size[1]):
        for k in range(grid_size[2]):
            if tensor[i, j, k] == 0:
                continue
            box = trimesh.creation.box(
                extents=[voxel_size, voxel_size, voxel_size],
                transform=trimesh.transformations.translation_matrix(
                    [
                        i * voxel_size + aabb[0, 0],
                        j * voxel_size + aabb[0, 1],
                        k * voxel_size + aabb[0, 2],
                    ]
                ),
            )
            box.visual.vertex_colors = np.tile(
                np.array([1.0, 0.0, 0.0, 1.0]) * tensor[i, j, k],
                (len(box.vertices), 1),
            )
            boxes.append(box)
boxes = trimesh.util.concatenate(boxes)

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

server.add_mesh_simple(
    name="/frames/boxes",
    vertices=boxes.vertices,
    vertex_colors=boxes.visual.vertex_colors[:, :3],
    faces=boxes.faces,
    color=None,
    opacity=0.8,
)

while True:
    time.sleep(10.0)
