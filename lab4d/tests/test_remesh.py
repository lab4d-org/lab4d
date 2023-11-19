import subprocess
import trimesh

import point_cloud_utils as pcu


path = "logdir/ama-bouncing-4v-fg-urdf/export_0000/fg-mesh.obj"

mesh = trimesh.load(path)

# vw, fw = pcu.make_mesh_watertight(mesh.vertices, mesh.faces, 2000)
# print("fw", fw.shape)
# # Compute the shortest distance between each point in p and the mesh:
# #   dists is a NumPy array of shape (P,) where dists[i] is the
# #   shortest distnace between the point p[i, :] and the mesh (v, f)
# dists, fid, bc = pcu.closest_points_on_mesh(vw, mesh.vertices, mesh.faces)

# # Interpolate the barycentric coordinates to get the coordinates of
# # the closest points on the mesh to each point in p
# vw = pcu.interpolate_barycentric_coords(mesh.faces, fid, bc, mesh.vertices)

vw, fw, v_c, f_c = pcu.decimate_triangle_mesh(mesh.vertices, mesh.faces, 10000)

mesh = trimesh.Trimesh(vw, fw)
mesh.export("tmp/manifold.obj")
print("done")
# print(
#     # ./manifold input.obj output.obj [resolution (Default 20000)]
#     subprocess.check_output(
#         [
#             "./lab4d/third_party/Manifold/build/manifold",
#             path,
#             "tmp/manifold.obj",
#             "10000",
#         ]
#     )
# )
print("exported to tmp/manifold.obj")

# # print(
# #     subprocess.check_output(
# #         [
# #             "./lab4d/third_party/Manifold/build/simplify",
# #             "-i",
# #             path,
# #             "-o",
# #             "tmp/simple.obj",
# #             "-m",
# #             "-f",
# #             "20000",
# #         ]
# #     )
# # )
