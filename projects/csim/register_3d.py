# goal: register a pre-trained PPR to template mesh.

import os, sys
import numpy as np
import trimesh
import pdb
import open3d as o3d

import torch

cwd = os.getcwd()
if cwd not in sys.path:
    sys.path.insert(0, cwd)
from absl import flags, app
from lab4d.export import export, get_config
from projects.csim.render_polycam import PolyCamRender
from lab4d.utils.mesh_loader import MeshLoader

seqname = "cat-pikachu-0-0000"
ppr_dir = "logdir/cat-pikachu-0-bg/"
template_dir = "database/polycam/Oct5at10-49AM-poly/"
camera_dir = "database/processed/Cameras/Full-Resolution"
sys.argv.append("--flagfile=%s/opts.log" % ppr_dir)

voxel_size = 0.02
max_correspondence_distance_coarse = voxel_size * 15
max_correspondence_distance_fine = voxel_size * 1.5


def pairwise_registration(source, target):
    print("Apply point-to-plane ICP")
    icp_coarse = o3d.registration.registration_icp(
        source,
        target,
        max_correspondence_distance_coarse,
        np.identity(4),
        o3d.registration.TransformationEstimationPointToPlane(),
    )
    icp_fine = o3d.registration.registration_icp(
        source,
        target,
        max_correspondence_distance_fine,
        icp_coarse.transformation,
        o3d.registration.TransformationEstimationPointToPlane(),
    )
    transformation_icp = icp_fine.transformation
    information_icp = o3d.registration.get_information_matrix_from_point_clouds(
        source, target, max_correspondence_distance_fine, icp_fine.transformation
    )
    return transformation_icp, information_icp


def full_registration(
    pcds, max_correspondence_distance_coarse, max_correspondence_distance_fine
):
    pose_graph = o3d.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.registration.PoseGraphNode(odometry))
    n_pcds = len(pcds)
    for source_id in range(n_pcds):
        for target_id in range(source_id + 1, n_pcds):
            transformation_icp, information_icp = pairwise_registration(
                pcds[source_id], pcds[target_id]
            )
            print("Build o3d.registration.PoseGraph")
            if target_id == source_id + 1:  # odometry case
                odometry = np.dot(transformation_icp, odometry)
                pose_graph.nodes.append(
                    o3d.registration.PoseGraphNode(np.linalg.inv(odometry))
                )
                pose_graph.edges.append(
                    o3d.registration.PoseGraphEdge(
                        source_id,
                        target_id,
                        transformation_icp,
                        information_icp,
                        uncertain=False,
                    )
                )
            else:  # loop closure case
                pose_graph.edges.append(
                    o3d.registration.PoseGraphEdge(
                        source_id,
                        target_id,
                        transformation_icp,
                        information_icp,
                        uncertain=True,
                    )
                )
    return pose_graph


def main(_):
    # step1: load ppr mesh, determine scale
    # export meshes
    opts = get_config()
    opts["load_suffix"] = "latest"
    # export(opts)
    # load mesh
    ref_mesh = trimesh.load("%s/export_0000/bg-mesh.obj" % ppr_dir)
    loader = MeshLoader("%s/export_0000/" % ppr_dir)
    loader.load_files()
    ref_mesh_0 = ref_mesh.copy()
    ref_mesh_0.apply_transform(loader.extr_dict[0])
    # TODO
    ref_mesh_0.vertices = ref_mesh_0.vertices * 0.25
    # step2: load template mesh
    tempalte_renderer = PolyCamRender(template_dir, image_size=(1024, 768))
    trg_mesh = tempalte_renderer.mesh
    # # TODO
    # trg_mesh.vertices = np.stack(
    #     [trg_mesh.vertices[:, 0], -trg_mesh.vertices[:, 1], -trg_mesh.vertices[:, 2]],
    #     axis=-1,
    # )
    # step3: load initial alignment
    world_to_view1 = np.load("%s/%s/aligned-00.npy" % (camera_dir, seqname))[0]
    view_to_world1 = np.linalg.inv(world_to_view1)
    ref_mesh_0.apply_transform(view_to_world1)
    # step4: register ppr to template mesh
    # TODO
    from pytorch3d.ops import iterative_closest_point

    pts1 = torch.from_numpy(ref_mesh_0.vertices).float()[None].cuda()
    pts2 = torch.from_numpy(trg_mesh.vertices).float()[None].cuda()
    solution = iterative_closest_point(
        X=pts1,
        Y=pts2,
        init_transform=None,
        max_iterations=100,
        relative_rmse_thr=1e-06,
        estimate_scale=False,
        allow_reflection=False,
        verbose=True,
    )
    print(solution.RTs)
    # show both meshes
    transform = np.eye(4)
    transform[:3, :3] = solution.RTs.R[0].cpu().numpy()
    transform[:3, 3] = solution.RTs.T[0].cpu().numpy()
    aln_mesh = ref_mesh_0.copy()
    aln_mesh.apply_transform(transform)
    ref_mesh_0.export("tmp/ref_mesh_0.obj")
    trg_mesh.export("tmp/trg_mesh.obj")
    aln_mesh.export("tmp/aln_mesh.obj")
    print("saved to tmp/ref_mesh_0.obj")
    print("saved to tmp/trg_mesh.obj")
    print("saved to tmp/aln_mesh.obj")

    # pcd1 = o3d.geometry.PointCloud()
    # pcd1.points = o3d.utility.Vector3dVector(ref_mesh_0.vertices)
    # pcd2 = o3d.geometry.PointCloud()
    # pcd2.points = o3d.utility.Vector3dVector(trg_mesh.vertices)
    # pcds_down = [pcd1, pcd2]

    # print("Full registration ...")
    # pose_graph = full_registration(
    #     pcds_down, max_correspondence_distance_coarse, max_correspondence_distance_fine
    # )

    # print("Optimizing PoseGraph ...")
    # option = o3d.registration.GlobalOptimizationOption(
    #     max_correspondence_distance=max_correspondence_distance_fine,
    #     edge_prune_threshold=0.25,
    #     reference_node=0,
    # )
    # o3d.registration.global_optimization(
    #     pose_graph,
    #     o3d.registration.GlobalOptimizationLevenbergMarquardt(),
    #     o3d.registration.GlobalOptimizationConvergenceCriteria(),
    #     option,
    # )

    # print("Transform points and display")
    # for point_id in range(len(pcds_down)):
    #     print(pose_graph.nodes[point_id].pose)
    #     pcds_down[point_id].transform(pose_graph.nodes[point_id].pose)
    # o3d.visualization.draw_geometries(pcds_down)

    # print("Make a combined point cloud")
    # pcds = load_point_clouds(voxel_size)
    # pcd_combined = o3d.geometry.PointCloud()
    # for point_id in range(len(pcds)):
    #     pcds[point_id].transform(pose_graph.nodes[point_id].pose)
    #     pcd_combined += pcds[point_id]
    # pcd_combined_down = pcd_combined.voxel_down_sample(voxel_size=voxel_size)
    # o3d.io.write_point_cloud("multiway_registration.pcd", pcd_combined_down)
    # o3d.visualization.draw_geometries([pcd_combined_down])


if __name__ == "__main__":
    app.run(main)
