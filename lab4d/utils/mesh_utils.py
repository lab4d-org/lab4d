import torch
import open3d as o3d
import numpy as np
import tqdm
import trimesh
import copy

def to_cam_open3d(width, height, intrinsics, extrinsics):
    camera_traj = []
    for i, viewpoint_cam in enumerate(extrinsics):
        extrinsic=extrinsics[i]
        intrinsic=intrinsics[i]
        intrinsic=o3d.camera.PinholeCameraIntrinsic(width=width, 
                    height=height, 
                    fx = intrinsic[0],
                    fy = intrinsic[1],
                    cx = intrinsic[2],
                    cy = intrinsic[3],
                    )
        camera = o3d.camera.PinholeCameraParameters()
        camera.extrinsic = extrinsic
        camera.intrinsic = intrinsic
        camera_traj.append(camera)

    return camera_traj

@torch.no_grad()
def extract_mesh_bounded(intrinsics, extrinsics, rgbs, depths, masks=None, voxel_size=0.02, depth_trunc=3, mask_backgrond=True):
    """
    Perform TSDF fusion given a fixed depth range, used in the paper.
    
    voxel_size: the voxel size of the volume
    sdf_trunc: truncation value
    depth_trunc: maximum depth range, should depended on the scene's scales
    mask_backgrond: whether to mask backgroud, only works when the dataset have masks

    return o3d.mesh
    """
    sdf_trunc = voxel_size * 5
    print("Running tsdf volume integration ...")
    print(f'voxel_size: {voxel_size}')
    print(f'sdf_trunc: {sdf_trunc}')
    print(f'depth_truc: {depth_trunc}')

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length= voxel_size,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )

    _, height, width,_ = rgbs.shape
    cams = to_cam_open3d(width, height, intrinsics, extrinsics)
    for i, cam_o3d in tqdm.tqdm(enumerate(cams), desc="TSDF integration progress"):
        rgb = rgbs[i]
        depth = depths[i]
        
        # if we have mask provided, use it
        if masks is not None:
            depth[~masks[i]] = 0

        # make open3d rgbd
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.asarray(rgb * 255, order="C", dtype=np.uint8)),
            o3d.geometry.Image(np.asarray(depth, order="C")),
            depth_trunc = depth_trunc, convert_rgb_to_intensity=False,
            depth_scale = 1.0
        )

        volume.integrate(rgbd, intrinsic=cam_o3d.intrinsic, extrinsic=cam_o3d.extrinsic)

    mesh = volume.extract_triangle_mesh()
    mesh_post = post_process_mesh(mesh)
    # mesh = trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles))
    o3d.io.write_triangle_mesh("tmp/0.ply", mesh_post)
    return mesh

def post_process_mesh(mesh, cluster_to_keep=50):
    """
    Post-process a mesh to filter out floaters and disconnected parts
    """
    print("post processing the mesh to have {} clusterscluster_to_kep".format(cluster_to_keep))
    mesh_0 = copy.deepcopy(mesh)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            triangle_clusters, cluster_n_triangles, cluster_area = (mesh_0.cluster_connected_triangles())

    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    n_cluster = np.sort(cluster_n_triangles.copy())[-cluster_to_keep]
    n_cluster = max(n_cluster, 50) # filter meshes smaller than 50
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < n_cluster
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    mesh_0.remove_unreferenced_vertices()
    mesh_0.remove_degenerate_triangles()
    print("num vertices raw {}".format(len(mesh.vertices)))
    print("num vertices post {}".format(len(mesh_0.vertices)))
    return mesh_0