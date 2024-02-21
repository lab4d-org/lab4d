import os, sys
import trimesh
import pdb
import argparse
import torch
import numpy as np
import glob

sys.path.insert(0, os.getcwd())
from lab4d.config import load_flags_from_file
from projects.csim.voxelize import VoxelGrid
from lab4d.engine.trainer import Trainer
from lab4d.utils.mesh_loader import MeshLoader


class SceneLoader:
    @torch.no_grad()
    def __init__(self, opts, inst_id=0):
        # get rest mesh
        opts["inst_id"] = inst_id
        model, data_info, ref_dict = Trainer.construct_test_model(opts)
        self.ref_feature = ref_dict["ref_feature"]
        meshes_rest = model.fields.extract_canonical_meshes(
            grid_size=opts["grid_size"],
            level=opts["level"],
            inst_id=inst_id,
            vis_thresh=opts["vis_thresh"],
            use_extend_aabb=opts["extend_aabb"],
        )
        bg_field = model.fields.field_params["bg"]
        # color = bg_field.extract_canonical_color(meshes_rest["bg"])
        # meshes_rest["bg"].visual.vertex_colors = color * 255

        # get dino feature
        bg_feature = bg_field.extract_canonical_feature(meshes_rest["bg"])
        # visualize the feature
        bg_feature_vis = data_info["apply_pca_fn"](bg_feature, normalize=True)
        meshes_rest["bg"].visual.vertex_colors = bg_feature_vis * 255
        self.bg_feature = bg_feature

        scale_bg = bg_field.logscale.exp().cpu().numpy()
        meshes_rest["bg"].apply_scale(1.0 / scale_bg)
        self.meshes_rest = meshes_rest

    @staticmethod
    def find_NN_point(feature_1, feature_2):
        # feature_1: T,H,W,F
        # feature_2: N,F
        # feature: ..., F
        shape_1 = feature_1.shape[:-1]
        shape_2 = feature_2.shape[:-1]
        feature_1 = feature_1.reshape(-1, feature_1.shape[-1])
        feature_2 = feature_2.reshape(-1, feature_2.shape[-1])
        # feature_1: N, F
        # feature_2: M, F
        chunk_size = 32
        dist = []
        for i in range(0, feature_1.shape[0], chunk_size):
            dist.append((feature_1[i : i + chunk_size, None] * feature_2[None]).sum(-1))
        dist = np.concatenate(dist, 0)
        dist = (feature_1[:, None] * feature_2[None]).sum(-1)
        dist = dist.reshape(shape_1 + shape_2)  # T,H,W,N
        nn_match = dist.argmax(-1)  # T,H,W
        return nn_match


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="script to render extraced meshes")
    parser.add_argument(
        "--logdir", default="logdir/home-2023-11-bg-adapt1/", help="logdir"
    )
    parser.add_argument("--inst_id", default=0, type=int, help="inst_id")
    args = parser.parse_args()

    # load flags from file with absl
    opts = load_flags_from_file("%s/opts.log" % args.logdir)
    opts["load_suffix"] = "latest"
    opts["logroot"] = "logdir"
    opts["grid_size"] = 128
    opts["level"] = 0
    opts["vis_thresh"] = -10
    opts["extend_aabb"] = False

    loader = SceneLoader(opts, inst_id=args.inst_id)
    mesh = loader.meshes_rest["bg"]
    bg_feature = loader.bg_feature

    # get root trajectory
    root_trajs = []
    cam_trajs = []
    # testdirs = sorted(glob.glob("%s/export_*" % args.logdir))
    testdirs = sorted(glob.glob("logdir/home-2023-11-compose-ft/export_*"))
    for it, loader_path in enumerate(testdirs):
        if "export_0000" in loader_path:
            continue
        root_loader = MeshLoader(loader_path)
        # load root poses
        root_traj = root_loader.query_camtraj(data_class="fg")
        root_trajs.append(root_traj)

        # load cam poses
        cam_traj = root_loader.query_camtraj(data_class="bg")
        cam_trajs.append(cam_traj)
        print("loaded %d frames from %s" % (len(root_loader), loader_path))
    root_trajs = np.linalg.inv(np.concatenate(root_trajs))
    cam_trajs = np.linalg.inv(np.concatenate(cam_trajs))  # T1+...+TN,4,4

    voxel_grid = VoxelGrid(mesh, res=0.1)

    voxel_grid.count_root_visitation(root_trajs[..., :3, 3])
    voxel_grid.count_cam_visitation(cam_trajs[..., :3, 3])

    # step 3: learn the mapping
    # find bg feature for each voxel

    # pdb.set_trace()
    # nn_match = loader.find_NN_point(loader.ref_feature[0], loader.bg_feature)
    # nn_pts = loader.meshes_rest["bg"].vertices[nn_match]
    # pixel_visitations = voxel_grid.readout_voxel(nn_pts, mode="root_visitation")
    # import cv2

    # cv2.imwrite("tmp/nn_mat.png", pixel_visitations * 255)

    pdb.set_trace()
    x = loader.bg_feature
    y = voxel_grid.readout_voxel(
        loader.meshes_rest["bg"].vertices, mode="root_visitation"
    )
    np.save("tmp/x.npy", {"x": x, "y": y})
    np.save("tmp/x_test.npy", {"x": loader.ref_feature})
    # save data to tmp folder
    # train a NN model

    voxel_grid.run_viser()
