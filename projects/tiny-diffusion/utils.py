import math
import cv2
import os, sys
import pdb
import numpy as np
import torch
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from einops import rearrange
import trimesh
import argparse
from omegaconf import OmegaConf


sys.path.insert(0, os.getcwd())
from lab4d.utils.vis_utils import get_pts_traj
from lab4d.utils.pyrender_wrapper import PyRenderWrapper
from lab4d.utils.io import save_vid
from projects.csim.voxelize import VoxelGrid, readout_voxel_fn


import matplotlib.pyplot as plt
import pickle as pkl
import numpy as np

from denoiser import EnvEncoder, TrajDenoiser


def get_data():
    x0_0 = torch.zeros((1000, 2))
    x0_0[:, 0] = torch.linspace(-1, 1, 1000)
    y_0 = torch.zeros((1000, 1))
    x0_1 = torch.zeros((1000, 2))
    x0_1[:, 1] = torch.linspace(-1, 1, 1000)
    y_1 = torch.ones((1000, 1))

    x0 = torch.cat((x0_0, x0_1), dim=0)
    y = torch.cat((y_0, y_1), dim=0)
    return x0, y


def get_lab4d_data():
    datapath = "/home/gengshay/code/guided-motion-diffusion/dataset/Custom/"
    pkldatafilepath = os.path.join(datapath, "customposes.pkl")
    data = pkl.load(open(pkldatafilepath, "rb"))

    pose = [x for x in data["poses"]]
    joints = [x for x in data["joints3D"]]
    world_to_root = [x for x in data["se3"]]
    world_to_cam = [x for x in data["cam_se3"]]

    # current frame
    idx0 = 8

    # goal list
    goal_idx = [15, 31, 47, 63]
    # goal_idx = [63]

    # load data: N, T, 3
    root_world_se3 = np.linalg.inv(np.stack(world_to_root, axis=0))
    root_world_se3 = torch.tensor(root_world_se3, dtype=torch.float32)
    root_world = root_world_se3[..., :3, 3]  # N,T,3
    cam_world = np.linalg.inv(np.stack(world_to_cam, axis=0))[..., :3, 3]
    cam_world = torch.tensor(cam_world, dtype=torch.float32)
    joints_ego = np.stack(joints, axis=0)  # N,T,K,3
    joints_ego = torch.tensor(joints_ego, dtype=torch.float32)  # [:, :, 0:4]
    # transform to world
    joints_world = (root_world_se3[:, :, None, :3, :3] @ joints_ego[..., None])[..., 0]
    joints_world = joints_world + root_world[:, :, None, :]
    # trimesh.Trimesh(joints_world[0][0]).export("tmp/0.obj")

    # transform to zero centered
    root_world_curr = root_world[:, idx0].clone()
    joints_world_curr = joints_world[:, idx0].clone()
    cam_world = cam_world - root_world[:, idx0 : idx0 + 1]
    joints_world = joints_world - root_world[:, idx0 : idx0 + 1, None]
    root_world = root_world - root_world[:, idx0 : idx0 + 1]

    # get input/label pairs
    njoints = joints_world.shape[-2]
    goal_world = root_world[:, goal_idx]
    goal_joints_world = joints_world[:, goal_idx]
    goal_joints_relative = goal_joints_world - goal_world[:, :, None]
    past_world = root_world[:, :idx0]
    past_joints_world = joints_world[:, :idx0]
    cam_world = cam_world[:, :idx0]  # camera position of the past frames

    ## merge
    # goal_world = torch.cat((goal_world[:, :, None], goal_joints_world), dim=2)
    # past_world = torch.cat((past_world[:, :, None], past_joints_world), dim=2)

    # reshape
    cam_world = cam_world.view(cam_world.shape[0], -1)
    goal_world = goal_world.view(goal_world.shape[0], -1)
    past_world = past_world.view(past_world.shape[0], -1)
    goal_joints_world = goal_joints_world.view(goal_joints_world.shape[0], -1)
    past_joints_world = past_joints_world.view(past_joints_world.shape[0], -1)
    goal_joints_relative = goal_joints_relative.view(goal_joints_relative.shape[0], -1)

    # move to cuda
    goal_world = goal_world.cuda()
    past_world = past_world.cuda()
    cam_world = cam_world.cuda()
    root_world_curr = root_world_curr.cuda()
    goal_joints_relative = goal_joints_relative.cuda()
    past_joints_world = past_joints_world.cuda()

    return (
        goal_world,
        past_world,
        cam_world,
        root_world_curr,
        goal_joints_relative,
        past_joints_world,
    )


class BGField:
    @torch.no_grad()
    def __init__(self):
        sys.path.insert(0, os.getcwd())
        from lab4d.config import load_flags_from_file
        from lab4d.engine.trainer import Trainer
        from lab4d.utils.mesh_loader import MeshLoader
        import glob

        # load flags from file with absl
        logdir = "logdir/home-2023-11-bg-adapt1/"
        opts = load_flags_from_file("%s/opts.log" % logdir)
        opts["load_suffix"] = "latest"
        opts["logroot"] = "logdir"
        opts["grid_size"] = 128
        opts["level"] = 0
        opts["vis_thresh"] = -10
        opts["extend_aabb"] = False

        opts["inst_id"] = 0
        model, data_info, ref_dict = Trainer.construct_test_model(opts)
        bg_field = model.fields.field_params["bg"]
        self.bg_field = bg_field

        meshes_rest = model.fields.extract_canonical_meshes(
            grid_size=opts["grid_size"],
            level=opts["level"],
            inst_id=opts["inst_id"],
            vis_thresh=opts["vis_thresh"],
            use_extend_aabb=opts["extend_aabb"],
        )
        scale_bg = bg_field.logscale.exp().cpu().numpy()
        meshes_rest["bg"].apply_scale(1.0 / scale_bg)
        self.bg_mesh = meshes_rest["bg"]

        # camera trajs
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
        self.root_trajs = root_trajs
        self.cam_trajs = cam_trajs

        voxel_grid = VoxelGrid(self.bg_mesh, res=0.1)
        voxel_grid.count_root_visitation(self.root_trajs[:, :3, 3])
        # voxel_grid.run_viser()
        self.voxel_grid = voxel_grid

    def compute_feat(self, x):
        return self.bg_field.compute_feat(x)["feature"]

    def get_bg_mesh(self):
        return self.bg_mesh


class TrajDataset(Dataset):
    """Dataset for loading trajectory data."""

    def __init__(self, x, y, cam, x_to_world, x_joints, y_joints):
        self.x = x
        self.y = y
        self.cam = cam
        self.x_to_world = x_to_world
        self.x_joints = x_joints
        self.y_joints = y_joints

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return (
            self.x[idx],
            self.y[idx],
            self.cam[idx],
            self.x_to_world[idx],
            self.x_joints[idx],
            self.y_joints[idx],
        )


def define_models(
    config,
    state_size,
    forecast_size,
    num_kps,
    mean_goal=None,
    std_goal=None,
    mean_wp=None,
    std_wp=None,
    mean_joints=None,
    std_joints=None,
):
    env_model = EnvEncoder()
    goal_model = TrajDenoiser(
        mean_goal,
        std_goal,
        hidden_size=config.hidden_size,
        hidden_layers=config.hidden_layers,
        feat_dim=[env_model.feat_dim],
        forecast_size=1,
        state_size=state_size,
        condition_dim=config.condition_dim,
    )
    waypoint_model = TrajDenoiser(
        mean_wp,
        std_wp,
        hidden_size=config.hidden_size,
        hidden_layers=config.hidden_layers,
        feat_dim=[env_model.feat_dim * forecast_size],
        forecast_size=forecast_size,
        state_size=state_size,
        cond_size=state_size,
        condition_dim=config.condition_dim,
    )
    fullbody_model = TrajDenoiser(
        mean_joints,
        std_joints,
        hidden_size=config.hidden_size,
        hidden_layers=config.hidden_layers,
        feat_dim=[env_model.feat_dim * forecast_size],
        forecast_size=forecast_size,
        state_size=state_size,
        cond_size=state_size * forecast_size,
        condition_dim=config.condition_dim,
        kp_size=num_kps,
    )
    return env_model, goal_model, waypoint_model, fullbody_model


def load_models(config, state_size, forecast_size, num_kps):
    # define models
    env_model, goal_model, waypoint_model, fullbody_model = define_models(
        config,
        state_size,
        forecast_size,
        num_kps,
    )
    # load models
    goal_path = "projects/tiny-diffusion/exps/%s/goal_model_%s.pth" % (
        config.logname,
        config.suffix,
    )
    goal_model.load_state_dict(torch.load(goal_path), strict=True)
    waypoint_path = "projects/tiny-diffusion/exps/%s/waypoint_model_%s.pth" % (
        config.logname,
        config.suffix,
    )
    waypoint_model.load_state_dict(torch.load(waypoint_path), strict=True)
    env_path = "projects/tiny-diffusion/exps/%s/env_model_%s.pth" % (
        config.logname,
        config.suffix,
    )
    env_model.load_state_dict(torch.load(env_path), strict=True)
    fullbody_path = "projects/tiny-diffusion/exps/%s/fullbody_model_%s.pth" % (
        config.logname,
        config.suffix,
    )
    fullbody_model.load_state_dict(torch.load(fullbody_path), strict=True)

    env_model.cuda()
    env_model.eval()
    goal_model = goal_model.cuda()
    goal_model.eval()
    waypoint_model.cuda()
    waypoint_model.eval()
    fullbody_model.cuda()
    fullbody_model.eval()
    return env_model, goal_model, waypoint_model, fullbody_model


def get_grid_xyz(x0, xsize, ysize, zsize):
    xmin, ymin, zmin = x0.view(-1, 3).min(0)[0].cpu().numpy()
    xmax, ymax, zmax = x0.view(-1, 3).max(0)[0].cpu().numpy()
    xzmin = min(xmin, zmin) - 1
    xzmax = max(xmax, zmax) + 1
    x = np.linspace(xzmin, xzmax, xsize)
    z = np.linspace(xzmin, xzmax, zsize)
    y = np.linspace(ymin, ymax, ysize)
    xyz = np.stack(np.meshgrid(x, y, z), axis=-1).reshape(-1, 3)
    return xyz, xzmin, xzmax
