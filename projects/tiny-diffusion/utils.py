import math
import cv2
import os, sys
import pdb
import numpy as np
import glob
from tqdm.auto import tqdm
from einops import rearrange
import trimesh
import argparse
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
import pickle as pkl

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter


sys.path.insert(0, os.getcwd())
from lab4d.utils.vis_utils import get_pts_traj
from lab4d.utils.pyrender_wrapper import PyRenderWrapper
from lab4d.utils.io import save_vid
from lab4d.utils.quat_transform import matrix_to_axis_angle

from denoiser import EnvEncoder, TrajDenoiser, TrajPredictor


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


def get_lab4d_data(pkldatafilepath):
    # datapath = "/home/gengshay/code/guided-motion-diffusion/dataset/Custom/customposes.pkl"
    data = pkl.load(open(pkldatafilepath, "rb"))

    pose = [x for x in data["poses"]]
    joints = [x for x in data["joints3D"]]
    world_to_root = [x for x in data["se3"]]
    world_to_cam = [x for x in data["cam_se3"]]
    print("loading dataset of length %d | seq length %d" % (len(pose), len(pose[0])))

    # current frame
    idx0 = 8

    # goal list
    goal_idx = [15, 31, 47, 63]
    # goal_idx = [63]
    # goal_idx = [7, 15, 23, 31, 39, 47, 55, 63]
    # goal_idx = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63]
    forecast_size = len(goal_idx)

    # ADD CONTEXT LENGTH
    add_length = int(pkldatafilepath.split("/")[-1].split("-L")[1].split("-")[0]) - 64
    idx0 = idx0 + add_length
    goal_idx = [add_length + x for x in goal_idx]

    # load data: N, T, 3
    root_world_se3 = np.linalg.inv(np.stack(world_to_root, axis=0))
    root_world_se3 = torch.tensor(root_world_se3, dtype=torch.float32)
    root_world_trans = root_world_se3[..., :3, 3]  # N,T,3
    root_world_rot = root_world_se3[..., :3, :3]  # N,T,3,3
    cam_world = np.linalg.inv(np.stack(world_to_cam, axis=0))[..., :3, 3]
    cam_world = torch.tensor(cam_world, dtype=torch.float32)
    joints_ego = np.stack(joints, axis=0)  # N,T,K,3
    joints_ego = torch.tensor(joints_ego, dtype=torch.float32)  # [:, :, 0:4]
    angles = torch.tensor(np.stack(pose, axis=0), dtype=torch.float32)  # N,T,K,3
    # transform to world
    # joints_world = (root_world_rot[:, :, None] @ joints_ego[..., None])[..., 0]
    # joints_world = joints_world + root_world_trans[:, :, None, :]
    # trimesh.Trimesh(joints_world[2000].reshape(-1, 3)).export("tmp/0.obj")

    # transform root to zero centered at t0
    root_world_trans_curr = root_world_trans[:, idx0].clone()
    cam_world = cam_world - root_world_trans[:, idx0 : idx0 + 1]
    root_world_trans = root_world_trans - root_world_trans[:, idx0 : idx0 + 1]

    # transform root rotation to have identity at t0
    root_world_rot_curr = root_world_rot[:, idx0].clone()
    root_world_rot = root_world_rot[:, idx0 : idx0 + 1].transpose(2, 3) @ root_world_rot
    root_world_so3 = matrix_to_axis_angle(root_world_rot)

    # get past/goal pairs
    goal_world_trans = root_world_trans[:, goal_idx]  # N, T, 3
    goal_world_so3 = root_world_so3[:, goal_idx]
    goal_joints_ego = joints_ego[:, goal_idx]  # N, T, K, 3
    goal_angles = angles[:, goal_idx]

    past_world_trans = root_world_trans[:, :idx0]
    past_world_so3 = root_world_so3[:, :idx0]
    past_joints_ego = joints_ego[:, :idx0]  # N, T, K, 3
    past_angles = angles[:, :idx0]
    cam_world = cam_world[:, :idx0]  # camera position of the past frames

    # reshape
    bs = goal_world_trans.shape[0]
    goal_world_trans = goal_world_trans.view(bs, forecast_size, 1, 3)
    goal_world_so3 = goal_world_so3.view(bs, forecast_size, 1, 3)
    goal_joints_ego = goal_joints_ego.view(bs, forecast_size, -1, 3)
    goal_angles = goal_angles.view(bs, forecast_size, -1, 3)

    past_world_trans = past_world_trans.view(bs, idx0, 1, 3)
    past_world_so3 = past_world_so3.view(bs, idx0, 1, 3)
    past_joints_ego = past_joints_ego.view(bs, idx0, -1, 3)
    past_angles = past_angles.view(bs, idx0, -1, 3)
    cam_world = cam_world.view(bs, idx0, 1, 3)

    root_world_trans_curr = root_world_trans_curr.view(bs, 1, 1, 3)
    root_world_rot_curr = root_world_rot_curr.view(bs, 1, 1, 3, 3)

    # move to cuda
    goal_world_trans = goal_world_trans.cuda()
    goal_world_so3 = goal_world_so3.cuda()
    goal_joints_ego = goal_joints_ego.cuda()
    goal_angles = goal_angles.cuda()

    past_world_trans = past_world_trans.cuda()
    past_world_so3 = past_world_so3.cuda()
    past_joints_ego = past_joints_ego.cuda()
    past_angles = past_angles.cuda()
    cam_world = cam_world.cuda()
    root_world_trans_curr = root_world_trans_curr.cuda()
    root_world_rot_curr = root_world_rot_curr.cuda()

    goal_joints_ego = goal_angles
    past_joints_ego = past_angles

    # combine so3 with joints
    # goal_angles = torch.cat((goal_world_so3, goal_angles), dim=1)
    # past_angles = torch.cat((past_world_so3, past_angles), dim=1)
    goal_angles = goal_world_so3
    past_angles = past_world_so3

    # TODO learn an autoencoder for joints

    return (
        goal_world_trans,
        past_world_trans,
        cam_world,
        root_world_trans_curr,
        goal_joints_ego,
        past_joints_ego,
        goal_angles,
        past_angles,
        root_world_rot_curr,
    )


class TrajDataset(Dataset):
    """Dataset for loading trajectory data."""

    def __init__(
        self,
        x,
        y,
        cam,
        x_to_world,
        x_joints,
        y_joints,
        goal_angles,
        past_angles,
        x0_angles_to_world,
    ):
        self.x = x
        self.y = y
        self.cam = cam
        self.x_to_world = x_to_world
        self.x_joints = x_joints
        self.y_joints = y_joints
        self.goal_angles = goal_angles
        self.past_angles = past_angles
        self.x0_angles_to_world = x0_angles_to_world

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
            self.goal_angles[idx],
            self.past_angles[idx],
            self.x0_angles_to_world[idx],
        )


def define_models(
    config,
    state_size,
    forecast_size,
    memory_size,
    num_kps,
    mean_goal=None,
    std_goal=None,
    mean_wp=None,
    std_wp=None,
    mean_joints=None,
    std_joints=None,
    mean_angles=None,
    std_angles=None,
    in_dim=1,
    env_feat_dim=384,
    use_env=True,
    model=TrajDenoiser,
):
    env_model = EnvEncoder(in_dim, feat_dim=env_feat_dim)
    if not use_env:
        env_model.feat_dim = 0
    goal_model = model(
        mean_goal,
        std_goal,
        hidden_size=config.hidden_size,
        hidden_layers=config.hidden_layers,
        feat_dim=[env_model.feat_dim],
        forecast_size=1,
        memory_size=memory_size,
        state_size=state_size,
        condition_dim=config.condition_dim,
    )
    waypoint_model = model(
        mean_wp,
        std_wp,
        hidden_size=config.hidden_size,
        hidden_layers=config.hidden_layers,
        feat_dim=[env_model.feat_dim * forecast_size],
        forecast_size=forecast_size,
        memory_size=memory_size,
        state_size=state_size,
        cond_size=state_size,
        condition_dim=config.condition_dim,
    )
    fullbody_model = model(
        mean_joints,
        std_joints,
        hidden_size=config.hidden_size,
        hidden_layers=config.hidden_layers,
        # feat_dim=[env_model.feat_dim * forecast_size],
        feat_dim=[],
        forecast_size=forecast_size,
        memory_size=memory_size,
        state_size=state_size,
        cond_size=state_size * forecast_size,
        condition_dim=config.condition_dim,
        kp_size=num_kps,
        is_angle=True,
    )
    angle_model = model(
        mean_angles,
        std_angles,
        hidden_size=config.hidden_size,
        hidden_layers=config.hidden_layers,
        feat_dim=[],
        forecast_size=forecast_size,
        memory_size=memory_size,
        state_size=state_size,
        cond_size=state_size * forecast_size,
        condition_dim=config.condition_dim,
        kp_size=1,
        is_angle=True,
    )
    return env_model, goal_model, waypoint_model, fullbody_model, angle_model


def define_models_regress(
    config,
    state_size,
    forecast_size,
    memory_size,
    num_kps,
    mean_goal=None,
    std_goal=None,
    mean_wp=None,
    std_wp=None,
    mean_joints=None,
    std_joints=None,
    mean_angles=None,
    std_angles=None,
    in_dim=1,
    env_feat_dim=384,
    use_env=True,
    model=TrajPredictor,
):
    env_model = EnvEncoder(in_dim, feat_dim=env_feat_dim)
    if not use_env:
        env_model.feat_dim = 0
    goal_model = model(
        mean_goal,
        std_goal,
        hidden_size=config.hidden_size,
        hidden_layers=config.hidden_layers,
        feat_dim=[env_model.feat_dim],
        forecast_size=1,
        memory_size=memory_size,
        state_size=state_size,
        condition_dim=config.condition_dim,
    )
    waypoint_model = model(
        mean_wp,
        std_wp,
        hidden_size=config.hidden_size,
        hidden_layers=config.hidden_layers,
        feat_dim=[env_model.feat_dim],
        forecast_size=forecast_size,
        memory_size=memory_size,
        state_size=state_size,
        cond_size=state_size,
        condition_dim=config.condition_dim,
    )
    fullbody_model = model(
        mean_joints,
        std_joints,
        hidden_size=config.hidden_size,
        hidden_layers=config.hidden_layers,
        # feat_dim=[env_model.feat_dim * forecast_size],
        feat_dim=[],
        forecast_size=forecast_size,
        memory_size=memory_size,
        state_size=state_size,
        cond_size=state_size * forecast_size,
        condition_dim=config.condition_dim,
        kp_size=num_kps,
        is_angle=True,
    )
    angle_model = model(
        mean_angles,
        std_angles,
        hidden_size=config.hidden_size,
        hidden_layers=config.hidden_layers,
        feat_dim=[],
        forecast_size=forecast_size,
        memory_size=memory_size,
        state_size=state_size,
        cond_size=state_size * forecast_size,
        condition_dim=config.condition_dim,
        kp_size=1,
        is_angle=True,
    )
    return env_model, goal_model, waypoint_model, fullbody_model, angle_model


def load_models_regress(
    config, state_size, forecast_size, memory_size, num_kps, use_env=True
):
    # define models
    env_model, goal_model, waypoint_model, fullbody_model, angle_model = (
        define_models_regress(
            config,
            state_size,
            forecast_size,
            memory_size,
            num_kps,
            use_env=use_env,
        )
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
    angle_model_path = "projects/tiny-diffusion/exps/%s/angle_model_%s.pth" % (
        config.logname,
        config.suffix,
    )
    angle_model.load_state_dict(torch.load(angle_model_path), strict=True)

    env_model.cuda()
    env_model.eval()
    goal_model = goal_model.cuda()
    goal_model.eval()
    waypoint_model.cuda()
    waypoint_model.eval()
    fullbody_model.cuda()
    fullbody_model.eval()
    angle_model.cuda()
    angle_model.eval()
    return env_model, goal_model, waypoint_model, fullbody_model, angle_model


def load_models(config, state_size, forecast_size, memory_size, num_kps, use_env=True):
    # define models
    env_model, goal_model, waypoint_model, fullbody_model, angle_model = define_models(
        config,
        state_size,
        forecast_size,
        memory_size,
        num_kps,
        use_env=use_env,
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
    angle_model_path = "projects/tiny-diffusion/exps/%s/angle_model_%s.pth" % (
        config.logname,
        config.suffix,
    )
    angle_model.load_state_dict(torch.load(angle_model_path), strict=True)

    env_model.cuda()
    env_model.eval()
    goal_model = goal_model.cuda()
    goal_model.eval()
    waypoint_model.cuda()
    waypoint_model.eval()
    fullbody_model.cuda()
    fullbody_model.eval()
    angle_model.cuda()
    angle_model.eval()
    return env_model, goal_model, waypoint_model, fullbody_model, angle_model


def get_xzbounds(x0):
    xmin, ymin, zmin = x0.view(-1, 3).min(0)[0].cpu().numpy()
    xmax, ymax, zmax = x0.view(-1, 3).max(0)[0].cpu().numpy()
    xzmin = min(xmin, zmin) - 1
    xzmax = max(xmax, zmax) + 1
    return xzmin, xzmax, ymin, ymax


def get_grid_xyz(x0, xsize, ysize, zsize):
    xzmin, xzmax, ymin, ymax = get_xzbounds(x0)
    x = np.linspace(xzmin, xzmax, xsize)
    z = np.linspace(xzmin, xzmax, zsize)
    y = np.linspace(ymin, ymax, ysize)
    xyz = np.stack(np.meshgrid(x, y, z), axis=-1).reshape(-1, 3)
    return xyz, xzmin, xzmax
