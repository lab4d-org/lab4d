import os, sys
import cv2
import pdb
import numpy as np
import torch
import trimesh
import argparse

import ddpm
from utils import get_lab4d_data, load_models_regress, get_grid_xyz
from denoiser import reverse_diffusion, simulate_forward_diffusion
from eval import eval_ADE, eval_all
from visualizer import DiffusionVisualizer, spline_interp
from config import get_config

sys.path.insert(0, os.getcwd())
from lab4d.utils.quat_transform import axis_angle_to_matrix
from projects.csim.voxelize import BGField

if __name__ == "__main__":
    # params
    config = get_config()
    nsamp = config.eval_batch_size
    drop_cam = config.drop_cam
    drop_past = config.drop_past
    sample_idx = config.sample_idx
    state_size = 3
    num_timesteps = 50
    timesteps = list(range(num_timesteps))[::-1]
    noise_scheduler = ddpm.NoiseScheduler(num_timesteps=num_timesteps).cuda()

    # data
    (
        x0_wp_all,
        past_wp_all,
        cam_all,
        x0_to_world_all,
        x0_joints_all,
        past_joints_all,
        x0_angles_all,
        past_angles_all,
        x0_angles_to_world_all,
        # ) = get_lab4d_data("database/motion/S26-test-L80-S10.pkl")
        # ) = get_lab4d_data("database/motion/S26-train-L80-S1.pkl")
        # ) = get_lab4d_data("database/motion/S26-train-L64-S1.pkl")
    ) = get_lab4d_data("database/motion/S26-test-L64-S10.pkl")
    x0_goal_all = x0_wp_all[:, -1:]

    x0_goal = x0_goal_all
    x0_wp = x0_wp_all
    x0_joints = x0_joints_all
    x0_angles = x0_angles_all
    x0_to_world = x0_to_world_all
    x0_angles_to_world = x0_angles_to_world_all

    past_wp = past_wp_all
    past_joints = past_joints_all
    past_angles = past_angles_all
    cam = cam_all
    # bs, T, K, 3

    # # old
    # x0_goal = x0_goal_all[sample_idx : sample_idx + 1]
    # x0_wp = x0_wp_all[sample_idx : sample_idx + 1]
    # x0_joints = x0_joints_all[sample_idx : sample_idx + 1]
    # x0_angles = x0_angles_all[sample_idx : sample_idx + 1]
    # x0_to_world = x0_to_world_all[sample_idx : sample_idx + 1]
    # x0_angles_to_world = x0_angles_to_world_all[sample_idx : sample_idx + 1]

    # past_wp = past_wp_all[sample_idx : sample_idx + 1]
    # past_joints = past_joints_all[sample_idx : sample_idx + 1]
    # past_angles = past_angles_all[sample_idx : sample_idx + 1]
    # cam = cam_all[sample_idx : sample_idx + 1]
    # # # bs, T, K, 3

    # model
    bs = x0_wp.shape[0]
    forecast_size = x0_wp.shape[1]
    num_kps = x0_joints.shape[2]
    memory_size = past_wp.shape[1]
    env_model, goal_model, waypoint_model, fullbody_model, angle_model = (
        load_models_regress(
            config,
            state_size,
            forecast_size,
            memory_size,
            num_kps,
            # use_env=False,
        )
    )

    # scene data
    bg_field = BGField()
    occupancy = bg_field.voxel_grid.data[None]
    occupancy = torch.tensor(occupancy, dtype=torch.float32, device="cuda")
    feat_volume = env_model.extract_features(occupancy)

    # define grid to visualize the gradient
    xsize, ysize, zsize = 30, 5, 30
    xyz, xzmin, xzmax = get_grid_xyz(x0_wp, xsize, ysize, zsize)
    xyz_cuda_goal = torch.tensor(xyz, dtype=torch.float32, device="cuda")
    xyz_cuda_wp = (
        xyz_cuda_goal[:, None]
        .repeat(1, forecast_size, 1)
        .view(xyz_cuda_goal.shape[0], -1)
    )
    xsize_joints, ysize_joints, zsize_joints = 15, 5, 15
    xyz_joints, xzmin_joints, xzmax_joints = get_grid_xyz(
        x0_joints, xsize_joints, ysize_joints, zsize_joints
    )
    xyz_cuda_joints = torch.tensor(xyz_joints, dtype=torch.float32, device="cuda")
    xyz_cuda_joints = (
        xyz_cuda_joints[:, None]
        .repeat(1, forecast_size * num_kps, 1)
        .view(xyz_cuda_joints.shape[0], -1)
    )

    # goal
    goal = None
    xyz_grid = xyz_cuda_goal
    goal_pred = goal_model.predict(
        past_wp,
        cam,
        x0_to_world,
        feat_volume,
        bg_field.voxel_grid,
        goal,
    )
    # waypoint | goal conditioning
    # goal = torch.tensor(reverse_samples_goal[-1][:1], device="cuda")
    goal = x0_goal  # gt
    # goal = None
    xyz_grid = xyz_cuda_wp
    wp_pred = waypoint_model.predict(
        past_wp,
        cam,
        x0_to_world,
        feat_volume,
        bg_field.voxel_grid,
        goal,
    )
    # full body | wp conditioning
    # goal = torch.tensor(reverse_samples_wp[-1][:1], device="cuda")
    goal = x0_wp  # GT
    goal_ego = (x0_angles_to_world.transpose(3, 4) @ goal[..., None])[..., 0]
    xyz_grid = xyz_cuda_joints
    joints_pred = fullbody_model.predict(
        past_joints,
        cam * 0,
        None,
        None,
        None,
        goal_ego,
    )
    # angle | wp conditioning
    angles_pred = angle_model.predict(
        past_angles,
        cam * 0,
        None,
        None,
        None,
        goal_ego,
    )

    reverse_goal_all = goal_pred.view(-1, bs, 1, 1, 1, 3)
    reverse_wp_all = wp_pred.view(-1, bs, 1, forecast_size, 1, 3)
    reverse_angles_all = angles_pred.view(-1, bs, 1, forecast_size, 1, 3)
    reverse_joints_all = joints_pred.view(-1, bs, 1, forecast_size, num_kps, 3)

    reverse_joints_abs_all = (
        x0_angles_to_world[None, :, None]
        @ axis_angle_to_matrix(reverse_angles_all)
        @ reverse_joints_all[..., None]
    )[..., 0] + reverse_wp_all

    # compute numbers
    # TODO single-stage = reverse_wp_all[-1][:, :, -1:]
    eval_all(reverse_goal_all[-1], x0_goal, reverse_wp_all[-1], x0_wp)
    # eval_all(reverse_wp_all[-1][:, :, -1:], x0_goal, reverse_wp_all[-1], x0_wp)

    for i in range(bs):
        reverse_goal = reverse_goal_all[:, i]
        reverse_wp = reverse_wp_all[:, i]
        reverse_angles = reverse_angles_all[:, i]
        reverse_joints = reverse_joints_all[:, i]
        reverse_joints_abs = reverse_joints_abs_all[:, i]
        ############# visualization
        # goal visualization
        save_prefix = "goal-%d" % i
        # forward process
        forward_samples_goal = []
        visualizer = DiffusionVisualizer(
            xzmax=xzmax,
            xzmin=xzmin,
            num_timesteps=num_timesteps,
            logdir="projects/tiny-diffusion/exps/%s/" % config.logname,
            bg_field=bg_field,
        )
        visualizer.render_trajectory(
            forward_samples_goal,
            reverse_goal,
            past_wp[i],
            x0_to_world[i],
            prefix=save_prefix,
        )
        visualizer.plot_trajectory_2d(
            forward_samples_goal,
            reverse_goal,
            [],
            x0_goal[i],
            xyz,
            ysize,
            past_wp[i],
            cam[i],
            prefix=save_prefix,
        )
        visualizer.delete()

        # waypoint visualization
        save_prefix = "wp-%d" % i
        # forward process
        forward_samples_waypoint = []
        visualizer = DiffusionVisualizer(
            xzmax=xzmax,
            xzmin=xzmin,
            num_timesteps=num_timesteps,
            logdir="projects/tiny-diffusion/exps/%s/" % config.logname,
            bg_field=bg_field,
        )
        visualizer.render_trajectory(
            forward_samples_waypoint,
            reverse_wp,
            past_wp[i],
            x0_to_world[i],
            prefix=save_prefix,
        )
        visualizer.plot_trajectory_2d(
            forward_samples_waypoint,
            reverse_wp,
            [],
            x0_wp[i],
            xyz,
            ysize,
            past_wp[i],
            cam[i],
            prefix=save_prefix,
        )
        visualizer.delete()

        # full body visualization
        save_prefix = "joints-%d" % i
        # forward process
        forward_samples_joints = []
        visualizer = DiffusionVisualizer(
            xzmax=xzmax_joints,
            xzmin=xzmin_joints,
            num_timesteps=num_timesteps,
            logdir="projects/tiny-diffusion/exps/%s/" % config.logname,
            bg_field=bg_field,
        )
        # 1,1,1,3,3
        past_angles = x0_angles_to_world @ axis_angle_to_matrix(past_angles)
        past_joints = (past_angles @ past_joints[..., None])[..., 0] + past_wp
        visualizer.render_trajectory(
            forward_samples_joints,
            reverse_joints_abs,
            past_joints[0],
            x0_to_world[0],
            prefix=save_prefix,
        )
        x0_angles = x0_angles_to_world @ axis_angle_to_matrix(x0_angles)
        x0_joints = (x0_angles @ x0_joints[..., None])[..., 0] + x0_wp
        visualizer.plot_trajectory_2d(
            forward_samples_joints,
            reverse_joints_abs,
            [],
            x0_joints[0],
            xyz_joints,
            ysize_joints,
            past_joints[0],
            cam[0],
            prefix=save_prefix,
        )
        visualizer.delete()
