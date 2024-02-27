import os, sys
import cv2
import pdb
import numpy as np
import torch
import trimesh
import argparse

import ddpm
from utils import get_lab4d_data, BGField, load_models, get_grid_xyz
from denoiser import reverse_diffusion, simulate_forward_diffusion
from visualizer import DiffusionVisualizer, spline_interp
from config import get_config

if __name__ == "__main__":
    # params
    config = get_config()
    bs = config.eval_batch_size
    drop_cam = config.drop_cam
    drop_past = config.drop_past
    sample_idx = config.sample_idx
    state_size = 3
    num_timesteps = 50
    timesteps = list(range(num_timesteps))[::-1]
    noise_scheduler = ddpm.NoiseScheduler(num_timesteps=num_timesteps).cuda()

    # data
    x0, past, cam, x0_to_world, x0_joints, past_joints = get_lab4d_data()
    x0_goal = x0[..., -state_size:]
    x0_sampled = x0[sample_idx : sample_idx + 1]
    x0_goal_sampled = x0_goal[sample_idx : sample_idx + 1]
    past_sampled = past[sample_idx : sample_idx + 1]
    cam_sampled = cam[sample_idx : sample_idx + 1]
    x0_to_world_sampled = x0_to_world[sample_idx : sample_idx + 1]
    x0_joints_sampled = x0_joints[sample_idx : sample_idx + 1]
    past_joints_sampled = past_joints[sample_idx : sample_idx + 1]

    # model
    forecast_size = int(x0.shape[1] / state_size)
    num_kps = int(x0_joints.shape[1] / state_size / forecast_size)
    env_model, goal_model, waypoint_model, fullbody_model = load_models(
        config, state_size, forecast_size, num_kps
    )

    # scene data
    bg_field = BGField()
    occupancy = bg_field.voxel_grid.data[None]
    occupancy = torch.tensor(occupancy, dtype=torch.float32, device="cuda")
    feat_volume = env_model.extract_features(occupancy)

    # define grid to visualize the gradient
    xsize, ysize, zsize = 30, 5, 30
    xyz, xzmin, xzmax = get_grid_xyz(x0, xsize, ysize, zsize)
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
    reverse_samples_goal, reverse_grad_grid_goal = goal_model.reverse_diffusion(
        bs,
        num_timesteps,
        noise_scheduler,
        past_sampled,
        cam_sampled,
        x0_to_world_sampled,
        feat_volume,
        bg_field,
        drop_cam,
        drop_past,
        goal,
        xyz_grid=xyz_grid,
    )

    # waypoint | goal conditioning
    goal = torch.tensor(reverse_samples_goal[-1][:1], device="cuda")
    # reversed_goal = x0_goal # gt
    xyz_grid = xyz_cuda_wp
    reverse_samples_wp, reverse_grad_grid_wp = waypoint_model.reverse_diffusion(
        bs,
        num_timesteps,
        noise_scheduler,
        past_sampled,
        cam_sampled,
        x0_to_world_sampled,
        feat_volume,
        bg_field,
        drop_cam,
        drop_past,
        goal,
        xyz_grid=xyz_grid,
    )
    # use spline to interpolate the dense trajectory
    reverse_samples_wp_dense = []
    for noisy_wp in reverse_samples_wp:
        sample_wp_dense = spline_interp(noisy_wp)
        reverse_samples_wp_dense.append(sample_wp_dense)

    # full body | wp conditioning
    goal = torch.tensor(reverse_samples_wp[-1][:1], device="cuda")
    # reversed_goal = x0_sampled # GT
    xyz_grid = xyz_cuda_joints
    reverse_samples_joints, reverse_grad_grid_joints = fullbody_model.reverse_diffusion(
        bs,
        num_timesteps,
        noise_scheduler,
        past_joints_sampled,
        cam_sampled,
        x0_to_world_sampled,
        feat_volume,
        bg_field,
        drop_cam,
        drop_past,
        goal,
        xyz_grid=xyz_grid,
    )
    # from ego t to ego at frame 0
    reverse_samples_joints_abs = []
    for i in range(len(reverse_samples_joints)):
        samples_joints_abs = (
            reverse_samples_joints[i].reshape(bs, -1, num_kps, 3)
            + goal.reshape(1, -1, 1, 3).cpu().numpy()
        ).reshape(bs, -1)
        reverse_samples_joints_abs.append(samples_joints_abs)
    # use spline to interpolate the dense trajectory
    reverse_samples_joints_dense = []
    for sample_joints in reverse_samples_joints_abs:
        sample_wp_dense = spline_interp(
            sample_joints, forecast_size=forecast_size, num_kps=num_kps
        )
        reverse_samples_joints_dense.append(sample_wp_dense)

    ############## visualization
    # goal visualization
    save_prefix = "goal"
    # forward process
    forward_samples_goal = goal_model.simulate_forward_diffusion(
        x0_goal, noise_scheduler
    )
    visualizer = DiffusionVisualizer(
        xzmax=xzmax,
        xzmin=xzmin,
        num_timesteps=num_timesteps,
        logdir="projects/tiny-diffusion/exps/%s/" % config.logname,
        prefix=save_prefix,
    )
    visualizer.render_trajectory(
        forward_samples_goal, reverse_samples_goal, past, x0_to_world
    )
    visualizer.plot_trajectory_2d(
        forward_samples_goal,
        reverse_samples_goal,
        reverse_grad_grid_goal,
        x0_goal_sampled[0],
        sample_idx,
        xyz,
        ysize,
        past,
        cam,
    )
    visualizer.delete()

    # waypoint visualization
    save_prefix = "wp"
    # forward process
    forward_samples_waypoint = waypoint_model.simulate_forward_diffusion(
        x0, noise_scheduler
    )
    visualizer = DiffusionVisualizer(
        xzmax=xzmax,
        xzmin=xzmin,
        num_timesteps=num_timesteps,
        logdir="projects/tiny-diffusion/exps/%s/" % config.logname,
        prefix=save_prefix,
    )
    visualizer.render_trajectory(
        forward_samples_waypoint, reverse_samples_wp_dense, past, x0_to_world
    )
    visualizer.plot_trajectory_2d(
        forward_samples_waypoint,
        reverse_samples_wp,
        reverse_grad_grid_wp,
        reverse_samples_goal[-1][0],
        sample_idx,
        xyz,
        ysize,
        past,
        cam,
    )
    visualizer.delete()

    # full body visualization
    save_prefix = "joints"
    # forward process
    forward_samples_joints = fullbody_model.simulate_forward_diffusion(
        x0_joints, noise_scheduler
    )
    visualizer = DiffusionVisualizer(
        xzmax=xzmax_joints,
        xzmin=xzmin_joints,
        num_timesteps=num_timesteps,
        logdir="projects/tiny-diffusion/exps/%s/" % config.logname,
        prefix=save_prefix,
    )
    visualizer.render_trajectory(
        forward_samples_joints, reverse_samples_joints_dense, past_joints, x0_to_world
    )
    visualizer.plot_trajectory_2d(
        forward_samples_joints,
        reverse_samples_joints_abs,
        reverse_grad_grid_joints,
        reverse_samples_wp[-1][0],
        sample_idx,
        xyz_joints,
        ysize_joints,
        past,
        cam,
    )
    visualizer.delete()
