import os, sys
import cv2
import pdb
import numpy as np
import torch
import trimesh

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
    x0_to_world_sampled = x0_to_world[sample_idx : sample_idx + 1]  # 1,3
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
    xsize_joints, ysize_joints, zsize_joints = 15, 5, 15
    xyz_joints, xzmin_joints, xzmax_joints = get_grid_xyz(
        x0_joints, xsize_joints, ysize_joints, zsize_joints
    )

    # visualize
    visualizer = DiffusionVisualizer(
        xzmax=xzmax,
        xzmin=xzmin,
        num_timesteps=num_timesteps,
        bg_field=bg_field,
        logdir="projects/tiny-diffusion/exps/%s/" % config.logname,
    )
    visualizer.run_viser()

    accumulated_traj = past_sampled.view(-1, 3)  # T', 3 in the latest ego coordinate
    while True:
        # for sample_idx in range(100):
        # goal
        goal = None
        xyz_grid = None
        reverse_samples_goal, _ = goal_model.reverse_diffusion(
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
        goal_samples = reverse_samples_goal[-1]

        # filter out goals that has nevel been visited
        goal_score = bg_field.voxel_grid.readout_voxel(
            goal_samples + x0_to_world_sampled, mode="root_visitation"
        )
        goal_samples = goal_samples[goal_score > 0]

        # select the first one
        goal = goal_samples[:1]

        reverse_samples_wp, _ = waypoint_model.reverse_diffusion(
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
            noisy_wp = noisy_wp.view(bs, -1, 3)
            noisy_wp = torch.cat([torch.zeros_like(noisy_wp[:, :1]), noisy_wp], 1)
            noisy_wp = noisy_wp.view(bs, -1)
            sample_wp_dense = spline_interp(noisy_wp, forecast_size=forecast_size + 1)
            reverse_samples_wp_dense.append(sample_wp_dense)
        reverse_samples_wp_dense = torch.stack(reverse_samples_wp_dense, 0)

        ############## visualization
        # # goal visualization
        # visualizer.render_trajectory(
        #     [],
        #     reverse_samples_goal,
        #     accumulated_traj.view(1, -1),
        #     x0_to_world_sampled,
        #     rotate=False,
        #     prefix="goal-%03d" % sample_idx,
        # )
        # # waypoint visualization
        # visualizer.render_trajectory(
        #     [],
        #     reverse_samples_wp_dense,
        #     accumulated_traj.view(1, -1),
        #     x0_to_world_sampled,
        #     rotate=False,
        #     prefix= "wp-%03d" % sample_idx,
        # )

        visualizer.render_trajectory_viser(
            reverse_samples_wp_dense[-1].view(bs, -1, 3),
            accumulated_traj.view(-1, 3),
            x0_to_world_sampled,
        )

        # add to accumulated_traj: ego coordinate
        predicted_traj = reverse_samples_wp_dense[-1, 0].view(-1, 3)
        accumulated_traj = torch.cat([accumulated_traj, predicted_traj[:-1]], 0)
        accumulated_traj = accumulated_traj - predicted_traj[-1]
        # update past_sampled, x0_to_world_sampled
        x0_to_world_sampled = x0_to_world_sampled + predicted_traj[-1]
        past_sampled = (accumulated_traj[-8:]).view(1, -1)
visualizer.delete()
