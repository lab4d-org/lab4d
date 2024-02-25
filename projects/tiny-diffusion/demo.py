import math
import cv2
import os, sys
import pdb
import numpy as np
import torch

from tqdm.auto import tqdm
import trimesh
import argparse

import ddpm
from arch import TemporalUnet, TimestepEmbedder
from ddpm import get_data, get_lab4d_data, BGField
from visualizer import DiffusionVisualizer, spline_interp

sys.path.insert(0, os.getcwd())
from lab4d.utils.vis_utils import get_pts_traj
from lab4d.utils.pyrender_wrapper import PyRenderWrapper
from lab4d.utils.io import save_vid
from projects.csim.voxelize import VoxelGrid, readout_voxel_fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--hidden_layers", type=int, default=4)
    parser.add_argument("--logname", type=str, default="base")
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--drop_cam", action="store_true")
    parser.add_argument("--drop_past", action="store_true")
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--suffix", type=str, default="latest")
    config = parser.parse_args()

    ### forward and reverse process animation
    num_timesteps = 50
    noise_scheduler = ddpm.NoiseScheduler(num_timesteps=num_timesteps).cuda()

    # forward
    x0, past, cam, x0_to_world, x0_joints, past_joints = get_lab4d_data()
    x0 = x0.cuda()
    past = past.cuda()
    cam = cam.cuda()
    x0_to_world = x0_to_world.cuda()
    x0_joints = x0_joints.cuda()
    past_joints = past_joints.cuda()
    sample_idx = config.sample_idx
    x0_sampled = x0[sample_idx : sample_idx + 1]
    past_sampled = past[sample_idx : sample_idx + 1]
    cam_sampled = cam[sample_idx : sample_idx + 1]
    x0_to_world_sampled = x0_to_world[sample_idx : sample_idx + 1]
    x0_joints_sampled = x0_joints[sample_idx : sample_idx + 1]
    past_joints_sampled = past_joints[sample_idx : sample_idx + 1]

    # define models
    bs = config.eval_batch_size
    state_size = 3
    forecast_size = int(x0.shape[1] / state_size)
    num_kps = int(x0_joints.shape[1] / state_size / forecast_size)
    env_model = ddpm.EnvEncoder()
    goal_model = ddpm.TrajDenoiser(
        feat_dim=[env_model.feat_dim],
        forecast_size=1,
        state_size=state_size,
        hidden_layers=config.hidden_layers,
        hidden_size=config.hidden_size,
    )
    waypoint_model = ddpm.TrajDenoiser(
        feat_dim=[env_model.feat_dim * forecast_size],
        forecast_size=forecast_size,
        state_size=state_size,
        cond_size=state_size,
        hidden_layers=config.hidden_layers,
        hidden_size=config.hidden_size,
    )
    fullbody_model = ddpm.TrajDenoiser(
        feat_dim=[env_model.feat_dim * forecast_size * num_kps],
        forecast_size=forecast_size,
        state_size=state_size * num_kps,
        cond_size=state_size * forecast_size,
        hidden_layers=config.hidden_layers,
        hidden_size=config.hidden_size,
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
    drop_cam = config.drop_cam
    drop_past = config.drop_past

    # scene data
    bg_field = BGField()
    occupancy = bg_field.voxel_grid.data[None]
    occupancy = torch.tensor(occupancy, dtype=torch.float32, device="cuda")
    feat_volume = env_model.extract_features(occupancy)

    # define grid to visualize the gradient
    xmin, ymin, zmin = x0.view(-1, 3).min(0)[0].cpu().numpy()
    xmax, ymax, zmax = x0.view(-1, 3).max(0)[0].cpu().numpy()
    xzmin = min(xmin, zmin) - 1
    xzmax = max(xmax, zmax) + 1
    x = np.linspace(xzmin, xzmax, 30)
    z = np.linspace(xzmin, xzmax, 30)
    y = np.linspace(ymin, ymax, 5)
    xyz = np.stack(np.meshgrid(x, y, z), axis=-1).reshape(-1, 3)
    xyz_cuda_goal = torch.tensor(xyz, dtype=torch.float32, device="cuda")
    xyz_cuda_wp = (
        xyz_cuda_goal[:, None]
        .repeat(1, forecast_size, 1)
        .view(xyz_cuda_goal.shape[0], -1)
    )

    xmin_body, ymin_body, zmin_body = x0_joints.view(-1, 3).min(0)[0].cpu().numpy()
    xmax_body, ymax_body, zmax_body = x0_joints.view(-1, 3).max(0)[0].cpu().numpy()
    xzmin_body = min(xmin_body, zmin_body) - 1
    xzmax_body = max(xmax_body, zmax_body) + 1
    x_body = np.linspace(xzmin_body, xzmax_body, 15)
    z_body = np.linspace(xzmin_body, xzmax_body, 15)
    y_body = np.linspace(ymin_body, ymax_body, 15)
    xyz_body = np.stack(np.meshgrid(x_body, y_body, z_body), axis=-1).reshape(-1, 3)
    xyz_cuda_joints = torch.tensor(xyz_body, dtype=torch.float32, device="cuda")
    xyz_cuda_joints = (
        xyz_cuda_joints[:, None]
        .repeat(1, forecast_size * num_kps, 1)
        .view(xyz_cuda_joints.shape[0], -1)
    )

    ################ goal
    # visulaize forward samples
    x0_goal = x0[..., -state_size:]
    forward_samples_goal = []
    forward_samples_goal.append(x0_goal.cpu().numpy())
    for t in range(len(noise_scheduler)):
        timesteps = torch.tensor(
            np.repeat(t, len(x0_goal)), dtype=torch.long, device="cuda"
        )
        noise_goal = torch.randn_like(x0_goal, device="cuda")
        noise_goal = noise_goal * goal_model.std + goal_model.mean
        sample_goal = noise_scheduler.add_noise(x0_goal, noise_goal, timesteps)
        forward_samples_goal.append(sample_goal.cpu().numpy())

    # start reverse process
    sample_goal = torch.randn(bs, state_size, device="cuda")
    sample_goal = sample_goal * goal_model.std + goal_model.mean
    timesteps = list(range(num_timesteps))[::-1]
    reverse_samples = []
    reverse_grad = []
    reverse_samples.append(sample_goal.cpu().numpy())
    for i, t in enumerate(tqdm(timesteps)):
        t_grid = torch.tensor(np.repeat(t, len(xyz)), dtype=torch.long, device="cuda")
        t = torch.tensor(
            np.repeat(t, len(sample_goal)), dtype=torch.long, device="cuda"
        )

        past_grid = past_sampled.repeat(len(xyz), 1)
        past = past_sampled.repeat(sample_goal.shape[0], 1)

        cam_grid = cam_sampled.repeat(len(xyz), 1)
        cam = cam_sampled.repeat(sample_goal.shape[0], 1)

        x0_to_world_grid = x0_to_world_sampled.repeat(len(xyz), 1)
        x0_to_world = x0_to_world_sampled.repeat(sample_goal.shape[0], 1)

        with torch.no_grad():
            ###### goal
            # 3D convs then query, # B1HWD => B3HWD
            feat = env_model.readout_in_world(
                feat_volume,
                sample_goal,
                x0_to_world,
                bg_field.voxel_grid.res,
                bg_field.voxel_grid.origin,
            )
            feat_grid = env_model.readout_in_world(
                feat_volume,
                xyz_cuda_goal,
                x0_to_world_grid,
                bg_field.voxel_grid.res,
                bg_field.voxel_grid.origin,
            )

            grad = goal_model(
                sample_goal,
                t[:, None] / num_timesteps,
                past,
                cam,
                [feat],
                drop_cam=drop_cam,
                drop_past=drop_past,
            )
            xy_grad = goal_model(
                xyz_cuda_goal,
                t_grid[:, None] / num_timesteps,
                past_grid,
                cam_grid,
                [feat_grid],
                drop_cam=drop_cam,
                drop_past=drop_past,
            )
            # xy_grad = voxel_grid.readout_voxel(
            #     (xyz_cuda + x0_to_world_grid).cpu().numpy(), mode="root_visitation_gradient"
            # )
            # xy_grad = torch.tensor(xy_grad, dtype=torch.float32, device="cuda")
            # xy_grad = -10 * xy_grad
        sample_goal = noise_scheduler.step(grad, t[0], sample_goal, goal_model.std)
        reverse_samples.append(sample_goal.cpu().numpy())
        reverse_grad.append(xy_grad.cpu().numpy())

    # goal visualization
    visualizer = DiffusionVisualizer(
        xzmax=xzmax,
        xzmin=xzmin,
        num_timesteps=num_timesteps,
        logdir="projects/tiny-diffusion/exps/%s/" % config.logname,
        prefix="goal",
    )
    visualizer.render_trajectory(
        forward_samples_goal, reverse_samples, past, bg_field, x0_to_world
    )
    visualizer.plot_trajectory_2d(
        forward_samples_goal,
        reverse_samples,
        reverse_grad,
        x0_sampled[0, -state_size:],
        sample_idx,
        xyz,
        y,
        past,
        cam,
    )
    visualizer.delete()

    ############### waypoint
    # visulaize forward samples
    x0_waypoint = x0
    forward_samples_waypoint = []
    forward_samples_waypoint.append(x0_waypoint.cpu().numpy())
    for t in range(len(noise_scheduler)):
        timesteps = torch.tensor(
            np.repeat(t, len(x0_waypoint)), dtype=torch.long, device="cuda"
        )
        noise_waypoint = torch.randn_like(x0_waypoint, device="cuda")
        noise_waypoint = noise_waypoint * waypoint_model.std + waypoint_model.mean
        sample_wp = noise_scheduler.add_noise(x0_waypoint, noise_waypoint, timesteps)
        forward_samples_waypoint.append(sample_wp.cpu().numpy())

    # start reverse process
    sample_wp = torch.randn(bs, forecast_size * state_size, device="cuda")
    sample_wp = sample_wp * waypoint_model.std + waypoint_model.mean
    timesteps = list(range(num_timesteps))[::-1]
    reverse_samples_wp = []
    reverse_grad = []
    reverse_samples_wp.append(sample_wp.cpu().numpy())
    # sampled_goal = sample_goal[:1]
    sampled_goal = x0_sampled[..., -state_size:]
    for i, t in enumerate(tqdm(timesteps)):
        t_grid = torch.tensor(np.repeat(t, len(xyz)), dtype=torch.long, device="cuda")
        t = torch.tensor(np.repeat(t, len(sample_wp)), dtype=torch.long, device="cuda")

        past_grid = past_sampled.repeat(len(xyz), 1)
        past = past_sampled.repeat(sample_wp.shape[0], 1)

        cam_grid = cam_sampled.repeat(len(xyz), 1)
        cam = cam_sampled.repeat(sample_wp.shape[0], 1)

        x0_to_world_grid = x0_to_world_sampled.repeat(len(xyz), 1)
        x0_to_world = x0_to_world_sampled.repeat(sample_wp.shape[0], 1)

        goal_grid = sampled_goal.repeat(len(xyz), 1)
        goal = sampled_goal.repeat(sample_wp.shape[0], 1)

        with torch.no_grad():
            ###### goal
            # 3D convs then query, # B1HWD => B3HWD
            feat = env_model.readout_in_world(
                feat_volume,
                sample_wp,
                x0_to_world,
                bg_field.voxel_grid.res,
                bg_field.voxel_grid.origin,
            )
            feat_grid = env_model.readout_in_world(
                feat_volume,
                xyz_cuda_wp,
                x0_to_world_grid,
                bg_field.voxel_grid.res,
                bg_field.voxel_grid.origin,
            )

            grad = waypoint_model(
                sample_wp,
                t[:, None] / num_timesteps,
                past,
                cam,
                [feat],
                drop_cam=drop_cam,
                drop_past=drop_past,
                goal=goal,
            )
            xy_grad = waypoint_model(
                xyz_cuda_wp,
                t_grid[:, None] / num_timesteps,
                past_grid,
                cam_grid,
                [feat_grid],
                drop_cam=drop_cam,
                drop_past=drop_past,
                goal=goal_grid,
            )
            # xy_grad = voxel_grid.readout_voxel(
            #     (xyz_cuda + x0_to_world_grid).cpu().numpy(), mode="root_visitation_gradient"
            # )
            # xy_grad = torch.tensor(xy_grad, dtype=torch.float32, device="cuda")
            # xy_grad = -10 * xy_grad
        sample_wp = noise_scheduler.step(grad, t[0], sample_wp, waypoint_model.std)
        reverse_samples_wp.append(sample_wp.cpu().numpy())
        reverse_grad.append(xy_grad.cpu().numpy())

    # use spline to interpolate the dense trajectory
    reverse_samples_wp_dense = []
    for sample_wp in reverse_samples_wp:
        # TODO sample_wp to full trajectory
        sample_wp_dense = spline_interp(sample_wp)
        reverse_samples_wp_dense.append(sample_wp_dense)

    # waypoint visualization
    visualizer = DiffusionVisualizer(
        xzmax=xzmax,
        xzmin=xzmin,
        num_timesteps=num_timesteps,
        logdir="projects/tiny-diffusion/exps/%s/" % config.logname,
        prefix="wp",
    )
    visualizer.render_trajectory(
        forward_samples_waypoint,
        reverse_samples_wp_dense,
        past,
        bg_field,
        x0_to_world,
    )
    visualizer.plot_trajectory_2d(
        forward_samples_waypoint,
        reverse_samples_wp,
        reverse_grad,
        sampled_goal[0],
        sample_idx,
        xyz,
        y,
        past,
        cam,
    )
    visualizer.delete()

    ############### full body
    # visulaize forward samples
    forward_samples_joints = []
    forward_samples_joints.append(x0_joints.cpu().numpy())
    for t in range(len(noise_scheduler)):
        timesteps = torch.tensor(
            np.repeat(t, len(x0_joints)), dtype=torch.long, device="cuda"
        )
        noise_joints = torch.randn_like(x0_joints, device="cuda")
        noise_joints = noise_joints * fullbody_model.std + fullbody_model.mean
        sample_joints = noise_scheduler.add_noise(x0_joints, noise_joints, timesteps)
        forward_samples_joints.append(sample_joints.cpu().numpy())

    # TODO: get gradient plot on a fixed body position at goal config?
    # start reverse process
    sample_joints = torch.randn(bs, forecast_size * state_size * num_kps, device="cuda")
    sample_joints = sample_joints * fullbody_model.std + fullbody_model.mean
    timesteps = list(range(num_timesteps))[::-1]
    # sampled_goal = sample_goal[:1]
    sampled_wp = x0_sampled

    reverse_samples_joints = []
    reverse_grad = []
    sample_joints_abs = (
        sample_joints.view(bs, -1, num_kps, 3) + sampled_wp.view(1, -1, 1, 3)
    ).view(bs, -1)
    # reverse_samples_joints.append(sample_joints.cpu().numpy())
    reverse_samples_joints.append(sample_joints_abs.cpu().numpy())
    for i, t in enumerate(tqdm(timesteps)):
        t_grid = torch.tensor(
            np.repeat(t, len(xyz_body)), dtype=torch.long, device="cuda"
        )
        t = torch.tensor(
            np.repeat(t, len(sample_joints)), dtype=torch.long, device="cuda"
        )

        past_joints_grid = past_joints_sampled.repeat(len(xyz_body), 1)
        past_joints = past_joints_sampled.repeat(sample_joints.shape[0], 1)

        cam_grid = cam_sampled.repeat(len(xyz_body), 1)
        cam = cam_sampled.repeat(sample_joints.shape[0], 1)

        x0_to_world_grid = x0_to_world_sampled.repeat(len(xyz_body), 1)
        x0_to_world = x0_to_world_sampled.repeat(sample_joints.shape[0], 1)

        wp_grid = sampled_wp.repeat(len(xyz_body), 1)
        wp = sampled_wp.repeat(sample_joints.shape[0], 1)

        with torch.no_grad():
            feat = env_model.readout_in_world(
                feat_volume,
                sample_joints.clone().view(
                    sample_joints.shape[0], -1, state_size * num_kps
                ),
                x0_to_world[:, None] + wp.view(wp.shape[0], -1, 3),
                bg_field.voxel_grid.res,
                bg_field.voxel_grid.origin,
            )
            feat = feat.reshape(feat.shape[0], -1)

            feat_grid = env_model.readout_in_world(
                feat_volume,
                xyz_cuda_joints,
                x0_to_world_grid + wp_grid.view(wp_grid.shape[0], -1, 3)[:, 0],
                bg_field.voxel_grid.res,
                bg_field.voxel_grid.origin,
            )

            grad = fullbody_model(
                sample_joints,
                t[:, None] / num_timesteps,
                past_joints,
                cam,
                [feat],
                drop_cam=drop_cam,
                drop_past=drop_past,
                goal=wp,
            )

            xy_grad = fullbody_model(
                xyz_cuda_joints,
                t_grid[:, None] / num_timesteps,
                past_joints_grid,
                cam_grid,
                [feat_grid],
                drop_cam=drop_cam,
                drop_past=drop_past,
                goal=wp_grid,
            )
        sample_joints = noise_scheduler.step(
            grad, t[0], sample_joints, fullbody_model.std
        )
        sample_joints_abs = (
            sample_joints.view(bs, -1, num_kps, 3) + wp.view(bs, -1, 1, 3)
        ).view(bs, -1)
        # reverse_samples_joints.append(sample_joints.cpu().numpy())
        reverse_samples_joints.append(sample_joints_abs.cpu().numpy())
        reverse_grad.append(xy_grad.cpu().numpy())

    # use spline to interpolate the dense trajectory
    reverse_samples_joints_dense = []
    for sample_joints in reverse_samples_joints:
        sample_wp_dense = spline_interp(
            sample_joints, forecast_size=forecast_size, num_kps=num_kps
        )
        reverse_samples_joints_dense.append(sample_wp_dense)

    # full body visualization
    visualizer = DiffusionVisualizer(
        xzmax=xzmax_body,
        xzmin=xzmin_body,
        num_timesteps=num_timesteps,
        logdir="projects/tiny-diffusion/exps/%s/" % config.logname,
        prefix="joints",
    )
    visualizer.render_trajectory(
        forward_samples_joints,
        reverse_samples_joints_dense,
        past_joints,
        bg_field,
        x0_to_world,
    )
    visualizer.plot_trajectory_2d(
        forward_samples_joints,
        reverse_samples_joints,
        reverse_grad,
        x0_sampled[0, -state_size:],
        sample_idx,
        xyz_body,
        y_body,
        past,
        cam,
    )
    visualizer.delete()
