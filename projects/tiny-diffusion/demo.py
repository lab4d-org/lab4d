import os, sys
import cv2
import pdb
import numpy as np
import torch
import trimesh
import argparse

import ddpm
from utils import get_lab4d_data, load_models, get_grid_xyz
from denoiser import reverse_diffusion, simulate_forward_diffusion
from eval import eval_ADE, eval_all
from visualizer import DiffusionVisualizer, spline_interp
from config import get_config

sys.path.insert(0, os.getcwd())
from lab4d.utils.quat_transform import axis_angle_to_matrix
from projects.csim.voxelize import BGField
from denoiser import TotalDenoiserThreeStage


if __name__ == "__main__":
    # params
    config = get_config()
    nsamp = config.eval_batch_size
    drop_cam = config.drop_cam
    drop_past = config.drop_past
    sample_idx = config.sample_idx
    state_size = 3
    num_timesteps = config.num_timesteps
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
        # ) = get_lab4d_data("database/motion/S26-test-L64-S10.pkl")
        # ) = get_lab4d_data("database/motion/S26-train-L240-S1.pkl")
    ) = get_lab4d_data("database/motion/S26-train-L64-S1.pkl")
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

    # model
    model = TotalDenoiserThreeStage(
        config,
        x0_wp,
        x0_joints,
        x0_angles,
        past_wp,
        # use_env=False,
    )
    model.load_ckpts(config)
    model = model.cuda()
    model.eval()
    feat_volume = model.extract_feature_grid()

    # old
    x0_goal = x0_goal_all[sample_idx : sample_idx + 1]
    x0_wp = x0_wp_all[sample_idx : sample_idx + 1]
    x0_joints = x0_joints_all[sample_idx : sample_idx + 1]
    x0_angles = x0_angles_all[sample_idx : sample_idx + 1]
    x0_to_world = x0_to_world_all[sample_idx : sample_idx + 1]
    x0_angles_to_world = x0_angles_to_world_all[sample_idx : sample_idx + 1]

    past_wp = past_wp_all[sample_idx : sample_idx + 1]
    past_joints = past_joints_all[sample_idx : sample_idx + 1]
    past_angles = past_angles_all[sample_idx : sample_idx + 1]
    cam = cam_all[sample_idx : sample_idx + 1]
    # # bs, T, K, 3

    bs = x0_wp.shape[0]
    # define grid to visualize the gradient
    xsize, ysize, zsize = 30, 5, 30
    xyz, xzmin, xzmax = get_grid_xyz(x0_wp, xsize, ysize, zsize)
    xyz_cuda_goal = torch.tensor(xyz, dtype=torch.float32, device="cuda")
    xyz_cuda_wp = (
        xyz_cuda_goal[:, None]
        .repeat(1, model.forecast_size, 1)
        .view(xyz_cuda_goal.shape[0], -1)
    )
    xsize_joints, ysize_joints, zsize_joints = 15, 5, 15
    xyz_joints, xzmin_joints, xzmax_joints = get_grid_xyz(
        x0_joints, xsize_joints, ysize_joints, zsize_joints
    )
    xyz_cuda_joints = torch.tensor(xyz_joints, dtype=torch.float32, device="cuda")
    xyz_cuda_joints = (
        xyz_cuda_joints[:, None]
        .repeat(1, model.forecast_size * model.num_kps, 1)
        .view(xyz_cuda_joints.shape[0], -1)
    )

    # goal
    goal = None
    xyz_grid = xyz_cuda_goal
    reverse_goal, reverse_grad_grid_goal = model.goal_model.reverse_diffusion(
        nsamp,
        num_timesteps,
        noise_scheduler,
        past_wp,
        cam,
        x0_to_world,
        # x0_angles_to_world,
        None,
        feat_volume,
        model.bg_field.voxel_grid,
        drop_cam,
        drop_past,
        goal,
        xyz_grid=xyz_grid[None],
    )

    # waypoint | goal conditioning
    # goal = torch.tensor(reverse_samples_goal[-1][:1], device="cuda")
    goal = x0_goal  # gt
    # goal = None
    xyz_grid = xyz_cuda_wp
    reverse_wp, reverse_grad_grid_wp = model.waypoint_model.reverse_diffusion(
        nsamp,
        num_timesteps,
        noise_scheduler,
        past_wp,
        cam,
        x0_to_world,
        # x0_angles_to_world,
        None,
        feat_volume,
        model.bg_field.voxel_grid,
        drop_cam,
        drop_past,
        goal,
        xyz_grid=xyz_grid[None],
    )
    # full body | wp conditioning
    # goal = torch.tensor(reverse_samples_wp[-1][:1], device="cuda")
    goal = x0_wp  # GT
    goal_ego = (x0_angles_to_world.transpose(3, 4) @ goal[..., None])[..., 0]
    xyz_grid = xyz_cuda_joints
    reverse_joints, reverse_grad_grid_joints = model.fullbody_model.reverse_diffusion(
        nsamp,
        num_timesteps,
        noise_scheduler,
        past_joints,
        cam * 0,
        None,
        None,
        None,
        None,
        drop_cam,
        drop_past,
        goal_ego,
        xyz_grid=xyz_grid[None],
    )
    # angle | wp conditioning
    reverse_angles, _ = model.angle_model.reverse_diffusion(
        nsamp,
        num_timesteps,
        noise_scheduler,
        past_angles,
        cam * 0,
        None,
        None,
        None,
        None,
        drop_cam,
        drop_past,
        goal_ego,
        None,
    )
    # _, _, t_frac = noise_scheduler.sample_noise(x0_angles_sampled, 0.1)
    # pdb.set_trace()
    # angles_pred = angle_model(
    #     torch.zeros_like(x0_angles_sampled),
    #     torch.zeros_like(t_frac),
    #     past_angles_sampled,
    #     cam_sampled,
    #     [],
    #     goal=goal,
    # )

    reverse_goal_all = reverse_goal.view(-1, bs, nsamp, 1, 1, 3)
    reverse_wp_all = reverse_wp.view(-1, bs, nsamp, model.forecast_size, 1, 3)
    reverse_angles_all = reverse_angles.view(-1, bs, nsamp, model.forecast_size, 1, 3)
    reverse_joints_all = reverse_joints.view(
        -1, bs, nsamp, model.forecast_size, model.num_kps, 3
    )
    # # TODO create axis
    # axis = trimesh.creation.axis(axis_length=0.1).vertices
    # axis = torch.tensor(axis, dtype=torch.float32, device="cuda").view(
    #     1, 1, 1, 1, -1, 3
    # )
    # reverse_joints_all = axis.repeat(
    #     reverse_joints_all.shape[0], bs, nsamp, forecast_size, 1, 1
    # )

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
        if reverse_grad_grid_goal is not None:
            reverse_grad_grid_goal = reverse_grad_grid_goal[:, i]
            reverse_grad_grid_wp = reverse_grad_grid_wp[:, i]
            reverse_grad_grid_joints = reverse_grad_grid_joints[:, i]

        reverse_goal = reverse_goal_all[:, i]
        reverse_wp = reverse_wp_all[:, i]
        reverse_angles = reverse_angles_all[:, i]
        reverse_joints = reverse_joints_all[:, i]
        reverse_joints_abs = reverse_joints_abs_all[:, i]
        ############# visualization
        # goal visualization
        save_prefix = "goal-%d" % i
        # forward process
        forward_samples_goal = model.goal_model.simulate_forward_diffusion(
            x0_goal_all, noise_scheduler
        )
        # forward_samples_goal = []
        visualizer = DiffusionVisualizer(
            xzmax=xzmax,
            xzmin=xzmin,
            num_timesteps=num_timesteps,
            logdir="projects/tiny-diffusion/exps/%s/" % config.logname,
            bg_field=model.bg_field,
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
            reverse_grad_grid_goal,
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
        forward_samples_waypoint = model.waypoint_model.simulate_forward_diffusion(
            x0_wp_all, noise_scheduler
        )
        # forward_samples_waypoint = []
        visualizer = DiffusionVisualizer(
            xzmax=xzmax,
            xzmin=xzmin,
            num_timesteps=num_timesteps,
            logdir="projects/tiny-diffusion/exps/%s/" % config.logname,
            bg_field=model.bg_field,
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
            reverse_grad_grid_wp,
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
            bg_field=model.bg_field,
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
            reverse_grad_grid_joints,
            x0_joints[0],
            xyz_joints,
            ysize_joints,
            past_joints[0],
            cam[0],
            prefix=save_prefix,
        )
        visualizer.delete()

        out_path = "projects/tiny-diffusion/exps/%s/" % config.logname
        for idx, joints in enumerate(reverse_joints_all[-1, 0]):
            wp = x0_to_world[0] + reverse_wp_all[-1, 0, idx]
            sample = torch.cat(
                [reverse_angles_all[-1, 0, idx], wp, joints],
                dim=-2,
            )
            sample = sample.reshape(sample.shape[0], -1).cpu().numpy()
            # T,81
            if not os.path.exists("%s/sample_%03d" % (out_path, idx)):
                os.mkdir("%s/sample_%03d" % (out_path, idx))
            np.save("%s/sample_%03d/sample.npy" % (out_path, idx), sample)

        # run the command
        ckpt_path = "logdir-12-05/home-2023-11-11--11-51-53-compose/"
        bash_cmd = f"'cd ../vid2sim/; source ~/miniconda3/etc/profile.d/conda.sh; conda activate lab4d; python projects/behavior/vis.py --gendir {out_path} --logdir {ckpt_path} --fps 30'"
        bash_cmd = f"/bin/bash -c {bash_cmd}"
        print(bash_cmd)
        os.system(bash_cmd)
        print("results are at %s" % out_path)
