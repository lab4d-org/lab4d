import os, sys
import cv2
import pdb
import numpy as np
import torch
import trimesh

import ddpm
from utils import get_lab4d_data, load_models, get_grid_xyz, get_xzbounds
from denoiser import reverse_diffusion, simulate_forward_diffusion
from visualizer import DiffusionVisualizer, spline_interp
from config import get_config

sys.path.insert(0, os.getcwd())
from lab4d.utils.quat_transform import axis_angle_to_matrix, matrix_to_axis_angle
from projects.csim.voxelize import BGField
from denoiser import TotalDenoiserThreeStage, TotalDenoiserTwoStage


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
        x0,
        past,
        cam,
        x0_to_world,
        x0_joints,
        past_joints,
        x0_angles,
        past_angles,
        x0_angles_to_world,
        # ) = get_lab4d_data("database/motion/S26-test-L64-S10.pkl")
        # ) = get_lab4d_data("database/motion/S26-test-L80-S10.pkl")
    ) = get_lab4d_data("database/motion/S26-train-L64-S1.pkl")
    # ) = get_lab4d_data("database/motion/S26-train-L80-S1.pkl")

    # model
    # model = TotalDenoiserThreeStage(
    #     config,
    #     x0,
    #     x0_joints,
    #     x0_angles,
    #     past,
    #     # use_env=False,
    # )
    model = TotalDenoiserTwoStage(
        config,
        x0,
        x0_joints,
        x0_angles,
        past,
    )
    model.load_ckpts(config)
    model = model.cuda()
    model.eval()
    feat_volume = model.extract_feature_grid()

    # data
    x0_wp = x0[sample_idx]
    past = past[sample_idx]
    cam = cam[sample_idx]
    x0_to_world = x0_to_world[sample_idx]
    x0_joints = x0_joints[sample_idx]
    past_joints = past_joints[sample_idx]
    x0_angles = x0_angles[sample_idx]
    past_angles = past_angles[sample_idx]
    x0_angles_to_world = x0_angles_to_world[sample_idx]
    # T, K, 3

    # visualize
    xzmin, xzmax, _, _ = get_xzbounds(x0)
    visualizer = DiffusionVisualizer(
        xzmax=xzmax,
        xzmin=xzmin,
        num_timesteps=num_timesteps,
        bg_field=model.bg_field,
        logdir="projects/tiny-diffusion/exps/%s/" % config.logname,
        lab4d_dir="logdir/home-2023-curated3-compose-ft/",
    )
    visualizer.run_viser()
    hit_list = torch.stack([cam[0, 0], cam[-1, 0]], 0)  # 2,3
    hit_list = hit_list + x0_to_world[:, 0]
    visualizer.userwp_list = hit_list.cpu().numpy().tolist()
    visualizer.show_control_points()

    accumulated_traj = past.clone()  # T',1, 3 in the latest ego coordinate
    current_joints = x0_joints[-1:]  # 1, K, 3
    # for sample_idx in range(100):
    while True:
        if not drop_cam:
            pdb.set_trace()
        # else:
        #     pdb.set_trace()
        # interpret user-specified hit points as camera path
        if len(visualizer.userwp_list) > 1:
            # interpolate t_-1 and t_-2
            cam = torch.tensor(
                visualizer.userwp_list, device="cuda", dtype=torch.float32
            )
            cam = spline_interp(cam.view(1, -1), 2, interp_size=model.memory_size)
            cam = (cam.view(-1, 3) - x0_to_world).view(-1, 1, 3)

        if len(visualizer.goal_list) > 0:
            selected_goal = torch.tensor(
                visualizer.goal_list, device="cuda", dtype=torch.float32
            )
            # delete frames/goal
            selected_goal = selected_goal - x0_to_world[0]
            selected_goal = selected_goal.view(1, 3)
            reverse_goal = selected_goal[None, None, None]
        else:
            # goal
            goal = None
            xyz_grid = None
            reverse_goal, _ = model.goal_model.reverse_diffusion(
                nsamp,
                num_timesteps,
                noise_scheduler,
                past[None],
                cam[None],
                x0_to_world[None],
                feat_volume,
                model.bg_field.voxel_grid,
                drop_cam,
                drop_past,
                goal,
                xyz_grid=xyz_grid,
            )
            reverse_goal = reverse_goal.view(-1, nsamp, 1, 1, 3)

            # waypoint | goal conditioning
            goal_samples = reverse_goal[-1, :, 0, 0]  # bs,3

            # filter goals that has nevel been visited
            goal_score = model.bg_field.voxel_grid.readout_voxel(
                goal_samples + x0_to_world[0, 0], mode="root_visitation"
            )
            goal_samples = goal_samples[goal_score > 0]
            # best_idx = goal_samples.norm(dim=-1).argmax()  # find the furthest goal
            best_idx = 0
            selected_goal = goal_samples[best_idx : best_idx + 1]

        # full body | goal conditioning
        reverse_wp, _ = model.fullbody_model.reverse_diffusion(
            nsamp,
            num_timesteps,
            noise_scheduler,
            past[None],
            cam[None],
            x0_to_world[None],
            feat_volume,
            model.bg_field.voxel_grid,
            drop_cam,
            drop_past,
            selected_goal[None],
            xyz_grid=xyz_grid,
            denoise_angles=True,
        )
        reverse_angles = reverse_wp[1]
        reverse_wp = reverse_wp[0]
        reverse_wp = reverse_wp.view(-1, nsamp, model.forecast_size, 1, 3)
        reverse_joints = reverse_angles[
            ..., model.fullbody_model.state_size * model.fullbody_model.forecast_size :
        ]
        reverse_angles = reverse_angles[
            ...,
            : model.fullbody_model.state_size * model.fullbody_model.forecast_size,
        ]
        reverse_joints = (
            reverse_joints.view(-1, nsamp, model.forecast_size, model.num_kps, 3) * 0
        )
        reverse_angles = reverse_angles.view(-1, nsamp, model.forecast_size, 1, 3)

        # # waypoint | goal conditioning
        # reverse_wp, _ = model.waypoint_model.reverse_diffusion(
        #     nsamp,
        #     num_timesteps,
        #     noise_scheduler,
        #     past[None],
        #     cam[None],
        #     x0_to_world[None],
        #     feat_volume,
        #     model.bg_field.voxel_grid,
        #     drop_cam,
        #     drop_past,
        #     selected_goal[None],
        #     xyz_grid=xyz_grid,
        # )
        # reverse_wp = reverse_wp.view(-1, nsamp, model.forecast_size, 1, 3)

        # # full body | wp conditioning
        # wp_samples = reverse_wp[-1, 0]  # T,1,3
        # # to ego
        # goal_wp = x0_angles_to_world.transpose(2, 3) @ wp_samples[..., None]
        # goal_wp = goal_wp[..., 0]
        # # reversed_goal = x0_sampled # GT
        # reverse_joints, _ = model.fullbody_model.reverse_diffusion(
        #     nsamp,
        #     num_timesteps,
        #     noise_scheduler,
        #     past_joints[None],
        #     cam[None] * 0,
        #     None,
        #     None,
        #     None,
        #     drop_cam,
        #     drop_past,
        #     goal_wp[None],
        #     xyz_grid=xyz_grid,
        # )
        # # angle | wp conditioning
        # reverse_angles, _ = model.angle_model.reverse_diffusion(
        #     nsamp,
        #     num_timesteps,
        #     noise_scheduler,
        #     past_angles[None],
        #     cam[None] * 0,
        #     None,
        #     None,
        #     None,
        #     drop_cam,
        #     drop_past,
        #     goal_wp[None],
        #     xyz_grid=xyz_grid,
        # )
        # reverse_joints = reverse_joints.view(
        #     -1, nsamp, model.forecast_size, model.num_kps, 3
        # )
        # reverse_angles = reverse_angles.view(-1, nsamp, model.forecast_size, 1, 3)

        # spline | wp
        reverse_wp_dense = []
        for noisy_wp in reverse_wp:
            noisy_wp = torch.cat([torch.zeros_like(noisy_wp[:, :1]), noisy_wp], 1)
            noisy_wp = noisy_wp.view(nsamp, -1)
            sample_wp_dense = spline_interp(
                noisy_wp, forecast_size=model.forecast_size + 1
            )
            reverse_wp_dense.append(sample_wp_dense)
        reverse_wp_dense = torch.stack(reverse_wp_dense, 0)
        reverse_wp_dense = reverse_wp_dense.view(
            reverse_wp_dense.shape[0], nsamp, -1, 1, 3
        )
        # spline | joints
        reverse_joints_dense = []
        for i, noisy_joints in enumerate(reverse_joints):
            noisy_joints = torch.cat(
                [current_joints[None].repeat(nsamp, 1, 1, 1), noisy_joints], 1
            )
            noisy_joints = noisy_joints.view(nsamp, -1)
            noisy_joints_dense = spline_interp(
                noisy_joints,
                forecast_size=model.forecast_size + 1,
                num_kps=model.num_kps,
            )
            reverse_joints_dense.append(noisy_joints_dense)
        reverse_joints_dense = torch.stack(reverse_joints_dense, 0)
        reverse_joints_dense = reverse_joints_dense.view(
            reverse_joints_dense.shape[0], nsamp, -1, model.num_kps, 3
        )
        # spline | angles
        reverse_angles_dense = []
        for i, noisy_angles in enumerate(reverse_angles):
            noisy_angles = torch.cat(
                [torch.zeros_like(noisy_angles[:, :1]), noisy_angles], 1
            )
            noisy_angles = noisy_angles.view(nsamp, -1)
            sample_angles_dense = spline_interp(
                noisy_angles, forecast_size=model.forecast_size + 1
            )
            reverse_angles_dense.append(sample_angles_dense)
        reverse_angles_dense = torch.stack(reverse_angles_dense, 0)
        reverse_angles_dense = reverse_angles_dense.view(
            reverse_angles_dense.shape[0], nsamp, -1, 1, 3
        )

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
        # # joints visualization
        # visualizer.render_trajectory(
        #     [],
        #     reverse_samples_joints_abs[:, :1],
        #     accumulated_joints.view(1, -1),
        #     x0_to_world_sampled,
        #     rotate=False,
        #     prefix="joints-%03d" % sample_idx,
        # )

        angles_world_dense = matrix_to_axis_angle(
            x0_angles_to_world @ axis_angle_to_matrix(reverse_angles_dense[-1, 0])
        )
        # joints_abs_dense = (
        #     x0_angles_to_world
        #     @ axis_angle_to_matrix(reverse_angles_dense[-1, 0])
        #     @ reverse_joints_dense[-1, 0][..., None]
        # )[..., 0] + reverse_wp_dense[-1, 0]

        visualizer.render_trajectory_viser(
            reverse_goal[-1].view(-1, 3),  # goal
            reverse_wp_dense[-1].view(nsamp, -1, 3),  # waypoints
            accumulated_traj.view(-1, 3),  # past trajectory
            # joints_abs_dense.view(-1, num_kps, 3),  # joints
            reverse_joints_dense[-1, 0].view(-1, model.num_kps, 3),  # joint angles
            angles_world_dense.view(-1, 1, 3),  # angles
            x0_to_world,
        )

        # update past
        pred_traj = reverse_wp_dense[-1, 0]
        accumulated_traj = torch.cat([accumulated_traj, pred_traj], 0)
        accumulated_traj = accumulated_traj - pred_traj[-1:]
        past = accumulated_traj[-model.memory_size - 1 : -1]
        # update past_joints, past_angles
        past_joints = reverse_joints_dense[-1, 0][-model.memory_size - 1 : -1]
        current_joints = reverse_joints_dense[-1, 0][-1:]
        pred_angles = reverse_angles_dense[-1, 0]
        past_angles = matrix_to_axis_angle(
            axis_angle_to_matrix(pred_angles[-1:]).transpose(2, 3)
            @ axis_angle_to_matrix(pred_angles[-model.memory_size - 1 : -1])
        )
        # update x0_to_world, x0_angles_to_world
        x0_to_world = x0_to_world + pred_traj[-1:]
        x0_angles_to_world = x0_angles_to_world @ axis_angle_to_matrix(pred_angles[-1:])
        # rectify it
        x0_angles_to_world = matrix_to_axis_angle(x0_angles_to_world)
        x0_angles_to_world[..., [0, 2]] *= 0
        x0_angles_to_world = axis_angle_to_matrix(x0_angles_to_world)
        print(x0_angles_to_world)
visualizer.delete()
