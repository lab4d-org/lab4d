import os, sys
import pdb
import math
import trimesh

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from einops import rearrange

import numpy as np

sys.path.insert(0, os.getcwd())
from projects.csim.voxelize import BGField

from utils import get_lab4d_data, TrajDataset, define_models_regress
from ddpm import NoiseScheduler
from config import get_config
from denoiser import TrajPredictor


if __name__ == "__main__":
    config = get_config()
    (
        x0,
        y,
        cam,
        x0_to_world,
        x0_joints,
        y_joints,
        x0_angles,
        past_angles,
        x0_angles_to_world,
        # ) = get_lab4d_data("database/motion/S26-train-L80-S1.pkl")
    ) = get_lab4d_data("database/motion/S26-train-L64-S1.pkl")

    # convert to bs -1
    x0 = x0.view(x0.shape[0], -1)
    y = y.view(y.shape[0], -1)
    cam = cam.view(cam.shape[0], -1)
    x0_to_world = x0_to_world.view(x0_to_world.shape[0], -1)
    x0_joints = x0_joints.view(x0_joints.shape[0], -1)
    y_joints = y_joints.view(y_joints.shape[0], -1)
    x0_angles = x0_angles.view(x0_angles.shape[0], -1)
    past_angles = past_angles.view(past_angles.shape[0], -1)
    x0_angles_to_world = x0_angles_to_world.view(x0_angles_to_world.shape[0], -1)

    dataset = TrajDataset(
        x0,
        y,
        cam,
        x0_to_world,
        x0_joints,
        y_joints,
        x0_angles,
        past_angles,
        x0_angles_to_world,
    )
    data_size = 20000  # 20k samples
    dataset = torch.utils.data.ConcatDataset([dataset] * (data_size // len(dataset)))

    # logging
    outdir = f"projects/tiny-diffusion/exps/{config.logname}"
    log = SummaryWriter(outdir, comment=config.logname)

    loader_train = DataLoader(
        dataset, batch_size=config.train_batch_size, shuffle=True, drop_last=True
    )
    loader_eval = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False)

    # model setup
    state_size = 3
    forecast_size = int(x0.shape[1] / state_size)
    num_kps = int(x0_joints.shape[1] / state_size / forecast_size)
    memory_size = int(y.shape[1] / state_size)
    print(f"state_size: {state_size}")
    print(f"forecast_size: {forecast_size}")
    print(f"num_kps: {num_kps}")
    print(f"memory_size: {memory_size}")
    # forecast_size = x0.shape[1]
    # num_kps = x0_joints.shape[2]
    mean = x0.mean(0)
    std = x0.std(0) * 3
    mean_joints = x0_joints.mean(0)
    std_joints = x0_joints.std(0)
    mean_angles = x0_angles.mean(0)
    std_angles = x0_angles.std(0)

    env_model, goal_model, waypoint_model, fullbody_model, angle_model = (
        define_models_regress(
            config,
            state_size,
            forecast_size,
            memory_size,
            num_kps,
            mean_goal=mean[-state_size:],
            std_goal=std[-state_size:],
            mean_wp=mean,
            std_wp=std,
            mean_joints=mean_joints,
            std_joints=std_joints,
            mean_angles=mean_angles,
            std_angles=std_angles,
            model=TrajPredictor,
            # use_env=False,
        )
    )
    env_model.cuda()
    goal_model = goal_model.cuda()
    waypoint_model = waypoint_model.cuda()
    fullbody_model = fullbody_model.cuda()
    angle_model = angle_model.cuda()

    voxel_grid = BGField().voxel_grid
    env_input = voxel_grid.data[None]
    env_input = torch.tensor(env_input, dtype=torch.float32, device="cuda")

    # optimization setup
    noise_scheduler = NoiseScheduler(num_timesteps=config.num_timesteps)
    noise_scheduler = noise_scheduler.cuda()

    params = (
        list(env_model.parameters())
        + list(goal_model.parameters())
        + list(waypoint_model.parameters())
        + list(fullbody_model.parameters())
        + list(angle_model.parameters())
    )
    optimizer = torch.optim.AdamW(
        params,
        lr=config.learning_rate,
    )
    total_steps = config.num_epochs * len(loader_train)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        config.learning_rate,
        total_steps,
        # pct_start=0.1,
        pct_start=1000 / total_steps,
        cycle_momentum=False,
        anneal_strategy="linear",
        # div_factor=25,
        # final_div_factor=1,
    )
    # scheduler = torch.optim.lr_scheduler.CyclicLR(
    #     optimizer,
    #     0.01 * config.learning_rate,
    #     config.learning_rate,
    #     step_size_up=1000,
    #     step_size_down=19000,
    #     mode="triangular",
    #     gamma=1.0,
    #     scale_mode="cycle",
    #     cycle_momentum=False,
    # )

    global_step = 0
    frames = []
    losses = []
    print("Training model...")
    for epoch in range(config.num_epochs):
        goal_model.train()
        progress_bar = tqdm(total=len(loader_train))
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in enumerate(loader_train):
            # get input data
            feat_volume = env_model.extract_features(env_input)
            clean = batch[0]
            past = batch[1]
            cam = batch[2]
            x0_to_world = batch[3]
            x0_joints = batch[4]
            past_joints = batch[5]
            x0_angles = batch[6]
            past_angles = batch[7]
            x0_angles_to_world = batch[8]

            ############ goal prediction
            clean_goal = clean[:, -state_size:]
            # get features
            if env_model.feat_dim > 0:
                feat = voxel_grid.readout_in_world(
                    feat_volume, torch.zeros_like(x0_to_world), x0_to_world
                )
                feat = [feat]
            else:
                feat = []
            # predict noise
            goal_pred = goal_model(past, cam, feat)
            loss_goal = F.mse_loss(goal_pred, clean_goal)
            ############################

            ############ waypoint prediction
            # get features
            if env_model.feat_dim > 0:
                feat = voxel_grid.readout_in_world(
                    feat_volume, torch.zeros_like(x0_to_world), x0_to_world
                )
                feat = [feat]
            else:
                feat = []
            wp_pred = waypoint_model(past, cam, feat, goal=clean_goal)
            loss_wp = F.mse_loss(wp_pred, clean)
            ############################

            ############ fullbody prediction
            # get features
            clean_ego = (
                x0_angles_to_world.view(-1, 1, 3, 3).transpose(2, 3)
                @ clean.view(-1, forecast_size, 3, 1)
            ).view(clean.shape)
            follow_wp = clean_ego
            # # N, T, K3 => N,T, K, F => N,TKF
            # feat = voxel_grid.readout_in_world(
            #     feat_volume,
            #     noisy_joints.view(noisy_joints.shape[0], -1, state_size * num_kps),
            #     x0_to_world[:, None] + follow_wp.view(follow_wp.shape[0], -1, 3),
            # )
            # feat = feat.reshape(feat.shape[0], -1)
            joints_pred = fullbody_model(past_joints, cam * 0, [], goal=follow_wp)
            loss_joints = F.mse_loss(joints_pred, x0_joints)
            ############################

            ############ angle prediction
            # get features
            follow_wp = clean_ego
            angles_pred = angle_model(past_angles, cam * 0, [], goal=follow_wp)
            loss_angles = F.mse_loss(angles_pred, x0_angles)
            ############################
            # ############ angle prediction
            # follow_wp = clean
            # angles_pred = angle_model(past_angles, follow_wp)
            # loss_angles = F.mse_loss(angles_pred, x0_angles) / angle_model.std.mean()
            # ############################

            loss = loss_goal + loss_wp + loss_joints + loss_angles
            loss.backward(loss)

            nn.utils.clip_grad_norm_(goal_model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "step": global_step}
            log.add_scalar("loss", loss, global_step)
            log.add_scalar("loss_goal", loss_goal, global_step)
            log.add_scalar("loss_wp", loss_wp, global_step)
            log.add_scalar("loss_joints", loss_joints, global_step)
            log.add_scalar("loss_angles", loss_angles, global_step)
            losses.append(loss.detach().item())
            progress_bar.set_postfix(**logs)
            global_step += 1
        progress_bar.close()

        # if epoch % config.save_images_step == 0 or epoch == config.num_epochs - 1:
        #     # generate data with the model to later visualize the learning process
        #     model.eval()
        #     x0, past, cam, x0_to_world = next(iter(loader_eval))
        #     sample = torch.randn(config.eval_batch_size, 3, device="cuda")
        #     past = past.repeat(config.eval_batch_size, 1)
        #     cam = cam.repeat(config.eval_batch_size, 1)
        #     x0_to_world = x0_to_world.repeat(config.eval_batch_size, 1)
        #     timesteps = list(range(len(noise_scheduler)))[::-1]
        #     for i, t in enumerate(tqdm(timesteps)):
        #         t = torch.tensor(
        #             np.repeat(t, config.eval_batch_size), device="cuda"
        #         ).long()
        #         t_frac = t[:, None] / noise_scheduler.num_timesteps
        #         with torch.no_grad():
        #             # feat = bg_field.compute_feat(sample + x0_to_world)
        #             feat = model.extract_env(
        #                 occupancy,
        #                 sample + x0_to_world,
        #                 bg_field.voxel_grid.res,
        #                 bg_field.voxel_grid.origin,
        #             )
        #             residual = model(sample, t_frac, past, cam, feat)
        #         sample = noise_scheduler.step(residual, t[0], sample)
        #     frames.append(sample.cpu().numpy())

        if epoch % config.save_model_epoch == 0 or epoch == config.num_epochs - 1:
            print("Saving model...")
            os.makedirs(outdir, exist_ok=True)
            param_path = f"{outdir}/env_model_%04d.pth" % epoch
            latest_path = f"{outdir}/env_model_latest.pth"
            torch.save(env_model.state_dict(), param_path)
            os.system("cp %s %s" % (param_path, latest_path))

            param_path = f"{outdir}/goal_model_%04d.pth" % epoch
            latest_path = f"{outdir}/goal_model_latest.pth"
            torch.save(goal_model.state_dict(), param_path)
            os.system("cp %s %s" % (param_path, latest_path))

            param_path = f"{outdir}/waypoint_model_%04d.pth" % epoch
            latest_path = f"{outdir}/waypoint_model_latest.pth"
            torch.save(waypoint_model.state_dict(), param_path)
            os.system("cp %s %s" % (param_path, latest_path))

            param_path = f"{outdir}/fullbody_model_%04d.pth" % epoch
            latest_path = f"{outdir}/fullbody_model_latest.pth"
            torch.save(fullbody_model.state_dict(), param_path)
            os.system("cp %s %s" % (param_path, latest_path))

            param_path = f"{outdir}/angle_model_%04d.pth" % epoch
            latest_path = f"{outdir}/angle_model_latest.pth"
            torch.save(angle_model.state_dict(), param_path)
            os.system("cp %s %s" % (param_path, latest_path))
