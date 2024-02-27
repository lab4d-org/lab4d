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

from utils import get_lab4d_data, TrajDataset, BGField, define_models
from ddpm import NoiseScheduler
from denoiser import readout_in_world
from config import get_config


if __name__ == "__main__":
    config = get_config()
    x0, y, cam, x0_to_world, x0_joints, y_joints = get_lab4d_data()
    dataset = TrajDataset(x0, y, cam, x0_to_world, x0_joints, y_joints)

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
    mean = x0.mean(0)
    std = x0.std(0) * 3
    mean_joints = x0_joints.mean(0)
    std_joints = x0_joints.std(0) * 3

    env_model, goal_model, waypoint_model, fullbody_model = define_models(
        config,
        state_size,
        forecast_size,
        num_kps,
        mean_goal=mean[-state_size:],
        std_goal=std[-state_size:],
        mean_wp=mean,
        std_wp=std,
        mean_joints=mean_joints,
        std_joints=std_joints,
    )

    env_model.cuda()
    goal_model = goal_model.cuda()
    waypoint_model = waypoint_model.cuda()
    fullbody_model = fullbody_model.cuda()

    bg_field = BGField()
    occupancy = bg_field.voxel_grid.data[None]
    occupancy = torch.tensor(occupancy, dtype=torch.float32, device="cuda")

    # optimization setup
    noise_scheduler = NoiseScheduler(
        num_timesteps=config.num_timesteps, beta_schedule=config.beta_schedule
    )
    noise_scheduler = noise_scheduler.cuda()

    params = (
        list(env_model.parameters())
        + list(goal_model.parameters())
        + list(waypoint_model.parameters())
        + list(fullbody_model.parameters())
    )
    optimizer = torch.optim.AdamW(
        params,
        lr=config.learning_rate,
    )

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        config.learning_rate,
        config.num_epochs * len(loader_train),
        pct_start=0.1,
        cycle_momentum=False,
        anneal_strategy="linear",
        div_factor=25,
        final_div_factor=1,
    )

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
            feat_volume = env_model.extract_features(occupancy)
            clean = batch[0]
            past = batch[1]
            cam = batch[2]
            x0_to_world = batch[3]
            x0_joints = batch[4]
            past_joints = batch[5]

            ############ goal prediction
            clean_goal = clean[:, -state_size:]
            noise_goal, noisy_goal, t_frac = noise_scheduler.sample_noise(
                clean_goal, std=goal_model.std
            )
            # get features
            feat = readout_in_world(
                feat_volume,
                noisy_goal,
                x0_to_world,
                bg_field.voxel_grid.res,
                bg_field.voxel_grid.origin,
            )
            # predict noise
            goal_delta = goal_model(noisy_goal, t_frac, past, cam, [feat])
            loss_goal = F.mse_loss(goal_delta, noise_goal) / goal_model.std.mean()
            ############################

            ############ waypoint prediction
            noise_wp, noisy_wp, t_frac = noise_scheduler.sample_noise(
                clean, std=waypoint_model.std
            )
            # get features
            feat = readout_in_world(
                feat_volume,
                noisy_wp,
                x0_to_world,
                bg_field.voxel_grid.res,
                bg_field.voxel_grid.origin,
            )
            wp_delta = waypoint_model(
                noisy_wp, t_frac, past, cam, [feat], goal=clean_goal
            )
            loss_wp = F.mse_loss(wp_delta, noise_wp) / waypoint_model.std.mean()
            ############################

            ############ fullbody prediction
            noise_joints, noisy_joints, t_frac = noise_scheduler.sample_noise(
                x0_joints, std=fullbody_model.std
            )
            # get features
            follow_wp = clean
            # N, T, K3 => N,T, K, F => N,TKF
            feat = readout_in_world(
                feat_volume,
                noisy_joints.view(noisy_joints.shape[0], -1, state_size * num_kps),
                x0_to_world[:, None] + follow_wp.view(follow_wp.shape[0], -1, 3),
                bg_field.voxel_grid.res,
                bg_field.voxel_grid.origin,
            )
            feat = feat.reshape(feat.shape[0], -1)
            joints_delta = fullbody_model(
                noisy_joints, t_frac, past_joints, cam, [feat], goal=follow_wp
            )
            loss_joints = (
                F.mse_loss(joints_delta, noise_joints) / fullbody_model.std.mean()
            )
            ############################

            loss = loss_goal + loss_wp + loss_joints
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
