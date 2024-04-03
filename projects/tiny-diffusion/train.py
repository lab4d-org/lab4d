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

from utils import get_lab4d_data, TrajDataset, define_models
from ddpm import NoiseScheduler
from config import get_config
from denoiser import TotalDenoiserThreeStage


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
        # ) = get_lab4d_data("database/motion/S26-train-L240-S1.pkl")
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
    x0_angles_to_world = x0_angles_to_world.reshape(x0_angles_to_world.shape[0], -1)

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

    model = TotalDenoiserThreeStage(
        config,
        x0,
        x0_joints,
        x0_angles,
        y,
        # use_env=False,
    )
    model = model.cuda()

    # optimization setup
    noise_scheduler = NoiseScheduler(
        num_timesteps=config.num_timesteps, beta_schedule=config.beta_schedule
    )
    noise_scheduler = noise_scheduler.cuda()

    params = list(model.parameters())
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
        model.train()
        progress_bar = tqdm(total=len(loader_train))
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in enumerate(loader_train):
            # get data
            clean = batch[0]
            past = batch[1]
            cam = batch[2]
            x0_to_world = batch[3]
            x0_joints = batch[4]
            past_joints = batch[5]
            x0_angles = batch[6]
            past_angles = batch[7]
            x0_angles_to_world = batch[8].reshape(-1, 3, 3)

            # get context
            feat_volume = model.extract_feature_grid()

            # # combine
            # x0_to_world = (x0_to_world, x0_angles_to_world)

            # goal
            clean_goal = clean[:, -model.state_size :]
            noise_goal, noisy_goal, t_frac = noise_scheduler.sample_noise(
                clean_goal, std=model.goal_model.std
            )
            goal_delta = model.forward_goal(
                noisy_goal, x0_to_world, t_frac, past, cam, feat_volume
            )
            # path + fullbody v1
            # path
            noise_wp, noisy_wp, t_frac = noise_scheduler.sample_noise(
                clean, std=model.waypoint_model.std
            )
            wp_delta = model.forward_path(
                noisy_wp,
                x0_to_world,
                t_frac,
                past,
                cam,
                feat_volume,
                clean_goal=clean_goal,
            )

            # fullbody
            noise_joints, noisy_joints, t_frac = noise_scheduler.sample_noise(
                torch.cat([x0_angles, x0_joints], 1),
                std=torch.cat([model.angle_model.std, model.fullbody_model.std], 0),
            )
            noise_angles = noise_joints[..., : x0_angles.shape[-1]]
            noisy_angles = noisy_joints[..., : x0_angles.shape[-1]]
            noise_joints = noise_joints[..., x0_angles.shape[-1] :]
            noisy_joints = noisy_joints[..., x0_angles.shape[-1] :]

            clean_ego = (
                x0_angles_to_world.view(-1, 1, 3, 3).transpose(2, 3)
                @ clean.view(-1, model.forecast_size, 3, 1)
            ).view(clean.shape)
            # clean_ego = clean
            joints_delta, angles_delta = model.forward_fullbody(
                noisy_joints,
                noisy_angles,
                t_frac,
                past_joints,
                past_angles,
                cam,
                follow_wp=clean_ego,
            )

            loss_goal = F.mse_loss(goal_delta, noise_goal) / model.goal_model.std.mean()
            loss_wp = F.mse_loss(wp_delta, noise_wp) / model.waypoint_model.std.mean()
            loss_joints = (
                F.mse_loss(joints_delta, noise_joints) / model.fullbody_model.std.mean()
            )
            loss_angles = (
                F.mse_loss(angles_delta, noise_angles) / model.angle_model.std.mean()
            )

            # sum up
            loss = loss_goal + loss_wp + loss_joints + loss_angles
            loss.backward(loss)

            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
            param_path = f"{outdir}/ckpt_%04d.pth" % epoch
            latest_path = f"{outdir}/ckpt_latest.pth"
            torch.save(model.state_dict(), param_path)
            os.system("cp %s %s" % (param_path, latest_path))
