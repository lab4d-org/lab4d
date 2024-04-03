import pdb
import os, sys
import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm.auto import tqdm
from einops import rearrange

sys.path.insert(0, os.getcwd())
from lab4d.nnutils.embedding import PosEmbedding
from lab4d.utils.quat_transform import axis_angle_to_matrix, matrix_to_axis_angle
from lab4d.nnutils.base import BaseMLP
from projects.csim.voxelize import BGField

from arch import TemporalUnet, TransformerPredictor


def simulate_forward_diffusion(x0, noise_scheduler, mean, std):
    forward_samples_goal = []
    forward_samples_goal.append(x0.cpu().numpy())
    for t in range(len(noise_scheduler)):
        timesteps = torch.tensor(np.repeat(t, len(x0)), dtype=torch.long, device="cuda")
        noise_goal = torch.randn_like(x0, device="cuda")
        noise_goal = noise_goal * std + mean
        sample_goal = noise_scheduler.add_noise(x0, noise_goal, timesteps)
        forward_samples_goal.append(sample_goal.cpu().numpy())
    return forward_samples_goal


def reverse_diffusion(
    sample_x0,
    num_timesteps,
    model,
    noise_scheduler,
    past,
    cam,
    x0_to_world,
    x0_angles_to_world,
    feat_volume,
    voxel_grid,
    drop_cam,
    drop_past,
    track_x0=True,
    goal=None,
    noisy_angles=None,
):
    """
    sample_x0: 1, nsamp,-1
    past: bs, 1,-1

    return: reverse_samples: num_timesteps+1, bs, nsamp, -1
    """
    timesteps = list(range(num_timesteps))[::-1]
    nsamp = sample_x0.shape[1]
    bs = past.shape[0]

    sample_x0 = sample_x0.repeat(bs, 1, 1).view(bs * nsamp, -1)
    if noisy_angles is not None:
        noisy_angles = noisy_angles.repeat(bs, 1, 1).view(bs * nsamp, -1)
        reverse_angles = [noisy_angles]

    past = past.repeat(1, nsamp, 1).view(bs * nsamp, -1)
    cam = cam.repeat(1, nsamp, 1).view(bs * nsamp, -1)
    if x0_to_world is not None:
        x0_to_world = x0_to_world.repeat(1, nsamp, 1).view(bs * nsamp, 3)
    if x0_angles_to_world is not None:
        x0_angles_to_world = x0_angles_to_world.repeat(1, nsamp, 1, 1).view(
            bs * nsamp, 3, 3
        )
        x0_to_world = (x0_to_world, x0_angles_to_world)
    if goal is not None:
        goal = goal.repeat(1, nsamp, 1).view(bs * nsamp, -1)

    reverse_samples = [sample_x0]
    reverse_grad = []
    for i, t in enumerate(tqdm(timesteps)):
        t = torch.tensor(np.repeat(t, len(sample_x0)), dtype=torch.long, device="cuda")
        with torch.no_grad():
            ###### goal
            # 3D convs then query, # B1HWD => B3HWD
            if feat_volume is None:
                feat = []
            else:
                feat = voxel_grid.readout_in_world(feat_volume, sample_x0, x0_to_world)
                feat = [feat]
            grad_cond = model(
                sample_x0,
                t[:, None] / num_timesteps,
                past,
                cam,
                feat,
                drop_cam=drop_cam,
                drop_past=drop_past,
                goal=goal,
                noisy_angles=noisy_angles,
            )
            grad_uncond = model(
                sample_x0,
                t[:, None] / num_timesteps,
                past,
                cam,
                feat,
                drop_cam=True,
                drop_past=True,
                goal=goal,
                noisy_angles=noisy_angles,
            )

            cfg_score = 2.5
            grad = cfg_score * (grad_cond - grad_uncond) + grad_uncond
            # grad = grad_cond

        if track_x0:
            if noisy_angles is not None:
                sample_x0 = torch.cat([sample_x0, noisy_angles], dim=-1)
                std = torch.cat([model.std, model.std_angle])
            else:
                std = model.std
            sample_x0 = noise_scheduler.step(grad, t[0], sample_x0, std)
            if noisy_angles is not None:
                noisy_angles = sample_x0[..., -noisy_angles.shape[-1] :]
                sample_x0 = sample_x0[..., : -noisy_angles.shape[-1]]
                reverse_angles.append(noisy_angles)
        else:
            if noisy_angles is not None:
                reverse_angles.append(noisy_angles)
        reverse_samples.append(sample_x0)
        reverse_grad.append(grad)
    reverse_samples = torch.stack(reverse_samples, 0)
    reverse_grad = torch.stack(reverse_grad, 0)
    reverse_samples = reverse_samples.view(num_timesteps + 1, bs, nsamp, -1)
    reverse_grad = reverse_grad.view(num_timesteps, bs, nsamp, -1)

    if noisy_angles is not None:
        reverse_angles = torch.stack(reverse_angles, 0)
        reverse_angles = reverse_angles.view(num_timesteps + 1, bs, nsamp, -1)
        reverse_samples = (reverse_samples, reverse_angles)
    return reverse_samples, reverse_grad


class EnvEncoder(nn.Module):
    def __init__(self, in_dim=1, feat_dim=384):
        super().__init__()
        self.unet_3d = UNet3D(in_dim, feat_dim)
        self.feat_dim = feat_dim

    def extract_features(self, occupancy):
        """
        x_world: N,3
        """
        # 3D convs then query B1HWD => B3HWD
        feature_vol = self.unet_3d(occupancy[None])[0]
        return feature_vol


class TrajPredictor(nn.Module):
    def __init__(
        self,
        mean=None,
        std=None,
        hidden_size: int = 128,
        hidden_layers: int = 3,
        feat_dim: list = [],
        condition_dim: int = 64,
        N_freq: int = 8,
        memory_size: int = 8,
        forecast_size: int = 1,
        state_size: int = 3,
        camera_state_size: int = 3,
        cond_size: int = 0,
        kp_size: int = 1,
        is_angle: bool = False,
    ):
        # store mean and std as buffers
        super().__init__()
        if mean is None:
            mean = torch.zeros(state_size * forecast_size * kp_size)
            print("Warning: mean not provided. Make sure to load from ckpts.")
        if std is None:
            std = torch.ones(state_size * forecast_size * kp_size)
            print("Warning: std not provided. Make sure to load from ckpts.")
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.forecast_size = forecast_size
        self.state_size = state_size
        self.kp_size = kp_size
        self.is_angle = is_angle

        # state embedding
        time_embed = PosEmbedding(1, N_freq)
        self.time_embed = nn.Sequential(
            time_embed, nn.Linear(time_embed.out_channels, condition_dim)
        )
        # latent_embed = PosEmbedding(state_size * forecast_size * kp_size, N_freq)
        latent_embed = PosEmbedding(state_size * kp_size, N_freq)
        self.latent_embed = nn.Sequential(
            latent_embed, nn.Linear(latent_embed.out_channels, condition_dim)
        )

        # condition embedding
        past_mlp = PosEmbedding(state_size * memory_size * kp_size, N_freq)
        self.past_mlp = nn.Sequential(
            past_mlp, nn.Linear(past_mlp.out_channels, condition_dim)
        )
        cam_mlp = PosEmbedding(camera_state_size * memory_size, N_freq)
        self.cam_mlp = nn.Sequential(
            cam_mlp, nn.Linear(cam_mlp.out_channels, condition_dim)
        )

        if np.sum(feat_dim) == 0:
            feat_dim = []
        if len(feat_dim) > 0:
            feat_proj = []
            for feat_dim_sub in feat_dim:
                feat_proj.append(nn.Linear(feat_dim_sub * kp_size, condition_dim))
            self.feat_proj = nn.ParameterList(feat_proj)

        # goal cond
        if cond_size > 0:
            cond_embed = PosEmbedding(cond_size, N_freq)
            self.cond_embed = nn.Sequential(
                cond_embed, nn.Linear(cond_embed.out_channels, condition_dim)
            )

        in_channels = condition_dim * (1 + 1 + len(feat_dim) + int(cond_size > 0))
        # # prediction heads
        # self.pred_head = self.define_head(
        #     in_channels,
        #     hidden_size,
        #     hidden_layers,
        #     forecast_size,
        #     state_size,
        #     kp_size,
        # )
        self.pred_head = BaseMLP(
            D=hidden_layers,
            W=hidden_size,
            in_channels=in_channels,
            out_channels=forecast_size * state_size * kp_size,
            skips=[4],
            activation=nn.ReLU(True),
            final_act=False,
        )
        # self.proj = nn.Linear(in_channels, hidden_size)
        # seqTransEncoderLayer = nn.TransformerEncoderLayer(
        #     d_model=hidden_size, nhead=4, batch_first=True
        # )
        # self.encoder = nn.TransformerEncoder(
        #     seqTransEncoderLayer,
        #     num_layers=hidden_layers,
        # )
        # self.pred_head = nn.Linear(hidden_size, state_size * kp_size)

    def define_head(
        self,
        concat_size,
        hidden_size,
        hidden_layers,
        forecast_size,
        state_size,
        kp_size,
    ):
        layers = [nn.Linear(concat_size, hidden_size), nn.GELU()]
        for _ in range(hidden_layers):
            layers.append(Block(hidden_size))
        layers.append(nn.Linear(hidden_size, state_size * forecast_size * kp_size))
        layers = nn.Sequential(*layers)
        return layers

    def forward(
        self,
        past,
        cam,
        feat,
        drop_cam=False,
        drop_past=False,
        goal=None,
    ):
        """
        past: N, M*3
        cam: N, M*3
        """
        bs = past.shape[0]
        device = past.device
        past = past.clone()
        cam = cam.clone()
        if goal is not None:
            goal = goal.clone()

        # rotaion augmentation
        if self.training and not self.is_angle:
            rand_roty = torch.rand(bs, 1, device=device) * 2 * np.pi
            rotxyz = torch.zeros(bs, 3, device=device)
            rotxyz[:, 1] = rand_roty[:, 0]
            rotmat = axis_angle_to_matrix(rotxyz)  # N, 3, 3

            past = past.view(past.shape[:-1] + (-1, 3))
            past = (rotmat[:, None] @ past[..., None]).squeeze(-1)
            past = past.view(past.shape[:-2] + (-1,))
            # # add random noise
            # past = past + torch.randn_like(past, device=device) * 0.1

            cam = cam.view(cam.shape[:-1] + (-1, 3))
            cam = (rotmat[:, None] @ cam[..., None]).squeeze(-1)
            cam = cam.view(cam.shape[:-2] + (-1,))
            # # add random noise
            # cam = cam + torch.randn_like(cam, device=device) * 0.1

            if goal is not None:
                goal = goal.view(goal.shape[:-1] + (-1, 3))
                goal = (rotmat[:, None] @ goal[..., None]).squeeze(-1)
                goal = goal.view(goal.shape[:-2] + (-1,))
                # # add random noise
                # goal = goal + torch.randn_like(goal, device=device) * 0.1

        # condition embedding
        past_emb = self.past_mlp(past.view(past.shape[0], -1))
        cam_emb = self.cam_mlp(cam.view(cam.shape[0], -1))

        # merge the embeddings
        emb = torch.cat((past_emb, cam_emb), dim=-1)

        if hasattr(self, "feat_proj"):
            feat_emb = []
            for i in range(len(self.feat_proj)):
                feat_sub = feat[i].reshape(feat[i].shape[0], -1)
                # # add noise to feat
                # if self.training:
                #     batchstd = feat_sub.reshape(-1).std().detach()
                #     feat_sub = (
                #         feat_sub
                #         + torch.randn_like(feat_sub, device=device) * batchstd * 0.1
                #     )
                feat_emb_sub = self.feat_proj[i](feat_sub)
                feat_emb.append(feat_emb_sub)
            feat_emb = torch.cat(feat_emb, dim=-1)

            emb = torch.cat((emb, feat_emb), dim=-1)

        if hasattr(self, "cond_embed"):
            if goal is not None:
                cond_emb = self.cond_embed(goal)

                # if self.training:
                #     drop_rate = 0.5
                #     rand_mask = (
                #         torch.rand(noisy.shape[0], 1, device=noisy.device) > drop_rate
                #     ).float()
                #     cond_emb = cond_emb * rand_mask
            else:
                cond_emb = self.cond_embed(
                    torch.zeros(bs, self.cond_embed[0].in_channels, device=device)
                )

            emb = torch.cat((emb, cond_emb), dim=-1)

        # prediction: N,F => N,T*K*3
        delta = self.pred_head(emb)

        delta = delta * self.std + self.mean

        # undo rotation augmentation
        if self.training and not self.is_angle:
            rotmat = axis_angle_to_matrix(-rotxyz)
            delta = delta.view(delta.shape[:-1] + (-1, 3))
            delta = (rotmat[:, None] @ delta[..., None]).squeeze(-1)  # same for so3
            delta = delta.view(delta.shape[:-2] + (-1,))
        return delta

    @torch.no_grad()
    def predict(
        self,
        past,
        cam,
        x0_to_world,
        feat_volume,
        voxel_grid,
        goal=None,
    ):
        """
        sample_x0: 1, nsamp,-1
        past: bs, 1,-1

        return: reverse_samples: num_timesteps+1, bs, nsamp, -1
        """
        bs = past.shape[0]
        past = past.view(bs, -1)
        cam = cam.view(bs, -1)
        if x0_to_world is not None:
            x0_to_world = x0_to_world.view(bs, -1)
        if goal is not None:
            goal = goal.view(bs, -1)

        if feat_volume is None:
            feat = []
        else:
            feat = voxel_grid.readout_in_world(
                feat_volume, torch.zeros_like(x0_to_world), x0_to_world
            )
            feat = [feat]
        pred = self.forward(
            past,
            cam,
            feat,
            goal=goal,
        )
        return pred


class TrajDenoiser(nn.Module):
    def __init__(
        self,
        mean=None,
        std=None,
        hidden_size: int = 128,
        hidden_layers: int = 3,
        feat_dim: list = [],
        condition_dim: int = 64,
        N_freq: int = 8,
        memory_size: int = 8,
        forecast_size: int = 1,
        state_size: int = 3,
        camera_state_size: int = 3,
        cond_size: int = 0,
        kp_size: int = 1,
        is_angle: bool = False,
        angle_size: int = 0,
        mean_angle=None,
        std_angle=None,
    ):
        # store mean and std as buffers
        super().__init__()
        if mean is None:
            mean = torch.zeros(state_size * forecast_size * kp_size)
            print("Warning: mean not provided. Make sure to load from ckpts.")
        if std is None:
            std = torch.ones(state_size * forecast_size * kp_size)
            print("Warning: std not provided. Make sure to load from ckpts.")

        if mean_angle is None:
            mean_angle = torch.zeros(state_size * angle_size * kp_size)
            print("Warning: mean_angle not provided. Make sure to load from ckpts.")
        if std_angle is None:
            std_angle = torch.ones(state_size * angle_size * kp_size)
            print("Warning: std_angle not provided. Make sure to load from ckpts.")

        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.register_buffer("mean_angle", mean_angle)
        self.register_buffer("std_angle", std_angle)
        self.forecast_size = forecast_size
        self.state_size = state_size
        self.kp_size = kp_size
        self.is_angle = is_angle

        # state embedding
        time_embed = PosEmbedding(1, N_freq)
        self.time_embed = nn.Sequential(
            time_embed, nn.Linear(time_embed.out_channels, condition_dim)
        )

        # condition embedding
        past_mlp = PosEmbedding(state_size * memory_size * kp_size, N_freq)
        self.past_mlp = nn.Sequential(
            past_mlp, nn.Linear(past_mlp.out_channels, condition_dim)
        )
        cam_mlp = PosEmbedding(camera_state_size * memory_size, N_freq)
        self.cam_mlp = nn.Sequential(
            cam_mlp, nn.Linear(cam_mlp.out_channels, condition_dim)
        )

        if np.sum(feat_dim) == 0:
            feat_dim = []
        if len(feat_dim) > 0:
            feat_proj = []
            for feat_dim_sub in feat_dim:
                feat_proj.append(nn.Linear(feat_dim_sub * kp_size, condition_dim))
            self.feat_proj = nn.ParameterList(feat_proj)

        # goal cond
        if cond_size > 0:
            cond_embed = PosEmbedding(cond_size, N_freq)
            self.cond_embed = nn.Sequential(
                cond_embed, nn.Linear(cond_embed.out_channels, condition_dim)
            )

        in_channels = condition_dim * (3 + 1 + len(feat_dim) + int(cond_size > 0))

        # # prediction heads
        # self.pred_head = self.define_head(
        #     in_channels,
        #     hidden_size,
        #     hidden_layers,
        #     forecast_size,
        #     state_size,
        #     kp_size,
        # )
        # self.pred_head = BaseMLP(
        #     D=hidden_layers,
        #     W=hidden_size,
        #     in_channels=in_channels,
        #     out_channels=forecast_size * state_size * kp_size,
        #     skips=[4],
        #     activation=nn.ReLU(True),
        #     final_act=False,
        # )

        # if forecast_size == 1:
        if True:
            self.pred_head = TransformerPredictor(
                in_channels,
                hidden_size,
                hidden_layers,
                state_size,
                kp_size,
                condition_dim,
                N_freq=N_freq,
            )
        else:
            self.pred_head = TemporalUnet(
                input_dim=state_size * kp_size, cond_dim=in_channels - condition_dim
            )

        # additional angles
        if angle_size > 0:
            angle_embed = PosEmbedding(state_size * angle_size, N_freq)
            self.angle_embed = nn.Sequential(
                angle_embed, nn.Linear(angle_embed.out_channels, condition_dim)
            )
            self.angle_head = nn.Linear(hidden_size, state_size * angle_size)

            in_channels += condition_dim
            self.angle_proj = nn.Linear(in_channels, hidden_size)
            seqTransEncoderLayer = nn.TransformerEncoderLayer(
                d_model=hidden_size, nhead=4, batch_first=True
            )
            self.angle_encoder = nn.TransformerEncoder(
                seqTransEncoderLayer,
                num_layers=hidden_layers,
            )

        self.angle_size = angle_size

    def define_head(
        self,
        concat_size,
        hidden_size,
        hidden_layers,
        forecast_size,
        state_size,
        kp_size,
    ):
        layers = [nn.Linear(concat_size, hidden_size), nn.GELU()]
        for _ in range(hidden_layers):
            layers.append(Block(hidden_size))
        layers.append(nn.Linear(hidden_size, state_size * forecast_size * kp_size))
        layers = nn.Sequential(*layers)
        return layers

    def forward(
        self,
        noisy,
        t,
        past,
        cam,
        feat,
        drop_cam=False,
        drop_past=False,
        goal=None,
        noisy_angles=None,
    ):
        """
        noisy: N, K*3
        t: N
        past: N, M*3
        cam: N, M*3
        """
        assert noisy.dim() == 2
        bs = noisy.shape[0]
        device = noisy.device
        noisy = noisy.clone()
        past = past.clone()
        cam = cam.clone()
        if goal is not None:
            goal = goal.clone()

        # rotaion augmentation
        if self.training and not self.is_angle:
            rand_roty = torch.rand(bs, 1, device=noisy.device) * 2 * np.pi
            rotxyz = torch.zeros(bs, 3, device=noisy.device)
            rotxyz[:, 1] = rand_roty[:, 0]
            rotmat = axis_angle_to_matrix(rotxyz)  # N, 3, 3

            noisy = noisy.view(noisy.shape[:-1] + (-1, 3))  # N, TK, 3
            noisy = (rotmat[:, None] @ noisy[..., None]).squeeze(-1)  # N, TK, 3
            noisy = noisy.view(noisy.shape[:-2] + (-1,))  # N, TK*3

            past = past.view(past.shape[:-1] + (-1, 3))
            past = (rotmat[:, None] @ past[..., None]).squeeze(-1)
            past = past.view(past.shape[:-2] + (-1,))
            # # add random noise
            # past = past + torch.randn_like(past, device=device) * 0.1

            cam = cam.view(cam.shape[:-1] + (-1, 3))
            cam = (rotmat[:, None] @ cam[..., None]).squeeze(-1)
            cam = cam.view(cam.shape[:-2] + (-1,))
            # # add random noise
            # cam = cam + torch.randn_like(cam, device=device) * 0.1

            if goal is not None:
                goal = goal.view(goal.shape[:-1] + (-1, 3))
                goal = (rotmat[:, None] @ goal[..., None]).squeeze(-1)
                goal = goal.view(goal.shape[:-2] + (-1,))
                # # add random noise
                # goal = goal + torch.randn_like(goal, device=device) * 0.1

        noisy = (noisy - self.mean) / (self.std + 1e-6)

        # state embedding
        t_goal_emb = self.time_embed(t)

        # condition embedding
        past_emb = self.past_mlp(past.view(past.shape[0], -1))
        cam_emb = self.cam_mlp(cam.view(cam.shape[0], -1))
        if self.training:
            drop_rate = 0.5
            rand_mask = (
                torch.rand(noisy.shape[0], 1, device=noisy.device) > drop_rate
            ).float()
            cam_emb = cam_emb * rand_mask
            rand_mask = (
                torch.rand(noisy.shape[0], 1, device=noisy.device) > drop_rate
            ).float()
            past_emb = past_emb * rand_mask
        if drop_cam:
            cam_emb = cam_emb * 0
        if drop_past:
            past_emb = past_emb * 0

        # merge the embeddings
        emb = torch.cat((t_goal_emb, past_emb, cam_emb), dim=-1)

        if hasattr(self, "feat_proj"):
            feat_emb = []
            for i in range(len(self.feat_proj)):
                feat_sub = feat[i].reshape(feat[i].shape[0], -1)
                # # add noise to feat
                # if self.training:
                #     batchstd = feat_sub.reshape(-1).std().detach()
                #     feat_sub = (
                #         feat_sub
                #         + torch.randn_like(feat_sub, device=device) * batchstd * 0.1
                #     )
                feat_emb_sub = self.feat_proj[i](feat_sub)
                feat_emb.append(feat_emb_sub)
            feat_emb = torch.cat(feat_emb, dim=-1)

            if self.training:
                drop_rate = 0.5
                rand_mask = (
                    torch.rand(noisy.shape[0], 1, device=noisy.device) > drop_rate
                ).float()
                feat_emb = feat_emb * rand_mask

            emb = torch.cat((emb, feat_emb), dim=-1)

        if hasattr(self, "cond_embed"):
            if goal is not None:
                cond_emb = self.cond_embed(goal)

                if self.training:
                    drop_rate = 0.5
                    rand_mask = (
                        torch.rand(noisy.shape[0], 1, device=noisy.device) > drop_rate
                    ).float()
                    cond_emb = cond_emb * rand_mask
            else:
                cond_emb = self.cond_embed(
                    torch.zeros(bs, self.cond_embed[0].in_channels, device=device)
                )

            emb = torch.cat((emb, cond_emb), dim=-1)

        noisy = noisy.reshape(bs, self.forecast_size, -1)
        delta = self.pred_head(noisy, emb)
        delta = delta.reshape(delta.shape[0], -1)

        delta = delta * self.std + self.mean

        # undo rotation augmentation
        if self.training and not self.is_angle:
            rotmat = axis_angle_to_matrix(-rotxyz)
            delta = delta.view(delta.shape[:-1] + (-1, 3))
            delta = (rotmat[:, None] @ delta[..., None]).squeeze(-1)  # same for so3
            delta = delta.view(delta.shape[:-2] + (-1,))

        if hasattr(self, "angle_head") and noisy_angles is not None:
            noisy_angles = (noisy_angles - self.mean_angle) / (self.std_angle + 1e-6)
            angle_emb = self.angle_embed(
                noisy_angles.reshape(bs, self.forecast_size, -1)
            )
            emb = torch.cat((emb, angle_emb), dim=-1)
            emb_out = self.angle_proj(emb)  # N,T,F
            emb_out = self.angle_encoder(emb_out)  # N,T,F

            angle_delta = self.angle_head(emb_out.view(-1, emb_out.shape[-1])).view(
                bs, -1
            )
            angle_delta = angle_delta * self.std_angle + self.mean_angle
            delta = torch.cat([delta, angle_delta], dim=-1)
        return delta

    def simulate_forward_diffusion(self, x0, noise_scheduler):
        x0 = x0.view(x0.shape[0], -1)
        forward_samples_goal = simulate_forward_diffusion(
            x0, noise_scheduler, self.mean, self.std
        )
        return forward_samples_goal

    def reverse_diffusion(
        self,
        nsamp,
        num_timesteps,
        noise_scheduler,
        past,
        cam,
        x0_to_world,
        x0_angles_to_world,
        feat_volume,
        voxel_grid,
        drop_cam,
        drop_past,
        goal,
        xyz_grid=None,
        denoise_angles=False,
    ):
        """
        past: bs, T,K,...
        xyz_grid: N, -1
        """
        bs = past.shape[0]
        noisy = torch.randn(
            1, nsamp, self.forecast_size * self.kp_size * self.state_size, device="cuda"
        )
        noisy = noisy * self.std + self.mean

        if denoise_angles:
            noisy_angles = torch.randn(
                1,
                nsamp,
                self.forecast_size * self.angle_size * self.state_size,
                device="cuda",
            )
            noisy_angles = self.std_angle * noisy_angles + self.mean_angle
        else:
            noisy_angles = None

        past = past.view(bs, 1, -1)
        cam = cam.view(bs, 1, -1)
        if x0_to_world is not None:
            x0_to_world = x0_to_world.view(bs, 1, -1)
        if x0_angles_to_world is not None:
            x0_angles_to_world = x0_angles_to_world.view(bs, 1, 3, 3)
        if goal is not None:
            goal = goal.view(bs, 1, -1)

        reverse_samples, _ = reverse_diffusion(
            noisy,
            num_timesteps,
            self,
            noise_scheduler,
            past,
            cam,
            x0_to_world,
            x0_angles_to_world,
            feat_volume,
            voxel_grid,
            drop_cam,
            drop_past,
            goal=goal,
            noisy_angles=noisy_angles,
        )
        if xyz_grid is not None:
            # when computing grad, only use the first sample in the batch
            past = past[:1]
            cam = cam[:1]
            if x0_to_world is not None:
                x0_to_world = x0_to_world[:1]
            if goal is not None:
                goal = goal[:1]
            _, reverse_grad_grid = reverse_diffusion(
                xyz_grid,
                num_timesteps,
                self,
                noise_scheduler,
                past,
                cam,
                x0_to_world,
                x0_angles_to_world,
                feat_volume,
                voxel_grid,
                drop_cam,
                drop_past,
                track_x0=False,
                goal=goal,
                noisy_angles=(
                    torch.zeros_like(
                        xyz_grid[..., :1].repeat(
                            (1,) * (xyz_grid.dim() - 1) + noisy_angles.shape[-1:]
                        )
                    )
                    if denoise_angles
                    else None
                ),
            )
            if denoise_angles:
                reverse_grad_grid = reverse_grad_grid[..., : -noisy_angles.shape[-1]]
        else:
            reverse_grad_grid = None
        return reverse_samples, reverse_grad_grid


class TotalDenoiser(nn.Module):
    def __init__(
        self,
        config,
        x0,
        x0_joints,
        x0_angles,
        y,
        in_dim=1,
        env_feat_dim=384,
        use_env=True,
        model=TrajDenoiser,
    ):
        super().__init__()

        # reshape
        x0 = x0.view(x0.shape[0], -1)
        x0_joints = x0_joints.view(x0_joints.shape[0], -1)
        x0_angles = x0_angles.view(x0_angles.shape[0], -1)
        y = y.view(y.shape[0], -1)

        # model setup
        state_size = 3
        forecast_size = int(x0.shape[1] / state_size)
        num_kps = int(x0_joints.shape[1] / state_size / forecast_size)
        memory_size = int(y.shape[1] / state_size)
        print(f"state_size: {state_size}")
        print(f"forecast_size: {forecast_size}")
        print(f"num_kps: {num_kps}")
        print(f"memory_size: {memory_size}")
        mean = x0.mean(0)
        std = x0.std(0) * 3
        mean_goal = mean[-state_size:]
        std_goal = std[-state_size:]

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

        bg_field = BGField()
        voxel_grid = bg_field.voxel_grid
        env_input = voxel_grid.data[None]
        env_input = torch.tensor(env_input, dtype=torch.float32, device="cuda")

        self.env_model = env_model
        self.goal_model = goal_model

        self.state_size = state_size
        self.memory_size = memory_size
        self.forecast_size = forecast_size
        self.num_kps = num_kps
        self.voxel_grid = voxel_grid
        self.bg_field = bg_field
        self.env_input = env_input

    def extract_feature_grid(self):
        feat_volume = self.env_model.extract_features(self.env_input)
        return feat_volume

    def forward_goal(self, noisy_goal, x0_to_world, t_frac, past, cam, feat_volume):
        # get features
        if self.env_model.feat_dim > 0:
            feat = self.voxel_grid.readout_in_world(
                feat_volume, noisy_goal, x0_to_world
            )
            feat = [feat]
        else:
            feat = []
        # predict noise
        goal_delta = self.goal_model(noisy_goal, t_frac, past, cam, feat)
        return goal_delta

    def load_ckpts(self, config):
        model_path = "projects/tiny-diffusion/exps/%s/ckpt_%s.pth" % (
            config.logname,
            config.suffix,
        )
        self.load_state_dict(torch.load(model_path), strict=True)


class TotalDenoiserTwoStage(TotalDenoiser):
    def __init__(
        self,
        config,
        x0,
        x0_joints,
        x0_angles,
        y,
        in_dim=1,
        env_feat_dim=384,
        use_env=True,
        model=TrajDenoiser,
    ):
        super(TotalDenoiserTwoStage, self).__init__(
            config,
            x0,
            x0_joints,
            x0_angles,
            y,
            in_dim,
            env_feat_dim,
            use_env,
            model,
        )
        # reshape
        x0 = x0.view(x0.shape[0], -1)
        x0_joints = x0_joints.view(x0_joints.shape[0], -1)
        x0_angles = x0_angles.view(x0_angles.shape[0], -1)

        mean = x0.mean(0)
        std = x0.std(0) * 3
        mean_wp = mean
        std_wp = std
        x0_angles = torch.cat([x0_angles, x0_joints], dim=-1)
        mean_angle = x0_angles.mean(0)
        std_angle = x0_angles.std(0)

        fullbody_model = model(
            mean_wp,
            std_wp,
            hidden_size=config.hidden_size,
            hidden_layers=config.hidden_layers,
            feat_dim=[self.env_model.feat_dim * self.forecast_size],
            forecast_size=self.forecast_size,
            memory_size=self.memory_size,
            state_size=self.state_size,
            cond_size=self.state_size,
            condition_dim=config.condition_dim,
            angle_size=self.num_kps + 1,
            mean_angle=mean_angle,
            std_angle=std_angle,
        )
        self.fullbody_model = fullbody_model

    def forward_fullbody(
        self,
        noisy_wp,
        x0_to_world,
        t_frac,
        past,
        cam,
        feat_volume,
        noisy_joints,
        noisy_angles,
        clean_goal=None,
    ):
        # get features
        if self.env_model.feat_dim > 0:
            feat = self.voxel_grid.readout_in_world(feat_volume, noisy_wp, x0_to_world)
            feat = [feat]
        else:
            feat = []
        noisy_angles = torch.cat([noisy_angles, noisy_joints], dim=-1)
        all_delta = self.fullbody_model(
            noisy_wp,
            t_frac,
            past,
            cam,
            feat,
            goal=clean_goal,
            noisy_angles=noisy_angles,
        )
        wp_delta, angles_delta = (
            all_delta[..., : self.state_size * self.forecast_size],
            all_delta[..., self.state_size * self.forecast_size :],
        )
        joints_delta = angles_delta[..., self.state_size * self.forecast_size :]
        angles_delta = angles_delta[..., : self.state_size * self.forecast_size]
        return wp_delta, angles_delta, joints_delta


class TotalDenoiserThreeStage(TotalDenoiser):
    def __init__(
        self,
        config,
        x0,
        x0_joints,
        x0_angles,
        y,
        in_dim=1,
        env_feat_dim=384,
        use_env=True,
        model=TrajDenoiser,
    ):
        super(TotalDenoiserThreeStage, self).__init__(
            config,
            x0,
            x0_joints,
            x0_angles,
            y,
            in_dim,
            env_feat_dim,
            use_env,
            model,
        )
        # reshape
        x0 = x0.view(x0.shape[0], -1)
        x0_joints = x0_joints.view(x0_joints.shape[0], -1)
        x0_angles = x0_angles.view(x0_angles.shape[0], -1)

        mean = x0.mean(0)
        std = x0.std(0) * 3
        mean_wp = mean
        std_wp = std
        mean_joints = x0_joints.mean(0)
        std_joints = x0_joints.std(0)
        mean_angles = x0_angles.mean(0)
        std_angles = x0_angles.std(0)

        waypoint_model = model(
            mean_wp,
            std_wp,
            hidden_size=config.hidden_size,
            hidden_layers=config.hidden_layers,
            feat_dim=[self.env_model.feat_dim * self.forecast_size],
            forecast_size=self.forecast_size,
            memory_size=self.memory_size,
            state_size=self.state_size,
            cond_size=self.state_size,
            condition_dim=config.condition_dim,
        )

        fullbody_model = model(
            mean_joints,
            std_joints,
            hidden_size=config.hidden_size,
            hidden_layers=config.hidden_layers,
            # feat_dim=[env_model.feat_dim * forecast_size],
            feat_dim=[],
            forecast_size=self.forecast_size,
            memory_size=self.memory_size,
            state_size=self.state_size,
            cond_size=self.state_size * self.forecast_size,
            condition_dim=config.condition_dim,
            kp_size=self.num_kps,
            is_angle=True,
        )
        angle_model = model(
            mean_angles,
            std_angles,
            hidden_size=config.hidden_size,
            hidden_layers=config.hidden_layers,
            feat_dim=[],
            forecast_size=self.forecast_size,
            memory_size=self.memory_size,
            state_size=self.state_size,
            cond_size=self.state_size * self.forecast_size,
            condition_dim=config.condition_dim,
            kp_size=1,
            is_angle=True,
        )

        self.waypoint_model = waypoint_model
        self.fullbody_model = fullbody_model
        self.angle_model = angle_model

    def forward_path(
        self, noisy_wp, x0_to_world, t_frac, past, cam, feat_volume, clean_goal=None
    ):
        ############ waypoint prediction
        # get features
        if self.env_model.feat_dim > 0:
            feat = self.voxel_grid.readout_in_world(feat_volume, noisy_wp, x0_to_world)
            feat = [feat]
        else:
            feat = []
        wp_delta = self.waypoint_model(
            noisy_wp, t_frac, past, cam, feat, goal=clean_goal
        )
        return wp_delta

    def forward_fullbody(
        self,
        noisy_joints,
        noisy_angles,
        t_frac,
        past_joints,
        past_angles,
        cam,
        follow_wp=None,
    ):
        ############ fullbody prediction
        # # N, T, K3 => N,T, K, F => N,TKF
        # feat = voxel_grid.readout_in_world(
        #     feat_volume,
        #     noisy_joints.view(noisy_joints.shape[0], -1, state_size * num_kps),
        #     x0_to_world[:, None] + follow_wp.view(follow_wp.shape[0], -1, 3),
        # )
        # feat = feat.reshape(feat.shape[0], -1)
        joints_delta = self.fullbody_model(
            noisy_joints, t_frac, past_joints, cam * 0, [], goal=follow_wp
        )
        ############################

        ############ angle prediction
        # get features
        angles_delta = self.angle_model(
            noisy_angles, t_frac, past_angles, cam * 0, [], goal=follow_wp
        )
        ############################
        # ############ angle prediction
        # follow_wp = clean
        # angles_pred = angle_model(past_angles, follow_wp)
        # loss_angles = F.mse_loss(angles_pred, x0_angles) / angle_model.std.mean()
        # ############################
        return joints_delta, angles_delta


class Block(nn.Module):
    def __init__(self, size: int):
        super().__init__()

        self.ff = nn.Linear(size, size)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor):
        return x + self.act(self.ff(x))


class UNet3D(nn.Module):
    """
    3d U-Net
    """

    def __init__(self, in_planes, out_planes):
        super(UNet3D, self).__init__()
        self.decoder1 = Conv3dBlock(in_planes, 16, stride=(2, 2, 2))  # 2x
        self.decoder2 = Conv3dBlock(16, 128, stride=(2, 2, 2))  # 4x
        self.decoder3 = Conv3dBlock(128, out_planes, stride=(2, 2, 2))  # 8x
        # self.out = Conv3d(256,512,3, (1,1,1),1,bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = (
                    m.kernel_size[0]
                    * m.kernel_size[1]
                    * m.kernel_size[2]
                    * m.out_channels
                )
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if hasattr(m.bias, "data"):
                    m.bias.data.zero_()

    def forward(self, x):
        shape = x.shape
        x = self.decoder1(x)
        x = self.decoder2(x)
        x = self.decoder3(x)
        x = F.interpolate(x, size=shape[2:], mode="trilinear", align_corners=False)
        return x


class Conv3dBlock(nn.Module):
    """
    3d convolution block as 2 convolutions and a projection
    layer
    """

    def __init__(self, in_planes, out_planes, stride=(1, 1, 1)):
        super(Conv3dBlock, self).__init__()
        if in_planes == out_planes and stride == (1, 1, 1):
            self.downsample = None
        else:
            # self.downsample = projfeat3d(in_planes, out_planes,stride)
            self.downsample = nn.Sequential(
                nn.Conv3d(in_planes, out_planes, 1, stride, 0),
                nn.BatchNorm3d(out_planes),
            )
        self.conv1 = Conv3d(in_planes, out_planes, 3, stride, 1)
        self.conv2 = Conv3d(out_planes, out_planes, 3, (1, 1, 1), 1)

    def forward(self, x):
        out = F.relu(self.conv1(x), inplace=True)
        if self.downsample:
            x = self.downsample(x)
        out = F.relu(x + self.conv2(out), inplace=True)
        return out


def Conv3d(in_planes, out_planes, kernel_size, stride, pad, bias=False):
    if bias:
        return nn.Sequential(
            nn.Conv3d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                padding=pad,
                stride=stride,
                bias=bias,
            )
        )
    else:
        return nn.Sequential(
            nn.Conv3d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                padding=pad,
                stride=stride,
                bias=bias,
            ),
            nn.BatchNorm3d(out_planes),
        )


# class FiLMedBlock(nn.Module):
#     """A block of neural network layers with FiLM applied."""

#     def __init__(self, in_features, out_features, past_embed_size):
#         super().__init__()
#         self.linear = nn.Linear(in_features, out_features)
#         self.activation = nn.GELU()
#         # FiLM layer specifically for this block
#         self.film = DenseFiLM(past_embed_size)

#     def forward(self, x, scale_shift):
#         x = self.linear(x)
#         x = self.activation(x)
#         # Apply FiLM conditioning
#         x = featurewise_affine(x, scale_shift)
#         return x


# class DenseFiLM(nn.Module):
#     """Feature-wise linear modulation (FiLM) generator."""

#     def __init__(self, embed_channels):
#         super().__init__()
#         self.embed_channels = embed_channels
#         self.block = nn.Sequential(
#             nn.Mish(), nn.Linear(embed_channels, embed_channels * 2)
#         )

#     def forward(self, position):
#         pos_encoding = self.block(position)
#         pos_encoding = rearrange(pos_encoding, "b c -> b 1 c")
#         scale_shift = pos_encoding.chunk(2, dim=-1)
#         return scale_shift


# def featurewise_affine(x, scale_shift):
#     scale, shift = scale_shift
#     return (scale + 1) * x + shift
