import argparse
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


import matplotlib.pyplot as plt
import pickle as pkl
import numpy as np

from positional_embeddings import PositionalEmbedding

# sys.path.insert(0, os.getcwd() + "/../../")
sys.path.insert(0, os.getcwd())
from lab4d.nnutils.embedding import PosEmbedding
from lab4d.utils.quat_transform import axis_angle_to_matrix
from lab4d.nnutils.base import BaseMLP
from projects.csim.voxelize import VoxelGrid, readout_voxel_fn


def get_data():
    x0_0 = torch.zeros((1000, 2))
    x0_0[:, 0] = torch.linspace(-1, 1, 1000)
    y_0 = torch.zeros((1000, 1))
    x0_1 = torch.zeros((1000, 2))
    x0_1[:, 1] = torch.linspace(-1, 1, 1000)
    y_1 = torch.ones((1000, 1))

    x0 = torch.cat((x0_0, x0_1), dim=0)
    y = torch.cat((y_0, y_1), dim=0)
    return x0, y


def get_lab4d_data():
    datapath = "/home/gengshay/code/guided-motion-diffusion/dataset/Custom/"
    pkldatafilepath = os.path.join(datapath, "customposes.pkl")
    data = pkl.load(open(pkldatafilepath, "rb"))

    pose = [x for x in data["poses"]]
    joints = [x for x in data["joints3D"]]
    world_to_root = [x for x in data["se3"]]
    world_to_cam = [x for x in data["cam_se3"]]

    # current frame
    idx0 = 8

    # goal list
    goal_idx = [15, 31, 47, 63]
    # goal_idx = [63]

    # load data: N, T, 3
    root_world_se3 = np.linalg.inv(np.stack(world_to_root, axis=0))
    root_world_se3 = torch.tensor(root_world_se3, dtype=torch.float32)
    root_world = root_world_se3[..., :3, 3]  # N,T,3
    cam_world = np.linalg.inv(np.stack(world_to_cam, axis=0))[..., :3, 3]
    cam_world = torch.tensor(cam_world, dtype=torch.float32)
    joints_ego = np.stack(joints, axis=0)  # N,T,K,3
    joints_ego = torch.tensor(joints_ego, dtype=torch.float32)  # [:, :, 0:4]
    # transform to world
    joints_world = (root_world_se3[:, :, None, :3, :3] @ joints_ego[..., None])[..., 0]
    joints_world = joints_world + root_world[:, :, None, :]
    # trimesh.Trimesh(joints_world[0][0]).export("tmp/0.obj")

    # transform to zero centered
    root_world_curr = root_world[:, idx0].clone()
    joints_world_curr = joints_world[:, idx0].clone()
    cam_world = cam_world - root_world[:, idx0 : idx0 + 1]
    joints_world = joints_world - root_world[:, idx0 : idx0 + 1, None]
    root_world = root_world - root_world[:, idx0 : idx0 + 1]

    # get input/label pairs
    njoints = joints_world.shape[-2]
    goal_world = root_world[:, goal_idx]
    goal_joints_world = joints_world[:, goal_idx]
    goal_joints_relative = goal_joints_world - goal_world[:, :, None]
    past_world = root_world[:, :idx0]
    past_joints_world = joints_world[:, :idx0]
    cam_world = cam_world[:, :idx0]  # camera position of the past frames

    ## merge
    # goal_world = torch.cat((goal_world[:, :, None], goal_joints_world), dim=2)
    # past_world = torch.cat((past_world[:, :, None], past_joints_world), dim=2)

    # reshape
    cam_world = cam_world.view(cam_world.shape[0], -1)
    goal_world = goal_world.view(goal_world.shape[0], -1)
    past_world = past_world.view(past_world.shape[0], -1)
    goal_joints_world = goal_joints_world.view(goal_joints_world.shape[0], -1)
    past_joints_world = past_joints_world.view(past_joints_world.shape[0], -1)
    goal_joints_relative = goal_joints_relative.view(goal_joints_relative.shape[0], -1)
    return (
        goal_world,
        past_world,
        cam_world,
        root_world_curr,
        goal_joints_relative,
        past_joints_world,
    )


class Block(nn.Module):
    def __init__(self, size: int):
        super().__init__()

        self.ff = nn.Linear(size, size)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor):
        return x + self.act(self.ff(x))


class FiLMedBlock(nn.Module):
    """A block of neural network layers with FiLM applied."""

    def __init__(self, in_features, out_features, past_embed_size):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = nn.GELU()
        # FiLM layer specifically for this block
        self.film = DenseFiLM(past_embed_size)

    def forward(self, x, scale_shift):
        x = self.linear(x)
        x = self.activation(x)
        # Apply FiLM conditioning
        x = featurewise_affine(x, scale_shift)
        return x


class DenseFiLM(nn.Module):
    """Feature-wise linear modulation (FiLM) generator."""

    def __init__(self, embed_channels):
        super().__init__()
        self.embed_channels = embed_channels
        self.block = nn.Sequential(
            nn.Mish(), nn.Linear(embed_channels, embed_channels * 2)
        )

    def forward(self, position):
        pos_encoding = self.block(position)
        pos_encoding = rearrange(pos_encoding, "b c -> b 1 c")
        scale_shift = pos_encoding.chunk(2, dim=-1)
        return scale_shift


def featurewise_affine(x, scale_shift):
    scale, shift = scale_shift
    return (scale + 1) * x + shift


class EnvEncoder(nn.Module):
    def __init__(self, feat_dim=384):
        super().__init__()
        self.unet_3d = UNet3D(1, feat_dim)
        self.feat_dim = feat_dim

    def extract_features(self, occupancy):
        """
        x_world: N,3
        """
        # 3D convs then query B1HWD => B3HWD
        feature_vol = self.unet_3d(occupancy[None])[0]
        return feature_vol

    def readout_in_world(self, feature_vol, x_ego, ego_to_world, res, origin):
        """
        x_ego: ...,KL
        ego_to_world: ...,L
        feat: ..., K, F
        """
        Ldim = ego_to_world.shape[-1]
        x_world = x_ego.view(x_ego.shape[:-1] + (-1, Ldim)) + ego_to_world[..., None, :]
        feat = self.readout_features(feature_vol, x_world, res, origin)
        feat = feat.reshape(x_ego.shape[:-1] + (-1,))
        return feat

    @staticmethod
    def readout_features(feature_vol, x_world, res, origin):
        """
        x_world: ...,3
        """
        # 3D convs then query B1HWD => B3HWD
        queried_feature = readout_voxel_fn(
            feature_vol, x_world.view(-1, 3), res, origin
        )
        queried_feature = queried_feature.T

        queried_feature = queried_feature.reshape(x_world.shape[:-1] + (-1,))
        return queried_feature


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
        self.forecase_size = forecast_size
        self.kp_size = kp_size

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
        self.proj = nn.Linear(in_channels, hidden_size)
        seqTransEncoderLayer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=4, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            seqTransEncoderLayer,
            num_layers=hidden_layers,
        )
        self.pred_head = nn.Linear(hidden_size, state_size * kp_size)

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
    ):
        """
        noisy: N, K*3
        t: N
        past: N, M*3
        cam: N, M*3
        """
        assert noisy.dim() == 2
        bs = noisy.shape[0]
        noisy = noisy.clone()
        past = past.clone()
        cam = cam.clone()
        if goal is not None:
            goal = goal.clone()

        # rotaion augmentation
        if self.training:
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

            cam = cam.view(cam.shape[:-1] + (-1, 3))
            cam = (rotmat[:, None] @ cam[..., None]).squeeze(-1)
            cam = cam.view(cam.shape[:-2] + (-1,))

            if goal is not None:
                goal = goal.view(goal.shape[:-1] + (-1, 3))
                goal = (rotmat[:, None] @ goal[..., None]).squeeze(-1)
                goal = goal.view(goal.shape[:-2] + (-1,))

        noisy = (noisy - self.mean) / self.std

        # state embedding
        # latent_emb = self.latent_embed(noisy)
        latent_emb = self.latent_embed(noisy.reshape(bs, self.forecase_size, -1))
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
                feat_emb_sub = self.feat_proj[i](feat[i].reshape(feat[i].shape[0], -1))
                feat_emb.append(feat_emb_sub)
            feat_emb = torch.cat(feat_emb, dim=-1)

            emb = torch.cat((emb, feat_emb), dim=-1)

        if goal is not None:
            cond_emb = self.cond_embed(goal)
            emb = torch.cat((emb, cond_emb), dim=-1)

        # # prediction: N,F => N,T*K*3
        # emb = torch.cat((latent_emb, emb), dim=-1)
        # delta = self.pred_head(emb)

        # Transformer
        emb = torch.cat(
            (latent_emb, emb[:, None].repeat(1, latent_emb.shape[1], 1)), dim=-1
        )  # N,T,(KF+F')
        emb = self.proj(emb)  # N,T,F
        emb = self.encoder(emb)  # N,T,F
        # NT,F=>NT,K3=>N,TK3
        delta = self.pred_head(emb.view(-1, emb.shape[-1])).view(bs, -1)

        delta = delta * self.std + self.mean

        # undo rotation augmentation
        if self.training:
            rotmat = axis_angle_to_matrix(-rotxyz)
            delta = delta.view(delta.shape[:-1] + (-1, 3))
            delta = (rotmat[:, None] @ delta[..., None]).squeeze(-1)
            delta = delta.view(delta.shape[:-2] + (-1,))
        return delta


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


class BGField:
    @torch.no_grad()
    def __init__(self):
        sys.path.insert(0, os.getcwd())
        from lab4d.config import load_flags_from_file
        from lab4d.engine.trainer import Trainer
        from lab4d.utils.mesh_loader import MeshLoader
        import glob

        # load flags from file with absl
        logdir = "logdir/home-2023-11-bg-adapt1/"
        opts = load_flags_from_file("%s/opts.log" % logdir)
        opts["load_suffix"] = "latest"
        opts["logroot"] = "logdir"
        opts["grid_size"] = 128
        opts["level"] = 0
        opts["vis_thresh"] = -10
        opts["extend_aabb"] = False

        opts["inst_id"] = 0
        model, data_info, ref_dict = Trainer.construct_test_model(opts)
        bg_field = model.fields.field_params["bg"]
        self.bg_field = bg_field

        meshes_rest = model.fields.extract_canonical_meshes(
            grid_size=opts["grid_size"],
            level=opts["level"],
            inst_id=opts["inst_id"],
            vis_thresh=opts["vis_thresh"],
            use_extend_aabb=opts["extend_aabb"],
        )
        scale_bg = bg_field.logscale.exp().cpu().numpy()
        meshes_rest["bg"].apply_scale(1.0 / scale_bg)
        self.bg_mesh = meshes_rest["bg"]

        # camera trajs
        # get root trajectory
        root_trajs = []
        cam_trajs = []
        # testdirs = sorted(glob.glob("%s/export_*" % args.logdir))
        testdirs = sorted(glob.glob("logdir/home-2023-11-compose-ft/export_*"))
        for it, loader_path in enumerate(testdirs):
            if "export_0000" in loader_path:
                continue
            root_loader = MeshLoader(loader_path)
            # load root poses
            root_traj = root_loader.query_camtraj(data_class="fg")
            root_trajs.append(root_traj)

            # load cam poses
            cam_traj = root_loader.query_camtraj(data_class="bg")
            cam_trajs.append(cam_traj)
            print("loaded %d frames from %s" % (len(root_loader), loader_path))
        root_trajs = np.linalg.inv(np.concatenate(root_trajs))
        cam_trajs = np.linalg.inv(np.concatenate(cam_trajs))  # T1+...+TN,4,4
        self.root_trajs = root_trajs
        self.cam_trajs = cam_trajs

        voxel_grid = VoxelGrid(self.bg_mesh, res=0.1)
        voxel_grid.count_root_visitation(self.root_trajs[:, :3, 3])
        # voxel_grid.run_viser()
        self.voxel_grid = voxel_grid

    def compute_feat(self, x):
        return self.bg_field.compute_feat(x)["feature"]

    def get_bg_mesh(self):
        return self.bg_mesh


class TrajDataset(Dataset):
    """Dataset for loading trajectory data."""

    def __init__(self, x, y, cam, x_to_world, x_joints, y_joints):
        self.x = x
        self.y = y
        self.cam = cam
        self.x_to_world = x_to_world
        self.x_joints = x_joints
        self.y_joints = y_joints

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return (
            self.x[idx],
            self.y[idx],
            self.cam[idx],
            self.x_to_world[idx],
            self.x_joints[idx],
            self.y_joints[idx],
        )


class NoiseScheduler(nn.Module):
    def __init__(
        self,
        num_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
    ):
        super().__init__()

        self.num_timesteps = num_timesteps
        if beta_schedule == "linear":
            betas = torch.linspace(
                beta_start, beta_end, num_timesteps, dtype=torch.float32
            )
        elif beta_schedule == "quadratic":
            betas = (
                torch.linspace(
                    beta_start**0.5, beta_end**0.5, num_timesteps, dtype=torch.float32
                )
                ** 2
            )

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # required for self.add_noise
        sqrt_alphas_cumprod = alphas_cumprod**0.5
        sqrt_one_minus_alphas_cumprod = (1 - alphas_cumprod) ** 0.5

        # required for reconstruct_x0
        sqrt_inv_alphas_cumprod = torch.sqrt(1 / alphas_cumprod)
        sqrt_inv_alphas_cumprod_minus_one = torch.sqrt(1 / alphas_cumprod - 1)

        # required for q_posterior
        posterior_mean_coef1 = (
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        posterior_mean_coef2 = (
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)
        )

        # convert to buffer
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", sqrt_alphas_cumprod)
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", sqrt_one_minus_alphas_cumprod
        )
        self.register_buffer("sqrt_inv_alphas_cumprod", sqrt_inv_alphas_cumprod)
        self.register_buffer(
            "sqrt_inv_alphas_cumprod_minus_one", sqrt_inv_alphas_cumprod_minus_one
        )
        self.register_buffer("posterior_mean_coef1", posterior_mean_coef1)
        self.register_buffer("posterior_mean_coef2", posterior_mean_coef2)

    def reconstruct_x0(self, x_t, t, noise):
        s1 = self.sqrt_inv_alphas_cumprod[t]
        s2 = self.sqrt_inv_alphas_cumprod_minus_one[t]
        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)
        return s1 * x_t - s2 * noise

    def q_posterior(self, x_0, x_t, t):
        s1 = self.posterior_mean_coef1[t]
        s2 = self.posterior_mean_coef2[t]
        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)
        mu = s1 * x_0 + s2 * x_t
        return mu

    def get_variance(self, t):
        if t == 0:
            return 0

        variance = (
            self.betas[t]
            * (1.0 - self.alphas_cumprod_prev[t])
            / (1.0 - self.alphas_cumprod[t])
        )
        variance = variance.clip(1e-20)
        return variance

    def step(self, model_output, timestep, sample, std):
        t = timestep
        pred_original_sample = self.reconstruct_x0(sample, t, model_output)
        pred_prev_sample = self.q_posterior(pred_original_sample, sample, t)

        variance = 0
        if t > 0:
            noise = torch.randn_like(model_output)
            noise = noise * std
            variance = (self.get_variance(t) ** 0.5) * noise

        pred_prev_sample = pred_prev_sample + variance

        return pred_prev_sample

    def add_noise(self, x_start, x_noise, timesteps):
        s1 = self.sqrt_alphas_cumprod[timesteps]
        s2 = self.sqrt_one_minus_alphas_cumprod[timesteps]

        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)

        return s1 * x_start + s2 * x_noise

    def sample_noise(self, clean, std):
        shape = clean.shape
        noise = torch.randn(shape, device="cuda")
        # dataset std
        noise = noise * std
        timesteps = torch.randint(0, self.num_timesteps, (shape[0],), device="cuda")
        timesteps = timesteps.long()
        noisy = self.add_noise(clean, noise, timesteps)
        t_frac = timesteps[:, None] / self.num_timesteps
        return noise, noisy, t_frac

    def __len__(self):
        return self.num_timesteps


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logname", type=str, default="base")
    parser.add_argument("--train_batch_size", type=int, default=1024)
    parser.add_argument("--eval_batch_size", type=int, default=1000)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--num_timesteps", type=int, default=50)
    parser.add_argument(
        "--beta_schedule", type=str, default="linear", choices=["linear", "quadratic"]
    )
    parser.add_argument("--condition_dim", type=int, default=64)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--hidden_layers", type=int, default=4)
    parser.add_argument(
        "--time_embedding",
        type=str,
        default="sinusoidal",
        choices=["sinusoidal", "learnable", "linear", "zero"],
    )
    parser.add_argument(
        "--input_embedding",
        type=str,
        default="sinusoidal",
        choices=["sinusoidal", "learnable", "linear", "identity"],
    )
    parser.add_argument("--save_images_step", type=int, default=1)
    parser.add_argument("--save_model_epoch", type=int, default=50)
    config = parser.parse_args()

    x0, y, cam, x0_to_world, x0_joints, y_joints = get_lab4d_data()
    x0 = x0.cuda()
    y = y.cuda()
    cam = cam.cuda()
    x0_to_world = x0_to_world.cuda()
    x0_joints = x0_joints.cuda()
    y_joints = y_joints.cuda()
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
    env_model = EnvEncoder()
    goal_model = TrajDenoiser(
        mean[-state_size:],
        std[-state_size:],
        hidden_size=config.hidden_size,
        hidden_layers=config.hidden_layers,
        feat_dim=[env_model.feat_dim],
        forecast_size=1,
        state_size=state_size,
        condition_dim=config.condition_dim,
    )
    waypoint_model = TrajDenoiser(
        mean,
        std,
        hidden_size=config.hidden_size,
        hidden_layers=config.hidden_layers,
        feat_dim=[env_model.feat_dim * forecast_size],
        forecast_size=forecast_size,
        state_size=state_size,
        cond_size=state_size,
        condition_dim=config.condition_dim,
    )
    mean_joints = x0_joints.mean(0)
    std_joints = x0_joints.std(0) * 3
    fullbody_model = TrajDenoiser(
        mean_joints,
        std_joints,
        hidden_size=config.hidden_size,
        hidden_layers=config.hidden_layers,
        feat_dim=[env_model.feat_dim * forecast_size],
        forecast_size=forecast_size,
        state_size=state_size,
        cond_size=state_size * forecast_size,
        condition_dim=config.condition_dim,
        kp_size=num_kps,
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
            feat = env_model.readout_in_world(
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
            feat = env_model.readout_in_world(
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
            feat = env_model.readout_in_world(
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
