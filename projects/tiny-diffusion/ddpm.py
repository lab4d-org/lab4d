import argparse
import os, sys
import pdb
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
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

    # load data
    root_world = np.linalg.inv(np.stack(world_to_root, axis=0))[..., :3, 3]
    root_world = torch.tensor(root_world, dtype=torch.float32)
    cam_world = np.linalg.inv(np.stack(world_to_cam, axis=0))[..., :3, 3]
    cam_world = torch.tensor(cam_world, dtype=torch.float32)

    # store old copy
    root_world_curr = torch.zeros_like(root_world[:, 0])
    root_world_all = root_world.clone()

    # transform to zero centered
    root_world_curr = root_world[:, 0].clone()
    cam_world = cam_world - root_world[:, 0:1]
    root_world = root_world - root_world[:, 0:1]

    # get input/label pairs
    goal_world = root_world[:, -1]
    past_world = root_world[:, 0]

    cam_world = cam_world[:, 0]  # camera position of the goal frame

    return goal_world, past_world, cam_world, root_world_curr, root_world_all


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


class MLP(nn.Module):
    def __init__(
        self, mean=None, std=None, hidden_size: int = 128, hidden_layers: int = 3
    ):
        super().__init__()
        if mean is None:
            mean = torch.zeros(3)
        if std is None:
            std = torch.ones(3)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.time_mlp = PosEmbedding(1, 8)
        self.input_mlp = PosEmbedding(3, 8)
        self.past_mlp = PosEmbedding(3, 8)
        self.cam_mlp = PosEmbedding(3, 8)

        in_dim = 384
        out_dim = 64
        self.feat_proj = nn.Linear(in_dim, out_dim)

        concat_size = (
            self.time_mlp.out_channels
            + self.input_mlp.out_channels
            + self.past_mlp.out_channels
            + self.cam_mlp.out_channels
            + out_dim
        )
        layers = [nn.Linear(concat_size, hidden_size), nn.GELU()]
        for _ in range(hidden_layers):
            layers.append(Block(hidden_size))
        layers.append(nn.Linear(hidden_size, 3))
        self.joint_mlp = nn.Sequential(*layers)

        self.env_model = UNet3D(1, 1)

    def forward(self, x, t, past, cam, feat, drop_cam=False, drop_past=False):
        x = (x - self.mean) / self.std

        # # rotaion augmentation
        # if self.training:
        #     rand_roty = torch.rand(x.shape[0], 1, device=x.device) * 2 * np.pi
        #     rotxyz = torch.zeros(x.shape[0], 3, device=x.device)
        #     rotxyz[:, 1] = rand_roty[:, 0]
        #     rotmat = axis_angle_to_matrix(rotxyz)
        #     x = (rotmat @ x[..., None]).squeeze(-1)
        #     past = (rotmat @ past[..., None]).squeeze(-1)
        #     cam = (rotmat @ cam[..., None]).squeeze(-1)

        x_emb = self.input_mlp(x)
        t_emb = self.time_mlp(t)
        past_emb = self.past_mlp(past)
        cam_emb = self.cam_mlp(cam)
        feat_emb = self.feat_proj(feat)
        if self.training:
            drop_rate = 0.5
            rand_mask = (torch.rand(x.shape[0], 1, device=x.device) > drop_rate).float()
            cam_emb = cam_emb * rand_mask
            rand_mask = (torch.rand(x.shape[0], 1, device=x.device) > drop_rate).float()
            past_emb = past_emb * rand_mask

        # drop_cam = True
        # drop_past = True

        if drop_cam:
            cam_emb = cam_emb * 0

        if drop_past:
            past_emb = past_emb * 0

        emb = torch.cat((x_emb, t_emb, past_emb, cam_emb, feat_emb), dim=-1)
        x_out = self.joint_mlp(emb)

        # # undo rotation augmentation
        # if self.training:
        #     rotmat = axis_angle_to_matrix(-rotxyz)
        #     x_out = (rotmat @ x_out[..., None]).squeeze(-1)

        x_out = x_out * self.std + self.mean
        return x_out

    def extract_env(self, occupancy, x_world, res, origin):
        # 3D convs then query B1HWD => B3HWD
        feature_vol = self.env_model(occupancy[None])[0]
        queried_feature = readout_voxel_fn(feature_vol, x_world, res, origin)
        queried_feature = queried_feature.T
        return queried_feature


class UNet3D(nn.Module):
    """
    3d U-Net
    """

    def __init__(self, in_planes, out_planes):
        super(UNet3D, self).__init__()
        self.decoder1 = Conv3dBlock(1, 16, stride=(2, 2, 2))  # 2x
        self.decoder2 = Conv3dBlock(16, 128, stride=(2, 2, 2))  # 4x
        self.decoder3 = Conv3dBlock(128, 384, stride=(2, 2, 2))  # 8x
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

    def __init__(self, x, y, cam, x_to_world):
        self.x = x
        self.y = y
        self.cam = cam
        self.x_to_world = x_to_world

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.cam[idx], self.x_to_world[idx]


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

    def step(self, model_output, timestep, sample):
        t = timestep
        pred_original_sample = self.reconstruct_x0(sample, t, model_output)
        pred_prev_sample = self.q_posterior(pred_original_sample, sample, t)

        variance = 0
        if t > 0:
            noise = torch.randn_like(model_output)
            variance = (self.get_variance(t) ** 0.5) * noise

        pred_prev_sample = pred_prev_sample + variance

        return pred_prev_sample

    def add_noise(self, x_start, x_noise, timesteps):
        s1 = self.sqrt_alphas_cumprod[timesteps]
        s2 = self.sqrt_one_minus_alphas_cumprod[timesteps]

        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)

        return s1 * x_start + s2 * x_noise

    def __len__(self):
        return self.num_timesteps


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, default="base")
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=1000)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--num_timesteps", type=int, default=50)
    parser.add_argument(
        "--beta_schedule", type=str, default="linear", choices=["linear", "quadratic"]
    )
    parser.add_argument("--embedding_size", type=int, default=128)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--hidden_layers", type=int, default=3)
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
    config = parser.parse_args()

    x0, y, cam, x0_to_world, root_world_all = get_lab4d_data()
    x0 = x0.cuda()
    y = y.cuda()
    cam = cam.cuda()
    x0_to_world = x0_to_world.cuda()
    dataset = TrajDataset(x0, y, cam, x0_to_world)

    loader_train = DataLoader(
        dataset, batch_size=config.train_batch_size, shuffle=True, drop_last=True
    )
    loader_eval = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False)

    mean = x0.mean(0)
    std = x0.std(0)
    model = MLP(
        mean, std, hidden_size=config.hidden_size, hidden_layers=config.hidden_layers
    )
    model = model.cuda()

    noise_scheduler = NoiseScheduler(
        num_timesteps=config.num_timesteps, beta_schedule=config.beta_schedule
    )
    noise_scheduler = noise_scheduler.cuda()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
    )

    bg_field = BGField()
    occupancy = bg_field.voxel_grid.data[None]
    occupancy = torch.tensor(occupancy, dtype=torch.float32, device="cuda")

    global_step = 0
    frames = []
    losses = []
    print("Training model...")
    for epoch in range(config.num_epochs):
        model.train()
        progress_bar = tqdm(total=len(loader_train))
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in enumerate(loader_train):
            clean = batch[0]
            past = batch[1]
            cam = batch[2]
            x0_to_world = batch[3]
            noise = torch.randn(clean.shape, device="cuda")
            timesteps = torch.randint(
                0, noise_scheduler.num_timesteps, (clean.shape[0],), device="cuda"
            ).long()

            # add noise
            noisy = noise_scheduler.add_noise(clean, noise, timesteps)
            t_frac = timesteps[:, None] / noise_scheduler.num_timesteps

            # 3D convs then query,   # B1HWD => B3HWD
            feat = model.extract_env(
                occupancy,
                noisy + x0_to_world,
                bg_field.voxel_grid.res,
                bg_field.voxel_grid.origin,
            )
            # # move to world and compute feature
            # feat = bg_field.compute_feat(noisy + x0_to_world)

            noise_pred = model(noisy, t_frac, past, cam, feat)
            loss = F.mse_loss(noise_pred, noise)
            loss.backward(loss)

            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "step": global_step}
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

    print("Saving model...")
    outdir = f"projects/tiny-diffusion/exps/{config.experiment_name}"
    os.makedirs(outdir, exist_ok=True)
    torch.save(model.state_dict(), f"{outdir}/model.pth")
