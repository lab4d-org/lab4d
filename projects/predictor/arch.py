import numpy as np
import torch
import time
from torch import nn
from projects.predictor.encoder import Encoder
import torchvision.transforms as T
import torch.nn.functional as F

from lab4d.nnutils.embedding import TimeInfo
from lab4d.nnutils.base import CondMLP
from lab4d.nnutils.embedding import PosEmbedding
from lab4d.utils.quat_transform import quaternion_translation_to_se3


class CameraPredictor(nn.Module):
    """Camera pose from image
    Args:
        rtmat: (N,4,4) Object to camera transform
        data_info (Dict): Metadata about the dataset
    """

    def __init__(self, rtmat, data_info, W=384, activation=nn.ReLU(True)):
        super().__init__()
        if data_info is None:
            num_frames = len(rtmat)
            frame_info = {
                "frame_offset": np.asarray([0, num_frames]),
                "frame_mapping": list(range(num_frames)),
                "frame_offset_raw": np.asarray([0, num_frames]),
            }
        else:
            frame_info = data_info["frame_info"]
        self.time_info = TimeInfo(frame_info)

        # cached input
        rgb_imgs = torch.tensor(data_info["rgb_imgs"], dtype=torch.float32)
        rgb_imgs = rgb_imgs.permute(0, 3, 1, 2)
        transforms = nn.Sequential(
            T.Resize((224, 224), antialias=True),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        )
        rgb_imgs = transforms(rgb_imgs)
        self.register_buffer("rgb_imgs", rgb_imgs)

        # encoder
        torch.manual_seed(0)
        # self.encoder = Encoder(out_channels=W)
        self.encoder = DINOv2Encoder(in_channels=3, out_channels=W, use_depth=False)

        # output layers
        self.trans = TranslationHead(W)
        self.quat = RotationHead(W)

        self.base_trans = nn.Parameter(torch.zeros(3))

        # initialization
        def loss_fn(gt):
            # randomly select subset of frames
            num_samples = 100
            all_frameid = self.time_info.frame_mapping
            rand_frameid = all_frameid[torch.randperm(len(all_frameid))[:num_samples]]
            gt = gt[rand_frameid]

            quat, trans = self.get_vals(rand_frameid)
            pred = quaternion_translation_to_se3(quat, trans)
            loss = F.mse_loss(pred, gt)
            return loss

        self.loss_fn = loss_fn
        self.init_vals = rtmat

    def init_weights(self, loss_fn=None, termination_loss=0.0001, max_iters=500):
        """Initialize the time embedding MLP to match external priors.
        `self.init_vals` is defined by the child class, and could be
        (nframes, 4, 4) camera poses or (nframes, 4) camera intrinsics
        """
        # init base trans
        if type(self.init_vals) is tuple:
            rtmat = self.init_vals[0]
        else:
            rtmat = self.init_vals
        self.base_trans.data = rtmat[:, :3, 3].mean(0)

        if loss_fn is None:
            loss_fn = self.loss_fn

        optimizer = torch.optim.Adam(self.parameters(), lr=5e-4)

        i = 0
        while True:
            optimizer.zero_grad()
            loss = loss_fn(self.init_vals)
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print(f"iter: {i}, loss: {loss.item():.4f}")
            i += 1
            if loss < termination_loss or i >= max_iters:
                break

    def get_vals(self, frame_id=None):
        """Compute camera pose at the given frames.

        Args:
            frame_id: (M,) Frame id. If None, compute values at all frames
        Returns:
            quat: (M, 4) Output camera rotations
            trans: (M, 3) Output camera translations
        """
        # TODO: frame_id is absolute, need to fix
        if frame_id is None:
            frame_id = self.time_info.frame_mapping
        shape = frame_id.shape

        frame_id = frame_id.reshape(-1)
        imgs = self.rgb_imgs[frame_id]

        # easier to debug without bn update
        def set_bn_eval(m):
            classname = m.__class__.__name__
            if classname.find("BatchNorm") != -1:
                m.eval()

        self.encoder.apply(set_bn_eval)

        # do it in chunks
        feat = []
        chunk_size = 100
        for chunk_id in range(0, len(frame_id), chunk_size):
            chunk_imgs = imgs[chunk_id : chunk_id + chunk_size]
            feat_chunk = self.encoder(chunk_imgs)
            feat.append(feat_chunk)
        feat = torch.cat(feat, dim=0)
        quat = self.quat(feat)
        trans = self.trans(feat)

        quat = quat.reshape(*shape, 4)
        trans = trans.reshape(*shape, 3)
        trans = trans + self.base_trans.view((1,) * len(shape) + (3,))

        self.feat = feat.reshape(*shape, -1)  # save feat
        return quat, trans


class TrajPredictor(CameraPredictor):
    """Trajectory (3N) from image
    Args:
        data_info (Dict): Metadata about the dataset
    """

    def __init__(
        self, rtmat, xyz, trajectory, data_info, W=384, activation=nn.ReLU(True)
    ):
        super().__init__(rtmat, data_info, W=W, activation=activation)
        self.init_vals = (self.init_vals, xyz, trajectory)

        # initialization
        def loss_fn(gt):
            dev = next(self.parameters()).device
            gt_rtmat = gt[0]
            gt_xyz = gt[1].to(dev)  # N, 3
            gt_traj = gt[2].to(dev)  # N, T, 3

            # randomly select 10 frames
            all_frameid = self.time_info.frame_mapping
            rand_frameid = all_frameid[torch.randperm(len(all_frameid))[:10]]
            gt_rtmat = gt_rtmat[rand_frameid]
            gt_traj = gt_traj[:, rand_frameid]

            quat, trans, motion = self.get_vals(rand_frameid, gt_xyz)
            pred = quaternion_translation_to_se3(quat, trans)
            loss_root = F.mse_loss(pred, gt_rtmat)
            loss_traj = F.mse_loss(motion, gt_traj)
            loss = (loss_root + loss_traj) / 2
            return loss

        self.loss_fn = loss_fn

        self.pos_embedding = PosEmbedding(3, N_freqs=10)
        self.forward_warp = CondMLP(
            num_inst=1,
            D=5,
            W=W,
            in_channels=W + self.pos_embedding.out_channels,
            out_channels=3,
        )

    def get_vals(self, frame_id=None, xyz=None):
        """Compute camera pose at the given frames.

        Args:
            frame_id: (M,) Frame id. If None, compute values at all frames
        Returns:
            quat: (M, 4) Output camera rotations
            trans: (M, 3) Output camera translations
        """
        quat, trans = super().get_vals(frame_id)
        if xyz is not None:
            feat = self.feat  # bs,F1
            shape = feat.shape[:-1]
            feat = feat.reshape(-1, feat.shape[-1])  # N,F1
            xyz_embed = self.pos_embedding(xyz).detach()  # N,F2

            # N, bs, F1+F2
            xyz_embed = xyz_embed[:, None].expand((-1,) + feat.shape[:1] + (-1,))
            feat = feat[None].expand((xyz.shape[0],) + (-1,) + (-1,))

            embed = torch.cat([xyz_embed, feat], dim=-1)

            # N, bs, 3 => ..., bs, 3
            motion = []
            chunk_size = 100
            for chunk_id in range(0, len(embed[0]), chunk_size):
                motion_sub = self.forward_warp(
                    embed[:, chunk_id : chunk_id + chunk_size], None
                )
                motion.append(motion_sub)
            motion = torch.cat(motion, dim=1)
            motion = motion.reshape(-1, *shape, 3)
            return quat, trans, motion
        else:
            return quat, trans


class RegressionHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.fc = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        return self.fc(x)


class RotationHead(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.head = RegressionHead(in_channels, 4)

    def forward(self, x):
        out = self.head(x)
        out = F.normalize(out, dim=-1)  # wxyz
        neg_out = -out
        # force positive w
        out = torch.where(out[:, 0:1] > 0, out, neg_out)
        return out


class TranslationHead(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.head = RegressionHead(in_channels, 3)

    def forward(self, x):
        out = self.head(x)
        return out


class UncertaintyHead(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.head = RegressionHead(in_channels, 1)

    def forward(self, x):
        out = self.head(x)
        out = out.exp() * 0.01  # variance of error
        return out


class DINOv2Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, use_depth=False):
        super().__init__()
        self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        # self.backbone = torch.hub.load(
        #     "facebookresearch/dinov2", "dinov2_vits14_reg", force_reload=True
        # )
        self.encoder = Encoder(in_channels=384, out_channels=out_channels)

        if use_depth:
            self.depth_proj = nn.Conv2d(1, 384, kernel_size=1, stride=1, padding=0)

    def forward(self, img, depth=None):
        with torch.no_grad():
            self.backbone.eval()
            masks = torch.zeros(1, 16 * 16, device="cuda").bool()
            feat = self.backbone.forward_features(img, masks=masks)[
                "x_norm_patchtokens"
            ]
            feat = feat.permute(0, 2, 1).reshape(-1, 384, 16, 16)  # N, 384, 16*16
        feat = F.interpolate(feat, size=(112, 112), mode="bilinear")

        if depth is not None and hasattr(self, "depth_proj"):
            depth = F.interpolate(depth, size=(112, 112), mode="bilinear")
            feat = feat + self.depth_proj(depth)

        feat = self.encoder(feat)
        return feat