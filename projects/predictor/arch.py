import pdb
import numpy as np
import torch
import time
from torch import nn
import einops
from projects.predictor.encoder import Encoder
import torchvision.transforms as T
import torch.nn.functional as F
from torch.nn.functional import interpolate

from lab4d.nnutils.embedding import TimeInfo
from lab4d.nnutils.base import CondMLP
from lab4d.nnutils.embedding import PosEmbedding
from lab4d.utils.quat_transform import quaternion_translation_to_se3
from lab4d.utils.torch_utils import zero_module


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
        # self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        # self.backbone = torch.hub.load(
        #     "facebookresearch/dinov2", "dinov2_vits14_reg", force_reload=True
        # )
        self.backbone = DINO(return_multilayer=True)
        self.encoder = Encoder(in_channels=384, out_channels=out_channels)

        if use_depth:
            self.depth_proj = nn.Conv2d(1, 384, kernel_size=1, stride=1, padding=0)

    def forward(self, img, depth=None):
        with torch.no_grad():
            self.backbone.eval()
            feats = self.backbone(img)
            feat = feats[-1]  # N, 384, 16*16

        feat = F.interpolate(feat, size=(112, 112), mode="bilinear")

        if depth is not None and hasattr(self, "depth_proj"):
            depth = F.interpolate(depth, size=(112, 112), mode="bilinear")
            feat = feat + self.depth_proj(depth)

        feat = self.encoder(feat)
        return feat, feats
    

class FeatureFusionBlock(nn.Module):
    def __init__(self, features, kernel_size, with_skip=True):
        super().__init__()
        self.with_skip = with_skip
        if self.with_skip:
            self.resConfUnit1 = ResidualConvUnit(features, kernel_size)

        self.resConfUnit2 = ResidualConvUnit(features, kernel_size)

    def forward(self, x, skip_x=None):
        if skip_x is not None:
            assert self.with_skip and skip_x.shape == x.shape
            x = self.resConfUnit1(x) + skip_x

        x = self.resConfUnit2(x)
        return x


class ResidualConvUnit(nn.Module):
    def __init__(self, features, kernel_size):
        super().__init__()
        assert kernel_size % 1 == 0, "Kernel size needs to be odd"
        padding = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv2d(features, features, kernel_size, padding=padding),
            nn.ReLU(True),
            nn.Conv2d(features, features, kernel_size, padding=padding),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.conv(x) + x


class DPT(nn.Module):
    """
    https://github.com/mbanani/probe3d/blob/main/evals/models/probes.py#L178
    """
    def __init__(self, input_dims, output_dim, hidden_dim=512, kernel_size=3):
        super().__init__()
        assert len(input_dims) == 4
        self.conv_0 = nn.Conv2d(input_dims[0], hidden_dim, 1, padding=0)
        self.conv_1 = nn.Conv2d(input_dims[1], hidden_dim, 1, padding=0)
        self.conv_2 = nn.Conv2d(input_dims[2], hidden_dim, 1, padding=0)
        self.conv_3 = nn.Conv2d(input_dims[3], hidden_dim, 1, padding=0)

        self.ref_0 = FeatureFusionBlock(hidden_dim, kernel_size)
        self.ref_1 = FeatureFusionBlock(hidden_dim, kernel_size)
        self.ref_2 = FeatureFusionBlock(hidden_dim, kernel_size)
        self.ref_3 = FeatureFusionBlock(hidden_dim, kernel_size, with_skip=False)

        self.out_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(hidden_dim, output_dim, 3, padding=1),
        )

        self.out_conv[-1] = zero_module(self.out_conv[-1])

    def forward(self, feats):
        """Prediction each pixel."""
        assert len(feats) == 4

        feats[0] = self.conv_0(feats[0])
        feats[1] = self.conv_1(feats[1])
        feats[2] = self.conv_2(feats[2])
        feats[3] = self.conv_3(feats[3])

        feats = [interpolate(x, scale_factor=2) for x in feats]

        out = self.ref_3(feats[3], None)
        out = self.ref_2(feats[2], out)
        out = self.ref_1(feats[1], out)
        out = self.ref_0(feats[0], out)

        out = interpolate(out, scale_factor=4)
        out = self.out_conv(out)
        out = interpolate(out, scale_factor=2)
        return out
    

class DINO(torch.nn.Module):
    def __init__(
        self,
        dino_name="dinov2",
        model_name="vits14",
        output="dense",
        layer=-1,
        return_multilayer=False,
    ):
        super().__init__()
        feat_dims = {
            "vitb8": 768,
            "vitb16": 768,
            "vitb14": 768,
            "vitb14_reg": 768,
            "vitl14": 1024,
            "vitg14": 1536,
            "vits14": 384,
        }

        # get model
        self.model_name = dino_name
        self.checkpoint_name = f"{dino_name}_{model_name}"
        dino_vit = torch.hub.load(f"facebookresearch/{dino_name}", self.checkpoint_name)
        self.vit = dino_vit.eval().to(torch.float32)
        self.has_registers = "_reg" in model_name

        assert output in ["cls", "gap", "dense", "dense-cls"]
        self.output = output
        self.patch_size = self.vit.patch_embed.proj.kernel_size[0]

        feat_dim = feat_dims[model_name]
        feat_dim = feat_dim * 2 if output == "dense-cls" else feat_dim

        num_layers = len(self.vit.blocks)
        multilayers = [
            num_layers // 4 - 1,
            num_layers // 2 - 1,
            num_layers // 4 * 3 - 1,
            num_layers - 1,
        ]

        if return_multilayer:
            self.feat_dim = [feat_dim, feat_dim, feat_dim, feat_dim]
            self.multilayers = multilayers
        else:
            self.feat_dim = feat_dim
            layer = multilayers[-1] if layer == -1 else layer
            self.multilayers = [layer]

        # define layer name (for logging)
        self.layer = "-".join(str(_x) for _x in self.multilayers)

    def forward(self, images):
        # pad images (if needed) to ensure it matches patch_size
        h, w = images.shape[-2:]
        h, w = h // self.patch_size, w // self.patch_size

        if self.model_name == "dinov2":
            x = self.vit.prepare_tokens_with_masks(images, None)
        else:
            x = self.vit.prepare_tokens(images)

        embeds = []
        for i, blk in enumerate(self.vit.blocks):
            x = blk(x)
            if i in self.multilayers:
                embeds.append(self.vit.norm(x))
                if len(embeds) == len(self.multilayers):
                    break

        num_spatial = h * w
        outputs = []
        for i, x_i in enumerate(embeds):
            cls_tok = x_i[:, 0]
            # ignoring register tokens
            spatial = x_i[:, -1 * num_spatial :]
            x_i = tokens_to_output(self.output, spatial, cls_tok, (h, w))
            outputs.append(x_i)

        return outputs[0] if len(outputs) == 1 else outputs

def tokens_to_output(output_type, dense_tokens, cls_token, feat_hw):
    if output_type == "cls":
        assert cls_token is not None
        output = cls_token
    elif output_type == "gap":
        output = dense_tokens.mean(dim=1)
    elif output_type == "dense":
        h, w = feat_hw
        dense_tokens = einops.rearrange(dense_tokens, "b (h w) c -> b c h w", h=h, w=w)
        output = dense_tokens.contiguous()
    elif output_type == "dense-cls":
        assert cls_token is not None
        h, w = feat_hw
        dense_tokens = einops.rearrange(dense_tokens, "b (h w) c -> b c h w", h=h, w=w)
        cls_token = cls_token[:, :, None, None].repeat(1, 1, h, w)
        output = torch.cat((dense_tokens, cls_token), dim=1).contiguous()
    else:
        raise ValueError()

    return output