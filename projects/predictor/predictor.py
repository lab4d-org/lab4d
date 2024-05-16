import numpy as np
import torch
import time
from torch import nn
from projects.predictor.encoder import Encoder
import torchvision.transforms as T
import torch.nn.functional as F
import pdb
import cv2
import tqdm
import trimesh

from projects.predictor.dataloader.dataset import PolyGenerator
from lab4d.nnutils.embedding import TimeInfo
from lab4d.nnutils.base import CondMLP, ScaleLayer
from lab4d.nnutils.embedding import PosEmbedding
from lab4d.utils.quat_transform import matrix_to_quaternion, quaternion_to_matrix
from lab4d.utils.loss_utils import rot_angle
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


class Predictor(nn.Module):
    def __init__(self, opts):
        super().__init__()
        self.transforms = nn.Sequential(
            T.Resize((224, 224), antialias=True),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        )
        self.transforms_depth = nn.Sequential(
            T.Resize((224, 224), antialias=True),
            T.Normalize(mean=(0.0), std=(3.0)),
        )
        # # dino
        self.encoder = DINOv2Encoder(in_channels=3, out_channels=384, use_depth=False)

        # regression heads
        self.head_trans = TranslationHead(384)
        self.head_quat = RotationHead(384)
        self.head_uncertainty = UncertaintyHead(384)

        # self.data_generator = PolyGenerator()
        assert len(opts["poly_1"])>0
        self.data_generator1 = PolyGenerator(poly_name=opts["poly_1"], inside_out = opts["inside_out"])
        self.data_generator2 = PolyGenerator(poly_name=opts["poly_2"], inside_out = opts["inside_out"])

        # hyper params
        self.azimuth_limit = np.pi
        self.first_idx = 0
        self.last_idx = -1

    def set_progress(self, steps, progress):
        pass
        # self.azimuth_limit = np.pi * progress
        # total_frames = len(self.data_generator.polycam_loader)
        # self.last_idx = int(np.clip(total_frames * progress, 1, total_frames))
        # print("set last index to %d/%d" % (self.last_idx, total_frames))

    def convert_img_to_pixel(self, batch):
        num_images = len(batch["index"])
        if np.random.binomial(1, 0.5):
            data_generator = self.data_generator1
        else:
            data_generator = self.data_generator2
        data_batch = data_generator.generate_batch(
            num_images,
            first_idx=self.first_idx,
            last_idx=self.last_idx,
            azimuth_limit=self.azimuth_limit,
        )
        # augment data
        # if self.training:
        data_batch["img"], data_batch["xyz"] = self.augment_data(
            data_batch["img"], data_batch["xyz"]
        )

        batch.update(data_batch)

        return batch

    @staticmethod
    def get_rand_bbox(lb, ub, h, w):
        sx = int(np.random.uniform(lb * w, ub * w))
        sy = int(np.random.uniform(lb * h, ub * h))
        cx = int(np.clip(np.random.uniform(0, w), sx, w - sx))
        cy = int(np.clip(np.random.uniform(0, h), sy, h - sy))
        return sx, sy, cx, cy

    @staticmethod
    def mask_aug(img, xyz, lb=0.1, ub=0.2):
        _, h, w = img.shape
        mean_img = img.mean(-1).mean(-1)[:, None, None]
        mean_xyz = xyz.mean(0).mean(0)[None, None]  # hw3
        if True:  # np.random.binomial(1, 0.5):
            for _ in range(5):
                sx, sy, cx, cy = Predictor.get_rand_bbox(lb, ub, h, w)
                img[:, cy - sy : cy + sy, cx - sx : cx + sx] = mean_img
                xyz[cy - sy : cy + sy, cx - sx : cx + sx] = mean_xyz
        return img, xyz

    def augment_data(self, img, xyz):
        """
        img: (N, 3, H, W)
        """
        # color
        color_aug = T.ColorJitter(brightness=0.5, contrast=0.2, saturation=0.2, hue=0.1)

        for i in range(img.shape[0]):
            img[i] = color_aug(img[i])
            # mask
            img[i], xyz[i] = self.mask_aug(img[i], xyz[i])

        return img, xyz

    def predict(self, img, depth):
        # image
        img = self.transforms(img)
        depth = self.transforms_depth(depth[:, None])
        feat = self.encoder(img)

        # prediction heads
        trans = self.head_trans(feat)
        quat = self.head_quat(feat)
        uncertainty = self.head_uncertainty(feat)[..., 0]
        return quat, trans, uncertainty

    @staticmethod
    def xyz_to_canonical(xyz, extrinsics):
        """
        xyz: N, H, W, 3
        extrinsics: N, 4, 4
        """
        xyz_homo = torch.cat([xyz, torch.ones_like(xyz[..., :1])], dim=-1)
        # N,HW,4
        extrinsics = torch.inverse(extrinsics)  # view to world
        xyz_canonical = extrinsics[:, None] @ xyz_homo.view(xyz_homo.shape[0], -1, 4, 1)
        xyz_canonical = xyz_canonical[..., :3, 0].view(xyz.shape)
        return xyz_canonical

    def forward(self, batch):
        loss_dict = {}

        # # rand image
        # img = torch.rand(1, 3, 256, 256, device="cuda")  # much faster than loader
        # rot_gt = torch.zeros(1, 4, device="cuda")
        # rot_gt[:, 0] = 1
        # trans_gt = torch.zeros(1, 3, device="cuda")
        # load data
        img = batch["img"]
        xyz = batch["xyz"]
        depth = xyz[..., 2]
        quat_gt = matrix_to_quaternion(batch["extrinsics"][:, :3, :3])
        trans_gt = batch["extrinsics"][:, :3, 3]
        xyz_gt = self.xyz_to_canonical(xyz, batch["extrinsics"])

        # # TODO unit test: load croped image / dinofeature and check thedifference
        # import numpy as np
        # feat = np.load(
        #     "database/processed/Features/Full-Resolution/Oct5at10-49AM-poly-0000/full-256-dinov2-00.npy"
        # )[:1]

        # img = np.load(
        #     "database/processed/JPEGImages/Full-Resolution/Oct5at10-49AM-poly-0000/full-256.npy"
        # )[:1]
        # img = torch.from_numpy(img).float().cuda().permute(0, 3, 1, 2)
        # feat = torch.from_numpy(feat).float().cuda()
        # pdb.set_trace()

        # pred = F.interpolate(pred.permute(0, 3, 1, 2), size=(112, 112), mode="bilinear")
        # pred = pred.permute(0, 2, 3, 1)
        # pred = F.normalize(pred, dim=-1)

        quat, trans, uncertainty = self.predict(img, depth)
        extrinsics_pred = torch.zeros_like(batch["extrinsics"])
        extrinsics_pred[:, :3, :3] = quaternion_to_matrix(quat)
        extrinsics_pred[:, :3, 3] = trans
        extrinsics_pred[:, 3, 3] = 1
        xyz_pred = self.xyz_to_canonical(xyz, extrinsics_pred)

        # loss
        # pdb.set_trace()
        # trimesh.Trimesh(vertices=xyz_pred[0].view(-1, 3).detach().cpu()).export(
        #     "tmp/0.obj"
        # )
        # trimesh.Trimesh(vertices=xyz_gt[0].view(-1, 3).detach().cpu()).export(
        #     "tmp/1.obj"
        # )
        loss_dict = self.compute_metrics(
            quat, quat_gt, trans, trans_gt, xyz_pred, xyz_gt, uncertainty=uncertainty
        )
        return loss_dict

    def compute_metrics(
        self, quat, quat_gt, trans, trans_gt, xyz=None, xyz_gt=None, uncertainty=None
    ):
        loss_dict = {}
        so3 = quaternion_to_matrix(quat)
        so3_gt = quaternion_to_matrix(quat_gt)
        rot_loss = rot_angle(so3 @ so3_gt.permute(0, 2, 1)).mean(-1) * 2e-4
        # rot_loss = (quat - quat_gt).pow(2).mean() * 1e-3
        trans_loss = (trans - trans_gt).pow(2).mean(-1) * 1e-4
        # camera_pose_loss = trans_loss + rot_loss
        # loss_dict["camera_pose_loss"] = camera_pose_loss.mean()
        loss_dict["rot_loss"] = rot_loss.mean()
        loss_dict["trans_loss"] = trans_loss.mean()
        if xyz is not None and xyz_gt is not None:
            loss_dict["xyz_loss"] = (xyz - xyz_gt).pow(2).mean() * 2e-4
            # not optimizing it
            loss_dict["xyz_loss"] = loss_dict["xyz_loss"].detach()

        # weight by uncertainty: error = error / uncertainty
        if uncertainty is not None:
            # loss_dict["rot_loss"] = (rot_loss / uncertainty.detach()).mean()
            # loss_dict["trans_loss"] = (trans_loss / uncertainty.detach()).mean()
            loss_dict["uncertainty_loss"] = (
                (uncertainty - (rot_loss + trans_loss).detach()).pow(2).mean()
            )

        return loss_dict

    def get_field_params(self):
        return {}

    @staticmethod
    def convert_se3_to_qt(se3):
        """
        Nx4x4 se3 to quaternion, translation
        """
        quat = matrix_to_quaternion(se3[:, :3, :3])
        trans = se3[:, :3, 3]
        return quat, trans

    @torch.no_grad()
    def predict_batch(self, batch):
        re_imgs = []
        pred_extrinsics = []
        pred_uncertainty = []
        # predict pose and re-render
        for idx, img in tqdm.tqdm(enumerate(batch["img"])):
            if "depth" in batch:
                depth = batch["depth"][idx]
            elif "xyz" in batch:
                depth = batch["xyz"][idx][..., 2]
            else:
                raise ValueError("no depth or xyz in batch")
            quat, trans, uncertainty = self.predict(img[None], depth[None])
            # get extrinsics
            extrinsics = np.eye(4)
            extrinsics[:3, :3] = quaternion_to_matrix(quat)[0].cpu().numpy()
            extrinsics[:3, 3] = trans[0].cpu().numpy()
            uncertainty = uncertainty.cpu().numpy()
            # re-render
            intrinsics = self.data_generator1.polycam_loader.intrinsics[0]
            re_img, _ = self.data_generator1.polycam_loader.render(
                None, extrinsics=extrinsics, intrinsics=intrinsics
            )
            re_img = cv2.putText(
                re_img.astype(np.uint8),
                "%.5f" % (uncertainty),
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                255,
            )
            re_img = re_img.astype(np.float32) / 255.0
            re_imgs.append(re_img)
            pred_extrinsics.append(extrinsics)
            pred_uncertainty.append(uncertainty)
        pred_extrinsics = np.stack(pred_extrinsics)
        pred_extrinsics = torch.tensor(
            pred_extrinsics, device="cuda", dtype=torch.float32
        )
        re_imgs = np.stack(re_imgs)
        pred_uncertainty = np.stack(pred_uncertainty)
        return re_imgs, pred_extrinsics, pred_uncertainty

    @torch.no_grad()
    def evaluate(self, batch):
        re_imgs, pred_extrinsics, pred_uncertainty = self.predict_batch(batch)
        rendered = {"re_rgb": re_imgs}
        if "extrinsics" in batch:
            quat, trans = self.convert_se3_to_qt(pred_extrinsics)
            quat_gt, trans_gt = self.convert_se3_to_qt(batch["extrinsics"])
            scalars = self.compute_metrics(quat, quat_gt, trans, trans_gt)
        else:
            scalars = {}
        return rendered, scalars
