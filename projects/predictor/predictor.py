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

from projects.predictor.dataloader.dataset import PolyGenerator
from lab4d.utils.quat_transform import matrix_to_quaternion, quaternion_to_matrix
from lab4d.utils.loss_utils import rot_angle


class RegressionHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.fc = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        if len(x.shape) == 4:
            pooled = x.max(dim=(2, 3))
        else:
            pooled = x
        pooled = pooled.view(pooled.shape[0], -1)

        return self.fc(pooled)


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


class Predictor(nn.Module):
    def __init__(self, opts):
        super().__init__()
        self.transforms = nn.Sequential(
            T.Resize(224, antialias=True),
            T.CenterCrop(224),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        )
        self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        self.encoder = nn.Sequential(
            # Encoder(in_channels=3, out_channels=384),
            Encoder(in_channels=384, out_channels=384),
        )

        self.head_trans = TranslationHead(384)
        self.head_quat = RotationHead(384)
        self.head_uncertainty = UncertaintyHead(384)

        self.data_generator = PolyGenerator()

        # hyper params
        self.azimuth_limit = np.pi
        self.first_idx = 0
        self.last_idx = -1

    def set_progress(self, steps, progress):
        # self.azimuth_limit = np.pi * progress
        total_frames = len(self.data_generator.polycam_loader)
        # self.last_idx = int(np.clip(total_frames * progress, 1, total_frames))
        # print("set last index to %d/%d" % (self.last_idx, total_frames))

    def convert_img_to_pixel(self, batch):
        num_images = len(batch["index"])
        data_batch = self.data_generator.generate_batch(
            num_images,
            first_idx=self.first_idx,
            last_idx=self.last_idx,
            azimuth_limit=self.azimuth_limit,
        )
        # augment data
        # if self.training:
        data_batch["img"] = self.augment_data(data_batch["img"])

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
    def mask_aug(rendered, lb=0.1, ub=0.2):
        _, h, w = rendered.shape
        feat_mean = rendered.mean(-1).mean(-1)[:, None, None]
        if True:  # np.random.binomial(1, 0.5):
            for _ in range(5):
                sx, sy, cx, cy = Predictor.get_rand_bbox(lb, ub, h, w)
                rendered[:, cy - sy : cy + sy, cx - sx : cx + sx] = feat_mean
        return rendered

    def augment_data(self, img):
        """
        img: (N, 3, H, W)
        """
        # color
        color_aug = T.ColorJitter(brightness=0.5, contrast=0.2, saturation=0.2, hue=0.1)

        for i in range(img.shape[0]):
            img[i] = color_aug(img[i])
            # mask
            img[i] = self.mask_aug(img[i])

        return img

    def predict(self, img):
        # transform pipeline
        img = self.transforms(img)

        # regression
        with torch.no_grad():
            self.backbone.eval()
            masks = torch.zeros(1, 16 * 16, device="cuda").bool()
            feat = self.backbone.forward_features(img, masks=masks)[
                "x_norm_patchtokens"
            ]
            feat = feat.permute(0, 2, 1)
            feat = feat.reshape(-1, 384, 16, 16)
            feat = F.interpolate(feat, size=(112, 112), mode="bilinear")
        feat = self.encoder(feat.detach())

        # feat = self.encoder(img)

        trans = self.head_trans(feat)
        quat = self.head_quat(feat)
        uncertainty = self.head_uncertainty(feat)[..., 0]
        return quat, trans, uncertainty

    def forward(self, batch):
        loss_dict = {}

        # # rand image
        # img = torch.rand(1, 3, 256, 256, device="cuda")  # much faster than loader
        # rot_gt = torch.zeros(1, 4, device="cuda")
        # rot_gt[:, 0] = 1
        # trans_gt = torch.zeros(1, 3, device="cuda")
        # load data
        img = batch["img"]
        quat_gt = matrix_to_quaternion(batch["extrinsics"][:, :3, :3])
        trans_gt = batch["extrinsics"][:, :3, 3]

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

        quat, trans, uncertainty = self.predict(img)

        # loss
        loss_dict = self.compute_metrics(
            quat, quat_gt, trans, trans_gt, uncertainty=uncertainty
        )
        return loss_dict

    def compute_metrics(self, quat, quat_gt, trans, trans_gt, uncertainty=None):
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
            quat, trans, uncertainty = self.predict(img[None])
            # get extrinsics
            extrinsics = np.eye(4)
            extrinsics[:3, :3] = quaternion_to_matrix(quat)[0].cpu().numpy()
            extrinsics[:3, 3] = trans[0].cpu().numpy()
            uncertainty = uncertainty.cpu().numpy()
            # re-render
            intrinsics = self.data_generator.polycam_loader.intrinsics[0]
            re_img, _ = self.data_generator.polycam_loader.render(
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
