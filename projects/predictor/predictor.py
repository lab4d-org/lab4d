import numpy as np
import torch
from torch import nn
from predictor.encoder import Encoder
import torchvision.transforms as T
import torch.nn.functional as F
import pdb
import tqdm

from projects.predictor.dataloader.dataset import PolyGenerator
from lab4d.utils.quat_transform import matrix_to_quaternion, quaternion_to_matrix


class RegressionHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.fc = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        if len(x.shape) == 4:
            pooled = x.mean(dim=(1, 2))
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


class Predictor(nn.Module):
    def __init__(self, opts):
        super().__init__()
        self.transforms = nn.Sequential(
            T.Resize(224, antialias=True),
            T.CenterCrop(224),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        )
        self.encoder = nn.Sequential(
            Encoder((112, 112), in_channels=3, out_channels=384),
        )
        # self.encoder = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")

        self.decoder_trans = TranslationHead(384)
        self.decoder_quat = RotationHead(384)

        self.data_generator = PolyGenerator()

        # hyper params
        self.azimuth_limit = np.pi

    def set_progress(self, steps, progress):
        self.azimuth_limit = np.pi * progress

    def convert_img_to_pixel(self, batch):
        num_images = len(batch["index"])
        data_batch = self.data_generator.generate_batch(
            num_images, azimuth_limit=self.azimuth_limit
        )
        batch.update(data_batch)
        return batch

    def predict(self, img):
        # transform pipeline
        img = self.transforms(img)

        # regression
        # masks = torch.zeros(1, 16 * 16, device="cuda").bool()
        # feat = self.encoder.forward_features(img, masks=masks)["x_norm_patchtokens"]
        # feat = feat.reshape(-1, 16, 16, 384)
        feat = self.encoder(img)

        trans = self.decoder_trans(feat)
        quat = self.decoder_quat(feat)
        return quat, trans

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

        quat, trans = self.predict(img)

        # loss
        rot_loss = (quat - quat_gt).pow(2).mean() * 1e-3
        trans_loss = (trans - trans_gt).pow(2).mean() * 1e-4
        camera_pose_loss = trans_loss + rot_loss
        loss_dict["camera_pose_loss"] = camera_pose_loss
        loss_dict["rot_loss"] = rot_loss
        loss_dict["trans_loss"] = trans_loss
        return loss_dict

    def get_field_params(self):
        return {}

    def evaluate(self, batch):
        re_imgs = []
        # predict pose and re-render
        for idx, frame_idx in tqdm.tqdm(enumerate(batch["index"])):
            quat, trans = self.predict(batch["img"][idx : idx + 1])
            extrinsics = np.eye(4)
            extrinsics[:3, :3] = quaternion_to_matrix(quat)[0].cpu().numpy()
            extrinsics[:3, 3] = trans[0].cpu().numpy()
            re_img, _ = self.data_generator.polycam_loader.render(
                frame_idx, extrinsics=extrinsics
            )
            re_img = re_img.astype(np.float32) / 255.0
            re_imgs.append(re_img)

        rendered = {"re_rgb": np.stack(re_imgs)}

        return rendered
