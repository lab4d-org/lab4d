import numpy as np
import time
import torch
import pdb

from torch.utils.data import Dataset
import torchvision.transforms as T

from projects.csim.render_polycam import PolyCamRender
from projects.csim.render_random import sample_extrinsics


class PolyGenerator:
    def __init__(self, poly_name="Oct31at1-13AM-poly", img_scale=0.25):
        image_size = (1024, 768)
        poly_path = "database/polycam/%s" % poly_name

        image_size = (int(image_size[0] * img_scale), int(image_size[1] * img_scale))
        polycam_loader = PolyCamRender(poly_path, image_size=image_size)
        polycam_loader.intrinsics *= img_scale

        polycam_loader.renderer.set_ambient_light()
        self.polycam_loader = polycam_loader

    def sample(self, frame_indices):
        color_batch = []
        extrinsics_batch = []
        for frame_idx in frame_indices:
            extrinsics = self.polycam_loader.extrinsics[frame_idx]

            # # random xyz direction from -1,1
            # extrinsics = sample_extrinsics(
            #     extrinsics,
            #     azimuth_limit=0,
            #     elevation_limit=0,
            #     roll_limit=0,
            # )

            color, _ = self.polycam_loader.render(frame_idx, extrinsics=extrinsics)
            color_batch.append(color)
            extrinsics_batch.append(extrinsics)
        color_batch = np.stack(color_batch)
        extrinsics_batch = np.stack(extrinsics_batch)
        return color_batch, extrinsics_batch

    def generate(
        self,
        first_idx=0,
        last_idx=-1,
        azimuth_limit=np.pi,
        crop_to_size=True,
        return_xyz=False,
    ):
        if last_idx == -1:
            last_idx = len(self.polycam_loader)
        frame_idx = np.random.randint(first_idx, last_idx)
        extrinsics_base = self.polycam_loader.extrinsics[frame_idx]

        # random xyz direction from -1,1
        extrinsics = sample_extrinsics(
            extrinsics_base,
            azimuth_limit=azimuth_limit,
            aabb=self.polycam_loader.aabb,
        )
        color, xyz = self.polycam_loader.render(
            frame_idx,
            extrinsics=extrinsics,
            crop_to_size=crop_to_size,
            return_xyz=return_xyz,
        )
        return color, xyz, extrinsics

    def convert_to_batch(
        self, color_batch, xyz_batch=None, depth_batch=None, extrinsics_batch=None
    ):
        # conver to torch
        batch = {}
        color_batch = torch.tensor(color_batch, dtype=torch.float32, device="cuda")
        batch["img"] = color_batch.permute(0, 3, 1, 2) / 255.0
        if xyz_batch is not None:
            xyz_batch = torch.tensor(xyz_batch, dtype=torch.float32, device="cuda")
            batch["xyz"] = xyz_batch
        if depth_batch is not None:
            depth_batch = torch.tensor(depth_batch, dtype=torch.float32, device="cuda")
            batch["depth"] = depth_batch
        if extrinsics_batch is not None:
            batch["extrinsics"] = torch.tensor(
                extrinsics_batch, dtype=torch.float32, device="cuda"
            )
        return batch

    def generate_batch(self, num_images, first_idx=0, last_idx=-1, azimuth_limit=np.pi):
        color_batch = []
        xyz_batch = []
        extrinsics_batch = []
        for i in range(num_images):
            color, xyz, extrinsics = self.generate(
                first_idx=first_idx,
                last_idx=last_idx,
                azimuth_limit=azimuth_limit,
                crop_to_size=False,
                return_xyz=True,
            )
            color_batch.append(color)
            xyz_batch.append(xyz)
            extrinsics_batch.append(extrinsics)
        color_batch = np.stack(color_batch)
        xyz_batch = np.stack(xyz_batch)
        extrinsics_batch = np.stack(extrinsics_batch)

        # randomly crop the image
        cropped_size = int(np.random.uniform(0.5, 1.0) * np.asarray(color.shape[1]))
        start_loc = np.random.randint(0, color.shape[1] - cropped_size)
        if np.random.binomial(1, 0.5):
            # crop lengthwise
            color_batch = color_batch[:, start_loc : start_loc + cropped_size, :, :]
            xyz_batch = xyz_batch[:, start_loc : start_loc + cropped_size, :, :]
        else:
            # crop widthwise
            color_batch = color_batch[:, :, start_loc : start_loc + cropped_size, :]
            xyz_batch = xyz_batch[:, :, start_loc : start_loc + cropped_size, :]

        batch = self.convert_to_batch(color_batch, xyz_batch, None, extrinsics_batch)
        return batch


class CustomDataset(Dataset):
    def __init__(self, opts) -> None:
        super().__init__()
        self.transform = T.Compose(
            [
                T.Resize(224),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

        self.data_generator = PolyGenerator()

    def __len__(self):
        # for infinite data generation
        return 200

    def __getitem__(self, index):
        batch = {}
        # img = (np.random.rand(256, 256, 3) * 255).astype(np.uint8)
        # img = Image.fromarray(img)
        # img = self.transform(img)
        # img = np.random.rand(3, 256, 256).astype(np.float32)
        # batch["img"] = img
        batch = {}

        # batch["img"], batch["extrinsics"] = self.data_generator.generate()
        batch["index"] = index
        return batch
