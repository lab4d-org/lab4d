import numpy as np
import torch
import pdb

from torch.utils.data import Dataset
import torchvision.transforms as T

from projects.csim.render_polycam import PolyCamRender
from projects.csim.render_random import sample_extrinsics


class PolyGenerator:
    def __init__(self, poly_name="Oct31at1-13AM-poly", img_scale=0.1):
        image_size = (1024, 768)
        poly_path = "database/polycam/%s" % poly_name

        image_size = (int(image_size[0] * img_scale), int(image_size[1] * img_scale))
        polycam_loader = PolyCamRender(poly_path, image_size=image_size)
        polycam_loader.intrinsics *= img_scale

        polycam_loader.renderer.set_ambient_light()
        self.polycam_loader = polycam_loader

    def generate(self, azimuth_limit=np.pi):
        frame_idx = np.random.randint(len(self.polycam_loader))
        extrinsics_base = self.polycam_loader.extrinsics[frame_idx]

        # random xyz direction from -1,1
        extrinsics = sample_extrinsics(
            extrinsics_base,
            azimuth_limit=azimuth_limit,
        )
        color, depth = self.polycam_loader.render(frame_idx, extrinsics=extrinsics)
        return color, extrinsics

    def generate_batch(self, num_images, azimuth_limit=np.pi):
        color_batch = []
        extrinsics_batch = []
        for i in range(num_images):
            color, extrinsics = self.generate(azimuth_limit=azimuth_limit)
            color_batch.append(color)
            extrinsics_batch.append(extrinsics)
        color_batch = np.stack(color_batch)
        extrinsics_batch = np.stack(extrinsics_batch)

        # conver to torch
        color_batch = (
            torch.from_numpy(color_batch).float().cuda().permute(0, 3, 1, 2) / 255.0
        )
        extrinsics_batch = torch.from_numpy(extrinsics_batch).float().cuda()

        batch = {}
        batch["img"] = color_batch
        batch["extrinsics"] = extrinsics_batch
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
