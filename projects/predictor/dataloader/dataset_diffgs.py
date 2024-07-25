import numpy as np
import time
import torch
import pdb

from torch.utils.data import Dataset
import torchvision.transforms as T

from lab4d.config import load_flags_from_file
from lab4d.utils.camera_utils import construct_batch
from lab4d.utils.torch_utils import frameid_to_vid
from projects.csim.render_random import sample_extrinsics_outside_in
from projects.diffgs.trainer import GSplatTrainer as Trainer

class DiffgsGenerator:
    def __init__(self, path):
        self.res = 256
        self.intrinsics = np.asarray([384., 384., 128., 128.])

        # init 3dgs
        opts = load_flags_from_file(path)
        opts["load_suffix"] = "latest"
        model, data_info, ref_dict = Trainer.construct_test_model(opts)
        self.model = model
        self.data_info = data_info
        self.ref_dict = ref_dict

    @torch.no_grad()
    def generate_batch(
        self,
        num_images,
        first_idx=0,
        last_idx=-1,
        azimuth_limit=np.pi,
    ):
        dev = self.model.device
        frame_offset = torch.tensor(self.data_info["frame_info"]["frame_offset_raw"], device=dev)
        if last_idx == -1:
            last_idx = int(self.data_info["total_frames"])
        frame_idx = torch.randint(first_idx, last_idx, (num_images,), device=dev)
        inst_id = frameid_to_vid(frame_idx, self.data_info["frame_info"]["frame_offset"])
        frame_idx_sub = frame_idx - frame_offset[inst_id]

        # compute extrinsics
        extrinsics_base = self.model.gaussians.get_extrinsics(frame_idx)
        # randomize depth
        extrinsics_base[:, 2, 3] *= (torch.randn_like(extrinsics_base[:, 2, 3]) * 0.2).exp()
        # randomlize trans
        extrinsics_base[:, :2, 3] = torch.randn_like(extrinsics_base[:, :2, 3]) * 0.05 * extrinsics_base[:, 2:3, 3]
        # randomlize rotation
        d_extrinsics = []
        for _ in range(num_images):
            d_ext = sample_extrinsics_outside_in(
                elevation_limit=np.pi / 4,
                azimuth_limit=np.pi,
                roll_limit=np.pi / 6,
            )
            d_extrinsics.append(d_ext)
        d_extrinsics = np.stack(d_extrinsics,0)
        d_extrinsics = torch.tensor(d_extrinsics, device=dev, dtype=torch.float32)
        extrinsics = extrinsics_base @ d_extrinsics

        # compute intrinsics
        # intrinsics = self.model.get_intrinsics(0).cpu().numpy()
        intrinsics = torch.tensor(self.intrinsics[None], device=dev).repeat(num_images,1)
        # # randomlize focal lengh a bit => no, since we should assume a canonical focal length due to depth/focal ambiguity
        # intrinsics[:,:2] *= (torch.randn_like(intrinsics[:,:2]) * 0.2).exp()

        outputs = self.render(inst_id, frame_idx_sub, extrinsics, intrinsics)
        batch = {
            "img": outputs["rgb"].permute(0,3,1,2),
            "xyz": outputs["xyz"],
            "depth": outputs["depth"][..., 0],
            "extrinsics": extrinsics,
        }
        return batch

    @torch.no_grad()
    def render(self, inst_id, frame_idx_sub, extrinsics, intrinsics):
        # TODO rendering script with 3dgs
        crop2raw = None
        batch = construct_batch(
            inst_id,
            frame_idx_sub,
            self.res,
            {"fg": extrinsics},
            intrinsics,
            crop2raw,
            self.model.device,
        )
        outputs, _ = self.model.evaluate(
            batch, is_pair=False, augment_nv=False, return_numpy=False
        )
        return outputs

    @torch.no_grad()
    def re_render(self, extrinsics):
        dev = self.model.device
        inst_id = torch.tensor([0], device=dev)
        frame_idx_sub = torch.tensor([0], device=dev)
        intrinsics = torch.tensor(self.intrinsics[None], device=dev)
        extrinsics = torch.tensor(extrinsics[None], device=dev)
        outputs = self.render(inst_id, frame_idx_sub, extrinsics, intrinsics)
        color = outputs["rgb"][0].cpu().numpy() * 255
        return color
