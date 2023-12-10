import os, sys
import numpy as np
import torch
import time
from torch import nn
import torchvision.transforms as T
import torch.nn.functional as F
import pdb
import cv2
import tqdm
import math
import pdb

from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)


sys.path.insert(0, os.getcwd())
from lab4d.engine.train_utils import get_local_rank
from lab4d.third_party.guidance.sd_utils import StableDiffusion
from lab4d.third_party.guidance.zero123_utils import Zero123
from projects.gsplat.gs_renderer import (
    GaussianModel,
    BasicPointCloud,
    getProjectionMatrix,
)
from projects.gsplat.sh_utils import eval_sh, SH2RGB, RGB2SH


class GSplatModel(nn.Module):
    def __init__(self, config, data_info):
        super().__init__()
        self.config = config
        self.device = get_local_rank()
        self.data_info = data_info

        # 3DGS
        sh_degree = config["sh_degree"] if "sh_degree" in config else 3
        white_background = (
            config["white_background"] if "white_background" in config else True
        )
        radius = config["radius"] if "radius" in config else 1
        num_pts = config["num_pts"] if "num_pts" in config else 5000

        self.sh_degree = sh_degree
        self.white_background = white_background
        self.radius = radius

        self.gaussians = GaussianModel(sh_degree)

        self.bg_color = torch.tensor(
            [1, 1, 1] if white_background else [0, 0, 0],
            dtype=torch.float32,
            device="cuda",
        )

        self.initialize(num_pts=num_pts)

        # diffusion
        if config["guidance_sd_wt"] > 0:
            self.guidance_sd = StableDiffusion(self.device)
            self.prompt = "a photo of a british shorthair cat"
            self.negative_prompt = ""
            self.guidance_sd.get_text_embeds([self.prompt], [self.negative_prompt])
        if config["guidance_zero123_wt"] > 0:
            self.guidance_zero123 = Zero123(self.device)

    def initialize(self, input=None, num_pts=5000, radius=0.5):
        # load checkpoint
        if input is None:
            # init from random point cloud

            phis = np.random.random((num_pts,)) * 2 * np.pi
            costheta = np.random.random((num_pts,)) * 2 - 1
            thetas = np.arccos(costheta)
            mu = np.random.random((num_pts,))
            radius = radius * np.cbrt(mu)
            x = radius * np.sin(thetas) * np.cos(phis)
            y = radius * np.sin(thetas) * np.sin(phis)
            z = radius * np.cos(thetas)
            xyz = np.stack((x, y, z), axis=1)
            # xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3

            shs = np.random.random((num_pts, 3)) / 255.0
            pcd = BasicPointCloud(
                points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3))
            )
            self.gaussians.create_from_pcd(pcd, 10)
        elif isinstance(input, BasicPointCloud):
            # load from a provided pcd
            self.gaussians.create_from_pcd(input, 1)
        else:
            # load from saved ply
            self.gaussians.load_ply(input)

    def render(
        self,
        camera_dict,
        scaling_modifier=1.0,
        bg_color=None,
        override_color=None,
        compute_cov3D_python=False,
        convert_SHs_python=False,
    ):
        FoVx = camera_dict["fovx"]
        FoVy = camera_dict["fovy"]
        image_height = int(camera_dict["render_resolution"])
        image_width = int(camera_dict["render_resolution"])
        viewmatrix = torch.tensor(camera_dict["w2c"]).cuda()
        # rectify...
        viewmatrix[1:3, :3] *= -1
        viewmatrix[:3, 3] *= -1
        viewmatrix = viewmatrix.transpose(0, 1)
        projmatrix = (
            getProjectionMatrix(
                znear=camera_dict["near"],
                zfar=camera_dict["far"],
                fovX=camera_dict["fovx"],
                fovY=camera_dict["fovy"],
            )
            .transpose(0, 1)
            .cuda()
        )
        projmatrix = viewmatrix @ projmatrix
        c2w = np.linalg.inv(camera_dict["w2c"])
        campos = -torch.tensor(c2w[:3, 3]).cuda()

        # FoVx = viewpoint_camera.FoVx
        # FoVy = viewpoint_camera.FoVy
        # image_height = int(viewpoint_camera.image_height)
        # image_width = int(viewpoint_camera.image_width)
        # viewmatrix = viewpoint_camera.world_view_transform
        # projmatrix = viewpoint_camera.full_proj_transform
        # campos = viewpoint_camera.camera_center

        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = (
            torch.zeros_like(
                self.gaussians.get_xyz,
                dtype=self.gaussians.get_xyz.dtype,
                requires_grad=True,
                device="cuda",
            )
            + 0
        )
        try:
            screenspace_points.retain_grad()
        except:
            pass

        # Set up rasterization configuration
        tanfovx = math.tan(FoVx * 0.5)
        tanfovy = math.tan(FoVy * 0.5)

        raster_settings = GaussianRasterizationSettings(
            image_height=image_height,
            image_width=image_width,
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=self.bg_color if bg_color is None else bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewmatrix,
            projmatrix=projmatrix,
            sh_degree=self.gaussians.active_sh_degree,
            campos=campos,
            prefiltered=False,
            debug=False,
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means3D = self.gaussians.get_xyz
        means2D = screenspace_points
        opacity = self.gaussians.get_opacity

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        if compute_cov3D_python:
            cov3D_precomp = self.gaussians.get_covariance(scaling_modifier)
        else:
            scales = self.gaussians.get_scaling
            rotations = self.gaussians.get_rotation

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        if colors_precomp is None:
            if convert_SHs_python:
                shs_view = self.gaussians.get_features.transpose(1, 2).view(
                    -1, 3, (self.gaussians.max_sh_degree + 1) ** 2
                )
                dir_pp = self.gaussians.get_xyz - campos.repeat(
                    self.gaussians.get_features.shape[0], 1
                )
                dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(
                    self.gaussians.active_sh_degree, shs_view, dir_pp_normalized
                )
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            else:
                shs = self.gaussians.get_features
        else:
            colors_precomp = override_color

        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        rendered_image, radii, rendered_depth, rendered_alpha = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
        )

        rendered_image = rendered_image.clamp(0, 1)

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        out_dict = {
            "rgb": rendered_image,
            "depth": rendered_depth,
            "alpha": rendered_alpha,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
        }
        for k, v in out_dict.items():
            out_dict[k] = v.unsqueeze(0)
        return out_dict

    def get_default_cam(self):
        # convert this to a dict
        render_resolution = 128
        fovx = np.pi / 4
        fovy = np.pi / 4
        near = 0.01
        far = 5
        w2c = np.eye(4).astype(np.float32)
        w2c[2, 3] = -2.5  # opengl convention
        cam_dict = {
            "w2c": w2c,
            "render_resolution": render_resolution,
            "fovx": fovx,
            "fovy": fovy,
            "near": near,
            "far": far,
        }
        return cam_dict

    def forward(self, batch):
        loss_dict = {}
        # batch_constructed = self.construct_rand_batch(1)
        # rendered = self.render(batch_constructed)["rendered"]

        # GS rendering
        bg_color = torch.tensor(
            [1, 1, 1] if np.random.rand() > 0.5 else [0, 0, 0],
            dtype=torch.float32,
            device=self.device,
        )
        cam_dict = self.get_default_cam()
        rendered = self.render(cam_dict, bg_color=bg_color)
        rgb = rendered["rgb"]

        # compute loss
        if self.config["guidance_sd_wt"] > 0:
            loss_guidance_sd = self.guidance_sd.train_step(
                rgb, step_ratio=self.progress
            )
            loss_dict["guidance_sd"] = loss_guidance_sd

        if self.config["guidance_zero123_wt"] > 0:
            ref_rgb = batch["rgb"][:1, :1]
            bs = ref_rgb.shape[0]
            res = self.config["train_res"]
            # M,N,C => (N, C, d1, d2, ...,dK)

            ref_rgb = ref_rgb.permute(0, 1, 3, 2).reshape(-1, 3, res, res)
            ref_rgb = F.interpolate(
                ref_rgb, (256, 256), mode="bilinear", align_corners=False
            )
            self.guidance_zero123.get_img_embeds(ref_rgb)

            elevation = 0
            min_ver = max(min(-30, -30 - elevation), -80 - elevation)
            max_ver = min(max(30, 30 - elevation), 80 - elevation)
            polar = [np.random.randint(min_ver, max_ver)] * bs
            azimuth = [np.random.randint(-180, 180)] * bs
            radius = [0] * bs
            loss_guidance_zero123 = self.guidance_zero123.train_step(
                rgb, polar, azimuth, radius, step_ratio=self.progress
            )
            loss_dict["guidance_zero123"] = loss_guidance_zero123

        # weight each loss term
        self.apply_loss_weights(loss_dict, self.config)
        return loss_dict

    @staticmethod
    def apply_loss_weights(loss_dict, config):
        """Weigh each loss term according to command-line configs

        Args:
            loss_dict (Dict): Loss values for each loss term
            config (Dict): Command-line options
        """
        for k, v in loss_dict.items():
            # scale with loss weights
            wt_name = k + "_wt"
            if wt_name in config.keys():
                loss_dict[k] *= config[wt_name]

    def construct_rand_batch(self, num_imgs):
        """
        Returns:
            batch (Dict): Batch of input metadata. Keys: "dataid" (M,),
                "frameid_sub" (M,), "crop2raw" (M,4), "feature" (M,N,16),
                "hxy" (M,N,3), and "frameid" (M,)
        """
        opts = self.config
        batch = {}
        inst_id = np.random.randint(
            0, len(self.data_info["frame_info"]["frame_offset"]) - 1, (num_imgs,)
        )
        frameid = np.random.randint(0, self.data_info["total_frames"], (num_imgs,))
        frameid[:] = 0
        frameid_sub = frameid - self.data_info["frame_info"]["frame_offset"][inst_id]

        camera_int = np.zeros((len(frameid_sub), 4))
        camera_int[:, :2] = opts["train_res"] * 2
        camera_int[:, 2:] = opts["train_res"] / 2

        from lab4d.utils.camera_utils import (
            get_object_to_camera_matrix,
            construct_batch,
        )

        field2cam = {"fg": []}
        for idx in range(len(frameid_sub)):
            theta = np.random.rand(1) * 360
            axis = np.random.rand(3)
            distance = 10
            rtmat = get_object_to_camera_matrix(theta, axis, distance)
            field2cam["fg"].append(rtmat)
        field2cam["fg"] = np.stack(field2cam["fg"], 0)

        batch = construct_batch(
            inst_id=inst_id,
            frameid_sub=frameid_sub,
            eval_res=opts["train_res"],
            field2cam=field2cam,
            camera_int=camera_int,
            crop2raw=None,
            device=self.device,
        )
        inst_id = torch.tensor(
            self.data_info["frame_info"]["frame_offset"][inst_id], device=self.device
        )
        batch["frameid"] = batch["frameid_sub"] + inst_id
        return batch

    @torch.no_grad()
    def evaluate(self, batch):
        """Evaluate model on a batch of data"""
        scalars = {}
        rendered = self.render_turntable()
        # cam_dict = self.get_default_cam()
        # rendered = self.render(cam_dict)
        # rendered["rgb"] = rendered["rgb"].permute(0, 2, 3, 1)
        # rendered["depth"] = rendered["depth"].permute(0, 2, 3, 1)
        # rendered["alpha"] = rendered["alpha"].permute(0, 2, 3, 1)
        # del rendered["viewspace_points"]
        # del rendered["visibility_filter"]
        # del rendered["radii"]
        return rendered, scalars

    def render_turntable(self, nframes=9):
        out_dict = {"rgb": [], "depth": [], "alpha": []}
        cam_dict = self.get_default_cam()

        for i in range(nframes):
            w2c = cam_dict["w2c"]
            w2c[:3, :3] = cv2.Rodrigues(
                np.asarray([0.0, 2 * i * np.pi / nframes, 0.0])
            )[0]
            cam_dict["w2c"] = w2c
            rendered = self.render(cam_dict)
            out_dict["rgb"].append(rendered["rgb"][0].permute(1, 2, 0).cpu().numpy())
            out_dict["depth"].append(
                rendered["depth"][0].permute(1, 2, 0).cpu().numpy()
            )
            out_dict["alpha"].append(
                rendered["alpha"][0].permute(1, 2, 0).cpu().numpy()
            )
        for k, v in out_dict.items():
            out_dict[k] = np.stack(v, 0)
        return out_dict

    def set_progress(self, current_steps, progress):
        """Adjust loss weights and other constants throughout training

        Args:
            current_steps (int): Number of optimization steps so far
            progress (float): Fraction of training completed (in the current stage)
        """
        self.progress = progress
        self.current_steps = current_steps

    @torch.no_grad()
    def get_field_params(self):
        """Get beta values for all neural fields

        Returns:
            betas (Dict): Beta values for each neural field
        """
        beta_dicts = {}
        return beta_dicts

    def convert_img_to_pixel(self, batch):
        """Check that batch shape matches pixel array, otherwise convert to expected shape
        The same as dvr_renderer.convert_img_to_pixel

        Args:
            batch (Dict): Batch of dataloader samples. Keys: "rgb" (M,2,N,3),
                "mask" (M,2,N,1), "depth" (M,2,N,1), "feature" (M,2,N,16),
                "flow" (M,2,N,2), "flow_uct" (M,2,N,1), "vis2d" (M,2,N,1),
                "crop2raw" (M,2,4), "dataid" (M,2), "frameid_sub" (M,2),
                "hxy" (M,2,N,3), and "is_detected" (M,2)
        """
        for k, v in batch.items():
            if len(v.shape) == 5:
                M, _, H, W, K = v.shape
                batch[k] = v.view(M, -1, H * W, K)

    def update_geometry_aux(self):
        """Extract proxy geometry for all neural fields"""
        self.gaussians.update_geometry_aux()

    def update_camera_aux(self):
        pass

    def export_geometry_aux(self, path):
        """Export proxy geometry for all neural fields"""
        return self.gaussians.export_geometry_aux(path)


if __name__ == "__main__":
    import cv2
    import os, sys

    sys.path.append(os.getcwd())
    from lab4d.utils.io import save_vid

    opts = {"sh_degree": 0}
    renderer = GSplatModel(opts, None)

    # convert this to a dict
    cam_dict = renderer.get_default_cam()

    # render turntable view
    nframes = 10
    frames = []
    for i in range(nframes):
        w2c = cam_dict["w2c"]
        w2c[:3, :3] = cv2.Rodrigues(np.asarray([0.0, 2 * i * np.pi / nframes, 0.0]))[0]
        cam_dict["w2c"] = w2c
        out = renderer.render(cam_dict)
        img = out["rgb"][0].permute(1, 2, 0).detach().cpu().numpy()
        frames.append(img)
    save_vid("tmp/vid", frames)
    print("saved to tmp/vid.mp4")
