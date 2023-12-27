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
import trimesh

from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)

sys.path.insert(0, os.getcwd())
from lab4d.engine.train_utils import get_local_rank
from lab4d.nnutils.intrinsics import IntrinsicsConst
from lab4d.utils.numpy_utils import interp_wt
from lab4d.utils.loss_utils import get_mask_balance_wt
from lab4d.utils.geom_utils import (
    fov_to_focal,
    focal_to_fov,
    K2mat,
    K2inv,
    pinhole_projection,
)
from lab4d.utils.quat_transform import (
    quaternion_mul,
    matrix_to_quaternion,
    quaternion_translation_to_se3,
)
from lab4d.third_party.guidance.sd_utils import StableDiffusion
from lab4d.third_party.guidance.zero123_utils import Zero123
from projects.gsplat.gs_renderer import (
    GaussianModel,
    BasicPointCloud,
    getProjectionMatrix_K,
)
from projects.gsplat.sh_utils import eval_sh, SH2RGB, RGB2SH
from projects.gsplat.cam_utils import orbit_camera


class GSplatModel(nn.Module):
    def __init__(self, config, data_info):
        super().__init__()
        self.config = config
        self.device = get_local_rank()
        self.data_info = data_info
        self.progress = 0.0

        # dataset info
        frame_info = data_info["frame_info"]
        frame_offset = data_info["frame_info"]["frame_offset"]
        frame_offset_raw = data_info["frame_info"]["frame_offset_raw"]
        self.frame_offset = frame_offset
        self.frame_offset_raw = frame_offset_raw
        self.frame_mapping = frame_info["frame_mapping"]
        self.num_frames = frame_offset[-1]

        # 3DGS
        sh_degree = config["sh_degree"] if "sh_degree" in config else 3
        white_background = (
            config["white_background"] if "white_background" in config else True
        )
        radius = config["radius"] if "radius" in config else 2  # dreamgaussian
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
        # # DEBUG
        # mesh = trimesh.load("tmp/0.obj")
        # pcd = BasicPointCloud(
        #     mesh.vertices,
        #     mesh.visual.vertex_colors[:, :3] / 255,
        #     np.zeros((mesh.vertices.shape[0], 3)),
        # )
        # self.initialize(input=pcd)

        # initialize temporal part: (dx,dy,dz)t
        if not config["fg_motion"] == "rigid":
            total_frames = data_info["total_frames"]
            if config["use_timesync"]:
                num_vids = len(data_info["frame_info"]["frame_offset"]) - 1
                total_frames = total_frames // num_vids
                self.use_timesync = True
            else:
                self.use_timesync = False
            self.gaussians.init_trajectory(total_frames)
        else:
            self.use_timesync = False

        self.gaussians.construct_stat_vars()

        # intrinsics and extrinsics
        self.construct_intrinsics()
        self.gaussians.construct_extrinsics(config, data_info)

        # diffusion
        if config["guidance_sd_wt"] > 0:
            self.guidance_sd = StableDiffusion(self.device)
            self.prompt = "a photo of a british shorthair cat that is walking"
            self.negative_prompt = ""
            self.guidance_sd.get_text_embeds([self.prompt], [self.negative_prompt])
        if config["guidance_zero123_wt"] > 0:
            self.guidance_zero123 = Zero123(self.device)
            # self.guidance_zero123 = Zero123(
            #     self.device, model_key="ashawkey/stable-zero123-diffusers"
            # )

    def construct_intrinsics(self):
        """Construct camera intrinsics module"""
        config = self.config
        if config["intrinsics_type"] == "const":
            self.intrinsics = IntrinsicsConst(
                self.data_info["intrinsics"],
                frame_info=self.data_info["frame_info"],
            )
        else:
            raise NotImplementedError

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
            self.gaussians.init_gaussians(xyz)
        elif isinstance(input, BasicPointCloud):
            # load from a provided pcd
            self.gaussians.create_from_pcd(input)
        else:
            raise NotImplementedError

    def get_screenspace_pts_placeholder(self):
        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = (
            torch.zeros(
                (self.gaussians.get_num_pts, 3),
                dtype=self.gaussians._xyz.dtype,
                requires_grad=True,
                device="cuda",
            )
            + 0
        )
        try:
            screenspace_points.retain_grad()
        except:
            pass
        return screenspace_points

    def get_rasterizer(self, camera_dict, bg_color=None):
        image_height = int(camera_dict["render_resolution"])
        image_width = int(camera_dict["render_resolution"])
        viewmatrix = torch.tensor(camera_dict["w2c"]).cuda()
        viewmatrix = viewmatrix.transpose(0, 1)

        projmatrix = getProjectionMatrix_K(
            znear=camera_dict["near"],
            zfar=camera_dict["far"],
            Kmat=torch.tensor(camera_dict["Kmat"]),
        )
        projmatrix = projmatrix.transpose(0, 1).cuda()
        projmatrix = viewmatrix @ projmatrix
        c2w = np.linalg.inv(camera_dict["w2c"])
        campos = -torch.tensor(c2w[:3, 3]).cuda()

        # Set up rasterization configuration
        FoVx = focal_to_fov(camera_dict["Kmat"][0, 0])
        FoVy = focal_to_fov(camera_dict["Kmat"][1, 1])
        tanfovx = math.tan(FoVx * 0.5)
        tanfovy = math.tan(FoVy * 0.5)

        raster_settings = GaussianRasterizationSettings(
            image_height=image_height,
            image_width=image_width,
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=self.bg_color if bg_color is None else bg_color,
            scale_modifier=1.0,
            viewmatrix=viewmatrix,
            projmatrix=projmatrix,
            sh_degree=self.gaussians.active_sh_degree,
            campos=campos,
            prefiltered=False,
            debug=False,
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)
        return rasterizer

    def render(
        self,
        camera_dict,
        w2c=None,
        bg_color=None,
        frameid=None,
        w2c_2=None,
        frameid_2=None,
        Kmat_2=None,
    ):
        assert "Kmat" in camera_dict

        rasterizer = self.get_rasterizer(camera_dict, bg_color)

        screenspace_points = self.get_screenspace_pts_placeholder()
        means3D = self.gaussians.get_xyz(frameid)
        means2D = screenspace_points
        opacity = self.gaussians.get_opacity
        cov3D_precomp = None
        scales = self.gaussians.get_scaling
        rotations = self.gaussians.get_rotation(frameid)

        if w2c is not None:
            means3D, rotations = self.gaussians.transform(means3D, rotations, w2c)

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = self.gaussians.get_features

        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        rendered_image, radii, rendered_depth, rendered_alpha = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=None,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
        )

        with torch.no_grad():
            field_vals = self.gaussians.get_xyz()
            #### render other quantities
            rasterizer = self.get_rasterizer(
                camera_dict, bg_color=torch.zeros_like(self.bg_color)
            )
            render_vals, _, _, _ = rasterizer(
                means3D=means3D,
                means2D=means2D,
                shs=None,
                colors_precomp=field_vals,
                opacities=opacity,
                scales=scales,
                rotations=rotations,
                cov3D_precomp=cov3D_precomp,
            )
            ####

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        out_dict = {
            "rgb": rendered_image.clamp(0, 1),
            "depth": rendered_depth,
            "alpha": rendered_alpha,
            "viewspace_points": screenspace_points,
            "visibility_mask": radii > 0,
            "radii": radii,
        }
        out_dict["rgb"] = out_dict["rgb"][None]
        out_dict["depth"] = out_dict["depth"][None]
        out_dict["alpha"] = out_dict["alpha"][None]
        out_dict["xyz"] = render_vals[None]

        # render flow
        if w2c_2 is not None and frameid_2 is not None and Kmat_2 is not None:
            Kmat = camera_dict["Kmat"]

            means3D_2 = self.gaussians.get_xyz(frameid_2)
            rotations_2 = self.gaussians.get_rotation(frameid_2)
            means3D_2, rotations_2 = self.gaussians.transform(
                means3D_2, rotations_2, w2c_2
            )
            xy_2 = pinhole_projection(Kmat_2[None], means3D_2[None])[0]
            xy_1 = pinhole_projection(Kmat[None], means3D[None])[0]
            render_xy1, _, _, _ = rasterizer(
                means3D=means3D,
                means2D=means2D,
                shs=None,
                colors_precomp=xy_1,
                opacities=opacity,
                scales=scales,
                rotations=rotations,
                cov3D_precomp=cov3D_precomp,
            )
            render_xy2, _, _, _ = rasterizer(
                means3D=means3D,
                means2D=means2D,
                shs=None,
                colors_precomp=xy_2,
                opacities=opacity,
                scales=scales,
                rotations=rotations,
                cov3D_precomp=cov3D_precomp,
            )
            flow = (render_xy2 - render_xy1)[None, :2]
            flow = flow * camera_dict["render_resolution"] / 2
            out_dict["flow"] = flow

        # save aux for densification
        if self.training:
            if hasattr(self, "rendered_aux"):
                self.rendered_aux.append(out_dict)
            else:
                self.rendered_aux = [out_dict]
        return out_dict

    def get_default_cam(self, render_resolution=256, Kmat=np.eye(3)):
        # focal=1 corresponds to 90 degree fov and identity projection matrix
        # convert this to a dict
        near = 0.01
        far = 100
        w2c = np.eye(4, dtype=np.float32)
        cam_dict = {
            "w2c": w2c,
            "render_resolution": render_resolution,
            "near": near,
            "far": far,
            "Kmat": Kmat,
        }
        return cam_dict

    def get_bg_color(self):
        bg_color = torch.tensor(
            [1, 1, 1] if np.random.rand() > 0.5 else [0, 0, 0],
            dtype=torch.float32,
            device=self.device,
        )
        return bg_color

    def compute_Kmat(self, crop_size, crop2raw, Kmat):
        # Kmat = Kmat_crop2raw^-1 @ Kmat_raw
        if not torch.is_tensor(Kmat):
            Kmat = torch.tensor(Kmat, dtype=torch.float32, device=self.device)
        Kmat = K2inv(crop2raw) @ Kmat  # THIS IS IN THE CROP SPACE (256x256 patch)
        # normlalize Kmat from pixel to screen space
        Kmat[..., :2, 2] = Kmat[..., :2, 2] - crop_size / 2
        Kmat[..., :2, :] = Kmat[..., :2, :] / crop_size * 2
        return Kmat

    def forward(self, batch):
        self.process_frameid(batch)
        loss_dict = {}

        # render reference view
        crop_size = self.config["train_res"]
        bg_color = self.get_bg_color()

        frameid_abs = batch["frameid"][0, 0]
        frameid = self.get_frameid(batch)[0, 0]
        crop2raw = batch["crop2raw"][0, 0]
        Kmat = K2mat(self.intrinsics.get_vals(frameid_abs))
        Kmat = self.compute_Kmat(crop_size, crop2raw, Kmat)

        w2c = self.gaussians.get_extrinsics(frameid_abs)
        cam_dict = self.get_default_cam(Kmat=Kmat)

        # flow args
        frameid_2 = self.get_frameid(batch)[0, 1]
        w2c_2 = self.gaussians.get_extrinsics(frameid_2)
        crop2raw_2 = batch["crop2raw"][0, 1]
        Kmat_2 = K2mat(self.intrinsics.get_vals(frameid_2))
        Kmat_2 = self.compute_Kmat(crop_size, crop2raw_2, Kmat_2)
        cam_dict_2 = self.get_default_cam(Kmat=Kmat_2)

        rendered = self.render(
            cam_dict,
            w2c=w2c,
            frameid=frameid,
            w2c_2=w2c_2,
            frameid_2=frameid_2,
            Kmat_2=Kmat_2,
        )

        rgb = rendered["rgb"]
        mask = rendered["alpha"]
        flow = rendered["flow"]

        # prepare reference view GT
        ref_rgb = batch["rgb"][:1, :1]
        ref_mask = batch["mask"][:1, :1].float()
        ref_vis2d = batch["vis2d"][:1, :1].float()
        ref_flow = batch["flow"][:1, :1]
        ref_flow_uct = batch["flow_uct"][:1, :1]
        white_img = torch.ones_like(ref_rgb)
        ref_rgb = torch.where(ref_mask > 0, ref_rgb, white_img)

        res = self.config["train_res"]
        # M,N,C => (N, C, d1, d2, ...,dK)
        ref_rgb = ref_rgb.permute(0, 1, 3, 2).reshape(-1, 3, res, res)
        ref_rgb = F.interpolate(
            ref_rgb, (256, 256), mode="bilinear", align_corners=False
        )
        ref_mask = ref_mask.permute(0, 1, 3, 2).reshape(-1, 1, res, res)
        ref_mask = F.interpolate(ref_mask, (256, 256), mode="nearest")
        ref_vis2d = ref_vis2d.permute(0, 1, 3, 2).reshape(-1, 1, res, res)
        ref_vis2d = F.interpolate(ref_vis2d, (256, 256), mode="nearest")
        ref_flow = ref_flow.permute(0, 1, 3, 2).reshape(-1, 2, res, res)
        ref_flow = F.interpolate(ref_flow, (256, 256), mode="bilinear")
        ref_flow_uct = ref_flow_uct.permute(0, 1, 3, 2).reshape(-1, 1, res, res)
        ref_flow_uct = F.interpolate(ref_flow_uct, (256, 256), mode="bilinear")

        rendered_2 = self.render(
            cam_dict_2,
            w2c=w2c_2,
            frameid=frameid_2,
            w2c_2=w2c,
            frameid_2=frameid,
            Kmat_2=Kmat,
        )
        rgb_2 = rendered_2["rgb"]
        mask_2 = rendered_2["alpha"]
        flow_2 = rendered_2["flow"]

        ref_rgb_2 = batch["rgb"][:1, 1:]
        ref_mask_2 = batch["mask"][:1, 1:].float()
        ref_vis2d_2 = batch["vis2d"][:1, 1:].float()
        ref_flow_2 = batch["flow"][:1, 1:]
        ref_flow_uct_2 = batch["flow_uct"][:1, 1:]
        ref_rgb_2 = torch.where(ref_mask_2 > 0, ref_rgb_2, white_img)

        ref_rgb_2 = ref_rgb_2.permute(0, 1, 3, 2).reshape(-1, 3, res, res)
        ref_rgb_2 = F.interpolate(
            ref_rgb_2, (256, 256), mode="bilinear", align_corners=False
        )
        ref_mask_2 = ref_mask_2.permute(0, 1, 3, 2).reshape(-1, 1, res, res)
        ref_mask_2 = F.interpolate(ref_mask_2, (256, 256), mode="nearest")
        ref_vis2d_2 = ref_vis2d_2.permute(0, 1, 3, 2).reshape(-1, 1, res, res)
        ref_vis2d_2 = F.interpolate(ref_vis2d_2, (256, 256), mode="nearest")
        ref_flow_2 = ref_flow_2.permute(0, 1, 3, 2).reshape(-1, 2, res, res)
        ref_flow_2 = F.interpolate(ref_flow_2, (256, 256), mode="bilinear")
        ref_flow_uct_2 = ref_flow_uct_2.permute(0, 1, 3, 2).reshape(-1, 1, res, res)
        ref_flow_uct_2 = F.interpolate(ref_flow_uct_2, (256, 256), mode="bilinear")

        # concat them
        rgb = torch.cat([rgb, rgb_2], 0)
        mask = torch.cat([mask, mask_2], 0)
        flow = torch.cat([flow, flow_2], 0)

        ref_rgb = torch.cat([ref_rgb, ref_rgb_2], 0)
        ref_mask = torch.cat([ref_mask, ref_mask_2], 0)
        ref_vis2d = torch.cat([ref_vis2d, ref_vis2d_2], 0)
        ref_flow = torch.cat([ref_flow, ref_flow_2], 0)
        ref_flow_uct = torch.cat([ref_flow_uct, ref_flow_uct_2], 0)

        # from flowutils.flowlib import point_vec, warp_flow
        # img_numpy = rgb[0].permute(1, 2, 0).detach().cpu().numpy()
        # flow_numpy = flow[0].permute(1, 2, 0).detach().cpu().numpy()
        # img_gt = ref_rgb[0].permute(1, 2, 0).cpu().numpy()
        # flow_gt = ref_flow[0].permute(1, 2, 0).cpu().numpy()
        # flow_vis = point_vec(img_numpy * 255, flow_numpy, skip=10)
        # flow_vis_gt = point_vec(img_gt * 255, flow_gt, skip=10)
        # cv2.imwrite("tmp/0.jpg", flow_vis)
        # cv2.imwrite("tmp/1.jpg", flow_vis_gt)

        # reference view loss
        loss_dict["rgb"] = (rgb - ref_rgb).pow(2)
        # pdb.set_trace()
        # cv2.imwrite(
        #     "tmp/0.jpg", rgb[0].permute(1, 2, 0).detach().cpu().numpy()[..., ::-1] * 255
        # )
        # cv2.imwrite(
        #     "tmp/1.jpg", ref_rgb[0].permute(1, 2, 0).cpu().numpy()[..., ::-1] * 255
        # )

        loss_dict["mask"] = (mask - ref_mask).pow(2)
        loss_dict["flow"] = (flow - ref_flow).norm(2, 1, keepdim=True)
        loss_dict["flow"] = loss_dict["flow"] * (ref_flow_uct > 0).float()
        # mask_balance_wt = get_mask_balance_wt(
        #     batch["mask"], batch["vis2d"], batch["is_detected"]
        # )
        # loss_dict["mask"] *= mask_balance_wt

        # mask loss
        self.mask_losses(loss_dict, ref_mask, ref_vis2d, self.config)

        # warp ref_rgb to normal aspect ratio and make it smaller
        full2crop = self.construct_uncrop_mat(crop2raw)
        ref_rgb = self.uncrop_img(ref_rgb, full2crop)
        # dataset intrinsics | if full2crop=I, then Kmat = Kmat
        full2crop[2:] *= 0  # no translation
        Kmat = K2inv(full2crop) @ Kmat.cpu().numpy()
        w2w0 = self.gaussians.get_extrinsics(frameid_abs)
        self.compute_reg_loss(loss_dict, ref_rgb, Kmat, w2w0, frameid, frameid_2)

        # weight each loss term
        self.apply_loss_weights(loss_dict, self.config)
        return loss_dict

    @staticmethod
    def construct_uncrop_mat(crop2raw):
        crop2raw = crop2raw.cpu().numpy()
        full2crop = np.zeros_like(crop2raw)
        full2crop[0] = 1
        full2crop[1] = crop2raw[0] / crop2raw[1]
        full2crop *= 1.2
        full2crop[2] = 256 * (1 - full2crop[0]) / 2
        full2crop[3] = 256 * (1 - full2crop[1]) / 2
        return full2crop

    @staticmethod
    def uncrop_img(ref_rgb, full2crop):
        dev = ref_rgb.device
        ref_rgb = ref_rgb.permute(0, 2, 3, 1).cpu().numpy()[0]
        x0, y0 = np.meshgrid(range(256), range(256))
        hp_full = np.stack([x0, y0, np.ones_like(x0)], -1)  # augmented coord
        hp_full = hp_full.astype(np.float32)
        hp_raw = hp_full @ K2mat(full2crop).T  # raw image coord
        x0 = hp_raw[..., 0].astype(np.float32)
        y0 = hp_raw[..., 1].astype(np.float32)
        ref_rgb = cv2.remap(
            ref_rgb,
            x0,
            y0,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(1, 1, 1),
        )
        # cv2.imwrite("tmp/ref_rgb.png", ref_rgb[..., ::-1] * 255)
        ref_rgb = torch.tensor(ref_rgb, device=dev).permute(2, 0, 1)[None]
        return ref_rgb

    def compute_reg_loss(self, loss_dict, ref_rgb, Kmat, w2w0, frameid, frameid_2):
        bg_color = self.get_bg_color()
        # render a random novel view
        bs = 1
        elevation = 0
        min_ver = max(min(-30, -30 - elevation), -80 - elevation)
        max_ver = min(max(30, 30 - elevation), 80 - elevation)

        # sample camera
        polar = [np.random.randint(min_ver, max_ver)] * bs
        azimuth = [np.random.randint(-180, 180)] * bs
        radius = [0] * bs
        c2w = orbit_camera(elevation + polar[0], azimuth[0], self.radius + radius[0])
        w2c = np.linalg.inv(c2w)
        # GL to CV for both obj and cam space
        w2c[1:3] *= -1  # left size: flip cam
        w2c[:, 1:3] *= -1  # right side: flip obj

        # dataset extrinsics at frameid | if w2c=I, then w2w0 = w2c
        w2c = torch.tensor(w2c, dtype=torch.float32, device=self.device)
        w2w0[:3, :3] = w2c[:3, :3] @ w2w0[:3, :3]
        w2c = w2w0

        cam_dict = self.get_default_cam(Kmat=Kmat)

        # render
        rendered = self.render(cam_dict, w2c=w2c, frameid=frameid)
        rgb_nv = rendered["rgb"]

        # compute loss
        if self.config["guidance_sd_wt"] > 0:
            loss_guidance_sd = self.guidance_sd.train_step(
                rgb_nv, step_ratio=self.progress
            )
            loss_dict["guidance_sd"] = loss_guidance_sd

        if self.config["guidance_zero123_wt"] > 0:
            # cv2.imwrite(
            #     "tmp/0.jpg", ref_rgb[0].permute(1, 2, 0).cpu().numpy()[..., ::-1] * 255
            # )
            # cv2.imwrite(
            #     "tmp/1.jpg",
            #     rgb_nv[0].permute(1, 2, 0).detach().cpu().numpy()[..., ::-1] * 255,
            # )
            # pdb.set_trace()
            self.guidance_zero123.get_img_embeds(ref_rgb)

            loss_guidance_zero123 = self.guidance_zero123.train_step(
                rgb_nv, polar, azimuth, radius, step_ratio=self.progress
            )
            loss_dict["guidance_zero123"] = loss_guidance_zero123

            # # DEBUG
            # outputs = self.guidance_zero123.refine(
            #     ref_rgb,
            #     elevation=polar,
            #     azimuth=azimuth,
            #     radius=radius,
            #     strength=0,
            # )
            # cv2.imwrite(
            #     "tmp/rgb_nv_refine.png",
            #     outputs[0]
            #     .permute(1, 2, 0)
            #     .detach()
            #     .cpu()
            #     .numpy()
            #     .astype(np.float32)[..., ::-1]
            #     * 255,
            # )
            # cv2.imwrite(
            #     "tmp/rgb_nv.png",
            #     rgb_nv[0].permute(1, 2, 0).detach().cpu().numpy()[..., ::-1] * 255,
            # )

        if self.config["reg_least_deform_wt"] > 0:
            loss_dict["reg_least_deform"] = self.gaussians.get_least_deform_loss()
        if self.config["reg_least_action_wt"] > 0:
            loss_dict["reg_least_action"] = self.gaussians.get_least_action_loss()
        if self.config["reg_arap_wt"] > 0:
            loss_dict["reg_arap"] = self.gaussians.get_arap_loss(
                frameid=frameid, frameid_2=frameid_2
            )

    @staticmethod
    def mask_losses(loss_dict, maskfg, vis2d, config):
        """Apply segmentation mask on dense losses

        Args:
            loss_dict (Dict): Dense losses. Keys: "mask" (M,N,1), "rgb" (M,N,3),
                "depth" (M,N,1), "flow" (M,N,1), "vis" (M,N,1), "feature" (M,N,1),
                "feat_reproj" (M,N,1), and "reg_gauss_mask" (M,N,1). Modified in
                place to multiply loss_dict["mask"] with the other losses
            batch (Dict): Batch of dataloader samples. Keys: "rgb" (M,2,N,3),
                "mask" (M,2,N,1), "depth" (M,2,N,1), "feature" (M,2,N,16),
                "flow" (M,2,N,2), "flow_uct" (M,2,N,1), "vis2d" (M,2,N,1),
                "crop2raw" (M,2,4), "dataid" (M,2), "frameid_sub" (M,2), and
                "hxy" (M,2,N,3)
            config (Dict): Command-line options
        """
        # always mask-out non-visible (out-of-frame) pixels
        keys_allpix = ["mask"]
        # field type specific keys
        keys_type_specific = ["rgb", "flow"]

        # type-specific masking rules
        if config["field_type"] == "bg":
            mask = (1 - maskfg) * vis2d
        elif config["field_type"] == "fg":
            mask = maskfg * vis2d
        elif config["field_type"] == "comp":
            mask = vis2d
        else:
            raise ("field_type %s not supported" % config["field_type"])
        # apply mask
        for k, v in loss_dict.items():
            if k in keys_allpix:
                loss_dict[k] = v * vis2d
            elif k in keys_type_specific:
                loss_dict[k] = v * mask
            else:
                raise ("loss %s not defined" % k)

    @staticmethod
    def apply_loss_weights(loss_dict, config):
        """Weigh each loss term according to command-line configs

        Args:
            loss_dict (Dict): Loss values for each loss term
            config (Dict): Command-line options
        """
        px_unit_keys = ["flow"]
        for k, v in loss_dict.items():
            # average over non-zero pixels
            v = v[v > 0]
            if v.numel() > 0:
                loss_dict[k] = v.mean()
            else:
                loss_dict[k] = v.sum()  # return zero

            # scale with image resolution
            if k in px_unit_keys:
                loss_dict[k] /= config["train_res"]

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

    def get_frameid(self, batch):
        if self.config["use_timesync"]:
            frameid = batch["frameid_sub"]
        else:
            frameid = batch["frameid"]
        return frameid

    @torch.no_grad()
    def evaluate(self, batch, is_pair=True):
        """Evaluate model on a batch of data"""
        crop_size = self.config["eval_res"]
        self.process_frameid(batch)
        scalars = {}
        out_dict = {"rgb": [], "depth": [], "alpha": [], "xyz": []}

        nframes = len(batch["frameid"])
        if is_pair:
            nframes = nframes // 2

        for idx in tqdm.tqdm(range(0, nframes)):
            # get dataset camera intrinsics and extrinsics
            if is_pair:
                idx = idx * 2
            frameid_abs = batch["frameid"][idx]
            frameid = self.get_frameid(batch)[idx]
            if "crop2raw" in batch.keys():
                crop2raw = batch["crop2raw"][idx]
            else:
                crop2raw = torch.tensor([1.0, 1.0, 0.0, 0.0], device=self.device)
            if "Kinv" in batch.keys():
                Kmat = batch["Kinv"][frameid].inverse()
            else:
                Kmat = K2mat(self.intrinsics.get_vals(frameid_abs))
            Kmat = self.compute_Kmat(crop_size, crop2raw, Kmat)
            if "field2cam" in batch.keys():
                quat_trans = batch["field2cam"]["fg"][idx]
                quat, trans = quat_trans[:4], quat_trans[4:]
                w2c = quaternion_translation_to_se3(quat, trans)
            else:
                w2c = self.gaussians.get_extrinsics(frameid_abs)
            cam_dict = self.get_default_cam(Kmat=Kmat)

            # # manually create cameras
            # w2c = np.eye(4, dtype=np.float32)
            # w2c[2, 3] = 3  # depth
            # w2c[:3, :3] = cv2.Rodrigues(
            #     np.asarray([0.0, 2 * idx * np.pi / nframes, 0.0])
            # )[0]
            # K = np.array([2, 2, 0, 0])
            # Kmat = K2mat(K)
            # cam_dict = self.get_default_cam(Kmat=Kmat)

            rendered = self.render(cam_dict, w2c=w2c, frameid=frameid)
            for k, v in rendered.items():
                if k in out_dict.keys():
                    out_dict[k].append(v[0].permute(1, 2, 0).cpu().numpy())
            # out_dict["rgb"].append(rendered["rgb"][0].permute(1, 2, 0).cpu().numpy())
            # out_dict["depth"].append(
            #     rendered["depth"][0].permute(1, 2, 0).cpu().numpy()
            # )
            # out_dict["alpha"].append(
            #     rendered["alpha"][0].permute(1, 2, 0).cpu().numpy()
            # )
        for k, v in out_dict.items():
            out_dict[k] = np.stack(v, 0)
        return out_dict, scalars

    def process_frameid(self, batch):
        """Convert frameid within each video to overall frame id

        Args:
            batch (Dict): Batch of input metadata. Keys: "dataid" (M,),
                "frameid_sub" (M,), "crop2raw" (M,4), "feature" (M,N,16), and
                "hxy" (M,N,3). This function modifies it in place to add key
                "frameid" (M,)
        """
        if not hasattr(self, "offset_cuda"):
            self.offset_cache = torch.tensor(
                self.data_info["frame_info"]["frame_offset_raw"],
                device=self.device,
                dtype=torch.long,
            )
        # convert frameid_sub to frameid
        if "motion_id" in batch.keys():
            # indicator for reanimation
            motion_id = batch["motion_id"]
            del batch["motion_id"]
        else:
            motion_id = batch["dataid"]
        batch["frameid"] = batch["frameid_sub"] + self.offset_cache[motion_id]

    def set_progress(self, current_steps, progress):
        """Adjust loss weights and other constants throughout training

        Args:
            current_steps (int): Number of optimization steps so far
            progress (float): Fraction of training completed (in the current stage)
        """
        self.progress = progress
        self.current_steps = current_steps
        config = self.config

        # local vs global arap loss
        if self.progress > config["inc_warmup_ratio"]:
            self.gaussians.is_inc_mode = False
        else:
            self.gaussians.is_inc_mode = True

        # # knn for arap
        # anchor_x = (0, 1.0)
        # anchor_y = (0.1, 0.0)
        # type = "linear"
        # ratio_knn = interp_wt(anchor_x, anchor_y, progress, type=type)
        # self.gaussians.ratio_knn = ratio_knn

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
        self.gaussians.export_geometry_aux(path)

    @torch.no_grad()
    def get_cameras(self, frame_id=None):
        """Compute camera matrices in world units

        Returns:
            field2cam (Dict): Maps field names ("fg" or "bg") to (M,4,4) cameras
        """
        field2cam = {}
        field2cam["fg"] = self.gaussians.get_extrinsics(frameid=frame_id)
        return field2cam

    @torch.no_grad()
    def get_intrinsics(self, frame_id=None):
        """Compute camera intrinsics at the given frames.

        Args:
            frame_id: (M,) Frame id. If None, compute at all frames
        Returns:
            intrinsics: (..., 4) Output camera intrinsics
        """
        return self.intrinsics.get_vals(frame_id=frame_id)

    @torch.no_grad()
    def get_aabb(self, inst_id=None):
        """Compute axis aligned bounding box
        Args:
            inst_id (int or tensor): Instance id. If None, return aabb for all instances

        Returns:
            aabb (Dict): Maps field names ("fg" or "bg") to (1/N,2,3) aabb
        """
        aabb = {}
        aabb["fg"] = self.gaussians.get_aabb()[None]
        return aabb

    def update_densification_stats(self):
        for rendered_aux in self.rendered_aux:
            if rendered_aux["viewspace_points"].grad is None:
                continue
            viewspace_point_grad, visibility_mask, radii = (
                rendered_aux["viewspace_points"].grad,
                rendered_aux["visibility_mask"],
                rendered_aux["radii"],
            )
            self.gaussians.max_radii2D[visibility_mask] = torch.max(
                self.gaussians.max_radii2D[visibility_mask],
                radii[visibility_mask],
            )
            self.gaussians.add_xyz_grad_stats(viewspace_point_grad, visibility_mask)
        # delete after update
        del self.rendered_aux


if __name__ == "__main__":
    import cv2
    import os, sys

    sys.path.append(os.getcwd())
    from lab4d.utils.io import save_vid

    opts = {"sh_degree": 0, "guidance_sd_wt": 0, "guidance_zero123_wt": 0}
    renderer = GSplatModel(opts, None)

    # convert this to a dict
    K = np.array([2, 2, 0, 0])
    Kmat = K2mat(K)
    cam_dict = renderer.get_default_cam(render_resolution=512, Kmat=Kmat)
    w2c = np.eye(4)
    w2c[2, 3] = 3  # depth

    # render turntable view
    nframes = 10
    frames = []
    for i in range(nframes):
        w2c[:3, :3] = cv2.Rodrigues(
            np.asarray([0.0, 2 * np.pi * (0.25 + i / nframes), 0.0])
        )[0]
        out = renderer.render(cam_dict, w2c=w2c)
        img = out["rgb"][0].permute(1, 2, 0).detach().cpu().numpy()
        frames.append(img)
        cv2.imwrite("tmp/%d.png" % i, img * 255)
    save_vid("tmp/vid", frames)
    print("saved to tmp/vid.mp4")
