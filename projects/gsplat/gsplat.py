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
    rot_angle,
)
from lab4d.utils.camera_utils import get_rotating_cam
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
from projects.predictor.predictor import CameraPredictor, TrajPredictor

from flowutils.flowlib import point_vec, warp_flow


def fake_a_pair(tensor):
    """Fake a pair of tensors by repeating the first dimension

    Args:
        tensor (torch.Tensor): Tensor with shape (M, ...)

    Returns:
        tensor (torch.Tensor): Tensor with shape (M*2, ...)
    """
    if torch.is_tensor(tensor):
        return tensor[:, None].repeat((1, 2) + tuple([1] * (tensor.ndim - 1)))
    elif isinstance(tensor, dict):
        for k, v in tensor.items():
            tensor[k] = fake_a_pair(v)
        return tensor
    else:
        raise NotImplementedError


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
        num_pts = config["num_pts"] if "num_pts" in config else 5000

        self.sh_degree = sh_degree
        self.white_background = white_background

        self.gaussians = GaussianModel(sh_degree)

        self.bg_color = torch.tensor(
            [1, 1, 1] if white_background else [0, 0, 0],
            dtype=torch.float32,
            device="cuda",
        )

        # load lab4d model
        self.gaussians.load_lab4d(config["lab4d_path"])
        if self.gaussians.lab4d_model is None:
            mean_depth = data_info["rtmat"][1][:, 2, 3].mean()
            self.initialize(num_pts=num_pts, radius=mean_depth * 0.2)
        else:
            mesh = self.gaussians.lab4d_model.fields.extract_canonical_meshes()["fg"]
            scale_fg = self.gaussians.lab4d_model.fields.field_params["fg"]
            self.gaussians.scale_fg = scale_fg.logscale.exp()
            pcd = BasicPointCloud(
                mesh.vertices / self.gaussians.scale_fg.detach().cpu().numpy(),
                mesh.visual.vertex_colors[:, :3] / 255,
                np.zeros((mesh.vertices.shape[0], 3)),
            )
            self.initialize(input=pcd)

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

        self.gaussians.init_background(config["train_res"])

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
            radius_rand = radius * np.cbrt(mu)
            x = radius_rand * np.sin(thetas) * np.cos(phis)
            y = radius_rand * np.sin(thetas) * np.sin(phis)
            z = radius_rand * np.cos(thetas)
            xyz = np.stack((x, y, z), axis=1)
            scales = np.log(np.ones_like(xyz) * radius * 0.1)
            self.gaussians.init_gaussians(xyz, scales=scales)
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

    def get_rasterizer(self, camera_dict, Kmat, bg_color=None):
        """
        Kmat is (3,3)
        """
        image_height = int(camera_dict["render_resolution"])
        image_width = int(camera_dict["render_resolution"])
        # viewmatrix = torch.tensor(camera_dict["w2c"]).cuda()
        # viewmatrix = viewmatrix.transpose(0, 1)
        viewmatrix = torch.eye(4, dtype=torch.float32, device="cuda")

        projmatrix = getProjectionMatrix_K(
            znear=camera_dict["near"],
            zfar=camera_dict["far"],
            Kmat=Kmat,
        )
        projmatrix = projmatrix.transpose(0, 1).cuda()
        projmatrix = viewmatrix @ projmatrix
        # c2w = np.linalg.inv(camera_dict["w2c"])
        # campos = -torch.tensor(c2w[:3, 3]).cuda()
        campos = torch.zeros(3, dtype=torch.float32, device="cuda")

        # Set up rasterization configuration
        FoVx = focal_to_fov(Kmat[0, 0])
        FoVy = focal_to_fov(Kmat[1, 1])
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

    def render_pair(
        self,
        camera_dict,
        Kmat,
        w2c,
        bg_color=None,
        frameid=None,
    ):
        """Render a batch of view pairs given batch with shape (M,2,...)
        Args:
            camera_dict (Dict): Camera parameters
            w2c (torch.Tensor): World to camera matrix
            bg_color (torch.Tensor): Background color
            frameid (torch.Tensor): Frame id
        """
        bs = frameid.shape[0]
        out_dicts = []
        for i in range(bs):
            Kmat_0 = Kmat[i, 0]
            Kmat_1 = Kmat[i, 1]
            w2c_0 = w2c[i, 0]
            w2c_1 = w2c[i, 1]
            frameid_0 = frameid[i, 0]
            frameid_1 = frameid[i, 1]
            out_dict_0 = self.render(
                camera_dict,
                Kmat_0,
                w2c_0,
                bg_color,
                frameid_0,
                w2c_1,
                frameid_1,
                Kmat_1,
            )
            out_dict_1 = self.render(
                camera_dict,
                Kmat_1,
                w2c_1,
                bg_color,
                frameid_1,
                w2c_0,
                frameid_0,
                Kmat_0,
            )
            for k, v in out_dict_0.items():
                out_dict_0[k] = torch.cat([v, out_dict_1[k]], 0)
            out_dicts.append(out_dict_0)
        out_dict = {}
        for k in out_dicts[0].keys():
            out_dict[k] = torch.stack([d[k] for d in out_dicts], 0)
        return out_dict

    def render(
        self,
        camera_dict,
        Kmat,
        w2c=None,
        bg_color=None,
        frameid=None,
        w2c_2=None,
        frameid_2=None,
        Kmat_2=None,
    ):
        rasterizer = self.get_rasterizer(camera_dict, Kmat, bg_color)

        screenspace_points = self.get_screenspace_pts_placeholder()
        means3D = self.gaussians.get_xyz(frameid)
        means2D = screenspace_points
        means2D_tmp = self.get_screenspace_pts_placeholder()
        opacity = self.gaussians.get_opacity
        cov3D_precomp = None
        scales = self.gaussians.get_scaling
        rotations = self.gaussians.get_rotation(frameid)

        if w2c is not None:
            means3D, rotations = self.gaussians.transform(means3D, rotations, w2c)
            xy_1 = pinhole_projection(Kmat, means3D[None])[0]
        else:
            raise NotImplementedError

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
            rasterizer_xyz = self.get_rasterizer(
                camera_dict, Kmat, bg_color=torch.zeros_like(self.bg_color)
            )
            render_vals, _, _, _ = rasterizer_xyz(
                means3D=means3D,
                means2D=torch.zeros_like(means2D),
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
            "reproj_xy": xy_1[None, ..., :2],
        }
        out_dict["rgb"] = out_dict["rgb"][None]
        out_dict["depth"] = out_dict["depth"][None]
        out_dict["alpha"] = out_dict["alpha"][None]
        out_dict["xyz"] = render_vals[None]

        # render flow
        if w2c_2 is not None and frameid_2 is not None and Kmat_2 is not None:
            rasterizer_flow = self.get_rasterizer(
                camera_dict, Kmat, torch.zeros_like(self.bg_color)
            )
            means3D_2 = self.gaussians.get_xyz(frameid_2)
            rotations_2 = self.gaussians.get_rotation(frameid_2)
            means3D_2, rotations_2 = self.gaussians.transform(
                means3D_2, rotations_2, w2c_2
            )
            xy_2 = pinhole_projection(Kmat_2[None], means3D_2[None])[0]
            flow, _, _, _ = rasterizer_flow(
                means3D=means3D,
                means2D=means2D_tmp,
                shs=None,
                colors_precomp=(xy_2 - xy_1),
                opacities=opacity,
                scales=scales,
                rotations=rotations,
                cov3D_precomp=cov3D_precomp,
            )
            out_dict["flow"] = flow[None, :2] * camera_dict["render_resolution"] / 2
            out_dict["means2D_tmp"] = means2D_tmp

        # save aux for densification
        if self.training:
            if hasattr(self, "rendered_aux"):
                self.rendered_aux.append(out_dict)
            else:
                self.rendered_aux = [out_dict]
        return out_dict

    def get_default_cam(self, render_resolution):
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
        }
        return cam_dict

    def compute_render_Kmat(self, crop_size, crop2raw, Kmat):
        # Kmat = Kmat_crop2raw^-1 @ Kmat_raw
        if not torch.is_tensor(Kmat):
            Kmat = torch.tensor(Kmat, dtype=torch.float32, device=self.device)
        Kmat = K2inv(crop2raw) @ Kmat  # THIS IS IN THE CROP SPACE (256x256 patch)
        # normlalize Kmat from pixel to screen space
        Kmat[..., :2, 2] = Kmat[..., :2, 2] - crop_size / 2
        Kmat[..., :2, :] = Kmat[..., :2, :] / crop_size * 2
        return Kmat

    def compute_camera_samples(self, batch, crop_size):
        """Compute camera extrinsics and intrinsics
        Args:
            batch (Dict): Items with shape (M, 2, ...)
        """
        frameid_abs = batch["frameid"]
        if "crop2raw" in batch.keys():
            crop2raw = batch["crop2raw"]
        else:
            crop2raw = torch.tensor([1.0, 1.0, 0.0, 0.0], device=self.device)
            crop2raw = crop2raw[None, None].repeat(frameid_abs.shape[0], 2, 1)
        if "Kinv" in batch.keys():
            Kmat_raw = batch["Kinv"].inverse()
        else:
            Kmat_raw = K2mat(self.intrinsics.get_vals(frameid_abs))
        Kmat_unit = self.compute_render_Kmat(crop_size, crop2raw, Kmat_raw)
        cam_dict = self.get_default_cam(crop_size)
        if "field2cam" in batch:
            w2c = batch["field2cam"]["fg"]
            w2c = quaternion_translation_to_se3(w2c[..., :4], w2c[..., 4:])
        else:
            w2c = self.gaussians.get_extrinsics(frameid_abs)
        return cam_dict, Kmat_unit, w2c

    def compute_recon_losses(self, loss_dict, rendered, batch):
        # reference view loss
        loss_dict["rgb"] = (rendered["rgb"] - batch["rgb"]).pow(2)
        loss_dict["mask"] = (rendered["alpha"] - batch["mask"].float()).pow(2)
        loss_dict["flow"] = (rendered["flow"] - batch["flow"]).norm(2, 1, keepdim=True)
        loss_dict["flow"] = loss_dict["flow"] * (batch["flow_uct"] > 0).float()
        # mask_balance_wt = get_mask_balance_wt(
        #     batch["mask"], batch["vis2d"], batch["is_detected"]
        # )
        # loss_dict["mask"] *= mask_balance_wt
        # pdb.set_trace()
        # cv2.imwrite(
        #     "tmp/0.jpg", rgb[0].permute(1, 2, 0).detach().cpu().numpy()[..., ::-1] * 255
        # )
        # cv2.imwrite(
        #     "tmp/1.jpg", ref_rgb[0].permute(1, 2, 0).cpu().numpy()[..., ::-1] * 255
        # )
        # img_numpy = rgb[0].permute(1, 2, 0).detach().cpu().numpy()
        # flow_numpy = flow[0].permute(1, 2, 0).detach().cpu().numpy()
        # img_gt = ref_rgb[0].permute(1, 2, 0).cpu().numpy()
        # flow_gt = ref_flow[0].permute(1, 2, 0).cpu().numpy()
        # flow_vis = point_vec(img_numpy * 255, flow_numpy, skip=10)
        # flow_vis_gt = point_vec(img_gt * 255, flow_gt, skip=10)
        # cv2.imwrite("tmp/0.jpg", flow_vis)
        # cv2.imwrite("tmp/1.jpg", flow_vis_gt)

    def update_visibility_stats(self, rendered, batch):
        """update pts visibility"""
        # oom: pts not in ref_mask and pts inside of vis2d
        reproj_xy = rendered["reproj_xy"][:, :, None]
        # resample ref_mask based of reproj_xy
        mask = batch["mask"].float()
        # dilate the mask
        mask = F.max_pool2d(mask, 3, stride=1, padding=1)
        outside_mask = F.grid_sample(mask, reproj_xy, align_corners=True) == 0
        # inside_vis2d = F.grid_sample(ref_vis2d, reproj_xy, align_corners=True) > 0
        # is_oom = (outside_mask & inside_vis2d)[:, 0, :].float().mean(0)
        is_oom = outside_mask[:, 0, :].float().mean(0)
        self.gaussians.add_xyz_vis_stats(is_oom, torch.ones_like(is_oom[..., 0]) > 0)

    def forward(self, batch):
        """Run forward pass and compute losses

        Args:
            batch (Dict): Batch of dataloader samples. Keys: "rgb" (M,2,N,3),
                "mask" (M,2,N,1), "depth" (M,2,N,1), "feature" (M,2,N,16),
                "flow" (M,2,N,2), "flow_uct" (M,2,N,1), "vis2d" (M,2,N,1),
                "crop2raw" (M,2,4), "dataid" (M,2), "frameid_sub" (M,2),
                "hxy" (M,2,N,3), and "is_detected" (M,2)
        Returns:
            loss_dict (Dict): Computed losses. Keys: "mask" (M,N,1),
                "rgb" (M,N,3), "depth" (M,N,1), "flow" (M,N,1), "vis" (M,N,1),
                "feature" (M,N,1), "feat_reproj" (M,N,1),
                "reg_gauss_mask" (M,N,1), "reg_visibility" (0,),
                "reg_eikonal" (0,), "reg_deform_cyc" (0,),
                "reg_soft_deform" (0,), "reg_gauss_skin" (0,),
                "reg_cam_prior" (0,), and "reg_skel_prior" (0,).
        """
        self.process_frameid(batch)
        cam_dict, Kmat, w2c = self.compute_camera_samples(
            batch, self.config["train_res"]
        )
        frameid = self.get_frameid(batch)

        # TODO: get deformation before rendering
        self.gaussians.update_trajectory(frameid)

        # render reference view
        rendered = self.render_pair(cam_dict, Kmat, w2c=w2c, frameid=frameid)
        self.reshape_batch(rendered)

        # prepare reference view GT
        self.reshape_batch(batch)
        self.NHWC2NCHW(batch)

        loss_dict = {}
        self.compute_recon_losses(loss_dict, rendered, batch)
        self.mask_losses(
            loss_dict, batch["mask"], batch["vis2d"], rendered["alpha"], self.config
        )

        # compute regularization loss
        self.compute_reg_loss(loss_dict, frameid)

        if self.config["guidance_sd_wt"] > 0 or self.config["guidance_zero123_wt"] > 0:
            self.compute_diffusion_loss(loss_dict, batch)

        # weight each loss term
        self.apply_loss_weights(loss_dict, self.config)

        self.update_visibility_stats(rendered, batch)
        return loss_dict

    @staticmethod
    def construct_uncrop_mat(crop2raw, extend_factor=1.2):
        """Restore aspect ratio and uncrop the image"""
        crop2raw = crop2raw.cpu().numpy()
        full2crop = np.zeros_like(crop2raw)
        full2crop[0] = 1
        full2crop[1] = crop2raw[0] / crop2raw[1]
        full2crop *= extend_factor
        return full2crop

    @staticmethod
    def uncrop_img(
        ref_rgb, full2crop, crop_size, mode=cv2.INTER_LINEAR, borderValue=(1, 1, 1)
    ):
        dev = ref_rgb.device
        # unit to pixel space
        full2crop = full2crop.copy()
        full2crop[2] = crop_size * (1 - full2crop[0]) / 2
        full2crop[3] = crop_size * (1 - full2crop[1]) / 2

        ref_rgb = ref_rgb.permute(1, 2, 0).cpu().numpy()
        x0, y0 = np.meshgrid(range(crop_size), range(crop_size))
        hp_full = np.stack([x0, y0, np.ones_like(x0)], -1)  # augmented coord
        hp_full = hp_full.astype(np.float32)
        hp_raw = hp_full @ K2mat(full2crop).T  # raw image coord
        x0 = hp_raw[..., 0].astype(np.float32)
        y0 = hp_raw[..., 1].astype(np.float32)
        ref_rgb = cv2.resize(ref_rgb, (crop_size, crop_size))
        ref_rgb = cv2.remap(
            ref_rgb,
            x0,
            y0,
            interpolation=mode,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=borderValue,
        ).reshape(ref_rgb.shape)
        # cv2.imwrite("tmp/ref_rgb.png", ref_rgb[..., ::-1] * 255)
        ref_rgb = torch.tensor(ref_rgb, device=dev).permute(2, 0, 1)
        return ref_rgb

    @staticmethod
    def sample_random_viewpoint(radius=2):
        # render a random novel view
        bs = 1
        elevation = 0
        min_ver = max(min(-30, -30 - elevation), -80 - elevation)
        max_ver = min(max(30, 30 - elevation), 80 - elevation)

        # sample camera
        polar = [np.random.randint(min_ver, max_ver)] * bs
        azimuth = [np.random.randint(-180, 180)] * bs
        radius_offset = [0] * bs
        c2w = orbit_camera(elevation + polar[0], azimuth[0], radius + radius_offset[0])
        w2c = np.linalg.inv(c2w)
        # GL to CV for both obj and cam space
        w2c[1:3] *= -1  # left size: flip cam
        w2c[:, 1:3] *= -1  # right side: flip obj
        return w2c, polar, azimuth, radius_offset

    def sample_rand_viewpoint_around(self, w2c0):
        w2c, polar, azimuth, radius_offset = self.sample_random_viewpoint()
        w2c = torch.tensor(w2c, dtype=torch.float32, device=self.device)
        nv_angle = rot_angle(w2c[:3, :3])
        nv_factor = nv_angle / np.pi
        # dataset extrinsics at frameid | if w2c=I, then w2w0 = w2c
        w2c0[:3, :3] = w2c[:3, :3] @ w2c0[:3, :3]
        w2c = w2c0
        return w2c, polar, azimuth, radius_offset, nv_factor

    def sample_rand_Kmat_around(self, Kmat, crop2raw, crop_size, ref_rgb):
        # warp ref_rgb to normal aspect ratio and make it smaller
        extend_factor = np.random.uniform(1.1, 1.5)
        # relative focal length
        full2crop = self.construct_uncrop_mat(crop2raw, extend_factor=extend_factor)
        # restore the image
        ref_rgb = self.uncrop_img(ref_rgb, full2crop, crop_size)
        # dataset intrinsics | if full2crop=I, then Kmat = Kmat
        full2crop = torch.tensor(full2crop, dtype=torch.float32, device=self.device)
        Kmat = K2inv(full2crop) @ Kmat
        return Kmat, ref_rgb

    def compute_diffusion_loss(self, loss_dict, batch):
        crop_size = self.config["train_res"]
        render_size = crop_size
        frameid = self.get_frameid(batch)
        # NOTE: gradient from diffusion might make the camera optimization unstable
        cam_dict, Kmat, w2c = self.compute_camera_samples(batch, crop_size)
        cam_dict["render_resolution"] = render_size

        # prepare for diffusion, only use one frame
        frameid = frameid[0]
        Kmat = Kmat[0]
        w2c = w2c[0]
        ref_rgb = torch.where(
            batch["mask"] > 0, batch["rgb"], torch.ones_like(batch["rgb"])
        )[0]
        crop2raw = batch["crop2raw"][0]

        # render novel view for diffusion guidance loss
        (
            w2c_nv,
            polar,
            azimuth,
            radius_offset,
            nv_factor,
        ) = self.sample_rand_viewpoint_around(w2c)
        Kmat_nv, ref_rgb_nv = self.sample_rand_Kmat_around(
            Kmat, crop2raw, crop_size, ref_rgb
        )
        rendered = self.render(cam_dict, Kmat_nv, w2c=w2c_nv, frameid=frameid)
        rgb_nv = rendered["rgb"]
        bg_color = F.interpolate(self.gaussians.get_bg_color()[None], render_size)
        rgb_nv = rgb_nv * rendered["alpha"] + bg_color * (1 - rendered["alpha"])

        # compute loss
        if self.config["guidance_sd_wt"] > 0:
            loss_guidance_sd = self.guidance_sd.train_step(
                rgb_nv, step_ratio=self.progress
            )
            loss_dict["guidance_sd"] = loss_guidance_sd

        if self.config["guidance_zero123_wt"] > 0:
            self.guidance_zero123.get_img_embeds(ref_rgb_nv[None], frameid)
            embeddings = self.guidance_zero123.get_embeddings(frameid)
            step_ratio = np.random.rand()
            # # map step_ratio to progress-1
            # progress_ratio = min(0.2, self.current_steps / 4000)
            # step_ratio = progress_ratio + (1 - progress_ratio) * step_ratio
            step_ratio = np.clip(step_ratio, 0.02, 0.98)

            loss_guidance_zero123 = self.guidance_zero123.train_step(
                rgb_nv, polar, azimuth, radius_offset, embeddings, step_ratio=step_ratio
            )
            loss_guidance_zero123 = loss_guidance_zero123 * (0.5 * nv_factor + 0.5)
            loss_dict["guidance_zero123"] = loss_guidance_zero123

            # # DEBUG
            # outputs = self.guidance_zero123.refine(
            #     ref_rgb_nv[None],
            #     elevation=polar,
            #     azimuth=azimuth,
            #     radius=radius_offset,
            #     embeddings=embeddings,
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
            # cv2.imwrite(
            #     "tmp/rgb_ref.jpg",
            #     ref_rgb_nv.permute(1, 2, 0).cpu().numpy()[..., ::-1] * 255,
            # )

    def compute_reg_loss(self, loss_dict, frameid):
        if self.config["reg_least_deform_wt"] > 0:
            loss_dict["reg_least_deform"] = self.gaussians.get_least_deform_loss(
                frameid=frameid[0, 0], frameid_2=frameid[0, 1]
            )
        if self.config["reg_least_action_wt"] > 0:
            loss_dict["reg_least_action"] = self.gaussians.get_least_action_loss()
        if self.config["reg_arap_wt"] > 0:
            loss_dict["reg_arap"] = self.gaussians.get_arap_loss(
                frameid=frameid[0, 0], frameid_2=frameid[0, 1]
            )
        if self.config["reg_lab4d_wt"] > 0:
            loss_dict["reg_lab4d"] = self.gaussians.get_lab4d_loss(frameid)

    @staticmethod
    def mask_losses(loss_dict, maskfg, vis2d, mask_pred, config):
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
        keys_allpix = ["mask", "flow", "rgb"]
        # field type specific keys
        keys_type_specific = []
        # rendered-mask weighted losses
        keys_mask_weighted = ["flow", "rgb"]

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

        # apply mask weights
        for k in keys_mask_weighted:
            loss_dict[k] *= mask_pred.detach()

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

    @staticmethod
    def reshape_batch_inv(batch):
        """Reshape a batch to merge the pair dimension into the batch dimension

        Args:
            batch (Dict): Arbitrary dataloader outputs (M*2, ...). This is
                modified in place to reshape each value to (M, 2, ...)
        """
        for k, v in batch.items():
            batch[k] = v.reshape(-1, 2, *v.shape[1:])

    @staticmethod
    def reshape_batch(batch):
        """Reshape a batch to merge the pair dimension into the batch dimension

        Args:
            batch (Dict): Arbitrary dataloader outputs (M, 2, ...). This is
                modified in place to reshape each value to (M*2, ...)
        """
        for k, v in batch.items():
            batch[k] = v.view(-1, *v.shape[2:])

    @staticmethod
    def NCHW2NHWC(batch):
        """Convert batch from NCHW to NHWC

        Args:
            batch (Dict): Arbitrary dataloader outputs (M, ...). This is
                modified in place to convert each value from NCHW to NHWC
        """
        for k, v in batch.items():
            if v.ndim == 4:
                batch[k] = v.permute(0, 2, 3, 1)

    @staticmethod
    def NHWC2NCHW(batch):
        """
        Args:
            batch (Dict): Arbitrary dataloader outputs (M, ...). This is
                modified in place to convert each value from NHWC to NCHW
        """
        for k, v in batch.items():
            if v.ndim == 4:
                batch[k] = v.permute(0, 3, 1, 2)

    def augment_visualization_nv(self, rendered, cam_dict, Kmat, w2c, frameid):
        # modify w2c to be a turn-table view
        num_nv = 8
        w2c_nv = get_rotating_cam(num_nv, max_angle=360)
        w2c_nv = torch.tensor(w2c_nv, dtype=torch.float32, device=w2c.device)
        w2c_nv[..., :3, 3] = w2c[0, 0, :3, 3]
        w2c_nv = w2c_nv.reshape(-1, 2, 4, 4)
        Kmat_nv = Kmat[:1, :1].repeat(num_nv // 2, 2, 1, 1)
        frameid_nv = frameid[:1, :1].repeat(num_nv // 2, 2)
        rendered_nv = self.render_pair(
            cam_dict, Kmat_nv, w2c=w2c_nv, frameid=frameid_nv
        )
        for k, v in rendered_nv.items():
            rendered[k] = torch.cat([rendered[k], v], 0)

    @torch.no_grad()
    def evaluate(self, batch, is_pair=True):
        """Evaluate model on a batch of data"""
        self.process_frameid(batch)
        if not is_pair:
            # fake a pair
            for k, v in batch.items():
                batch[k] = fake_a_pair(v)
        else:
            self.reshape_batch_inv(batch)
        # render mode or eval mode during training
        if "render_res" in self.config.keys():
            crop_size = self.config["render_res"]
        else:
            crop_size = self.config["eval_res"]
        cam_dict, Kmat, w2c = self.compute_camera_samples(batch, crop_size)
        frameid = self.get_frameid(batch)

        # TODO: get deformation before rendering
        self.gaussians.update_trajectory(frameid)

        rendered = self.render_pair(cam_dict, Kmat, w2c=w2c, frameid=frameid)
        self.augment_visualization_nv(rendered, cam_dict, Kmat, w2c, frameid)

        # if is_pair:
        #     self.reshape_batch(rendered)
        # else:
        for k, v in rendered.items():
            rendered[k] = v[:, 0]

        scalars = {}
        out_dict = {"rgb": [], "depth": [], "alpha": [], "xyz": [], "flow": []}
        for k, v in rendered.items():
            if k in out_dict.keys():
                out_dict[k] = v.permute(0, 2, 3, 1).cpu().numpy()
        bg_color = self.gaussians.get_bg_color().permute(1, 2, 0)[None].cpu().numpy()
        out_dict["bg_color"] = bg_color
        bg_color = cv2.resize(bg_color[0], (crop_size,) * 2)[None]
        out_dict["rgb_nv"] = out_dict["rgb"] * out_dict["alpha"] + bg_color * (
            1 - out_dict["alpha"]
        )

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

    def set_progress(self, current_steps, progress, sub_progress):
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

        # knn for arap
        anchor_x = (0, 1.0)
        anchor_y = (1.0, 0.0)
        type = "linear"
        if self.progress > config["inc_warmup_ratio"]:
            ratio_knn = interp_wt(anchor_x, anchor_y, progress, type=type)
        else:
            ratio_knn = interp_wt(anchor_x, anchor_y, sub_progress, type=type)
        self.gaussians.ratio_knn = ratio_knn

        # arap wt
        loss_name = "reg_arap_wt"
        anchor_x = (0, 200.0)
        anchor_y = (0.0, 1.0)
        type = "linear"
        self.set_loss_weight(loss_name, anchor_x, anchor_y, current_steps, type=type)

        # least action wt
        loss_name = "reg_least_action_wt"
        anchor_x = (0, 1.0)
        anchor_y = (1.0, 0.0)
        type = "linear"
        if self.progress > config["inc_warmup_ratio"]:
            self.set_loss_weight(loss_name, anchor_x, anchor_y, progress, type=type)
        else:
            self.set_loss_weight(loss_name, anchor_x, anchor_y, sub_progress, type=type)

        # diffusion wt
        loss_name = "guidance_zero123_wt"
        anchor_x = (0, 200.0)
        anchor_y = (0.0, 1.0)
        type = "linear"
        self.set_loss_weight(loss_name, anchor_x, anchor_y, current_steps, type=type)

        # # flow wt
        # loss_name = "flow_wt"
        # anchor_x = (
        #     config["inc_warmup_ratio"],
        #     min(config["inc_warmup_ratio"] + 0.001, 1),
        # )
        # anchor_y = (1.0, 0.0)
        # type = "linear"
        # self.set_loss_weight(loss_name, anchor_x, anchor_y, progress, type=type)

        # reg_lab4d_wt
        loss_name = "reg_lab4d_wt"
        anchor_x = (0, 2000.0)
        anchor_y = (1.0, 0.0)
        type = "linear"
        self.set_loss_weight(loss_name, anchor_x, anchor_y, current_steps, type=type)

    def set_loss_weight(
        self, loss_name, anchor_x, anchor_y, progress_ratio, type="linear"
    ):
        """Set a loss weight according to the current training step

        Args:
            loss_name (str): Name of loss weight to set
            anchor_x: Tuple of optimization steps [x0, x1]
            anchor_y: Tuple of loss values [y0, y1]
            progress_ratio (float): Current optimization ratio, 0 to 1
            type (str): Interpolation type ("linear" or "log")
        """
        if "%s_init" % loss_name not in self.config.keys():
            self.config["%s_init" % loss_name] = self.config[loss_name]
        factor = interp_wt(anchor_x, anchor_y, progress_ratio, type=type)
        self.config[loss_name] = self.config["%s_init" % loss_name] * factor

    @torch.no_grad()
    def get_field_params(self):
        """Get beta values for all neural fields

        Returns:
            betas (Dict): Beta values for each neural field
        """
        beta_dicts = {"num_pts": self.gaussians.get_num_pts}
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
        grads = []
        vis_masks = []
        for aux in self.rendered_aux:
            if aux["viewspace_points"].grad is None:
                continue
            grad = aux["viewspace_points"].grad
            # if "means2D_tmp" in aux.keys():
            #     grad = grad + aux["means2D_tmp"].grad
            grads.append(grad)
            vis_masks.append(aux["visibility_mask"])
        len_grads = len(grads)
        for grad, vis_mask in zip(grads, vis_masks):
            # d(L1+L2)/dx = dL1/dx + dL2/dx
            self.gaussians.add_xyz_grad_stats(grad * len_grads, vis_mask)
        # delete after update
        del self.rendered_aux

    def mlp_init(self):
        """Initialize camera transforms, geometry, articulations, and camera
        intrinsics for all neural fields from external priors
        """
        if isinstance(self.gaussians.camera_mlp, CameraPredictor):
            self.gaussians.camera_mlp.init_weights()


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
