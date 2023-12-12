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
from lab4d.utils.loss_utils import get_mask_balance_wt
from lab4d.utils.geom_utils import fov_to_focal, focal_to_fov, K2mat, K2inv
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

        # 3DGS
        sh_degree = config["sh_degree"] if "sh_degree" in config else 3
        white_background = (
            config["white_background"] if "white_background" in config else True
        )
        radius = config["radius"] if "radius" in config else 2.5
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
        # DEBUG
        # mesh = trimesh.load("tmp/0.obj")
        # pcd = BasicPointCloud(
        #     mesh.vertices,
        #     mesh.visual.vertex_colors[:, :3] / 255,
        #     np.zeros((mesh.vertices.shape[0], 3)),
        # )
        # self.initialize(input=pcd)

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
        w2c=None,
        scaling_modifier=1.0,
        bg_color=None,
        override_color=None,
    ):
        assert "Kmat" in camera_dict

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
        cov3D_precomp = None
        scales = self.gaussians.get_scaling
        rotations = self.gaussians.get_rotation

        if w2c is not None:
            if not torch.is_tensor(w2c):
                w2c = torch.tensor(w2c, dtype=torch.float, device=means3D.device)
            means3D = means3D @ w2c[:3, :3].T + w2c[:3, 3][None]
            # TODO verify this rotation is Nx4 quaternion
            # rotations = rotations @ w2c[:3, :3].T

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        if colors_precomp is None:
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

    def get_default_cam(self, render_resolution=256, Kmat=np.eye(3)):
        # focal=1 corresponds to 90 degree fov and identity projection matrix
        # convert this to a dict
        near = 0.01
        far = 5
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
        loss_dict = {}

        # render reference view
        crop_size = self.config["train_res"]
        bg_color = self.get_bg_color()

        frameid = batch["frameid_sub"][0, 0]
        crop2raw = batch["crop2raw"][0, 0]
        Kmat = K2mat(self.data_info["intrinsics"][frameid])
        Kmat = self.compute_Kmat(crop_size, crop2raw, Kmat)

        w2c = self.data_info["rtmat"][1][frameid].astype(np.float32)
        cam_dict = self.get_default_cam(Kmat=Kmat)

        rendered = self.render(cam_dict, w2c=w2c, bg_color=bg_color)
        rgb = rendered["rgb"]
        mask = rendered["alpha"]

        # prepare reference view GT
        ref_rgb = batch["rgb"][:1, :1]
        ref_mask = batch["mask"][:1, :1].float()
        ref_vis2d = batch["vis2d"][:1, :1].float()
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

        # reference view loss
        loss_dict["rgb"] = (rgb - ref_rgb).pow(2)
        loss_dict["mask"] = (mask - ref_mask).pow(2)
        # mask_balance_wt = get_mask_balance_wt(
        #     batch["mask"], batch["vis2d"], batch["is_detected"]
        # )
        # loss_dict["mask"] *= mask_balance_wt

        # mask loss
        self.mask_losses(loss_dict, ref_mask, ref_vis2d, self.config)

        self.compute_reg_loss(loss_dict, ref_rgb)

        # weight each loss term
        self.apply_loss_weights(loss_dict, self.config)
        return loss_dict

    def compute_reg_loss(self, loss_dict, ref_rgb):
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
        pose = orbit_camera(elevation + polar[0], azimuth[0], self.radius + radius[0])
        pose = np.linalg.inv(pose)
        # GL to CV
        pose[1:3] *= -1
        Kmat = np.eye(3)
        Kmat[0, 0] = fov_to_focal(np.pi / 4)
        Kmat[1, 1] = fov_to_focal(np.pi / 4)
        cam_dict = self.get_default_cam(Kmat=Kmat)
        cam_dict["w2c"] = pose

        # render
        rendered = self.render(cam_dict, bg_color=bg_color)
        rgb_nv = rendered["rgb"]

        # compute loss
        if self.config["guidance_sd_wt"] > 0:
            loss_guidance_sd = self.guidance_sd.train_step(
                rgb_nv, step_ratio=self.progress
            )
            loss_dict["guidance_sd"] = loss_guidance_sd

        if self.config["guidance_zero123_wt"] > 0:
            self.guidance_zero123.get_img_embeds(ref_rgb)

            loss_guidance_zero123 = self.guidance_zero123.train_step(
                rgb_nv, polar, azimuth, radius, step_ratio=self.progress
            )
            loss_dict["guidance_zero123"] = loss_guidance_zero123

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
        keys_type_specific = ["rgb"]

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
        for k, v in loss_dict.items():
            # average over non-zero pixels
            v = v[v > 0]
            if v.numel() > 0:
                loss_dict[k] = v.mean()
            else:
                loss_dict[k] = v.sum()  # return zero

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
    def evaluate(self, batch, is_pair=True):
        """Evaluate model on a batch of data"""
        crop_size = self.config["eval_res"]
        self.process_frameid(batch)
        scalars = {}
        out_dict = {"rgb": [], "depth": [], "alpha": []}

        for idx in tqdm.tqdm(range(0, len(batch["frameid"]) // 2)):
            if is_pair:
                idx = idx * 2
            frameid = batch["frameid_sub"][idx]
            crop2raw = batch["crop2raw"][idx]
            Kmat = K2mat(self.data_info["intrinsics"][frameid])
            Kmat = self.compute_Kmat(crop_size, crop2raw, Kmat)

            w2c = self.data_info["rtmat"][1][frameid].astype(np.float32)
            cam_dict = self.get_default_cam(Kmat=Kmat)

            rendered = self.render(cam_dict, w2c=w2c)
            out_dict["rgb"].append(rendered["rgb"][0].permute(1, 2, 0).cpu().numpy())
            out_dict["depth"].append(
                rendered["depth"][0].permute(1, 2, 0).cpu().numpy()
            )
            out_dict["alpha"].append(
                rendered["alpha"][0].permute(1, 2, 0).cpu().numpy()
            )
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
