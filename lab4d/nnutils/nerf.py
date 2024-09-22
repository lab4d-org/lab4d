# Copyright (c) 2023 Gengshan Yang, Carnegie Mellon University.
import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from torch import nn
from torch.autograd.functional import jacobian


from lab4d.nnutils.appearance import AppearanceEmbedding
from lab4d.nnutils.base import CondMLP
from lab4d.nnutils.embedding import PosEmbedding
from lab4d.nnutils.pose import (
    CameraMLP,
    CameraMLP_so3,
    CameraConst,
    CameraExplicit,
    CameraMix,
    CameraMixSE3,
)
from lab4d.nnutils.visibility import VisField
from lab4d.utils.decorator import train_only_fields
from lab4d.utils.geom_utils import (
    Kmatinv,
    apply_se3mat,
    extend_aabb,
    get_near_far,
    marching_cubes,
    pinhole_projection,
    check_inside_aabb,
    compute_rectification_se3,
)
from lab4d.utils.loss_utils import align_tensors, compute_se3_smooth_loss_2nd
from lab4d.utils.quat_transform import (
    quaternion_apply,
    quaternion_translation_inverse,
    quaternion_translation_mul,
    quaternion_translation_to_se3,
    dual_quaternion_to_quaternion_translation,
)
from lab4d.utils.render_utils import sample_cam_rays, sample_pdf, compute_weights
from lab4d.utils.torch_utils import compute_gradient, flip_pair, compute_gradients_sdf
from lab4d.utils.vis_utils import append_xz_plane


class NeRF(nn.Module):
    """A static neural radiance field with an MLP backbone.

    Args:
        vid_info (Dict): Dataset metadata from get_data_info()
        D (int): Number of linear layers for density (sigma) encoder
        W (int): Number of hidden units in each MLP layer
        num_freq_xyz (int): Number of frequencies in position embedding
        num_freq_dir (int): Number of frequencies in direction embedding
        appr_channels (int): Number of channels in the global appearance code
            (captures shadows, lighting, and other environmental effects)
        appr_num_freq_t (int): Number of frequencies in the time embedding of
            the global appearance code
        num_inst (int): Number of distinct object instances. If --nosingle_inst
            is passed, this is equal to the number of videos, as we assume each
            video captures a different instance. Otherwise, we assume all videos
            capture the same instance and set this to 1.
        inst_channels (int): Number of channels in the instance code
        skips (List(int)): List of layers to add skip connections at
        activation (Function): Activation function to use (e.g. nn.ReLU())
        init_beta (float): Initial value of beta, from Eqn. 3 of VolSDF.
            We transform a learnable signed distance function into density using
            the CDF of the Laplace distribution with zero mean and beta scale.
        init_scale (float): Initial geometry scale factor.
        color_act (bool): If True, apply sigmoid to the output RGB
        field_arch (Class): Field architecture to use
        extrinsics_type (str): Camera pose initialization method. Options:
            "mlp", "const", "explicit", "mix", "mixse3"
        invalid_vid (int): Video that are not used for training.
    """

    def __init__(
        self,
        data_info,
        D=5,
        W=256,
        num_freq_xyz=10,
        num_freq_dir=4,
        appr_channels=32,
        appr_num_freq_t=6,
        num_inst=1,
        inst_channels=32,
        skips=[4],
        activation=nn.ReLU(True),
        init_beta=0.1,
        init_scale=0.1,
        color_act=True,
        field_arch=CondMLP,
        extrinsics_type="mlp",
        invalid_vid=-1,
    ):
        rtmat = data_info["rtmat"]
        frame_info = data_info["frame_info"]
        frame_offset = data_info["frame_info"]["frame_offset"]
        frame_offset_raw = data_info["frame_info"]["frame_offset_raw"]
        geom_path = data_info["geom_path"]

        super().__init__()

        # dataset info
        self.frame_offset = frame_offset
        self.frame_offset_raw = frame_offset_raw
        self.frame_mapping = frame_info["frame_mapping"]
        self.num_frames = frame_offset[-1]
        self.num_inst = num_inst
        self.invalid_vid = invalid_vid

        # position and direction embedding
        self.pos_embedding = PosEmbedding(3, num_freq_xyz)
        self.dir_embedding = PosEmbedding(3, num_freq_dir)

        # xyz encoding layers
        # TODO: add option to replace with instNGP
        self.basefield = field_arch(
            num_inst=self.num_inst,
            D=D,
            W=W,
            in_channels=self.pos_embedding.out_channels,
            inst_channels=inst_channels,
            out_channels=W,
            skips=skips,
            activation=activation,
            final_act=True,
        )

        # color
        self.pos_embedding_color = PosEmbedding(3, num_freq_xyz + 2)
        self.colorfield = field_arch(
            num_inst=self.num_inst,
            D=2,
            W=W,
            in_channels=self.pos_embedding_color.out_channels,
            inst_channels=inst_channels,
            out_channels=W,
            skips=skips,
            activation=activation,
            final_act=True,
        )
        self.color_act = color_act

        # non-directional appearance code (shadow, lighting etc.)
        if appr_channels > 0:
            self.appr_embedding = AppearanceEmbedding(
                frame_info, appr_channels, num_freq_t=appr_num_freq_t
            )
        self.appr_channels = appr_channels

        # output layers
        self.sdf = nn.Linear(W, 1)
        self.rgb = nn.Sequential(
            nn.Linear(W + self.dir_embedding.out_channels + self.appr_channels, W // 2),
            activation,
            nn.Linear(W // 2, 3),
        )

        self.in_channels = 6
        self.out_channels = 4

        beta = torch.tensor([init_beta])
        self.logibeta = nn.Parameter(-beta.log())  # beta: transparency

        scale = torch.tensor([init_scale])  # scale of the field
        self.logscale = nn.Parameter(scale.log())
        self.scale_const = 0.2  # "ideal" space for reconstruction to metric space

        # # initialize with per-sequence pose
        # for i in range(1, len(frame_offset) - 1):
        #     rtmat[frame_offset[i] : frame_offset[i + 1]] = rtmat[frame_offset[i] : frame_offset[i + 1]] @ np.linalg.inv(rtmat[frame_offset[i] : frame_offset[i] + 1])

        # camera pose: field to camera
        rtmat[..., :3, 3] *= init_scale
        self.construct_extrinsics(rtmat, frame_info, extrinsics_type)
        # visibility mlp
        self.vis_mlp = VisField(self.num_inst, field_arch=field_arch)

        # load initial mesh, define aabb
        self.init_proxy(geom_path, init_scale)
        self.init_aabb()

        # non-parameters are not synchronized
        self.register_buffer(
            "near_far", torch.zeros(frame_offset_raw[-1], 2), persistent=False
        )

        field2world = torch.zeros(4, 4)[None].expand(self.num_inst, -1, -1).clone()
        self.register_buffer("field2world", field2world, persistent=True)

        # inverse sampling
        self.use_importance_sampling = True
        self.symm_ratio = 0.0

    def construct_extrinsics(self, rtmat, frame_info, extrinsics_type):
        if extrinsics_type == "mlp":
            self.camera_mlp = CameraMLP_so3(rtmat, frame_info=frame_info)
        elif extrinsics_type == "mlp_nodelta":
            self.camera_mlp = CameraMLP(rtmat, frame_info=frame_info)
        elif extrinsics_type == "const":
            self.camera_mlp = CameraConst(rtmat, frame_info=frame_info)
        elif extrinsics_type == "explicit":
            self.camera_mlp = CameraExplicit(rtmat, frame_info=frame_info)
        elif extrinsics_type == "mix":
            self.camera_mlp = CameraMix(rtmat, frame_info=frame_info, const_vid_id=0)
        elif extrinsics_type == "mixse3":
            self.camera_mlp = CameraMixSE3(rtmat, frame_info=frame_info, const_vid_id=0)
        else:
            raise NotImplementedError

    def forward(self, xyz, dir=None, frame_id=None, inst_id=None):
        """
        Args:
            xyz: (M,N,D,3) Points along ray in object canonical space
            dir: (M,N,D,3) Ray direction in object canonical space
            frame_id: (M,) Frame id. If None, render at all frames
            inst_id: (M,) Instance id. If None, render for the average instance
        Returns:
            rgb: (M,N,D,3) Rendered RGB
            sdf: (M,N,D,1) Signed distance (negative inside)
            sigma: (M,N,D,1) Denstiy
        """
        if frame_id is not None and not isinstance(frame_id, str):
            assert frame_id.ndim == 1
        if inst_id is not None:
            assert inst_id.ndim == 1

        xyz_sdf = xyz.clone()
        # symmetrize canonical shape
        if self.symm_ratio > 0.0 and self.category == "fg":
            xyz_x = xyz_sdf[..., :1].clone()
            symm_mask = torch.rand_like(xyz_x) < self.symm_ratio
            xyz_x[symm_mask] = -xyz_x[symm_mask]
            xyz_sdf = torch.cat([xyz_x, xyz_sdf[..., 1:3]], -1)

        sdf, xyz_feat = self.forward_sdf(xyz_sdf, inst_id=inst_id)
        # ideal space to metric space
        sdf = sdf / self.logscale.exp() * self.scale_const

        ibeta = self.logibeta.exp()
        # density = torch.sigmoid(-sdf * ibeta) * ibeta  # neus
        density = (
            0.5 + 0.5 * sdf.sign() * torch.expm1(-sdf.abs() * ibeta)
        ) * ibeta  # volsdf
        out = sdf, density

        if dir is not None:
            dir_embed = self.dir_embedding(dir)
            if self.appr_channels > 0:
                appr_embed = self.appr_embedding.get_vals(frame_id)
                extra_dims = len(dir_embed.shape) - len(appr_embed.shape)
                appr_embed = appr_embed.reshape(
                    appr_embed.shape[:1] + (1,) * extra_dims + appr_embed.shape[1:]
                )
                appr_embed = appr_embed.expand(
                    dir_embed.shape[:-1] + (appr_embed.shape[-1],)
                )
                appr_embed = torch.cat([dir_embed, appr_embed], -1)
            else:
                appr_embed = dir_embed

            xyz_embed = self.pos_embedding_color(xyz)
            xyz_feat = xyz_feat + self.colorfield(xyz_embed, inst_id)

            rgb = self.rgb(torch.cat([xyz_feat, appr_embed], -1))
            if self.color_act:
                rgb = rgb.sigmoid()
            out = (rgb,) + out
        return out

    def forward_sdf(self, xyz, inst_id=None):
        """Forward pass for signed distance function
        Args:
            xyz: (M,N,D,3) Points along ray in object canonical space
            inst_id: (M,) Instance id. If None, render for the average instance

        Returns:
            sdf: (M,N,D,1) Signed distance (negative inside)
            xyz_feat: (M,N,D,W) Features from the xyz encoder
        """
        xyz_embed = self.pos_embedding(xyz)
        xyz_feat = self.basefield(xyz_embed, inst_id)

        sdf = self.sdf(xyz_feat)  # negative inside, positive outside
        return sdf, xyz_feat

    def get_init_sdf_fn(self):
        """Initialize signed distance function from mesh geometry

        Returns:
            sdf_fn_torch (Function): Signed distance function
        """
        from pysdf import SDF
        sdf_fn_numpy = SDF(self.proxy_geometry.vertices, self.proxy_geometry.faces)

        def sdf_fn_torch(pts):
            sdf = -sdf_fn_numpy(pts.cpu().numpy())[:, None]  # negative inside
            sdf = torch.tensor(sdf, device=pts.device, dtype=pts.dtype)
            return sdf

        return sdf_fn_torch

    def mlp_init(self):
        """Initialize camera transforms and geometry from external priors"""
        self.camera_mlp.mlp_init()
        self.update_near_far(beta=0)
        sdf_fn_torch = self.get_init_sdf_fn()

        self.geometry_init(sdf_fn_torch)

    def init_proxy(self, geom_path, init_scale):
        """Initialize the geometry from a mesh

        Args:
            geom_path (List(str)): paths to initial shape mesh
            init_scale (float): Geometry scale factor
        """
        import os

        if os.path.isfile(geom_path[0]):
            mesh = trimesh.load(geom_path[0])
            mesh.vertices = mesh.vertices * init_scale
        else:
            mesh = trimesh.creation.uv_sphere(radius=0.12 * init_scale / 0.2, count=[4, 4])

        self.proxy_geometry = mesh

    def get_proxy_geometry(self):
        """Get proxy geometry

        Returns:
            proxy_geometry (Trimesh): Proxy geometry
        """
        return self.proxy_geometry

    def init_aabb(self):
        """Initialize axis-aligned bounding box"""
        self.register_buffer("aabb", torch.zeros(2, 3))
        self.update_aabb(beta=0)

    def geometry_init(self, sdf_fn, nsample=4096):
        """Initialize SDF using tsdf-fused geometry if radius is not given.
        Otherwise, initialize sdf using a unit sphere

        Args:
            sdf_fn (Function): Maps vertices to signed distances
            nsample (int): Number of samples
        """
        device = next(self.parameters()).device
        # setup optimizer
        total_steps = 5000
        lr = 2e-3
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            lr,
            total_steps,
            pct_start=0.1,
            cycle_momentum=False,
            anneal_strategy="linear",
        )

        def sdf_pred_fn(pts):
            sdf, _ = self.forward_sdf(pts, inst_id=inst_id)
            return sdf

        # optimize
        for i in range(total_steps):
            optimizer.zero_grad()

            # sample points and gt sdf
            inst_id = None # torch.randint(0, self.num_inst, (nsample,), device=device)

            # sample points
            pts, _, _ = self.sample_points_aabb(nsample//4*3, extend_factor=0.0)
            pts_neg, _, _ = self.sample_points_aabb(nsample//4, extend_factor=0.5)
            pts = torch.cat([pts, pts_neg], 0)

            # get sdf from proxy geometry
            sdf_gt = sdf_fn(pts)

            # evaluate sdf loss
            sdf = sdf_pred_fn(pts)
            scale = align_tensors(sdf, sdf_gt)
            sdf_loss = (sdf * scale.detach() - sdf_gt).pow(2).mean()

            # evaluate visibility loss
            vis = self.vis_mlp(pts, inst_id=inst_id)
            vis_loss = -F.logsigmoid(vis).mean()
            vis_loss = vis_loss * 0.01

            # evaluate eikonal loss
            eikonal_loss, _ = self.compute_eikonal(
                pts[:, None, None], inst_id=inst_id, sample_ratio=1
            )
            eikonal_loss = eikonal_loss[eikonal_loss > 0].mean()
            eikonal_loss = eikonal_loss * 1e-4

            total_loss = sdf_loss + vis_loss + eikonal_loss
            total_loss.backward()
            optimizer.step()
            scheduler.step()
            if i % 100 == 0:
                print(f"iter {i}, loss {total_loss.item()}")
                # aabb = self.get_aabb(inst_id=inst_id)
                # mesh_gt = marching_cubes(sdf_fn, aabb[0], level=0.0)
                # mesh_pred = marching_cubes(sdf_pred_fn, aabb[0], level=0.0)
                # mesh_gt.export("tmp/%04d-gt.obj"%i)
                # mesh_pred.export("tmp/%04d-pred.obj"%i)

    def update_proxy(self):
        """Extract proxy geometry using marching cubes"""
        if self.category == "fg":
            if self.invalid_vid >= 0 and self.num_inst > 1:
                inst_id = list(range(self.num_inst))
                inst_id.remove(self.invalid_vid)
                inst_id = inst_id[0]
            else:
                inst_id = None
            mesh = self.extract_canonical_mesh(
                level=0.005, vis_thresh=-10.0, inst_id=inst_id
            )
        else:
            mesh = self.extract_canonical_mesh(level=0.005)
        if len(mesh.vertices) > 3:
            self.proxy_geometry = mesh

    def visibility_func(self, xyz, inst_id=None):
        """Compute visibility function

        Args:
            xyz: (M,N,D,3) Points along ray in object canonical space
            inst_id: (M,) Instance id. If None, render for the average instance
        Returns:
            vis: (M,N,D,1) Visibility score
        """
        vis = self.vis_mlp(xyz, inst_id=inst_id)
        return vis

    @torch.no_grad()
    def extract_canonical_mesh(
        self,
        grid_size=64,
        level=0.0,
        inst_id=None,
        vis_thresh=0.0,
        use_extend_aabb=True,
    ):
        """Extract canonical mesh using marching cubes

        Args:
            grid_size (int): Marching cubes resolution
            level (float): Contour value to search for isosurfaces on the signed
                distance function
            inst_id: (int) Instance id. If None, extract for the average instance
            vis_thresh (float): threshold for visibility value to remove invisible pts.
            use_extend_aabb (bool): If True, extend aabb by 50% to get a loose proxy.
              Used at training time.
        Returns:
            mesh (Trimesh): Extracted mesh
        """
        if inst_id is not None:
            inst_id = torch.tensor([inst_id], device=next(self.parameters()).device)
            aabb = self.get_aabb(inst_id=inst_id)  # 2,3
        else:
            aabb = self.get_aabb()
        sdf_func = lambda xyz: self.forward_sdf(xyz, inst_id=inst_id)[0]
        vis_func = lambda xyz: self.visibility_func(xyz, inst_id=inst_id) > vis_thresh
        if use_extend_aabb:
            aabb = extend_aabb(aabb, factor=0.5)
        if self.category == "fg" and self.use_cc:
            apply_connected_component=True
        else:
            apply_connected_component=False
        mesh = marching_cubes(
            sdf_func,
            aabb[0],
            visibility_func=vis_func,
            grid_size=grid_size,
            level=level,
            apply_connected_component=apply_connected_component,
        )
        # cut bg
        # if self.category == "bg":
        #     # for bg
        #     bounds = np.asarray(mesh.bounds.tolist())
        #     bounds[0,1] = -0.05
        #     box = trimesh.creation.box(bounds=bounds)
        #     mesh = mesh.slice_plane(box.facets_origin, -box.facets_normal)
        return mesh

    @torch.no_grad()
    def extract_canonical_color(self, mesh, frame_id=None):
        """Extract color on canonical mesh vertices

        Args:
            mesh (Trimesh): Canonical mesh
        Returns:
            color (np.ndarray): Color on vertices
        """
        device = next(self.parameters()).device
        if torch.is_tensor(mesh):
            verts = mesh
        else:
            verts = torch.tensor(mesh.vertices, dtype=torch.float32, device=device)
        dir = torch.zeros_like(verts)
        if frame_id is None:
            frame_id = "mean"
        else:
            frame_id = torch.tensor([frame_id], device=device)
        color = self.forward(verts, dir=dir, frame_id=frame_id)[0]
        return color.cpu().numpy()

    @torch.no_grad()
    def extract_canonical_sdf(self, pts):
        """Extract signed distance function on canonical pts

        Args:
            pts (np.ndarray): Points in canonical space
        Returns:
            sdf (np.ndarray): Signed distance function
        """
        device = next(self.parameters()).device
        pts = torch.tensor(pts, dtype=torch.float32, device=device)
        sdf = self.forward_sdf(pts)[0]
        return sdf.cpu().numpy()

    def get_aabb(self, inst_id=None, return_cube=False):
        """Get axis-aligned bounding box
        Args:
            inst_id: (N,) Instance id
        Returns:
            aabb: (2,3) Axis-aligned bounding box if inst_id is None, (N,2,3) otherwise
        """
        aabb = self.aabb
        if return_cube:
            aabb[0] = aabb[0].min()
            aabb[1] = aabb[1].max()
        if inst_id is None:
            return aabb[None]
        else:
            return aabb[None].repeat(len(inst_id), 1, 1)

    def get_scale(self):
        """Get scale of the proxy geometry"""
        assert self.category == "fg"
        aabb = self.get_aabb()[0]
        return (aabb[1] - aabb[0]).mean()

    def update_aabb(self, beta=0.9):
        """Update axis-aligned bounding box by interpolating with the current
        proxy geometry's bounds

        Args:
            beta (float): Interpolation factor between previous/current values
        """
        device = self.aabb.device
        bounds = self.proxy_geometry.bounds
        if bounds is not None:
            aabb = torch.tensor(bounds, dtype=torch.float32, device=device)
            aabb = extend_aabb(aabb, factor=0.2)  # 1.4x larger
            self.aabb = self.aabb * beta + aabb * (1 - beta)

    def update_near_far(self, beta=0.9):
        """Update near-far bounds by interpolating with the current near-far bounds

        Args:
            beta (float): Interpolation factor between previous/current values
        """
        # get camera
        device = next(self.parameters()).device
        with torch.no_grad():
            quat, trans = self.camera_mlp.get_vals()  # (B, 4, 4)
            rtmat = quaternion_translation_to_se3(quat, trans)

        # verts = self.proxy_geometry.vertices
        verts = trimesh.bounds.corners(self.proxy_geometry.bounds)
        if verts is not None:
            proxy_pts = torch.tensor(verts, dtype=torch.float32, device=device)
            near_far = get_near_far(proxy_pts, rtmat).to(device)
            frame_mapping = self.frame_mapping
            self.near_far.data[frame_mapping] = self.near_far.data[
                frame_mapping
            ] * beta + near_far * (1 - beta)

    def sample_points_aabb(self, nsample, extend_factor=1.0, aabb=None, return_cube=False):
        """Sample points within axis-aligned bounding box

        Args:
            nsample (int): Number of samples
            extend_factor (float): Extend aabb along each side by factor of
                the previous size
            aabb: (2,3) Axis-aligned bounding box to sample from, optional
        Returns:
            pts: (nsample, 3) Sampled points
        """
        device = next(self.parameters()).device
        frame_id = torch.randint(0, self.num_frames, (nsample,), device=device)
        inst_id = torch.randint(0, self.num_inst, (nsample,), device=device)
        if aabb is None:
            aabb = self.get_aabb(inst_id=inst_id, return_cube=return_cube)
            aabb = extend_aabb(aabb, factor=extend_factor)
        pts = (
            torch.rand(nsample, 3, dtype=torch.float32, device=device)
            * (aabb[..., 1, :] - aabb[..., 0, :])
            + aabb[..., 0, :]
        )
        return pts, frame_id, inst_id

    def visibility_decay_loss(self, nsample=512):
        """Encourage visibility to be low at random points within the aabb. The
        effect is that invisible / occluded points are assigned -inf visibility

        Args:
            nsample (int): Number of points to sample
        Returns:
            loss: (0,) Visibility decay loss
        """
        # sample random points
        pts, _, inst_id = self.sample_points_aabb(nsample)

        # evaluate loss
        vis = self.vis_mlp(pts, inst_id=inst_id)
        loss = -F.logsigmoid(-vis).mean()
        return loss

    def timesync_cam_loss(self, bg_rtmat):
        # import pdb;pdb.set_trace()
        from lab4d.utils.geom_utils import rot_angle
        quat, trans = self.camera_mlp.get_vals()
        trans = trans / self.logscale.exp()
        rtmat = quaternion_translation_to_se3(quat, trans)
        frame_offset = self.frame_offset
        bg_rtmat = torch.tensor(bg_rtmat, device=rtmat.device, dtype=rtmat.dtype)
        # camk vs cam1 x cam1_to_k_gt
        loss = []
        for i in range(1, len(frame_offset) - 1):
            cami = rtmat[frame_offset[i] : frame_offset[i + 1]]
            leni = len(cami)
            cam1_to_cami = bg_rtmat[frame_offset[i] : frame_offset[i + 1]] @ \
                        (bg_rtmat[frame_offset[0] : frame_offset[1]])[:leni].inverse()
            cami_gt = cam1_to_cami @ rtmat[frame_offset[0] : frame_offset[1]][:leni]
            assert cami.dim()==3 and cami_gt.dim()==3
            loss_rot = rot_angle(cami[:,:3,:3]@cami_gt[:,:3,:3].permute(0,2,1)).mean()
            loss_trn = (cami[:,:3,3] - cami_gt[:,:3,3]).norm(2,-1).mean()
            loss.append(loss_rot + loss_trn)

            # from lab4d.utils.vis_utils import draw_cams
            # draw_cams(rtmat[frame_offset[0] : frame_offset[1]][:leni].detach().cpu().numpy()[:1]).export("tmp/before.obj")
            # draw_cams(cami_gt.detach().cpu().numpy()[:1]).export("tmp/gt.obj")
            # draw_cams(cami.detach().cpu().numpy()[:1]).export("tmp/pred.obj")

        loss = torch.stack(loss).mean()
        return loss

    def compute_eikonal(self, xyz, inst_id=None, sample_ratio=16):
        """Compute eikonal loss and normal in the canonical space

        Args:
            xyz: (M,N,D,3) Input coordinates in canonical space
            inst_id: (M,) Instance id, or None to use the average instance
            sample_ratio (int): Fraction to subsample to make it more efficient
        Returns:
            eikonal_loss: (M,N,D,1) Squared magnitude of SDF gradient
        """
        M, N, D, _ = xyz.shape
        xyz = xyz.reshape(-1, D, 3)
        sample_size = xyz.shape[0] // sample_ratio
        if sample_size < 1:
            sample_size = 1
        if inst_id is not None:
            inst_id = inst_id[:, None].expand(-1, N)
            inst_id = inst_id.reshape(-1)
        eikonal_loss = torch.zeros_like(xyz[..., 0])
        normal = torch.zeros_like(xyz)

        # subsample to make it more efficient
        if M * N > sample_size:
            probs = torch.ones(M * N)
            rand_inds = torch.multinomial(probs, sample_size, replacement=False)
            xyz = xyz[rand_inds]
            if inst_id is not None:
                inst_id = inst_id[rand_inds]
        else:
            rand_inds = Ellipsis

        xyz = xyz.detach()
        fn_sdf = lambda x: self.forward_sdf(x, inst_id=inst_id)[0]
        g = compute_gradients_sdf(fn_sdf, xyz, training=self.training)
        # g = compute_gradient(fn_sdf, xyz)[..., 0]

        # def fn_sdf(x):
        #     sdf, _ = self.forward(x, inst_id=inst_id)
        #     sdf_sum = sdf.sum()
        #     return sdf_sum

        # g = jacobian(fn_sdf, xyz, create_graph=True, strict=True)

        eikonal_loss[rand_inds] = (g.norm(2, dim=-1) - 1) ** 2
        eikonal_loss = eikonal_loss.reshape(M, N, D, 1)
        normal[rand_inds] = g  # self.grad_to_normal(g)
        normal = normal.reshape(M, N, D, 3)
        return eikonal_loss, normal

    def compute_eikonal_view(
        self, xyz_cam, dir_cam, field2cam, frame_id=None, inst_id=None, samples_dict={}
    ):
        """Compute eikonal loss and normals in camera space

        Args:
            xyz_cam: (M,N,D,3) Points along rays in camera space
            dir_cam: (M,N,D,3) Ray directions in camera space
            field2cam: (M,SE(3)) Object-to-camera SE(3) transform
            frame_id: (M,) Frame id to query articulations, or None to use all frames
            inst_id: (M,) Instance id, or None to use the average instance
            samples_dict (Dict): Time-dependent bone articulations. Keys:
                "rest_articulation": ((M,B,4), (M,B,4)) and
                "t_articulation": ((M,B,4), (M,B,4))
        Returns:
            normal: (M,N,D,3) Normal vector field in camera space
        """
        M, N, D, _ = xyz_cam.shape

        xyz_cam = xyz_cam.detach()
        dir_cam = dir_cam.detach()
        field2cam = (field2cam[0].detach(), field2cam[1].detach())
        samples_dict_copy = {}
        for k, v in samples_dict.items():
            if isinstance(v, tuple):
                samples_dict_copy[k] = (v[0].detach(), v[1].detach())
            else:
                samples_dict_copy[k] = v.detach()
        samples_dict = samples_dict_copy

        def fn_sdf(xyz_cam):
            xyz = self.backward_warp(
                xyz_cam,
                dir_cam,
                field2cam,
                frame_id=frame_id,
                inst_id=inst_id,
                samples_dict=samples_dict,
            )["xyz"]
            sdf, _ = self.forward_sdf(xyz, inst_id=inst_id)
            return sdf

        # g = compute_gradient(fn_sdf, xyz_cam)[..., 0]
        g = compute_gradients_sdf(fn_sdf, xyz_cam, training=self.training)

        eikonal = (g.norm(2, dim=-1, keepdim=True) - 1) ** 2
        normal = g  # self.grad_to_normal(g)
        return eikonal, normal

    @staticmethod
    def grad_to_normal(g):
        """
        Args:
            g: (...,3) Gradient of sdf
        Returns:
            normal: (...,3) Normal vector field
        """
        normal = F.normalize(g, dim=-1)

        # Multiply by [1, -1, -1] to match normal conventions from ECON
        # https://github.com/YuliangXiu/ECON/blob/d98e9cbc96c31ecaa696267a072cdd5ef78d14b8/apps/infer.py#L257
        normal = normal * torch.tensor([1, -1, -1], device="cuda")
        return normal

    @torch.no_grad()
    def get_valid_idx(self, xyz, xyz_t=None, vis_score=None, samples_dict={}):
        """Return a mask of valid points by thresholding visibility score

        Args:
            xyz: (M,N,D,3) Points in object canonical space to query
            xyz_t: (M,N,D,3) Points in object time t space to query
            vis_score: (M,N,D,1) Predicted visibility score, not used
        Returns:
            valid_idx: (M,N,D) Visibility mask, bool
        """
        # check whether the point is inside the aabb
        aabb = self.get_aabb(samples_dict["inst_id"])
        aabb = extend_aabb(aabb)
        # (M,N,D), whether the point is inside the aabb
        inside_aabb = check_inside_aabb(xyz, aabb)

        # valid_idx = inside_aabb & (vis_score[..., 0] > -5)
        valid_idx = inside_aabb

        if xyz_t is not None and "t_articulation" in samples_dict.keys():
            # for time t points, we set aabb based on articulation
            t_bones = dual_quaternion_to_quaternion_translation(
                samples_dict["t_articulation"]
            )[1][0]
            t_aabb = torch.stack([t_bones.min(0)[0], t_bones.max(0)[0]], 0)
            t_aabb = extend_aabb(t_aabb, factor=1.0)
            inside_aabb = check_inside_aabb(xyz_t, t_aabb[None])
            valid_idx = valid_idx & inside_aabb

        # temporally disable visibility mask
        if self.category == "bg":
            valid_idx = None

        return valid_idx

    def reset_beta(self, beta):
        """Reset beta to initial value"""
        print(f"Resetting beta to {beta}")
        beta = torch.tensor([beta], device=next(self.parameters()).device)
        self.logibeta.data = -beta.log()

    def get_samples(self, Kinv, batch):
        """Compute time-dependent camera and articulation parameters.

        Args:
            Kinv: (N,3,3) Inverse of camera matrix
            batch (Dict): Batch of inputs. Keys: "dataid", "frameid_sub",
                "crop2raw", "feature", "hxy", and "frameid"
        Returns:
            samples_dict (Dict): Input metadata and time-dependent outputs.
                Keys: "Kinv" (M,3,3), "field2cam" (M,SE(3)), "frame_id" (M,),
                "inst_id" (M,), "near_far" (M,2), "hxy" (M,N,2), and
                "feature" (M,N,16).
        """
        device = next(self.parameters()).device
        hxy = batch["hxy"]
        frame_id = batch["frameid"]
        inst_id = batch["dataid"]

        # get camera pose: (1) read from batch (obj only), (2) read from mlp,
        # (3) read from mlp and apply delta to first frame
        if "field2cam" in batch.keys():
            # quaternion_translation representation, (N, 7)
            field2cam = (batch["field2cam"][..., :4], batch["field2cam"][..., 4:])
            field2cam = (field2cam[0], field2cam[1] * self.logscale.exp())
        else:
            field2cam = self.camera_mlp.get_vals(frame_id)

        # # compute near-far
        # if self.training:
        #     near_far = self.near_far[frame_id]
        # else:
        #     near_far = self.get_near_far(frame_id, field2cam)
        near_far = self.get_near_far(frame_id, field2cam)

        # auxiliary outputs
        samples_dict = {}
        samples_dict["Kinv"] = Kinv
        samples_dict["field2cam"] = field2cam
        samples_dict["frame_id"] = frame_id
        samples_dict["inst_id"] = inst_id
        samples_dict["near_far"] = near_far

        samples_dict["hxy"] = hxy
        if "feature" in batch.keys():
            samples_dict["feature"] = batch["feature"]
        return samples_dict

    def get_near_far(self, frame_id, field2cam):
        device = next(self.parameters()).device
        corners = trimesh.bounds.corners(self.proxy_geometry.bounds)
        corners = torch.tensor(corners, dtype=torch.float32, device=device)
        field2cam_mat = quaternion_translation_to_se3(field2cam[0], field2cam[1])
        near_far = get_near_far(corners, field2cam_mat, tol_fac=1.5)
        return near_far

    def query_field(self, samples_dict, flow_thresh=None, n_depth=64):
        """Render outputs from a neural radiance field.

        Args:
            samples_dict (Dict): Input metadata and time-dependent outputs.
                Keys: "Kinv" (M,3,3), "field2cam" (M,SE(3)), "frame_id" (M,),
                "inst_id" (M,), "near_far" (M,2), "hxy" (M,N,2), and
                "feature" (M,N,16)
            flow_thresh (float): Flow magnitude threshold, for `compute_flow()`
        Returns:
            feat_dict (Dict): Neural field outputs. Keys: "rgb" (M,N,D,3),
                "density" (M,N,D,1), "density_{fg,bg}" (M,N,D,1), "vis" (M,N,D,1),
                "cyc_dist" (M,N,D,1), "xyz" (M,N,D,3), "xyz_cam" (M,N,D,3),
                "depth" (M,1,D,1)
            deltas: (M,N,D,1) Distance along rays between adjacent samples
            aux_dict (Dict): Auxiliary neural field outputs. Used in Deformable
        """
        Kinv = samples_dict["Kinv"]  # (M,3,3)
        field2cam = samples_dict["field2cam"]  # (M,SE(3))
        frame_id = samples_dict["frame_id"]  # (M,)
        inst_id = samples_dict["inst_id"]  # (M,)
        near_far = samples_dict["near_far"]  # (M,2)
        hxy = samples_dict["hxy"]  # (M,N,2)

        # sample camera space rays
        if self.use_importance_sampling:
            # importance sampling
            xyz_cam, dir_cam, deltas, depth = self.importance_sampling(
                hxy,
                Kinv,
                near_far,
                field2cam,
                frame_id,
                inst_id,
                samples_dict,
                n_depth=n_depth,
            )
        else:
            xyz_cam, dir_cam, deltas, depth = sample_cam_rays(
                hxy,
                Kinv,
                near_far,
                n_depth=n_depth,
                perturb=False,
            )  # (M, N, D, x)

        # backward warping
        backwarp_dict = self.backward_warp(
            xyz_cam, dir_cam, field2cam, frame_id, inst_id, samples_dict=samples_dict
        )
        xyz = backwarp_dict["xyz"]
        dir = backwarp_dict["dir"]
        xyz_t = backwarp_dict["xyz_t"]

        # visibility
        vis_score = self.visibility_func(xyz.detach(), inst_id=inst_id)  # (M, N, D, 1)

        # compute valid_indices to speed up querying fields
        if self.training:
            valid_idx = None
        else:
            valid_idx = self.get_valid_idx(xyz, xyz_t, vis_score, samples_dict)

        # NeRF
        feat_dict = self.query_nerf(xyz, dir, frame_id, inst_id, valid_idx=valid_idx)

        # visibility
        feat_dict["vis"] = vis_score

        # flow
        flow_dict = self.compute_flow(
            hxy,
            xyz,
            frame_id,
            inst_id,
            field2cam,
            Kinv,
            samples_dict,
            flow_thresh=flow_thresh,
        )
        feat_dict.update(flow_dict)

        # cycle loss
        cyc_dict = self.cycle_loss(
            xyz, xyz_t, frame_id, inst_id, samples_dict=samples_dict
        )
        for k in cyc_dict.keys():
            if k == "dual_quat":
                continue
            if k in backwarp_dict.keys():
                # 'skin_entropy', 'delta_skin'
                feat_dict[k] = (cyc_dict[k] + backwarp_dict[k]) / 2
            else:
                # 'cyc_dist'
                feat_dict[k] = cyc_dict[k]

        # jacobian
        jacob_dict = self.compute_jacobian(
            xyz, xyz_cam, dir_cam, field2cam, frame_id, inst_id, samples_dict
        )
        feat_dict.update(jacob_dict)


        # def lerp(a, b, gamma):
        #     return a * (1-gamma) + b*gamma

        # # ray direction
        # # removed since it is memory intensive, and did not give performance boost
        # _, normal = self.compute_eikonal(xyz, inst_id=inst_id, sample_ratio=1)
        # normal = F.normalize(normal, 2, -1)
        # projected_area = torch.abs(-(dir * normal).sum(-1, keepdim=True))
        # projected_area = lerp(0.5, projected_area, self.pos_embedding.alpha)
        # feat_dict["density"] *= projected_area

        # canonical point
        feat_dict["xyz"] = xyz
        feat_dict["xyz_t"] = xyz_t
        feat_dict["xyz_cam"] = xyz_cam

        # depth
        feat_dict["depth"] = depth / self.logscale.exp()  # world scale
        # to metric space
        deltas = deltas / self.logscale.exp() * self.scale_const

        # auxiliary outputs
        aux_dict = {}
        return feat_dict, deltas, aux_dict

    def importance_sampling(
        self,
        hxy,
        Kinv,
        near_far,
        field2cam,
        frame_id,
        inst_id,
        samples_dict,
        n_depth,
    ):
        """
        importance sampling coarse
        """
        with torch.no_grad():
            # sample camera space rays
            xyz_cam, dir_cam, deltas, depth = sample_cam_rays(
                hxy, Kinv, near_far, n_depth // 2, perturb=False
            )  # (M, N, D, x)

            # backward warping
            xyz = self.backward_warp(
                xyz_cam,
                dir_cam,
                field2cam,
                frame_id,
                inst_id,
                samples_dict=samples_dict,
            )["xyz"]

            # get pdf
            _, density = self.forward(
                xyz,
                dir=None,
                frame_id=frame_id,
                inst_id=inst_id,
            )  # (M, N, D, x)
            weights, _ = compute_weights(density, deltas)  # (M, N, D, 1)
            weights = weights.view(-1, n_depth // 2)[:, 1:-1]  # (M*N, D-2)
            # modify the weights such that only do is when there is a clear surface (wt is high)
            weights_fill = 1 - weights.sum(-1, keepdim=True)
            weights = weights + weights_fill / (n_depth // 2 - 2)
            # assert torch.allclose(weights.sum(-1), torch.ones_like(weights[:, 0]))

            depth_mid = 0.5 * (depth[:, :, :-1] + depth[:, :, 1:])  # (M, N, D-1)
            depth_mid = depth_mid.view(-1, n_depth // 2 - 1)  # (M*N, D-1)

            depth_ = sample_pdf(depth_mid, weights, n_depth // 2, det=True)
            depth_ = depth_.reshape(depth.shape)  # (M, N, D, 1)

            depth, _ = torch.sort(torch.cat([depth, depth_], -2), -2)  # (M, N, D, 1)

            # # plot depth and depth_
            # import matplotlib.pyplot as plt
            # import pdb

            # pdb.set_trace()

            # valid_ind = weights.sum(-1) > 0
            # plt.figure()
            # depth_vis = depth[0, :, :, 0][valid_ind].cpu().numpy()

            # plt.plot(depth_vis[::10].T)
            # plt.show()
            # plt.savefig("tmp/depth.png")

            # plt.figure()
            # weights_vis = weights[valid_ind].cpu().numpy()
            # plt.plot(weights_vis[::10].T)
            # plt.show()
            # plt.savefig("tmp/weights.png")

        # sample camera space rays
        xyz_cam, dir_cam, deltas, depth = sample_cam_rays(
            hxy, Kinv, near_far, None, depth=depth, perturb=False
        )

        return xyz_cam, dir_cam, deltas, depth

    def compute_jacobian(
        self, xyz, xyz_cam, dir_cam, field2cam, frame_id, inst_id, samples_dict
    ):
        """Compute eikonal and normal fields from Jacobian of SDF

        Args:
            xyz: (M,N,D,3) Points along rays in object canonical space. Only for training
            xyz_cam: (M,N,D,3) Points along rays in camera space. Only for rendering
            dir_cam: (M,N,D,3) Ray directions in camera space. Only for rendering
            field2cam: (M,SE(3)) Object-to-camera SE(3) transform. Only for rendering
            frame_id: (M,) Frame id to query articulations, or None to use all frames.
                Only for rendering
            inst_id: (M,) Instance id. If None, compute for the average instance
            samples_dict (Dict): Time-dependent bone articulations. Only for rendering. Keys:
                "rest_articulation": ((M,B,4), (M,B,4)) and
                "t_articulation": ((M,B,4), (M,B,4))
        Returns:
            jacob_dict (Dict): Jacobian fields. Keys: "eikonal" (M,N,D,1). Only when
                rendering, "normal" (M,N,D,3)
        """
        jacob_dict = {}
        if self.training:
            # For efficiency, compute subsampled eikonal loss in canonical space
            jacob_dict["eikonal"], jacob_dict["normal"] = self.compute_eikonal(
                xyz, inst_id=inst_id
            )
            # convert to camera space
            jacob_dict["normal"] = quaternion_apply(
                field2cam[0][:, None, None]
                .expand(jacob_dict["normal"].shape[:-1] + (4,))
                .clone(),
                jacob_dict["normal"],
            )
        else:
            # For rendering, compute full eikonal loss and normals in camera space
            jacob_dict["eikonal"], jacob_dict["normal"] = self.compute_eikonal_view(
                xyz_cam, dir_cam, field2cam, frame_id, inst_id, samples_dict
            )
            # jacob_dict["eikonal"], jacob_dict["normal"] = self.compute_eikonal(
            #     xyz, inst_id=inst_id, sample_ratio=1.0
            # )
        return jacob_dict

    def query_nerf(self, xyz, dir, frame_id, inst_id, valid_idx=None):
        """Neural radiance field rendering

        Args:
            xyz: (M,N,D,3) Points along rays in object canonical space
            dir: (M,N,D,3) Ray directions in object canonical space
            frame_id: (M,) Frame id. If None, render at all frames
            inst_id: (M,) Instance id. If None, render for the average instance
            valid_idx: (M,N,D) Mask of whether each point is visible to camera
        Returns:
            field_dict (Dict): Field outputs. Keys: "rgb" (M,N,D,3),
                "density" (M,N,D,1), and "density_{fg,bg}" (M,N,D,1)
        """
        if valid_idx is not None:
            if valid_idx.sum() == 0:
                field_dict = {
                    "rgb": torch.zeros(valid_idx.shape + (3,), device=xyz.device),
                    "density": torch.zeros(valid_idx.shape + (1,), device=xyz.device),
                    "density_%s"
                    % self.category: torch.zeros(
                        valid_idx.shape + (1,), device=xyz.device
                    ),
                    "sdf": torch.zeros(valid_idx.shape + (1,), device=xyz.device),
                }
                return field_dict
            # reshape
            shape = xyz.shape
            xyz = xyz[valid_idx][:, None, None]  # MND,1,1,3
            dir = dir[valid_idx][:, None, None]
            frame_id = frame_id[:, None, None].expand(shape[:3])[valid_idx]
            inst_id = inst_id[:, None, None].expand(shape[:3])[valid_idx]

        rgb, sdf, density = self.forward(
            xyz,
            dir=dir,
            frame_id=frame_id,
            inst_id=inst_id,
        )  # (M, N, D, x)

        # # density drop out, to enforce motion to explain the missing density
        # # get aabb
        # ratio = 4
        # aabb = self.get_aabb()
        # # select a random box from aabb with 1/ratio size
        # aabb_size = aabb[..., 1, :] - aabb[..., 0, :]
        # aabb_size_sub = aabb_size / ratio
        # aabb_sub_min = aabb[..., 0, :] + torch.rand_like(aabb_size) * (
        #     aabb_size - aabb_size_sub
        # )
        # aabb_sub_max = aabb_sub_min + aabb_size_sub
        # aabb_sub = torch.stack([aabb_sub_min, aabb_sub_max], -2)
        # # check whether the point is inside the aabb
        # inside_aabb = check_inside_aabb(xyz, aabb_sub)
        # density[inside_aabb] = 0

        # reshape
        field_dict = {
            "rgb": rgb,
            "sdf": sdf,
            "density": density,
            "density_%s" % self.category: density,  # (0,1)
        }

        if valid_idx is not None:
            for k, v in field_dict.items():
                tmpv = torch.zeros(valid_idx.shape + (v.shape[-1],), device=v.device)
                tmpv[valid_idx] = v.view(-1, v.shape[-1])
                field_dict[k] = tmpv
        return field_dict

    def wipe_loss(self, nsample=512):
        # density drop out, to enforce motion to explain the missing density
        # get aabb
        ratio = 4
        aabb = self.get_aabb()
        # select a random box from aabb with 1/ratio size
        aabb_size = aabb[..., 1, :] - aabb[..., 0, :]
        aabb_size_sub = aabb_size / ratio
        aabb_sub_min = aabb[..., 0, :] + torch.rand_like(aabb_size) * (
            aabb_size - aabb_size_sub
        )
        aabb_sub_max = aabb_sub_min + aabb_size_sub
        aabb_sub = torch.stack([aabb_sub_min, aabb_sub_max], -2)
        pts, frame_id, inst_id = self.sample_points_aabb(nsample, aabb=aabb_sub)
        # check whether the point is inside the aabb
        sdf, _ = self.forward_sdf(pts, frame_id=frame_id, inst_id=inst_id)
        wipe_loss = (-sdf).exp().mean()
        return wipe_loss

    @staticmethod
    def cam_to_field(xyz_cam, dir_cam, field2cam):
        """Transform rays from camera SE(3) to object SE(3)

        Args:
            xyz_cam: (M,N,D,3) Points along rays in camera SE(3)
            dir_cam: (M,N,D,3) Ray directions in camera SE(3)
            field2cam: (M,SE(3)) Object-to-camera SE(3) transform
        Returns:
            xyz: (M,N,D,3) Points along rays in object SE(3)
            dir: (M,N,D,3) Ray directions in object SE(3)
        """
        # warp camera space points to canonical space
        # scene/object space rays # (M,1,1,4,4) * (M,N,D,3) = (M,N,D,3)
        shape = xyz_cam.shape
        cam2field = quaternion_translation_inverse(field2cam[0], field2cam[1])
        cam2field = (
            cam2field[0][:, None, None].expand(shape[:-1] + (4,)).clone(),
            cam2field[1][:, None, None].expand(shape[:-1] + (3,)).clone(),
        )
        xyz = apply_se3mat(cam2field, xyz_cam)
        cam2field = (cam2field[0], torch.zeros_like(cam2field[1]))
        dir = apply_se3mat(cam2field, dir_cam)
        return xyz, dir

    def field_to_cam(self, xyz, field2cam):
        """Transform points from object SE(3) to camera SE(3)

        Args:
            xyz: (M,N,D,3) Points in object SE(3)
            field2cam: (M,SE(3)) Object to camera SE(3) transform
        Returns:
            xyz_cam: (M,N,D,3) Points in camera SE(3)
        """
        # transform from canonical to next frame camera space
        # (M,1,1,3,4) @ (M,N,D,3) = (M,N,D,3)
        shape = xyz.shape
        field2cam = (
            field2cam[0][:, None, None].expand(shape[:-1] + (4,)).clone(),
            field2cam[1][:, None, None].expand(shape[:-1] + (3,)).clone(),
        )
        xyz_cam_next = apply_se3mat(field2cam, xyz)
        return xyz_cam_next

    def backward_warp(
        self, xyz_cam, dir_cam, field2cam, frame_id, inst_id, samples_dict={}
    ):
        """Warp points from camera space to object canonical space

        Args:
            xyz_cam: (M,N,D,3) Points along rays in camera space
            dir_cam: (M,N,D,3) Ray directions in camera space
            field2cam: (M,SE(3)) Object-to-camera SE(3) transform
            frame_id: (M,) Frame id. If None, warp for all frames
            inst_id: (M,) Instance id. If None, warp for the average instance
            samples_dict (Dict): Only used in Deformable

        Returns:
            xyz: (M,N,D,3) Points along rays in object canonical space
            dir: (M,N,D,3) Ray directions in object canonical space
            xyz_t: (M,N,D,3) Points along rays in object time-t space. Same
                as canonical space for static fields
        """
        xyz, dir = self.cam_to_field(xyz_cam, dir_cam, field2cam)

        backwarp_dict = {"xyz": xyz, "dir": dir, "xyz_t": xyz}
        return backwarp_dict

    def forward_warp(self, xyz, field2cam, frame_id, inst_id, samples_dict={}):
        """Warp points from object canonical space to camera space

        Args:
            xyz: (M,N,D,3) Points along rays in object canonical space
            field2cam: (M,SE(3)) Object-to-camera SE(3) transform
            frame_id: (M,) Frame id. If None, warp for all frames
            inst_id: (M,) Instance id. If None, warp for the average instance
            samples_dict (Dict): Only used in Deformable

        Returns:
            xyz_cam: (M,N,D,3) Points along rays in camera space
        """
        xyz_cam = self.field_to_cam(xyz, field2cam)
        return xyz_cam

    def cycle_loss(self, xyz, xyz_t, frame_id, inst_id, samples_dict={}):
        """Compute cycle-consistency loss between points in object canonical
        space, and points that have been warped backward and then forward

        Args:
            xyz: (M,N,D,3) Points along rays in object canonical space
            xyz_t: (M,N,D,3) Points along rays in object time-t space
            frame_id: (M,) Frame id. If None, render at all frames
            inst_id: (M,) Instance id. If None, render for the average instance
            samples_dict (Dict): Used in Deformable

        Returns:
            cyc_dict (Dict): Cycle consistency loss. Keys: "cyc_dist" (M,N,D,1)
        """
        cyc_dist = torch.zeros_like(xyz[..., :1])
        l2_motion = torch.zeros_like(xyz[..., :1])
        delta_skin = torch.zeros_like(xyz[..., :1])
        skin_entropy = torch.zeros_like(xyz[..., :1])
        cyc_dict = {
            "cyc_dist": cyc_dist,
            "l2_motion": l2_motion,
            "delta_skin": delta_skin,
            "skin_entropy": skin_entropy,
        }
        return cyc_dict

    # @train_only_fields
    def compute_flow(
        self,
        hxy,
        xyz,
        frame_id,
        inst_id,
        field2cam,
        Kinv,
        samples_dict,
        flow_thresh=None,
    ):
        """Compute optical flow proposal by (1) projecting to next camera
        image plane, and (2) taking difference with xy

        Args:
            hxy: (M,N,D,3) Homogeneous pixel coordinates on the image plane
            xyz_t: (M,N,D,3) Canonical field coordinates at time t
            Kinv: (M,3,3) Inverse of camera intrinsics
            flow_thresh (float): Threshold for flow magnitude

        Returns:
            flow: (M,N,D,2) Optical flow proposal
        """
        # flip the frame id
        frame_id_next = flip_pair(frame_id)
        field2cam_next = (flip_pair(field2cam[0]), flip_pair(field2cam[1]))
        Kinv_next = flip_pair(Kinv)
        samples_dict_next = flip_pair(samples_dict)

        # forward warp points to camera space
        xyz_cam_next = self.forward_warp(
            xyz, field2cam_next, frame_id_next, inst_id, samples_dict=samples_dict_next
        )
        # xyz_cam_next = self.flow_warp(
        #     xyz_t, field2cam_next, frame_id, inst_id, samples_dict
        # )

        # project to next camera image plane
        Kmat_next = Kmatinv(Kinv_next)  # (M,1,1,3,3) @ (M,N,D,3) = (M,N,D,3)
        hxy_next = pinhole_projection(Kmat_next, xyz_cam_next)

        # compute 2d difference
        flow = (hxy_next - hxy.unsqueeze(-2))[..., :2]
        xyz_valid = xyz_cam_next[..., -1:] > 1e-6
        if flow_thresh is not None:
            flow_thresh = float(flow_thresh)
            xyz_valid = xyz_valid & (flow.norm(dim=-1, keepdim=True) < flow_thresh)

        flow = torch.cat([flow, xyz_valid.float()], dim=-1)

        flow_dict = {"flow": flow}
        return flow_dict

    def cam_prior_loss(self):
        """Encourage camera transforms over time to match external priors.

        Returns:
            loss: (0,) Mean squared error of camera SE(3) transforms to priors
        """
        if isinstance(self.camera_mlp, CameraConst) or isinstance(
            self.camera_mlp, CameraMixSE3
        ):
            return torch.zeros(1, device=self.parameters().__next__().device).mean()
        if isinstance(self.camera_mlp, CameraMix):
            return self.camera_mlp.camera_mlp.compute_distance_to_prior()
        loss = self.camera_mlp.compute_distance_to_prior()
        return loss

    def cam_prior_relative_loss(self):
        """Encourage camera transforms over time to match external priors.

        Returns:
            loss: (0,) Mean squared error of camera SE(3) transforms to priors
        """
        if isinstance(self.camera_mlp, CameraConst):
            return torch.zeros(1, device=self.parameters().__next__().device)
        if isinstance(self.camera_mlp, CameraMix):
            return self.camera_mlp.camera_mlp.compute_distance_to_prior_relative()
        loss = self.camera_mlp.compute_distance_to_prior_relative()
        return loss

    def cam_smooth_loss(self):
        """Encourage camera transforms over time to be smooth.

        Returns:
            loss: (0,) Mean squared error of camera SE(3) transforms to priors
        """
        if isinstance(self.camera_mlp, CameraConst):
            return torch.zeros(1, device=self.parameters().__next__().device)

        # compute smoothness
        extrinsics = self.get_camera(metric_scale=False)
        loss = compute_se3_smooth_loss_2nd(extrinsics, self.frame_offset)
        return loss

    def get_camera(self, frame_id=None, metric_scale=True):
        """Compute camera matrices in world units

        Returns:
            field2cam (Dict): Maps field names ("fg" or "bg") to (M,4,4) cameras
        """
        quat, trans = self.camera_mlp.get_vals(frame_id=frame_id)
        if metric_scale:
            trans = trans / self.logscale.exp()
        field2cam = quaternion_translation_to_se3(quat, trans)
        return field2cam

    @torch.no_grad()
    def compute_field2world(self, up_direction=[0, -1, 0]):
        """Compute SE(3) to transform points in the scene space to world space
        For background, this is computed by detecting planes with ransac.

        Returns:
            rect_se3: (4,4) SE(3) transform
        """
        for inst_id in range(self.num_inst):
            # TODO: move this to background nerf, and use each proxy geometry
            mesh = self.extract_canonical_mesh(
                level=0.0, inst_id=inst_id, vis_thresh=-10, grid_size=128
            )
            # get mesh at frame 0
            device = next(self.parameters()).device
            frame_id = torch.tensor([self.frame_offset_raw[inst_id]], device=device)
            field2view_0 = (
                self.get_camera(frame_id=frame_id, metric_scale=False)[0].cpu().numpy()
            )
            mesh.apply_transform(field2view_0)
            scale = self.logscale.exp().cpu().numpy()
            view2world = compute_rectification_se3(
                mesh, up_direction, threshold=0.05 * scale
            )
            self.field2world[inst_id] = view2world @ field2view_0

    def get_field2world(self, inst_id=None):
        """Compute SE(3) to transform points in the scene space to world space
        For background, this is computed by detecting planes with ransac.

        Returns:
            rect_se3: (4,4) SE(3) transform
        """
        if inst_id is None:
            field2world = self.field2world
        else:
            field2world = self.field2world[inst_id]
        field2world = field2world.clone()
        field2world[..., :3, 3] /= self.logscale.exp()
        return field2world

    @torch.no_grad()
    def visualize_floor_mesh(self, inst_id, to_world=False):
        """Visualize floor and canonical mesh in the world space
        Args:
            inst_id: (int) Instance id
        """
        field2world = self.get_field2world(inst_id)
        world2field = field2world.inverse().cpu()
        mesh = self.extract_canonical_mesh(
            level=0.0, inst_id=inst_id, vis_thresh=-10, grid_size=128
        )
        scale = self.logscale.exp().cpu().numpy()
        mesh.vertices /= scale
        mesh = append_xz_plane(mesh, world2field, gl=False, scale=1.0 / scale)
        if to_world:
            mesh.apply_transform(field2world.cpu().numpy())
        return mesh

    def valid_field2world(self):
        if self.field2world.abs().sum() == 0:
            return False
        else:
            return True
