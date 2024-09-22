# Copyright (c) 2023 Gengshan Yang, Carnegie Mellon University.
import numpy as np
import torch
import trimesh
from torch import nn
from torch.nn import functional as F
import sys
import os

os.environ["CUDA_PATH"] = sys.prefix  # needed for geomloss
from geomloss import SamplesLoss

from lab4d.nnutils.feature import FeatureNeRF
from lab4d.nnutils.warping import SkinningWarp, create_warp
from lab4d.utils.decorator import train_only_fields
from lab4d.utils.geom_utils import extend_aabb, check_inside_aabb
from lab4d.utils.quat_transform import dual_quaternion_to_quaternion_translation, axis_angle_to_matrix
from lab4d.utils.loss_utils import align_tensors

class Deformable(FeatureNeRF):
    """A dynamic neural radiance field

    Args:
        fg_motion (str): Foreground motion type ("rigid", "dense", "bob",
            "skel-{human,quad}", or "comp_skel-{human,quad}_{bob,dense}")
        data_info (Dict): Dataset metadata from get_data_info()
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
        skips (List(int): List of layers to add skip connections at
        activation (Function): Activation function to use (e.g. nn.ReLU())
        init_beta (float): Initial value of beta, from Eqn. 3 of VolSDF.
            We transform a learnable signed distance function into density using
            the CDF of the Laplace distribution with zero mean and beta scale.
        init_scale (float): Initial geometry scale factor.
        color_act (bool): If True, apply sigmoid to the output RGB
        feature_channels (int): Number of feature field channels
        time_sync (bool): If True, assume all videos captured at same time
    """

    def __init__(
        self,
        fg_motion,
        data_info,
        D=8,
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
        feature_channels=16,
        extrinsics_type="mlp",
        use_timesync=False,
        invalid_vid=-1,
    ):
        self.fg_motion = fg_motion
        self.use_timesync = use_timesync
        super().__init__(
            data_info,
            D=D,
            W=W,
            num_freq_xyz=num_freq_xyz,
            num_freq_dir=num_freq_dir,
            appr_channels=appr_channels,
            appr_num_freq_t=appr_num_freq_t,
            num_inst=num_inst,
            inst_channels=inst_channels,
            skips=skips,
            activation=activation,
            init_beta=init_beta,
            init_scale=init_scale,
            color_act=color_act,
            feature_channels=feature_channels,
            extrinsics_type=extrinsics_type,
            invalid_vid=invalid_vid,
        )

        self.warp = create_warp(fg_motion, data_info, num_inst)

    # def update_aabb(self, beta=0.5):
    #     """Update axis-aligned bounding box by interpolating with the current
    #     proxy geometry's bounds

    #     Args:
    #         beta (float): Interpolation factor between previous/current values
    #     """
    #     super().update_aabb(beta=beta)

    def init_proxy(self, geom_path, init_scale):
        """Initialize proxy geometry as a sphere

        Args:
            geom_path (str): Unused
            init_scale (float): Unused
        """
        if "smpl" in self.fg_motion:
            scale = self.logscale.detach().exp()[0]

            from projects.csim.smpl import SMPL
            self.smpl_model = SMPL().eval()
            global_orient = torch.eye(3)
            global_orient[1:] *= -1
            out = self.smpl_model(global_orient=global_orient[None])
            verts = (out["vertices"][0] * scale).numpy()
            faces = self.smpl_model.faces
            self.proxy_geometry = trimesh.Trimesh(vertices=verts, faces=faces)
        else:
            self.proxy_geometry = trimesh.creation.uv_sphere(radius=0.12 * init_scale / 0.2, count=[4, 4]) # zoe depth
            # self.proxy_geometry = trimesh.creation.uv_sphere(radius=0.12 * init_scale / 0.5, count=[4, 4]) # metric depth

    def get_init_sdf_fn(self):
        """Initialize signed distance function as a skeleton or sphere

        Returns:
            sdf_fn_torch (Function): Signed distance function
        """

        def sdf_fn_torch_sphere(pts):
            radius = 0.1
            # l2 distance to a unit sphere
            dis = (pts).pow(2).sum(-1, keepdim=True)
            sdf = torch.sqrt(dis) - radius  # negative inside, postive outside
            return sdf

        @torch.no_grad()
        def sdf_fn_torch_skel(pts):
            sdf = self.warp.get_gauss_sdf(pts)
            return sdf

        if "skel-" in self.fg_motion or "urdf-" in self.fg_motion:
            if "-smpl" in self.fg_motion:
                return super().get_init_sdf_fn() # use proxy-geometry from smpl
            else:
                return sdf_fn_torch_skel
        else:
            return sdf_fn_torch_sphere

    def frame_id_to_sub(self, frame_id, inst_id):
        """Convert frame id to frame id relative to the video.
        This forces all videos to share the same pose embedding.

        Args:
            frame_id: (M,) Frame id
            inst_id: (M,) Instance id
        Returns:
            frameid_sub: (M,) Frame id relative to the video
        """
        frame_offset_raw = torch.tensor(self.frame_offset_raw, device=frame_id.device)
        frameid_sub = frame_id - frame_offset_raw[inst_id]
        return frameid_sub

    def backward_warp(
        self, xyz_cam, dir_cam, field2cam, frame_id, inst_id, samples_dict={}
    ):
        """Warp points from camera space to object canonical space. This
        requires "un-articulating" the object from observed time-t to rest.

        Args:
            xyz_cam: (M,N,D,3) Points along rays in camera space
            dir_cam: (M,N,D,3) Ray directions in camera space
            field2cam: (M,SE(3)) Object-to-camera SE(3) transform
            frame_id: (M,) Frame id. If None, warp for all frames
            inst_id: (M,) Instance id. If None, warp for the average instance.
            samples_dict (Dict): Time-dependent bone articulations. Keys:
                "rest_articulation": ((M,B,4), (M,B,4)) and
                "t_articulation": ((M,B,4), (M,B,4))
        Returns:
            xyz: (M,N,D,3) Points along rays in object canonical space
            dir: (M,N,D,3) Ray directions in object canonical space
            xyz_t: (M,N,D,3) Points along rays in object time-t space.
        """
        xyz_t, dir = self.cam_to_field(xyz_cam, dir_cam, field2cam)
        samples_dict_copy = {}
        for k, v in enumerate(samples_dict):
            if k == "rest_articulation" or k == "t_articulation":
                samples_dict_copy[k] = v
        if self.use_timesync:
            frame_id = self.frame_id_to_sub(frame_id, inst_id)
            inst_id = inst_id.clone() * 0  # assume all videos capture the same instance
        xyz, warp_dict = self.warp(
            xyz_t,
            frame_id,
            inst_id,
            type="backward",
            samples_dict=samples_dict_copy,
            return_aux=True,
        )

        # TODO: apply se3 to dir
        backwarp_dict = {"xyz": xyz, "dir": dir, "xyz_t": xyz_t}
        backwarp_dict.update(warp_dict)
        return backwarp_dict

    def forward_warp(self, xyz, field2cam, frame_id, inst_id, samples_dict={}):
        """Warp points from object canonical space to camera space. This
        requires "re-articulating" the object from rest to observed time-t.

        Args:
            xyz: (M,N,D,3) Points along rays in object canonical space
            field2cam: (M,SE(3)) Object-to-camera SE(3) transform
            frame_id: (M,) Frame id. If None, warp for all frames
            inst_id: (M,) Instance id. If None, warp for the average instance
            samples_dict (Dict): Time-dependent bone articulations. Keys:
                "rest_articulation": ((M,B,4), (M,B,4)) and
                "t_articulation": ((M,B,4), (M,B,4))
        Returns:
            xyz_cam: (M,N,D,3) Points along rays in camera space
        """
        xyz_next = self.warp(xyz, frame_id, inst_id, samples_dict=samples_dict)
        xyz_cam = self.field_to_cam(xyz_next, field2cam)
        return xyz_cam

    def flow_warp(
        self,
        xyz_1,
        field2cam_flip,
        frame_id,
        inst_id,
        samples_dict={},
    ):
        """Warp points from camera space from time t1 to time t2

        Args:
            xyz_1: (M,N,D,3) Points along rays in canonical space at time t1
            field2cam_flip: (M,SE(3)) Object-to-camera SE(3) transform at time t2
            frame_id: (M,) Frame id. If None, warp for all frames
            inst_id: (M,) Instance id. If None, warp for the average instance
            samples_dict (Dict): Time-dependent bone articulations. Keys:
                "rest_articulation": ((M,B,4), (M,B,4)) and
                "t_articulation": ((M,B,4), (M,B,4))

        Returns:
            xyz_2: (M,N,D,3) Points along rays in camera space at time t2
        """
        xyz_2 = self.warp(
            xyz_1, frame_id, inst_id, type="flow", samples_dict=samples_dict
        )
        xyz_2 = self.field_to_cam(xyz_2, field2cam_flip)
        return xyz_2

    @train_only_fields
    def cycle_loss(self, xyz, xyz_t, frame_id, inst_id, samples_dict={}):
        """Enforce cycle consistency between points in object canonical space,
        and points warped from canonical space, backward to time-t space, then
        forward to canonical space again

        Args:
            xyz: (M,N,D,3) Points along rays in object canonical space
            xyz_t: (M,N,D,3) Points along rays in object time-t space
            frame_id: (M,) Frame id. If None, render at all frames
            inst_id: (M,) Instance id. If None, render for the average instance
            samples_dict (Dict): Time-dependent bone articulations. Keys:
                "rest_articulation": ((M,B,4), (M,B,4)) and
                "t_articulation": ((M,B,4), (M,B,4))
        Returns:
            cyc_dict (Dict): Cycle consistency loss. Keys: "cyc_dist" (M,N,D,1)
        """
        cyc_dict = super().cycle_loss(xyz, xyz_t, frame_id, inst_id, samples_dict)

        xyz_cycled, warp_dict = self.warp(
            xyz, frame_id, inst_id, samples_dict=samples_dict, return_aux=True
        )
        cyc_dist = (xyz_cycled - xyz_t).norm(2, -1, keepdim=True)
        cyc_dict["cyc_dist"] = cyc_dist
        cyc_dict["l2_motion"] = ((xyz_t - xyz).norm(2, -1, keepdim=True) + \
                            (xyz_cycled - xyz).norm(2, -1, keepdim=True)) * 0.5
        cyc_dict.update(warp_dict)
        return cyc_dict

    def gauss_skin_consistency_loss(self, type="optimal_transport"):
        """Enforce consistency between the NeRF's SDF and the SDF of Gaussian bones,

        Args:
            type (str): "optimal_transport" or "density"
        Returns:
            loss: (0,) Skinning consistency loss
        """
        if type == "optimal_transport":
            return self.gauss_optimal_transport_loss()
        elif type == "density":
            return self.gauss_skin_density_loss()
        else:
            raise NotImplementedError

    def gauss_skin_density_loss(self, nsample=4096):
        """Enforce consistency between the NeRF's SDF and the SDF of Gaussian bones,
        based on density.

        Args:
            nsample (int): Number of samples to take from both distance fields
        Returns:
            loss: (0,) Skinning consistency loss
        """
        pts, frame_id, _ = self.sample_points_aabb(nsample, extend_factor=0.5)
        inst_id = None
        samples_dict = {}
        (
            samples_dict["t_articulation"],
            samples_dict["rest_articulation"],
        ) = self.warp.articulation.get_vals_and_mean(frame_id)

        # match the gauss density to the reconstructed density
        bones2obj = samples_dict["t_articulation"]
        bones2obj = (
            torch.cat([bones2obj[0], samples_dict["rest_articulation"][0]], 0),
            torch.cat([bones2obj[1], samples_dict["rest_articulation"][1]], 0),
        )
        pts_gauss = torch.cat([pts, pts], dim=0)
        density_gauss = self.warp.get_gauss_density(pts_gauss, bone2obj=bones2obj)

        with torch.no_grad():
            density = torch.zeros_like(density_gauss)
            pts_warped = self.warp(
                pts[:, None, None],
                frame_id,
                inst_id,
                type="backward",
                samples_dict=samples_dict,
                return_aux=False,
            )[:, 0, 0]
            pts = torch.cat([pts_warped, pts], dim=0)

            # check whether the point is inside the aabb
            aabb = self.get_aabb()
            aabb = extend_aabb(aabb)
            inside_aabb = check_inside_aabb(pts, aabb)

            _, density[inside_aabb] = self.forward(pts[inside_aabb], inst_id=inst_id)
            density = density / self.logibeta.exp()  # (0,1)

        # loss = ((density_gauss - density).pow(2)).mean()
        # binary cross entropy loss to align gauss density to the reconstructed density
        # weight the loss such that:
        # wp lp = wn ln
        # wp lp + wn ln = lp + ln
        weight_pos = 0.5 / (1e-6 + density.mean())
        weight_neg = 0.5 / (1e-6 + 1 - density).mean()
        weight = density * weight_pos + (1 - density) * weight_neg
        loss = ((density_gauss - density).pow(2) * weight.detach()).mean()
        # loss = F.binary_cross_entropy(
        #     density_gauss, density.detach(), weight=weight.detach()
        # )

        # if get_local_rank() == 0:
        #     is_inside = density > 0.5
        #     mesh = trimesh.Trimesh(vertices=pts[is_inside[..., 0]].detach().cpu())
        #     mesh.export("tmp/0.obj")

        #     is_inside = density_gauss > 0.5
        #     mesh = trimesh.Trimesh(vertices=pts[is_inside[..., 0]].detach().cpu())
        #     mesh.export("tmp/1.obj")
        return loss

    def gauss_optimal_transport_loss(self, nsample=1024):
        """Enforce consistency between the NeRF's proxy rest shape
         and the gaussian bones, based on optimal transport.

        Args:
            nsample (int): Number of samples to take from proxy geometry
        Returns:
            loss: (0,) Gaussian optimal transport loss
        """
        # optimal transport loss
        device = self.parameters().__next__().device
        pts = self.get_proxy_geometry().vertices
        # sample points from the proxy geometry
        pts = pts[np.random.choice(len(pts), nsample)]
        pts = torch.tensor(pts, device=device, dtype=torch.float32)
        pts_gauss = self.warp.get_gauss_pts()
        samploss = SamplesLoss(
            loss="sinkhorn", p=2, blur=0.002, scaling=0.5, truncate=1
        )
        scale_proxy = self.get_scale()  # to normalize pts to 1
        loss = samploss(2 * pts_gauss / scale_proxy, 2 * pts / scale_proxy).mean()
        # if get_local_rank() == 0:
        #     mesh = trimesh.Trimesh(vertices=pts.detach().cpu())
        #     mesh.export("tmp/0.obj")
        #     mesh = trimesh.Trimesh(vertices=pts_gauss.detach().cpu())
        #     mesh.export("tmp/1.obj")
        return loss

    def soft_deform_loss(self, nsample=1024):
        """Minimize soft deformation so it doesn't overpower the skeleton.
        Compute L2 distance of points before and after soft deformation

        Args:
            nsample (int): Number of samples to take from both distance fields
        Returns:
            loss: (0,) Soft deformation loss
        """
        pts, frame_id, inst_id = self.sample_points_aabb(nsample, extend_factor=1.0)
        dist2 = self.warp.compute_post_warp_dist2(pts[:, None, None], frame_id, inst_id)
        return dist2.mean()

    def pose_prior_loss(self):
        """Encourage body pose over time to match external priors.

        Returns:
            loss: (0,) Mean squared error of camera SE(3) transforms to priors
        """
        if isinstance(self.warp, SkinningWarp):
            loss = self.warp.articulation.compute_distance_to_prior()
            return loss
        else:
            return torch.zeros(1, device=self.parameters().__next__().device).mean()

    def shape_prior_loss(self, nsample=2048):
        from lab4d.utils.loss_utils import align_tensors
        sdf_fn = self.get_init_sdf_fn()

        # sample points
        pts, _, inst_id = self.sample_points_aabb(nsample, extend_factor=0.5)

        # get sdf from proxy geometry
        sdf_gt = sdf_fn(pts)

        # evaluate sdf loss
        sdf, _ = self.forward_sdf(pts, inst_id=inst_id)
        scale = align_tensors(sdf, sdf_gt)
        sdf_loss = (sdf * scale.detach() - sdf_gt).pow(2).mean()
        return sdf_loss

    def get_samples(self, Kinv, batch):
        """Compute time-dependent camera and articulation parameters.

        Args:
            Kinv: (N,3,3) Inverse of camera matrix
            Batch (Dict): Batch of inputs. Keys: "dataid", "frameid_sub",
                "crop2raw", "feature", "hxy", and "frameid"
        Returns:
            samples_dict (Dict): Input metadata and time-dependent outputs.
                Keys: "Kinv" (M,3,3), "field2cam" (M,SE(3)), "frame_id" (M,),
                "inst_id" (M,), "near_far" (M,2), "hxy" (M,N,2),
                "feature" (M,N,16), "rest_articulation" ((M,B,4), (M,B,4)), and
                "t_articulation" ((M,B,4), (M,B,4))
        """
        samples_dict = super().get_samples(Kinv, batch)

        if isinstance(self.warp, SkinningWarp):
            # cache the articulation values
            # mainly to avoid multiple fk computation
            # (M,K,4)x2, # (M,K,4)x2
            inst_id = samples_dict["inst_id"]
            frame_id = samples_dict["frame_id"]
            if self.use_timesync:
                frame_id = self.frame_id_to_sub(frame_id, inst_id)
                inst_id = inst_id.clone() * 0
            if "joint_so3" in batch.keys():
                override_so3 = batch["joint_so3"]
                samples_dict["rest_articulation"] = (
                    self.warp.articulation.get_mean_vals()
                )
                samples_dict["t_articulation"] = self.warp.articulation.get_vals(
                    frame_id, override_so3=override_so3
                )
            else:
                (
                    samples_dict["t_articulation"],
                    samples_dict["rest_articulation"],
                ) = self.warp.articulation.get_vals_and_mean(frame_id)
        return samples_dict

    def mlp_init(self):
        """For skeleton fields, initialize bone lengths and rest joint angles
        from an external skeleton
        """
        if "skel-" in self.fg_motion or "urdf-" in self.fg_motion:
            self.warp.articulation.mlp_init()
        super().mlp_init()

    def query_field(self, samples_dict, flow_thresh=None, n_depth=64):
        """Render outputs from a neural radiance field.

        Args:
            samples_dict (Dict): Input metadata and time-dependent outputs.
                Keys: "Kinv" (M,3,3), "field2cam" (M,SE(3)), "frame_id" (M,),
                "inst_id" (M,), "near_far" (M,2), "hxy" (M,N,2), and
                "feature" (M,N,16), "rest_articulation" ((M,B,4), (M,B,4)),
                and "t_articulation" ((M,B,4), (M,B,4))
            flow_thresh (float): Flow magnitude threshold, for `compute_flow()`
        Returns:
            feat_dict (Dict): Neural field outputs. Keys: "rgb" (M,N,D,3),
                "density" (M,N,D,1), "density_{fg,bg}" (M,N,D,1), "vis" (M,N,D,1),
                "cyc_dist" (M,N,D,1), "xyz" (M,N,D,3), "xyz_cam" (M,N,D,3),
                "depth" (M,1,D,1) TODO
            deltas: (M,N,D,1) Distance along rays between adjacent samples
            aux_dict (Dict): Auxiliary neural field outputs. Keys: TODO
        """
        feat_dict, deltas, aux_dict = super().query_field(
            samples_dict, flow_thresh=flow_thresh, n_depth=n_depth,
        )

        # xyz = feat_dict["xyz"].detach()  # don't backprop to cam/dfm fields
        xyz = feat_dict["xyz"]
        xyz_t = feat_dict["xyz_t"]
        gauss_field = self.compute_gauss_density(xyz, xyz_t, samples_dict)
        feat_dict.update(gauss_field)

        return feat_dict, deltas, aux_dict

    def compute_gauss_density(self, xyz, xyz_t, samples_dict):
        """If this is a SkinningWarp, compute density from Gaussian bones

        Args:
            xyz: (M,N,D,3) Points in object canonical space
            samples_dict (Dict): Input metadata and time-dependent outputs.
                Keys: "Kinv" (M,3,3), "field2cam" (M,SE(3)), "frame_id" (M,),
                "inst_id" (M,), "near_far" (M,2), "hxy" (M,N,2), and
                "feature" (M,N,16), "rest_articulation" ((M,B,4), (M,B,4)),
                and "t_articulation" ((M,B,4), (M,B,4))
        Returns:
            gauss_field (Dict): Density. Keys: "gauss_density" (M,N,D,1)
        """
        M, N, D, _ = xyz.shape
        gauss_field = {}
        if isinstance(self.warp, SkinningWarp):
            # supervise t articulation
            xyz_t = xyz_t.view(-1, 3).detach()
            t_articulation = (
                samples_dict["t_articulation"][0][:, None]
                .repeat(1, N * D, 1, 1)
                .view(M * N * D, -1, 4),
                samples_dict["t_articulation"][1][:, None]
                .repeat(1, N * D, 1, 1)
                .view(M * N * D, -1, 4),
            )
            gauss_density = self.warp.get_gauss_density(xyz_t, bone2obj=t_articulation)

            # supervise rest articulation
            # rest_articulation = (
            #     samples_dict["rest_articulation"][0][:1],
            #     samples_dict["rest_articulation"][1][:1],
            # )
            # xyz = xyz.view(-1, 3).detach()
            # gauss_density = self.warp.get_gauss_density(xyz, bone2obj=rest_articulation)
            # gauss_density = gauss_density * 100  # [0,100] heuristic value
            gauss_density = gauss_density * self.warp.logibeta.exp()
            gauss_field["gauss_density"] = gauss_density.view((M, N, D, 1))

        return gauss_field

    def visibility_func(self, xyz, inst_id=None):
        """Compute visibility function

        Args:
            xyz: (M,N,D,3) Points along ray in object canonical space
            inst_id: (M,) Instance id. If None, render for the average instance
        Returns:
            vis: (M,N,D,1) Visibility score
        """
        if isinstance(self.warp, SkinningWarp):
            # use gaussians aabb to exclude points outside the skeleton
            articulation = self.warp.articulation.get_mean_vals()
            center = dual_quaternion_to_quaternion_translation(articulation)[1][0]
            gauss_aabb = torch.stack([center.min(0)[0], center.max(0)[0]], 0)
            gauss_aabb = extend_aabb(gauss_aabb, factor=0.5)
            vis = check_inside_aabb(xyz, gauss_aabb[None])
            vis = (vis[..., None].float() - 0.5) * 20  # to -10,10
        else:
            vis = self.vis_mlp(xyz, inst_id=inst_id)
        return vis


    def geometry_init(self, sdf_fn, nsample=2048):
        super().geometry_init(sdf_fn, nsample=nsample)
        device = self.parameters().__next__().device
        if not "-smpl" in self.fg_motion:
            return
        # time-varying sdf initialization
        from pysdf import SDF
        sys.path.insert(0, "projects/gdmdm/")
        from smpl import SMPL

        # smpl
        smpl_scale = self.logscale.exp().detach().cpu().numpy()
        self.smpl_model = SMPL().cuda().eval()
        global_orient = torch.eye(3).cuda()
        global_orient[1:] *= -1
        faces = self.smpl_model.faces

        # prepare poses
        # body_pose_all = axis_angle_to_matrix(self.warp.articulation.init_vals)
        with torch.no_grad():
            frame_ids = torch.arange(0, self.num_frames, device=device)
            t_articulation, rest_articulation = self.warp.articulation.get_vals_and_mean(frame_ids)
            t_embed = self.warp.articulation.time_embedding(frame_id=frame_ids)
            so3 = self.warp.articulation.forward(t_embed, None, return_so3=True)
            body_pose_all = axis_angle_to_matrix(so3)

            zero_pose = torch.zeros_like(so3[:1])
            nframes = so3.shape[0]
            xyz_rest = self.smpl_model(body_pose=axis_angle_to_matrix(zero_pose), global_orient=global_orient[None])["vertices"][0]
            xyz_t = self.smpl_model(body_pose=body_pose_all, global_orient=global_orient[None].repeat(nframes, 1, 1))["vertices"]


        def sdf_fn(pts):
            verts = xyz_t[frame_id].cpu() * smpl_scale
            sdf_fn_numpy = SDF(verts, faces)
            sdf = -sdf_fn_numpy(pts.cpu().numpy())[:, None]  # negative inside
            sdf = torch.tensor(sdf, device=pts.device, dtype=pts.dtype)
            return sdf

        def sdf_pred_fn(pts):
            frame_id_rep = frame_id[None].repeat(len(pts))
            samples_dict = {}
            samples_dict["t_articulation"] = (t_articulation[0][frame_id_rep], 
                                              t_articulation[1][frame_id_rep])
            samples_dict["rest_articulation"] = (rest_articulation[0][frame_id_rep], 
                                                 rest_articulation[1][frame_id_rep])
            pts_warped, warp_dict = self.warp(
                pts[:, None, None],
                frame_id_rep,
                inst_id,
                type="backward",
                samples_dict=samples_dict,
                return_aux=True,
            )
            pts_warped = pts_warped[:, 0, 0]
            delta_skin = warp_dict["delta_skin"]
            sdf, _ = self.forward_sdf(pts_warped, inst_id=inst_id)
            return sdf, delta_skin

        def compute_corresp_loss(frame_id):
            frame_id_rep = frame_id[None].repeat(len(xyz_rest))
            samples_dict = {}
            samples_dict["t_articulation"] = (t_articulation[0][frame_id_rep], 
                                              t_articulation[1][frame_id_rep])
            samples_dict["rest_articulation"] = (rest_articulation[0][frame_id_rep], 
                                                 rest_articulation[1][frame_id_rep])

            # forward: xyz->warp->xyz_t
            # nsample = 512
            # npts = xyz_rest.shape[0]
            # rand_id =  torch.randint(0, npts, (nsample,), device=device)
            pts_warped_t = self.warp(
                xyz_rest[:, None, None],
                frame_id_rep,
                inst_id,
                type="forward",
                samples_dict=samples_dict,
                return_aux=False,
            )[:, 0, 0]
            loss_fw = F.mse_loss(xyz_t[frame_id], pts_warped_t)

            # backward: xyz_t->warp->xyz
            pts_warped_c = self.warp(
                xyz_t[frame_id][:, None, None],
                frame_id_rep,
                inst_id,
                type="backward",
                samples_dict=samples_dict,
                return_aux=False,
            )[:, 0, 0]
            loss_bw = F.mse_loss(xyz_rest, pts_warped_c)

            loss = (loss_fw + loss_bw) / 2
            return loss

        # setup optimizer
        total_steps = 2000
        lr = 2e-3
        opt_model = self.warp.skinning_model
        optimizer = torch.optim.AdamW(opt_model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            lr,
            total_steps,
            pct_start=0.1,
            cycle_momentum=False,
            anneal_strategy="linear",
        )

        # optimize
        for i in range(total_steps):
            optimizer.zero_grad()

            # sample points and gt sdf
            pts, frame_id, _ = self.sample_points_aabb(nsample//4*3, extend_factor=0.0, return_cube=True)
            pts_neg, _, _ = self.sample_points_aabb(nsample//4, extend_factor=0.5, return_cube=True)
            pts = torch.cat([pts, pts_neg], 0)
            # frame_id = frame_id[0] * 0 + 40
            frame_id = frame_id[0]
            inst_id = None

            # get sdf from proxy geometry
            sdf_gt = sdf_fn(pts)

            # evaluate sdf loss
            sdf, delta_skin = sdf_pred_fn(pts)

            # sdf loss
            scale = align_tensors(sdf, sdf_gt)
            sdf_loss = (sdf * scale.detach() - sdf_gt).pow(2).mean()

            # dskin loss
            skin_loss = delta_skin.mean()

            # correspondence loss
            corresp_loss = compute_corresp_loss(frame_id)

            total_loss = sdf_loss * 0.5 + corresp_loss * 0.1 + skin_loss * 0.01
            total_loss.backward()
            nn.utils.clip_grad_norm_(opt_model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            if i % 100 == 0:
                print(f"iter {i}, loss {total_loss.item()}")
                # run marching cubes
                from lab4d.utils.geom_utils import marching_cubes
                aabb = self.get_aabb(inst_id=inst_id, return_cube=True)
                mesh_gt = marching_cubes(sdf_fn, aabb[0], level=0.0)
                mesh_pred = marching_cubes(lambda x: sdf_pred_fn(x)[0], aabb[0], level=0.0)
                mesh_gt.export("tmp/%04d-gt.obj"%i)
                mesh_pred.export("tmp/%04d-pred.obj"%i)