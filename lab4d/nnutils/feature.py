# Copyright (c) 2023 Gengshan Yang, Carnegie Mellon University.
import numpy as np
import torch
import trimesh
import cv2
from torch import nn

from lab4d.nnutils.base import BaseMLP, CondMLP
from lab4d.nnutils.embedding import PosEmbedding
from lab4d.nnutils.nerf import NeRF
from lab4d.utils.decorator import train_only_fields
from lab4d.utils.geom_utils import (
    Kmatinv,
    pinhole_projection,
    extend_aabb,
    check_inside_aabb,
)


class FeatureNeRF(NeRF):
    """A neural radiance field that renders features in addition to RGB

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
        init_scale (float): Initial geometry scale factor
        color_act (bool): If True, apply sigmoid to the output RGB
        feature_channels (int): Number of feature field channels
    """

    def __init__(
        self,
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
        field_arch=CondMLP,
        extrinsics_type="mlp",
        invalid_vid=-1,
    ):
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
            field_arch=field_arch,
            extrinsics_type=extrinsics_type,
            invalid_vid=invalid_vid,
        )

        if feature_channels <= 0:
            self.use_feature = False
            return

        self.use_feature = True

        self.feat_pos_embedding = PosEmbedding(3, N_freqs=6)  # lower frequency
        self.feature_field = BaseMLP(
            # self.feature_field = CondMLP(
            # num_inst=self.num_inst,
            D=5,
            W=128,
            in_channels=self.feat_pos_embedding.out_channels,
            out_channels=feature_channels,
            final_act=False,
        )

        sigma = torch.tensor([1.0])
        self.logsigma = nn.Parameter(sigma.log())
        self.set_match_region(sample_around_surface=True)

    def set_match_region(self, sample_around_surface):
        self.sample_around_surface = sample_around_surface

    def query_field(self, samples_dict, flow_thresh=None):
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
                "depth" (M,1,D,1) TODO
            deltas: (M,N,D,1) Distance along rays between adjacent samples
            aux_dict (Dict): Auxiliary neural field outputs. Keys: TODO
        """
        feat_dict, deltas, aux_dict = super(FeatureNeRF, self).query_field(
            samples_dict, flow_thresh=flow_thresh
        )
        if not self.use_feature:
            return feat_dict, deltas, aux_dict
        xyz = feat_dict["xyz"]
        field2cam = samples_dict["field2cam"]
        Kinv = samples_dict["Kinv"]
        frame_id = samples_dict["frame_id"]
        inst_id = samples_dict["inst_id"]

        # samples feature field
        feat_field_dict = self.compute_feat(xyz, inst_id)
        feat_dict.update(feat_field_dict)

        # global matching
        if "feature" in samples_dict and "feature" in feat_dict:
            feature = feat_dict["feature"]
            sdf = feat_dict["sdf"]
            feature, xyz = self.propose_matches(feature, xyz.detach(), sdf)
            xyz_matches = self.global_match(samples_dict["feature"], feature, xyz)
            xy_reproj, xyz_reproj = self.forward_project(
                xyz_matches,
                field2cam,
                Kinv,
                frame_id,
                inst_id,
                samples_dict=samples_dict,
            )
            aux_dict["xyz_matches"] = xyz_matches
            aux_dict["xyz_reproj"] = xyz_reproj
            hxy = samples_dict["hxy"][..., :2]
            aux_dict["xy_reproj"] = (xy_reproj - hxy).norm(2, -1, keepdim=True)
            # # visualize matches
            # if not self.training:
            #     img = self.plot_xy_matches(xy_reproj, samples_dict)
            #     cv2.imwrite("tmp/arrow.png", img)
            #     trimesh.Trimesh(vertices=xyz_matches[0].cpu().numpy()).export(
            #         "tmp/matches.obj"
            #     )
            # import pdb

            # pdb.set_trace()
        return feat_dict, deltas, aux_dict

    def plot_xy_matches(self, xy_reproj, samples_dict):
        # plot arrow from hxy to xy_reproj
        res = int(np.sqrt(samples_dict["hxy"].shape[1]))
        img = np.zeros((res * 16, res * 16, 3), dtype=np.uint8)
        hxy_vis = samples_dict["hxy"].view(-1, res, res, 3)[..., :2].cpu().numpy()
        xy_reproj_vis = xy_reproj.view(-1, res, res, 2).cpu().numpy()
        feature_vis = samples_dict["feature"].view(-1, res, res, 16)
        for i in range(res):
            for j in range(res):
                if feature_vis[0, i, j].norm(2, -1) == 0:
                    continue
                # draw a line
                img = cv2.arrowedLine(
                    img,
                    tuple(hxy_vis[0, i, j] * 16),
                    tuple(xy_reproj_vis[0, i, j] * 16),
                    (0, 255, 0),
                    1,
                )
        return img

    def propose_matches(self, feature, xyz, sdf, num_candidates=8192):
        """Sample canonical points for global matching
        Args:
            feature: (M,N,D,feature_channels) Pixel features
            xyz: (M,N,D,3) Points in field coordinates
            num_candidates: Number of candidates to sample
        Returns:
            feature: (num_candidates, feature_channels) Canonical features
            xyz: (num_candidates, 3) Points in field coordinates
        """
        # threshold
        if self.sample_around_surface:
            thresh = 0.005
        else:
            thresh = 1
        # sample canonical points
        feature = feature.view(-1, feature.shape[-1])  # (M*N*D, feature_channels)
        xyz = xyz.view(-1, 3)  # (M*N*D, 3)

        # remove points outsize aabb
        aabb = self.get_aabb()
        aabb = extend_aabb(aabb, 0.1)
        inside_aabb = check_inside_aabb(xyz, aabb)
        feature = feature[inside_aabb]
        xyz = xyz[inside_aabb]
        sdf = sdf.view(-1)[inside_aabb]

        # remove points far from the surface beyond a sdf threshold
        is_near_surface = sdf.abs() < thresh
        feature = feature[is_near_surface]
        xyz = xyz[is_near_surface]

        num_candidates = min(num_candidates, feature.shape[0])
        idx = torch.randperm(feature.shape[0])[: num_candidates // 2]
        feature = feature[idx]  # (num_candidates, feature_channels)
        xyz = xyz[idx]  # (num_candidates, 3)

        # sample additional points
        if self.sample_around_surface:
            # sample from proxy geometry on the surface
            proxy_geometry = self.get_proxy_geometry()
            rand_xyz, _ = trimesh.sample.sample_surface(
                proxy_geometry, num_candidates // 2
            )
            rand_xyz = torch.tensor(rand_xyz, dtype=torch.float32, device=xyz.device)
        else:
            # sample from aabb
            rand_xyz, _, _ = self.sample_points_aabb(
                num_candidates // 2, extend_factor=0.1
            )
        rand_feat = self.compute_feat(rand_xyz, None)["feature"]

        # combine
        feature = torch.cat([feature, rand_feat], dim=0)
        xyz = torch.cat([xyz, rand_xyz], dim=0)
        return feature, xyz

    def compute_feat(self, xyz, inst_id):
        """Render feature field

        Args:
            xyz: (M,N,D,3) Points in field coordinates
        Returns:
            feat_field_dict: Feature field. Keys: "feature" (M,N,D,16)
        """
        feat_field_dict = {}
        xyz_embed = self.feat_pos_embedding(xyz)
        # feature = self.feature_field(xyz_embed, inst_id)
        feature = self.feature_field(xyz_embed)
        feature = feature / feature.norm(dim=-1, keepdim=True)
        feat_field_dict["feature"] = feature
        return feat_field_dict

    def global_match(
        self,
        feat_px,
        feat_canonical,
        xyz_canonical,
        num_grad=0,
    ):
        """Match pixel features to canonical features, which combats local
        minima in differentiable rendering optimization

        Args:
            feat: (M,N,feature_channels) Pixel features
            feat_canonical: (...,feature_channels) Canonical features
            xyz_canonical: (...,3) Canonical points
        Returns:
            xyz_matched: (M,N,3) Matched xyz
        """
        shape = feat_px.shape
        feat_px = feat_px.view(-1, shape[-1])  # (M*N, feature_channels)

        # compute similarity
        score = torch.matmul(feat_px, feat_canonical.t())  # (M*N, num_candidates)

        # find top K candidates
        if num_grad > 0:
            num_grad = min(num_grad, score.shape[1])
            score, idx = torch.topk(score, num_grad, dim=1, largest=True)
            xyz_canonical = xyz_canonical[idx]

        # soft argmin
        # score = score.detach()  # do not backprop to features
        score = score * self.logsigma.exp()  # temperature
        prob = torch.softmax(score, dim=1)
        xyz_matched = torch.sum(prob.unsqueeze(-1) * xyz_canonical, dim=1)
        xyz_matched = xyz_matched.view(shape[:-1] + (-1,))
        return xyz_matched

    def forward_project(self, xyz, field2cam, Kinv, frame_id, inst_id, samples_dict={}):
        """Project xyz to image plane

        Args:
            xyz: (M,N,3) Points in field coordinates
            Kinv: (M,3,3) Inverse of camera intrinsics
            field2cam: (M,1,1,4,4) Field to camera transformation
            frame_id: (M,) Frame id. If None, warp for all frames
            inst_id: (M,) Instance id. If None, warp for the average instance

        Returns:
            xy: (M,N,2) Points in image plane
        """
        # TODO: make the format consistent
        xyz = xyz[:, :, None]
        # inst_id = inst_id[..., :1]
        # transform xyz to camera coordinates
        xyz_cam = self.forward_warp(
            xyz, field2cam, frame_id, inst_id, samples_dict=samples_dict
        )
        xyz_cam = xyz_cam[:, :, 0]

        # project
        Kmat = Kmatinv(Kinv)
        xy_reproj = pinhole_projection(Kmat, xyz_cam)[..., :2]
        return xy_reproj, xyz_cam

    @torch.no_grad()
    def extract_canonical_feature(self, mesh, inst_id):
        """Extract color on canonical mesh vertices

        Args:
            mesh (Trimesh): Canonical mesh
        Returns:
            feature (np.ndarray): Feature on vertices
        """
        device = next(self.parameters()).device
        verts = torch.tensor(mesh.vertices, dtype=torch.float32, device=device)
        feature = self.compute_feat(verts, inst_id)["feature"]
        return feature.cpu().numpy()
