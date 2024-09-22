# Copyright (c) 2023 Gengshan Yang, Carnegie Mellon University.
import torch
import trimesh
from torch import nn
from torch.nn import functional as F

from lab4d.nnutils.base import CondMLP
from lab4d.nnutils.embedding import PosEmbedding, TimeEmbedding
from lab4d.nnutils.pose import (
    ArticulationFlatMLP,
    ArticulationSkelMLP,
    ArticulationURDFMLP,
)
from lab4d.nnutils.skinning import SkinningField
from lab4d.third_party.nvp import NVP
from lab4d.utils.geom_utils import (
    dual_quaternion_skinning,
    marching_cubes,
    extend_aabb,
    linear_blend_skinning,
)
from lab4d.utils.quat_transform import dual_quaternion_inverse, dual_quaternion_mul, quaternion_translation_apply, quaternion_translation_to_dual_quaternion
from lab4d.utils.transforms import get_xyz_bone_distance, get_bone_coords
from lab4d.utils.loss_utils import entropy_loss, cross_entropy_skin_loss
from lab4d.utils.torch_utils import flip_pair


def create_warp(fg_motion, data_info, num_inst):
    """Construct a warping field.

    Args:
        fg_motion (str): Foreground motion type ("rigid", "dense", "bob",
            "skel-{human,quad}", or "comp_skel-{human,quad}_{bob,dense}")
        data_info (Dict): Dataset metadata from get_data_info()
    Returns:
        warp: An IdentityWarp, DenseWarp, SkinningWarp, or ComposedWarp
    """
    # joint angles initialization
    frame_info = data_info["frame_info"]
    if "joint_angles" in data_info.keys():
        joint_angles = data_info["joint_angles"]
    else:
        joint_angles = None

    if fg_motion == "rigid":
        warp = IdentityWarp(frame_info, num_inst)
    elif fg_motion == "dense":
        warp = DenseWarp(frame_info, num_inst)
    elif fg_motion == "se3":
        warp = SE3Warp(frame_info, num_inst)
    elif fg_motion == "bob":
        # use a lower freq since bob is too flexible
        warp = SkinningWarp(frame_info, num_inst, num_freq_t=6, D=5, W=256, skips=[])
    elif fg_motion.startswith("skel-"):
        warp = SkinningWarp(
            frame_info,
            num_inst,
            skel_type=fg_motion,
            joint_angles=joint_angles,
        )
    elif fg_motion.startswith("urdf-"):
        warp = SkinningWarp(
            frame_info,
            num_inst,
            skel_type=fg_motion,
            joint_angles=joint_angles,
        )
    elif fg_motion.startswith("comp"):
        warp = ComposedWarp(
            data_info,
            frame_info,
            num_inst,
            warp_type=fg_motion,
            joint_angles=joint_angles,
        )
    else:
        raise NotImplementedError
    return warp


class IdentityWarp(nn.Module):
    """A rigid warp with no deformation.

    Args:
        frame_info (FrameInfo): Metadata about the frames in a dataset
        num_freq_xyz (int): Number of frequencies in position embedding
        num_freq_t (int): Number of frequencies in time embedding
    """

    def __init__(self, frame_info, num_inst, num_freq_xyz=10, num_freq_t=6):
        super().__init__()
        self.num_frames = frame_info["frame_offset"][-1]
        self.num_inst = num_inst

    def forward(
        self, xyz, frame_id, inst_id, type="forward", samples_dict={}, return_aux=False
    ):
        """
        Args:
            xyz: (M,N,D,3) Points in object canonical space
            frame_id: (M,) Frame id. If None, warp for all frames
            inst_id: (M,) Instance id. If None, warp for the average instance
            type (str): Forward (=> deformed), backward (=> canonical), or flow (t1=>t2)
            samples_dict (Dict): Only used for SkeletonWarp
        Returns:
            xyz: (M,N,D,3) Warped xyz coordinates
        """
        warp_dict = {}
        if return_aux:
            return xyz, warp_dict
        else:
            return xyz

    def get_template_vis(self, aabb):
        """Visualize Gaussian density and SDF as meshes.

        Args:
            aabb: (2,3) Axis-aligned bounding box
        Returns:
            mesh_gauss (Trimesh): Gaussian density mesh
            mesh_sdf (Trimesh): SDF mesh
        """
        mesh = trimesh.Trimesh()
        return mesh, mesh


class DenseWarp(IdentityWarp):
    """Predict dense translation fields, using separate MLPs for forward and
    backward warping. Used by DNeRF.

    Args:
        frame_info (FrameInfo): Metadata about the frames in a dataset
        num_freq_xyz (int): Number of frequencies in position embedding
        num_freq_t (int): Number of frequencies in time embedding
        D (int): Number of linear layers
        W (int): Number of hidden units in each MLP layer
    """

    def __init__(self, frame_info, num_inst, num_freq_xyz=6, num_freq_t=6, D=6, W=256):
        super().__init__(
            frame_info=frame_info, num_inst=num_inst, num_freq_xyz=num_freq_xyz, num_freq_t=num_freq_t
        )

        self.pos_embedding = PosEmbedding(3, num_freq_xyz)
        self.time_embedding = TimeEmbedding(num_freq_t, frame_info)

        self.forward_map = CondMLP(
            self.num_inst,
            D=D,
            W=W,
            in_channels=self.pos_embedding.out_channels
            + self.time_embedding.out_channels,
            out_channels=3,
        )

        self.backward_map = CondMLP(
            self.num_inst,
            D=D,
            W=W,
            in_channels=self.pos_embedding.out_channels
            + self.time_embedding.out_channels,
            out_channels=3,
        )

    def forward(
        self, xyz, frame_id, inst_id, type="forward", samples_dict={}, return_aux=False
    ):
        """
        Args:
            xyz: (M,N,D,3) Points in object canonical space
            frame_id: (M,) Frame id. If None, warp for all frames
            inst_id: (M,) Instance id. If None, warp for the average instance
            type (str): Forward (=> deformed), backward (=> canonical), or flow (t1=>t2)
            samples_dict (Dict): Only used for SkeletonWarp
        Returns:
            xyz: (M,N,D,3) Warped xyz coordinates
        """
        xyz_embed = self.pos_embedding(xyz)
        t_embed = self.time_embedding(frame_id)
        t_embed = t_embed.reshape(-1, 1, 1, t_embed.shape[-1])
        t_embed = t_embed.expand(xyz.shape[:-1] + (-1,))
        embed = torch.cat([xyz_embed, t_embed], dim=-1)
        if type == "backward":
            motion = self.backward_map(embed, inst_id)
            out = xyz + motion * 0.1  # control the scale
        elif type == "forward":
            motion = self.forward_map(embed, inst_id)
            out = xyz + motion * 0.1  # control the scale
        elif type == "flow":
            # TODO: use dx/dt to compute flow
            raise NotImplementedError
            # motion = self.backward_map(embed, inst_id)
            # xyz_canonical = xyz + motion * 0.1
            # xyz_canonical_embed = self.pos_embedding(xyz_canonical)
            # t_embed_flip = flip_pair(t_embed)
            # embed_flip = torch.cat([xyz_canonical_embed, t_embed_flip], dim=-1)
            # motion = self.forward_map(embed_flip, inst_id)
            # out = xyz_canonical + motion * 0.1  # control the scale
        else:
            raise NotImplementedError

        warp_dict = {}
        if return_aux:
            return out, warp_dict
        else:
            return out

class SE3Warp(IdentityWarp):
    """Predict dense SE(3) fields, using separate MLPs for forward and
    backward warping. Used by Nerfies.

    Args:
        frame_info (FrameInfo): Metadata about the frames in a dataset
        num_freq_xyz (int): Number of frequencies in position embedding
        num_freq_t (int): Number of frequencies in time embedding
        D (int): Number of linear layers
        W (int): Number of hidden units in each MLP layer
    """

    def __init__(self, frame_info, num_inst, num_freq_xyz=6, num_freq_t=6, D=6, W=256):
        super().__init__(
            frame_info=frame_info, num_inst=num_inst, num_freq_xyz=num_freq_xyz, num_freq_t=num_freq_t
        )

        self.pos_embedding = PosEmbedding(3, num_freq_xyz)
        self.time_embedding = TimeEmbedding(num_freq_t, frame_info)

        self.forward_map = CondMLP(
            self.num_inst,
            D=D,
            W=W,
            in_channels=self.pos_embedding.out_channels
            + self.time_embedding.out_channels,
            out_channels=7,
        )

        self.backward_map = CondMLP(
            self.num_inst,
            D=D,
            W=W,
            in_channels=self.pos_embedding.out_channels
            + self.time_embedding.out_channels,
            out_channels=7,
        )   

    def forward(
        self, xyz, frame_id, inst_id, type="forward", samples_dict={}, return_aux=False
    ):
        """
        Args:
            xyz: (M,N,D,3) Points in object canonical space
            frame_id: (M,) Frame id. If None, warp for all frames
            inst_id: (M,) Instance id. If None, warp for the average instance
            type (str): Forward (=> deformed), backward (=> canonical), or flow (t1=>t2)
            samples_dict (Dict): Only used for SkeletonWarp
        Returns:
            xyz: (M,N,D,3) Warped xyz coordinates
        """
        xyz_embed = self.pos_embedding(xyz)
        t_embed = self.time_embedding(frame_id)
        t_embed = t_embed.reshape(-1, 1, 1, t_embed.shape[-1])
        t_embed = t_embed.expand(xyz.shape[:-1] + (-1,))
        embed = torch.cat([xyz_embed, t_embed], dim=-1)
        if type == "backward":
            map = self.backward_map
        elif type == "forward":
            map = self.forward_map
        elif type == "flow":
            # TODO: use dx/dt to compute flow
            raise NotImplementedError
            # motion = self.backward_map(embed, inst_id)
            # xyz_canonical = xyz + motion * 0.1
            # xyz_canonical_embed = self.pos_embedding(xyz_canonical)
            # t_embed_flip = flip_pair(t_embed)
            # embed_flip = torch.cat([xyz_canonical_embed, t_embed_flip], dim=-1)
            # motion = self.forward_map(embed_flip, inst_id)
            # out = xyz_canonical + motion * 0.1  # control the scale
        else:
            raise NotImplementedError

        quat_trans = map(embed, inst_id)
        quat, trans = quat_trans.split([4, 3], -1)
        quat = F.normalize(quat, dim=-1)
        trans = trans * 0.1 # control the scale

        out = quaternion_translation_apply(quat, trans, xyz)
        warp_dict = {}
        bs = quat.shape[0]
        warp_dict["dual_quat"] = quaternion_translation_to_dual_quaternion(quat.view(bs, -1,4), trans.view(bs, -1,3))
        if return_aux:
            return out, warp_dict
        else:
            return out

class NVPWarp(IdentityWarp):
    """Predict dense translation fields, using a single invertible MLP for
    forward and backward warping. Used by RealNVP, CaDeX.

    Args:
        frame_info (FrameInfo): Metadata about the frames in a dataset
        num_freq_xyz (int): Number of frequencies in position embedding
        num_freq_t (int): Number of frequencies in time embedding
        D (int): Number of hidden layers
    """

    def __init__(self, frame_info, num_freq_xyz=6, num_freq_t=6, D=2):
        super().__init__(
            frame_info=frame_info, num_freq_xyz=num_freq_xyz, num_freq_t=num_freq_t
        )
        self.time_embedding = TimeEmbedding(num_freq_t, frame_info)

        self.map = NVP(
            n_layers=D,
            feature_dims=self.time_embedding.out_channels,
            hidden_size=[32, 16, 16, 8, 8],
            proj_dims=32,
            code_proj_hidden_size=[32, 32, 32],
            proj_type="simple",
            block_normalize=False,
            normalization=False,
        )

    def forward(
        self, xyz, frame_id, inst_id, type="forward", samples_dict={}, return_aux=False
    ):
        """
        Args:
            xyz: (M,N,D,3) Points in object canonical space
            frame_id: (M) Frame id. If None, warp for all frames
            inst_id: (M) Instance id. If None, warp for the average instance
            type (str): Forward (=> deformed), backward (=> canonical), or flow (t1=>t2)
            samples_dict (Dict): Only used for SkeletonWarp
        Returns:
            out: (..., 3) Warped xyz coordinates
        """
        t_embed = self.time_embedding(frame_id)
        t_embed = t_embed.reshape(-1, 1, 1, t_embed.shape[-1])
        t_embed = t_embed.expand(xyz.shape[:-1] + (-1,))  # (M, N, D, x)
        t_embed = t_embed[:, 0]  # (M, D, x) vs (M, N, D, 3)
        if type == "backward":
            out = self.map.inverse(t_embed, xyz)
        elif type == "forward":
            out = self.map.forward(t_embed, xyz)
        elif type == "flow":
            # TODO: use dx/dt to compute flow
            raise NotImplementedError
            # out = self.map.inverse(t_embed, xyz)
            # t_embed_flip = flip_pair(t_embed)
            # out = self.map.forward(t_embed_flip, out)
        else:
            raise NotImplementedError
        warp_dict = {}
        if return_aux:
            return out, warp_dict
        else:
            return out


class SkinningWarp(IdentityWarp):
    """Neural blend skinning warping field. Supports bag of bones (bob) or skeleton

    Args:
        frame_info (FrameInfo): Metadata about the frames in a dataset
        skel_type (str): Skeleton type ("flat", "skel-human", or "skel-quad")
        joint_angles: (B, 3) If provided, initial joint angles
        num_freq_xyz (int): Number of frequencies in position embedding
        num_freq_t (int): Number of frequencies in time embedding
        num_se3 (int): Number of bones
        init_gauss_scale (float): Initial scale/variance of the Gaussian bones
        init_beta (float): Initial transparency for bone rendering
    """

    def __init__(
        self,
        frame_info,
        num_inst,
        skel_type="flat",
        joint_angles=None,
        D = 2,
        W = 256,
        skips = [1, 2],
        num_freq_xyz=10,
        num_freq_t=10,
        num_se3=25,
        init_gauss_scale=0.03,
        init_beta=0.01,
    ):
        super().__init__(
            frame_info=frame_info, num_inst=num_inst, num_freq_xyz=num_freq_xyz, num_freq_t=num_freq_t
        )
        if skel_type == "flat":
            self.articulation = ArticulationFlatMLP(frame_info, num_se3, num_freq_t=num_freq_t, D=D, W=W, skips=skips)
            symm_idx = None
        elif skel_type.startswith("skel-"):
            skel_type = skel_type.split("-")[1]
            self.articulation = ArticulationSkelMLP(frame_info, skel_type, joint_angles, self.num_inst, num_freq_t=num_freq_t, D=D, W=W, skips=skips)
            num_se3 = self.articulation.num_se3
            symm_idx = self.articulation.symm_idx
        elif skel_type.startswith("urdf-"):
            skel_type = skel_type.split("-")[1]
            self.articulation = ArticulationURDFMLP(frame_info, skel_type, joint_angles, self.num_inst, num_freq_t=num_freq_t, D=D, W=W, skips=skips)
            num_se3 = self.articulation.num_se3
            symm_idx = self.articulation.symm_idx
            init_gauss_scale = (
                self.articulation.bone_sizes * self.articulation.logscale.exp() * init_gauss_scale / 0.03
            )
        else:
            raise NotImplementedError

        self.skinning_model = SkinningField(
            num_se3,
            frame_info,
            self.num_inst,
            init_scale=init_gauss_scale,
            symm_idx=symm_idx,
        )

        # beta: transparency for bone rendering
        beta = torch.tensor([init_beta])
        self.logibeta = nn.Parameter(-beta.log())  # beta: transparency

        # device = self.parameters().__next__().device
        # articulation = self.articulation.forward(
        #     None,
        #     None,
        #     override_so3=torch.zeros(1, num_se3, 3, device=device),
        #     override_log_bone_len=torch.zeros(1, num_se3, device=device),
        # )
        # articulation = (articulation[0][0], articulation[1][0])
        # mesh_gauss = self.skinning_model.draw_gaussian(articulation, self.articulation.edges)
        # mesh_gauss.export("tmp/0.obj")

    def forward(
        self, xyz, frame_id, inst_id, type="forward", samples_dict={}, return_aux=False
    ):
        """Warp points according to a skinning field and articulated bones

        Args:
            xyz: (M,N,D,3) Points in object canonical space
            frame_id: (M,) Frame id. If None, warp for all frames
            inst_id: (M,) Instance id. If None, warp for the mean instance
            type (str): Forward (=> deformed), backward (=> canonical), or flow (t1=>t2)
            samples_dict: Time-dependent bone articulations. Keys:
                "rest_articulation": ((M,B,4), (M,B,4)) and
                "t_articulation": ((M,B,4), (M,B,4))
        Returns:
            out: (M,N,D,3) Warped xyz coordinates
        """
        # compute part-to-object space transformations
        if "rest_articulation" in samples_dict and "t_articulation" in samples_dict:
            rest_articulation = samples_dict["rest_articulation"]
            t_articulation = samples_dict["t_articulation"]
        else:
            (
                t_articulation,
                rest_articulation,
            ) = self.articulation.get_vals_and_mean(frame_id)

        # compute per bone se3
        rest_articulation = (
            rest_articulation[0].expand(t_articulation[0].shape),
            rest_articulation[1].expand(t_articulation[0].shape),
        )
        if type == "backward":
            se3 = dual_quaternion_mul(
                rest_articulation, dual_quaternion_inverse(t_articulation)
            )
            articulation = t_articulation
        elif type == "forward":
            se3 = dual_quaternion_mul(
                t_articulation, dual_quaternion_inverse(rest_articulation)
            )
            articulation = rest_articulation
            frame_id = None
        elif type == "flow":
            t_articulation_flip = flip_pair(t_articulation)
            se3 = dual_quaternion_mul(
                t_articulation_flip, dual_quaternion_inverse(t_articulation)
            )
            articulation = t_articulation
        else:
            raise NotImplementedError

        articulation = (
            articulation[0][:, None, None].expand(xyz.shape[:3] + (-1, -1)),
            articulation[1][:, None, None].expand(xyz.shape[:3] + (-1, -1)),
        )

        # skinning weights
        skin, delta_skin = self.skinning_model(xyz, articulation, frame_id, inst_id)
        # # temperature scaling -> smooth control
        # skin *= 0.2
        # # debug: hard selection
        # max_bone = 3
        # topk, indices = skin.topk(max_bone, 3, largest=True)
        # skin = torch.zeros_like(skin).fill_(-torch.inf)
        # skin = skin.scatter(3, indices, topk)
        skin_prob = skin.softmax(-1)

        # left-multiply per-point se3
        out, dual_quat = dual_quaternion_skinning(se3, xyz, skin_prob)

        # # linear blend skinning
        # out = linear_blend_skinning(se3, xyz, skin_prob)

        warp_dict = {}
        # warp_dict["skin_entropy"] = cross_entropy_skin_loss(skin)[..., None]
        warp_dict["skin_entropy"] = torch.zeros_like(skin[..., :1])  # TODO: remove
        warp_dict["dual_quat"] = dual_quat
        if delta_skin is not None:
            # (M, N, D, 1)
            warp_dict["delta_skin"] = delta_skin.pow(2).mean(-1, keepdims=True)
        if return_aux:
            return out, warp_dict
        else:
            return out

    def get_gauss_sdf(self, xyz, bias=0.0):
        """Calculate signed distance to Gaussian bones

        Args:
            xyz: (N, 3) Points in object canonical space
            bias (float): Value to add to signed distances, to control the
                surface to be different from the skinning weights
        Returns:
            sdf: (N,) Signed distance to Gaussian bones, -inf to +inf
        """
        density = self.get_gauss_density(xyz)
        density = density.clamp(1e-6, 1 - 1e-6)
        sdf = -density.logit()  # (N,), -inf, inf
        # control the surface to be different from skinning weights
        sdf = sdf + bias
        return sdf

    def get_xyz_bone_distance(self, xyz, bone2obj=None):
        """
        Args:
            xyz: (N, 3) Points in object canonical space
            bone2obj: ((1/N,B,4), (M,B,4)) Bone-to-object SE(3) transforms,
        Returns:
            dist2: (N,B) Squared distance to each bone
        """
        if isinstance(self.articulation, ArticulationURDFMLP):
            # # gauss bones only
            # xyz_bone = self.skinning_model.get_gauss_bone_coords(xyz, bone2obj)
            # dist2 = xyz_bone.pow(2).sum(dim=-1)
            # gauss bones + skinning
            xyz = xyz[:, None, None]  # (N,1,1,3)
            bone2obj = (
                bone2obj[0][:, None, None].expand(xyz.shape[0], -1, -1, -1, -1),
                bone2obj[1][:, None, None].expand(xyz.shape[0], -1, -1, -1, -1),
            )  # (N,1,1,K,4)
            dist2 = -self.skinning_model.forward(xyz, bone2obj, None, None)[0][:, 0, 0]
        else:
            dist2 = get_xyz_bone_distance(xyz, bone2obj)  # N,K
            dist2 = dist2 / (0.01) ** 2  # assuming spheres of radius 0.01
        return dist2

    def get_gauss_density(self, xyz, bone2obj=None):
        """Sample volumetric density at Gaussian bones

        Args:
            xyz: (M,3) Points in object canonical space
            bone2obj: ((M,B,4), (M,B,4)) Bone-to-object SE(3) transforms,
                written as dual quaternions
        Returns:
            density: (M,3) Volumetric density of bones at each point
        """
        if bone2obj is None:
            bone2obj = self.articulation.get_mean_vals()  # 1,K,4,4
        dist2 = self.get_xyz_bone_distance(xyz, bone2obj)  # (N,K)
        score = (-0.5 * dist2).exp()  # (N,K)

        # # hard selection
        # density = score.max(-1)[0]  # (N,)

        # soften the density selection
        mix_prob = F.softmax(-dist2, -1)
        density = (score * mix_prob).sum(-1)

        density = density[..., None]
        return density

    def get_gauss_pts(self, radius=1):
        """Sample points from Gaussian bones"""
        articulation = self.articulation.get_mean_vals()  # (1,K,4,4)
        articulation = (articulation[0][0], articulation[1][0])
        pts = self.skinning_model.get_gauss_pts(articulation, radius=radius)
        return pts

    def get_gauss_vis(self, show_joints=False):
        """Visualize Gaussians as meshes.

        Args:
            aabb: (2,3) Axis-aligned bounding box
        Returns:
            mesh_gauss (Trimesh): Gaussian density mesh
        """
        articulation = self.articulation.get_mean_vals()  # (1,K,4,4)
        articulation = (articulation[0][0], articulation[1][0])
        mesh_gauss = self.skinning_model.draw_gaussian(
            articulation, self.articulation.edges, show_joints=show_joints
        )
        return mesh_gauss

    def get_template_vis(self, aabb):
        """Visualize Gaussian density and SDF as meshes.

        Args:
            aabb: (2,3) Axis-aligned bounding box
        Returns:
            mesh_gauss (Trimesh): Gaussian density mesh
            mesh_sdf (Trimesh): SDF mesh
        """
        mesh_gauss = self.get_gauss_vis()

        sdf_func = lambda xyz: self.get_gauss_sdf(xyz)
        mesh_sdf = marching_cubes(sdf_func, aabb, level=0.0)
        return mesh_gauss, mesh_sdf


class ComposedWarp(SkinningWarp):
    """Compose a skeleton warp and soft warp. The skeleton warp handles large
    articulations and the soft warp handles fine details (e.g. ears)

    Args:
        vid_info (Dict): Dataset metadata from get_data_info()
        frame_info (FrameInfo): Metadata about the frames in a dataset
        warp_type (str): Type of warp: "comp_skel-{human,quad}_{bob,dense}"
        joint_angles: (B, 3) If provided, initial joint angles
    """

    def __init__(
        self,
        data_info,
        frame_info,
        num_inst,
        warp_type,
        joint_angles=None,
    ):
        # e.g., comp_skel-human_dense, limited to skel+another type of field
        type_list = warp_type.split("_")[1:]
        assert len(type_list) == 2
        assert type_list[0] in ["skel-human", "skel-quad", "urdf-human", "urdf-quad"]
        assert type_list[1] in ["bob", "dense"]
        if type_list[1] == "bob":
            raise NotImplementedError

        super().__init__(
            frame_info,
            num_inst,
            skel_type=type_list[0],
            joint_angles=joint_angles,
        )
        # self.post_warp = DenseWarp(frame_info, D=2, W=64)
        # self.post_warp = DenseWarp(frame_info, D=2, W=128)
        self.post_warp = DenseWarp(frame_info, D=2, W=256)
        # self.post_warp = NVPWarp(frame_info)

    def forward(
        self, xyz, frame_id, inst_id, type="forward", samples_dict={}, return_aux=False
    ):
        """Warp points according to a skinning field and articulated bones

        Args:
            xyz: (M,N,D,3) Points in object canonical space
            frame_id: (M,) Frame id. If None, warp for all frames
            inst_id: (M,) Instance id. If None, warp for the mean instance
            type (str): Forward (=> deformed), backward (=> canonical), or flow (t1=>t2)
            samples_dict: Time-dependent bone articulations. Keys:
                "rest_articulation": ((M,B,4), (M,B,4)) and
                "t_articulation": ((M,B,4), (M,B,4))
        Returns:
            out: (M,N,D,3) Warped xyz coordinates
        """
        # if forward, and has frame_id
        if type == "flow":
            raise NotImplementedError
        if (type == "forward" or type == "flow") and frame_id is not None:
            xyz = self.post_warp.forward(
                xyz, frame_id, inst_id, type="forward", samples_dict=samples_dict
            )

        out, warp_dict = super().forward(
            xyz,
            frame_id,
            inst_id,
            type=type,
            samples_dict=samples_dict,
            return_aux=True,
        )

        if (type == "backward" or type == "flow") and frame_id is not None:
            out = self.post_warp.forward(
                out, frame_id, inst_id, type="backward", samples_dict=samples_dict
            )
        if return_aux:
            return out, warp_dict
        else:
            return out

    def compute_post_warp_dist2(self, xyz, frame_id, inst_id):
        """Compute L2 distance between points before soft deformation and points
        after soft deformation

        Args:
            xyz: (M, ..., 3) Points after skeleton articulation
            frame_id: (M,) Frame id. If None, warp for all frames
            inst_id: (M,) Instance id. If None, warp for the mean instance
        Returns:
            dist2: (M, ...) Squared soft deformation distance
        """
        xyz_t = self.post_warp.forward(xyz, frame_id, inst_id, type="forward")
        dist2 = (xyz_t - xyz).pow(2).sum(-1)

        # additional cycle consistency regularization for soft deformation
        if isinstance(self.post_warp, DenseWarp):
            xyz_back = self.post_warp.forward(xyz_t, frame_id, inst_id, type="backward")
            dist2 = (dist2 + (xyz_t - xyz_back).pow(2).sum(-1)) * 0.5
        return dist2
