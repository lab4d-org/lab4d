# Copyright (c) 2023 Gengshan Yang, Carnegie Mellon University.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import trimesh

from lab4d.nnutils.base import CondMLP, BaseMLP, ScaleLayer
from lab4d.nnutils.time import TimeMLP
from lab4d.nnutils.embedding import TimeEmbedding, TimeInfo
from lab4d.utils.geom_utils import (
    so3_to_exp_map,
    rot_angle,
    interpolate_slerp,
    interpolate_linear,
)
from lab4d.utils.quat_transform import (
    axis_angle_to_quaternion,
    axis_angle_to_matrix,
    matrix_to_quaternion,
    matrix_to_axis_angle,
    quaternion_mul,
    quaternion_translation_to_dual_quaternion,
    dual_quaternion_mul,
    quaternion_translation_to_se3,
    dual_quaternion_to_quaternion_translation,
    quaternion_translation_mul,
    symmetric_orthogonalization,
    quaternion_to_axis_angle,
)
from lab4d.utils.skel_utils import (
    fk_se3,
    get_predefined_skeleton,
    rest_joints_to_local,
    shift_joints_to_bones_dq,
    apply_root_offset,
)
from lab4d.utils.vis_utils import draw_cams
from lab4d.utils.torch_utils import reinit_model


class CameraMixSE3(nn.Module):
    """Mix multiple camera models based on sequence id
    Args:
        rtmat: (N,4,4) Object to camera transform
        frame_info (Dict): Metadata about the frames in a dataset
        const_id: (N,) Sequence id whose camera pose is constant
    """

    def __init__(self, rtmat, frame_info=None, const_vid_id=0):
        super().__init__()
        self.camera_const1 = CameraConst(rtmat, frame_info)
        self.camera_const2 = CameraExplicit(rtmat, frame_info)

        # per-video sim3
        if frame_info is None:
            num_frames = len(rtmat)
            frame_info = {
                "frame_offset": np.asarray([0, num_frames]),
                "frame_mapping": list(range(num_frames)),
                "frame_offset_raw": np.asarray([0, num_frames]),
            }
        self.time_info = TimeInfo(frame_info)

        self.global_trans = nn.Parameter(torch.zeros(self.time_info.num_vids, 3))
        global_quat = torch.zeros(self.time_info.num_vids, 4)
        global_quat[:, 0] = 1  # xxyz
        self.global_quat = nn.Parameter(global_quat)
        self.global_logscale = nn.Parameter(torch.zeros(self.time_info.num_vids))

        self.frame_info = frame_info
        self.const_vid_id = const_vid_id

    def mlp_init(self):
        pass

    def get_vals(self, frame_id=None, apply_global_rt=True):
        """Compute camera pose at the given frames.

        Args:
            frame_id: (M,) Frame id. If None, compute values at all frames
        Returns:
            quat: (M, 4) Output camera rotations
            trans: (M, 3) Output camera translations
        """
        # compute all
        quat_const1, trans_const1 = self.camera_const1.get_vals(frame_id)
        quat_const2, trans_const2 = self.camera_const2.get_vals(frame_id)

        # mix based on sequence id
        if frame_id is None:
            frame_id = self.time_info.frame_mapping
        frame_id = frame_id.long()
        raw_fid_to_vid = self.time_info.raw_fid_to_vid

        if apply_global_rt:
            quat_const2, trans_const2 = self.apply_global_rt(
                quat_const2, trans_const2, frame_id
            )
        else:
            quat_const2, trans_const2 = quat_const2.clone(), trans_const2.clone()

        # replace with constant camera poses
        const1_frame_id = raw_fid_to_vid[frame_id] == self.const_vid_id
        if const1_frame_id.sum() > 0:
            quat_const2[const1_frame_id] = quat_const1[const1_frame_id]
            trans_const2[const1_frame_id] = trans_const1[const1_frame_id]

        return quat_const2, trans_const2

    def apply_global_rt(self, quat_const2, trans_const2, frame_id):
        # expand global se3 to have same size as frame_id
        raw_fid_to_vid = self.time_info.raw_fid_to_vid
        global_quat = []
        global_trans = []
        global_scale = []
        for vid_id in range(self.time_info.num_vids):
            vid_frame_id = raw_fid_to_vid == vid_id
            vid_frame_id_len = vid_frame_id.sum()
            global_quat.append(
                self.global_quat[vid_id : vid_id + 1].repeat(vid_frame_id_len, 1)
            )
            global_trans.append(
                self.global_trans[vid_id : vid_id + 1].repeat(vid_frame_id_len, 1)
            )
            global_scale.append(
                self.global_logscale[vid_id : vid_id + 1].repeat(vid_frame_id_len).exp()
            )
        global_quat = torch.cat(global_quat, dim=0)[frame_id]
        global_trans = torch.cat(global_trans, dim=0)[frame_id]
        global_scale = torch.cat(global_scale, dim=0)[frame_id]
        global_trans = global_trans * global_scale[:, None]

        # multiply with per-video global se3
        qt = (global_quat, global_trans)
        qt2 = (quat_const2, trans_const2)
        qt2 = quaternion_translation_mul(qt2, qt)
        quat_const2 = qt2[0]
        trans_const2 = qt2[1]
        return quat_const2, trans_const2

    def compute_distance_to_prior_relative(self):
        quat_const, trans_const = self.camera_const1.get_vals()
        y = quaternion_translation_to_se3(quat_const, trans_const)
        quat, trans = self.get_vals(apply_global_rt=False)
        x = quaternion_translation_to_se3(quat, trans)

        loss = []
        frame_offset = self.time_info.frame_offset
        for vidid in range(len(frame_offset[:-1])):
            x_rel = x[frame_offset[vidid] : frame_offset[vidid + 1]]
            y_rel = y[frame_offset[vidid] : frame_offset[vidid + 1]]

            loss_sub = F.mse_loss(x_rel, y_rel)
            loss.append(loss_sub)
        loss = torch.stack(loss).mean()
        return loss


class CameraMix(nn.Module):
    """Mix multiple camera models based on sequence id
    Args:
        rtmat: (N,4,4) Object to camera transform
        frame_info (Dict): Metadata about the frames in a dataset
        const_id: (N,) Sequence id whose camera pose is constant
    """

    def __init__(self, rtmat, frame_info=None, const_vid_id=0):
        super().__init__()
        self.camera_const = CameraConst(rtmat, frame_info)
        self.camera_mlp = CameraMLP_so3(rtmat, frame_info=frame_info)
        self.frame_info = frame_info
        self.const_vid_id = const_vid_id

    def mlp_init(self):
        self.camera_mlp.mlp_init()

    def get_vals(self, frame_id=None):
        """Compute camera pose at the given frames.

        Args:
            frame_id: (M,) Frame id. If None, compute values at all frames
        Returns:
            quat: (M, 4) Output camera rotations
            trans: (M, 3) Output camera translations
        """
        # compute all
        quat_const, trans_const = self.camera_const.get_vals(frame_id)
        quat, trans = self.camera_mlp.get_vals(frame_id)
        # mix based on sequence id
        if frame_id is None:
            frame_id = self.camera_mlp.time_embedding.frame_mapping

        raw_fid_to_vid = self.camera_mlp.time_embedding.raw_fid_to_vid
        const_frame_id = raw_fid_to_vid[frame_id.long()] == self.const_vid_id
        if const_frame_id.sum() > 0:
            quat[const_frame_id] = quat_const[const_frame_id]
            trans[const_frame_id] = trans_const[const_frame_id]

        return quat, trans


class CameraConst(nn.Module):
    """Constant camera pose

    Args:
        rtmat: (N,4,4) Object to camera transform
        frame_info (Dict): Metadata about the frames in a dataset
    """

    def __init__(self, rtmat, frame_info=None):
        super().__init__()
        if frame_info is None:
            num_frames = len(rtmat)
            frame_info = {
                "frame_offset": np.asarray([0, num_frames]),
                "frame_mapping": list(range(num_frames)),
                "frame_offset_raw": np.asarray([0, num_frames]),
            }
        self.frame_info = frame_info
        self.time_info = TimeInfo(frame_info)
        frame_mapping = torch.tensor(frame_info["frame_mapping"])
        frame_mapping_inv = torch.full((frame_mapping.max().item() + 1,), 0)
        frame_mapping_inv[frame_mapping] = torch.arange(len(frame_mapping))
        self.register_buffer("frame_mapping_inv", frame_mapping_inv, persistent=True)

        # camera pose: field to camera
        if not torch.is_tensor(rtmat):
            rtmat = torch.tensor(rtmat, dtype=torch.float32)
        trans = rtmat[:, :3, 3]
        quat = matrix_to_quaternion(rtmat[:, :3, :3])
        self.register_buffer("trans", trans, persistent=True)
        self.register_buffer("quat", quat, persistent=True)
        self.register_buffer("init_vals", rtmat, persistent=False)

    def mlp_init(self):
        pass

    def get_vals(self, frame_id=None):
        """Compute camera pose at the given frames.

        Args:
            frame_id: (M,) Frame id. If None, compute values at all frames
        Returns:
            quat: (M, 4) Output camera rotations
            trans: (M, 3) Output camera translations
        """
        # TODO: frame_id is absolute, need to fix
        # frame_mapping_inv = torch.full((frame_mapping.max().item() + 1,), 0)
        if frame_id is None:
            quat = self.quat
            trans = self.trans
        else:
            frame_id = self.frame_mapping_inv[frame_id.long()]
            quat = self.quat[frame_id]
            trans = self.trans[frame_id]
        return quat, trans


class CameraExplicit(CameraConst):
    """Explicit camera pose that can be optimized

    Args:
        rtmat: (N,4,4) Object to camera transform
        frame_info (Dict): Metadata about the frames in a dataset
    """

    def __init__(self, rtmat, frame_info=None):
        super().__init__(rtmat, frame_info=frame_info)
        trans = self.trans.data
        quat = self.quat.data
        self.trans = nn.Parameter(trans)
        self.quat = nn.Parameter(quat)


class CameraMLP_old(TimeMLP):
    """Encode camera pose over time (rotation + translation) with an MLP

    Args:
        rtmat: (N,4,4) Object to camera transform
        frame_info (Dict): Metadata about the frames in a dataset
        D (int): Number of linear layers
        W (int): Number of hidden units in each MLP layer
        num_freq_t (int): Number of frequencies in time Fourier embedding
        skips (List(int)): List of layers to add skip connections at
        activation (Function): Activation function to use (e.g. nn.ReLU())
    """

    def __init__(
        self,
        rtmat,
        frame_info=None,
        D=5,
        W=256,
        num_freq_t=6,
        skips=[],
        activation=nn.ReLU(True),
    ):
        if frame_info is None:
            num_frames = len(rtmat)
            frame_info = {
                "frame_offset": np.asarray([0, num_frames]),
                "frame_mapping": list(range(num_frames)),
                "frame_offset_raw": np.asarray([0, num_frames]),
            }
        # xyz encoding layers
        super().__init__(
            frame_info,
            D=D,
            W=W,
            num_freq_t=num_freq_t,
            skips=skips,
            activation=activation,
        )

        # output layers
        self.trans = nn.Sequential(
            nn.Linear(W, W // 2),
            activation,
            nn.Linear(W // 2, 3),
        )
        self.quat = nn.Sequential(
            nn.Linear(W, W // 2),
            activation,
            nn.Linear(W // 2, 4),
        )

        # camera pose: field to camera
        self.base_quat = nn.Parameter(torch.zeros(self.time_embedding.num_vids, 4))
        self.base_trans = nn.Parameter(torch.zeros(self.time_embedding.num_vids, 3))
        self.register_buffer(
            "init_vals", torch.tensor(rtmat, dtype=torch.float32), persistent=False
        )
        self.base_init()

        # override the loss function
        def loss_fn(gt):
            quat, trans = self.get_vals()
            pred = quaternion_translation_to_se3(quat, trans)
            loss = F.mse_loss(pred, gt)
            return loss

        self.loss_fn = loss_fn

    def base_init(self):
        """Initialize base camera rotations from initial camera trajectory"""
        rtmat = self.init_vals
        frame_offset = self.get_frame_offset()
        base_rmat = rtmat[frame_offset[:-1], :3, :3]
        base_quat = matrix_to_quaternion(base_rmat)
        self.base_quat.data = base_quat
        self.base_trans.data = rtmat[frame_offset[:-1], :3, 3]

    def mlp_init(self):
        """Initialize camera SE(3) transforms from external priors"""
        super().mlp_init()

        # with torch.no_grad():
        #     os.makedirs("tmp", exist_ok=True)
        #     draw_cams(rtmat.cpu().numpy()).export("tmp/cameras_gt.obj")
        #     quat, trans = self.get_vals()
        #     rtmat_pred = quaternion_translation_to_se3(quat, trans)
        #     draw_cams(rtmat_pred.cpu()).export("tmp/cameras_pred.obj")

    def forward(self, t_embed):
        """
        Args:
            t_embed: (M, self.W) Input Fourier time embeddings
        Returns:
            quat: (M, 4) Output camera rotation quaternions
            trans: (M, 3) Output camera translations
        """
        t_feat = super().forward(t_embed)
        trans = self.trans(t_feat)
        quat = self.quat(t_feat)
        quat = F.normalize(quat, dim=-1)
        return quat, trans

    def get_vals(self, frame_id=None):
        """Compute camera pose at the given frames.

        Args:
            frame_id: (M,) Frame id. If None, compute values at all frames
        Returns:
            quat: (M, 4) Output camera rotations
            trans: (M, 3) Output camera translations
        """
        t_embed = self.time_embedding(frame_id)
        quat, trans = self.forward(t_embed)
        if frame_id is None:
            inst_id = self.time_embedding.frame_to_vid
        else:
            inst_id = self.time_embedding.raw_fid_to_vid[frame_id.long()]

        # multiply with per-instance base rotation
        base_quat = self.base_quat[inst_id]
        base_quat = F.normalize(base_quat, dim=-1)
        quat = quaternion_mul(quat, base_quat)

        base_trans = self.base_trans[inst_id]
        trans = trans + base_trans
        return quat, trans


class CameraMLP(TimeMLP):
    """Encode camera pose over time (rotation + translation) with an MLP

    Args:
        rtmat: (N,4,4) Object to camera transform
        frame_info (Dict): Metadata about the frames in a dataset
        D (int): Number of linear layers
        W (int): Number of hidden units in each MLP layer
        num_freq_t (int): Number of frequencies in time Fourier embedding
        skips (List(int)): List of layers to add skip connections at
        activation (Function): Activation function to use (e.g. nn.ReLU())
    """

    def __init__(
        self,
        rtmat,
        frame_info=None,
        D=5,
        W=256,
        num_freq_t=6,
        skips=[],
        activation=nn.ReLU(True),
    ):
        if frame_info is None:
            num_frames = len(rtmat)
            frame_info = {
                "frame_offset": np.asarray([0, num_frames]),
                "frame_mapping": list(range(num_frames)),
                "frame_offset_raw": np.asarray([0, num_frames]),
            }
        # xyz encoding layers
        super().__init__(
            frame_info,
            D=D,
            W=W,
            num_freq_t=num_freq_t,
            skips=skips,
            activation=activation,
        )

        self.time_embedding_rot = TimeEmbedding(
            num_freq_t,
            frame_info,
            out_channels=W,
            time_scale=1,
        )

        self.base_rot = BaseMLP(
            D=D,
            W=W,
            in_channels=W,
            out_channels=W,
            skips=skips,
            activation=activation,
            final_act=True,
        )

        # output layers
        self.trans = nn.Sequential(
            nn.Linear(W, W // 2),
            activation,
            nn.Linear(W // 2, 3),
        )
        self.quat = nn.Sequential(
            nn.Linear(W, W // 2),
            activation,
            nn.Linear(W // 2, 4),
        )

        # camera pose: field to camera
        self.base_quat = nn.Parameter(torch.zeros(self.time_embedding.num_vids, 4))
        self.register_buffer(
            "init_vals", torch.tensor(rtmat, dtype=torch.float32), persistent=False
        )
        self.base_init()

        # override the loss function
        def loss_fn(gt):
            quat, trans = self.get_vals()
            pred = quaternion_translation_to_se3(quat, trans)
            loss = F.mse_loss(pred, gt)
            return loss

        self.loss_fn = loss_fn

    def base_init(self):
        """Initialize base camera rotations from initial camera trajectory"""
        rtmat = self.init_vals
        frame_offset = self.get_frame_offset()
        base_rmat = rtmat[frame_offset[:-1], :3, :3]
        base_quat = matrix_to_quaternion(base_rmat)
        self.base_quat.data = base_quat

    def mlp_init(self):
        """Initialize camera SE(3) transforms from external priors"""
        super().mlp_init()

        # with torch.no_grad():
        #     os.makedirs("tmp", exist_ok=True)
        #     draw_cams(rtmat.cpu().numpy()).export("tmp/cameras_gt.obj")
        #     quat, trans = self.get_vals()
        #     rtmat_pred = quaternion_translation_to_se3(quat, trans)
        #     draw_cams(rtmat_pred.cpu()).export("tmp/cameras_pred.obj")

    def forward(self, t_embed, t_embed_rot):
        """
        Args:
            t_embed: (M, self.W) Input Fourier time embeddings
        Returns:
            quat: (M, 4) Output camera rotation quaternions
            trans: (M, 3) Output camera translations
        """
        t_feat = super().forward(t_embed)
        trans = self.trans(t_feat)
        quat = self.quat(self.base_rot(t_embed_rot))
        quat = F.normalize(quat, dim=-1)
        return quat, trans

    def get_vals(self, frame_id=None):
        """Compute camera pose at the given frames.

        Args:
            frame_id: (M,) Frame id. If None, compute values at all frames
        Returns:
            quat: (M, 4) Output camera rotations
            trans: (M, 3) Output camera translations
        """
        t_embed = self.time_embedding(frame_id)
        t_embed_rot = self.time_embedding_rot(frame_id)
        quat, trans = self.forward(t_embed, t_embed_rot)
        if frame_id is None:
            inst_id = self.time_embedding.frame_to_vid
        else:
            inst_id = self.time_embedding.raw_fid_to_vid[frame_id.long()]

        # multiply with per-instance base rotation
        base_quat = self.base_quat[inst_id]
        base_quat = F.normalize(base_quat, dim=-1)
        quat = quaternion_mul(quat, base_quat)
        return quat, trans


class SO3Layer(nn.Module):
    def __init__(self, W=256, out_rots=1, out_type="mat"):
        super().__init__()
        self.fc = nn.Linear(W, 9 * out_rots)
        self.out_rots = out_rots
        self.out_type = out_type

    def forward(self, x):
        """
        output: quaternion
        """
        shape = x.shape  # ..., F
        x = self.fc(x)
        x = x.reshape(x.shape[:-1] + (self.out_rots, -1))  # ..., B, K
        x = symmetric_orthogonalization(x)
        if self.out_type == "quat":
            x = matrix_to_quaternion(x)
        elif self.out_type == "mat":
            pass
        elif self.out_type == "so3":
            x = matrix_to_axis_angle(x)
        x = x.reshape(shape[:-1] + (-1,))
        return x

    @staticmethod
    def decode_6d(rot6):
        """ssss
        decode 6D rotation to matrix

        Args:
            rot6: (..., 6)
        Returns:
            rot3x3: (..., 3, 3)
        """
        shape = rot6.shape
        rot6 = rot6.view(-1, 6)
        # decode 6D rotation to matrix
        a1, a2 = rot6[:, :3], rot6[:, 3:]
        b1 = F.normalize(a1, dim=-1)
        b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
        b2 = F.normalize(b2, dim=-1)  # -1, 3
        b3 = torch.cross(b1, b2, dim=-1)  # -1, 3
        rot3x3 = torch.stack([b1, b2, b3], dim=-2).view(-1, 3, 3)
        rot3x3 = rot3x3.view(*shape[:-1], 3, 3)
        return rot3x3


class CameraMLP_so3(TimeMLP):
    """Encode camera pose over time (rotation + translation) with an MLP

    Args:
        rtmat: (N,4,4) Object to camera transform
        frame_info (Dict): Metadata about the frames in a dataset
        D (int): Number of linear layers
        W (int): Number of hidden units in each MLP layer
        num_freq_t (int): Number of frequencies in time Fourier embedding
        skips (List(int)): List of layers to add skip connections at
        activation (Function): Activation function to use (e.g. nn.ReLU())
    """

    def __init__(
        self,
        rtmat,
        frame_info=None,
        D=5,
        W=256,
        num_freq_t=6,
        skips=[],
        activation=nn.ReLU(True),
    ):
        if frame_info is None:
            num_frames = len(rtmat)
            frame_info = {
                "frame_offset": np.asarray([0, num_frames]),
                "frame_mapping": list(range(num_frames)),
                "frame_offset_raw": np.asarray([0, num_frames]),
            }
        # xyz encoding layers
        super().__init__(
            frame_info,
            D=D,
            W=W,
            num_freq_t=num_freq_t,
            skips=skips,
            activation=activation,
        )

        self.time_embedding_rot = TimeEmbedding(
            num_freq_t,
            frame_info,
            out_channels=W,
            time_scale=1,
        )

        self.base_rot = BaseMLP(
            D=D,
            W=W,
            in_channels=W,
            out_channels=W,
            skips=skips,
            activation=activation,
            final_act=True,
        )

        # output layers
        self.trans = nn.Sequential(
            nn.Linear(W, 3),
        )
        # self.so3 = nn.Sequential(
        #     nn.Linear(W, 3),
        # )
        self.so3 = SO3Layer(out_rots=1, out_type="quat")

        # camera pose: field to camera
        base_quat = torch.zeros(frame_info["frame_offset"][-1], 4)
        base_trans = torch.zeros(frame_info["frame_offset"][-1], 3)
        self.register_buffer("base_quat", base_quat)
        self.register_buffer("base_trans", base_trans)
        self.register_buffer(
            "init_vals", torch.tensor(rtmat, dtype=torch.float32), persistent=False
        )
        self.base_init()

        # override the loss function
        def loss_fn(gt):
            quat, trans = self.get_vals()
            pred = quaternion_translation_to_se3(quat, trans)
            loss = F.mse_loss(pred, gt)
            return loss

        self.loss_fn = loss_fn

        def loss_fn_relative(y):
            """compute relative trajectory loss averaged over all videos
            y: (..., 4, 4)
            """
            quat, trans = self.get_vals()
            x = quaternion_translation_to_se3(quat, trans)

            loss = []
            frame_offset = self.time_embedding.frame_offset
            for vidid in range(len(frame_offset[:-1])):
                x_world = x[frame_offset[vidid] : frame_offset[vidid + 1]].inverse()
                y_world = y[frame_offset[vidid] : frame_offset[vidid + 1]].inverse()

                x_rel = x_world[:1].inverse() @ x_world
                y_rel = y_world[:1].inverse() @ y_world

                loss_sub = F.mse_loss(x_rel, y_rel)
                loss.append(loss_sub)
            loss = torch.stack(loss).mean()
            return loss

        self.loss_fn_relative = loss_fn_relative

    def base_init(self):
        """Initialize base camera rotations from initial camera trajectory"""
        rtmat = self.init_vals

        # initialize with corresponding frame rotation
        # self.base_quat.data = matrix_to_quaternion(rtmat[:, :3, :3])
        self.base_trans.data = rtmat[:, :3, 3]

        # initialize with per-sequence pose
        frame_offset = self.get_frame_offset()
        for i in range(len(frame_offset) - 1):
            base_rmat = rtmat[frame_offset[i], :3, :3]
            base_quat = matrix_to_quaternion(base_rmat)
            self.base_quat.data[frame_offset[i] : frame_offset[i + 1]] = base_quat

    def mlp_init(self):
        """Initialize camera SE(3) transforms from external priors"""
        super().mlp_init()

        # with torch.no_grad():
        #     os.makedirs("tmp", exist_ok=True)
        #     draw_cams(rtmat.cpu().numpy()).export("tmp/cameras_gt.obj")
        #     quat, trans = self.get_vals()
        #     rtmat_pred = quaternion_translation_to_se3(quat, trans)
        #     draw_cams(rtmat_pred.cpu()).export("tmp/cameras_pred.obj")

    def forward(self, t_embed, t_embed_rot):
        """
        Args:
            t_embed: (M, self.W) Input Fourier time embeddings
        Returns:
            quat: (M, 4) Output camera rotation quaternions
            trans: (M, 3) Output camera translations
        """
        t_feat = super().forward(t_embed)
        trans = self.trans(t_feat)
        # so3 = self.so3(self.base_rot(t_embed_rot))
        # quat = axis_angle_to_quaternion(so3)
        quat = self.so3(self.base_rot(t_embed_rot))
        return quat, trans

    def get_vals(self, frame_id=None):
        """Compute camera pose at the given frames.

        Args:
            frame_id: (M,) Frame id. If None, compute values at all frames
        Returns:
            quat: (M, 4) Output camera rotations
            trans: (M, 3) Output camera translations
        """
        t_embed = self.time_embedding(frame_id)
        t_embed_rot = self.time_embedding_rot(frame_id)
        quat, trans = self.forward(t_embed, t_embed_rot)

        # multiply with per-instance base rotation
        if frame_id is None:
            base_quat = self.base_quat
            base_trans = self.base_trans
        else:
            base_quat, base_trans = self.interpolate_base(frame_id)
        base_quat = F.normalize(base_quat, dim=-1)
        quat = quaternion_mul(quat, base_quat)
        trans = trans + base_trans

        # ensure quaternions has positive w
        neg_w = quat[..., 0] < 0
        quat[neg_w] = -quat[neg_w]
        return quat, trans

    def update_base_quat(self):
        """Update base camera rotations from current camera trajectory"""
        self.base_quat.data, self.base_trans.data = self.get_vals()
        # reinit the mlp head
        reinit_model(self.so3, std=0.01)
        reinit_model(self.trans, std=0.01)

    def interpolate_base(self, frame_id):
        idx = self.time_embedding.frame_mapping_inv[frame_id.long()]
        idx_ceil = idx + 1
        idx_ceil.clamp_(max=self.time_embedding.num_frames - 1)
        t_len = (
            self.time_embedding.frame_mapping[idx_ceil]
            - self.time_embedding.frame_mapping[idx]
        )
        t_frac = frame_id - self.time_embedding.frame_mapping[idx]
        t_frac = t_frac / (1e-6 + t_len)
        base_quat = interpolate_slerp(self.base_quat, idx, idx + 1, t_frac)
        base_trans = interpolate_linear(self.base_trans, idx, idx + 1, t_frac)
        return base_quat, base_trans

    def compute_distance_to_prior_relative(self):
        """Compute L2-distance from current SE(3) / intrinsics values to
        external priors.

        Returns:
            loss (0,): Mean squared error to priors
        """
        return self.loss_fn_relative(self.init_vals)


class ArticulationBaseMLP(TimeMLP):
    """Base class for bone articulation model (bag-of-bones or skeleton)

    Args:
        frame_info (FrameInfo): Metadata about the frames in a dataset
        num_se3 (int): Number of bones
        D (int): Number of linear layers
        W (int): Number of hidden units in each MLP layer
        num_freq_t (int): Number of frequencies in time Fourier embedding
        skips (List(int)): List of layers to add skip connections at
        activation (Function): Activation function to use (e.g. nn.ReLU())
    """

    def __init__(
        self,
        frame_info,
        num_se3,
        D=5,
        W=256,
        num_freq_t=6,
        skips=[],
        activation=nn.ReLU(True),
    ):
        # xyz encoding layers
        super().__init__(
            frame_info,
            D=D,
            W=W,
            num_freq_t=num_freq_t,
            skips=skips,
            activation=activation,
        )
        self.edges = None

        self.num_se3 = num_se3

    def forward(self, t_embed):
        """
        Args:
            t_embed: (M, self.W) Time Fourier embedding
        Returns:
            t_feat: (M, self.W) Time-dependent features
        """
        t_feat = super().forward(t_embed)
        return t_feat

    def get_vals(self, frame_id=None):
        """Compute articulation parameters at the given frames.

        Args:
            frame_id: (M,) Frame id. If None, compute values at all frames
        Returns:
            pred: ((M,B,4), (M,B,4)) Predicted bone-to-object transforms at each
                frame, written as dual quaternions
        """
        if frame_id is None:
            inst_id = self.time_embedding.frame_to_vid
        else:
            inst_id = self.time_embedding.raw_fid_to_vid[frame_id]
        t_embed = self.time_embedding(frame_id)
        pred = self.forward(t_embed, inst_id)
        return pred

    def get_mean_vals(self, inst_id=None):
        """Compute bone-to-object transforms for the rest shape

        Args:
            inst_id: (M,) Instance id. If None, compute values at the mean instance
        Returns:
            pred: ((1,B,4), (1,B,4)) Predicted bone-to-object transform for the rest
                shape, written as dual quaternions
        """
        device = self.parameters().__next__().device
        t_embed = self.time_embedding.get_mean_embedding(device)
        pred = self.forward(t_embed, inst_id)
        return pred

    def get_vals_and_mean(self, frame_id=None):
        """Compute bone-to-object transforms at the given frames, and also for
        the rest shape

        Args:
            frame_id: (M,) Frame id. If None, compute values at all frames
        """
        raise NotImplementedError


class ArticulationFlatMLP(ArticulationBaseMLP):
    """Encode a bag of bones over time using an MLP

    Args:
        frame_info (FrameInfo): Metadata about the frames in a dataset
        num_se3 (int): Number of bones
        D (int): Number of linear layers
        W (int): Number of hidden units in each MLP layer
        num_freq_t (int): Number of frequencies in time Fourier embedding
        skips (List(int)): List of layers to add skip connections at
        activation (Function): Activation function to use (e.g. nn.ReLU())
    """

    def __init__(
        self,
        frame_info,
        num_se3,
        D=5,
        W=256,
        num_freq_t=6,
        skips=[],
        activation=nn.ReLU(True),
    ):
        # xyz encoding layers
        super().__init__(
            frame_info,
            num_se3,
            D=D,
            W=W,
            num_freq_t=num_freq_t,
            skips=skips,
            activation=activation,
        )

        # output layers
        self.trans = nn.Sequential(
            nn.Linear(W, W // 2),
            activation,
            nn.Linear(W // 2, 3 * num_se3),
            ScaleLayer(0.1),
        )
        self.so3 = nn.Sequential(
            nn.Linear(W, W // 2),
            activation,
            nn.Linear(W // 2, 3 * num_se3),
        )

    def forward(self, t_embed, inst_id):
        """
        Args:
            t_embed: (M, num_freq_t) Time Fourier embedding
            inst_id: (M,) Instance id. If None, evaluate for the mean instance
        Returns:
            out: ((M,B,4), (M,B,4)): Predicted bone-to-object transforms for
                each bone, written as dual quaternions
        """
        t_feat = super().forward(t_embed)
        trans = self.trans(t_feat).reshape(*t_embed.shape[:-1], self.num_se3, 3)
        so3 = self.so3(t_feat).reshape(*t_embed.shape[:-1], self.num_se3, 3)

        # convert to rigid transformation
        qr = axis_angle_to_quaternion(so3)
        dq = quaternion_translation_to_dual_quaternion(qr, trans)
        return dq

    def get_vals_and_mean(self, frame_id=None):
        """Compute bone-to-object transforms at the given frames, and also for
        the rest shape

        Args:
            frame_id: (M,) Frame id. If None, compute values at all frames
        Returns:
            pred_t: ((M,B,4), (M,B,4)) Predicted bone-to-object transforms at
                each frame, written as dual quaternions
            pred_mean: ((M,B,4), (M,B,4)) Predicted bone-to-object transforms
                for the rest shape
        """
        pred_t = self.get_vals(frame_id)  # (M,K,4,4)
        pred_mean = self.get_mean_vals()  # (M,K,4,4)
        pred_mean = (
            pred_mean[0].expand_as(pred_t[0]).contiguous(),
            pred_mean[1].expand_as(pred_t[1]).contiguous(),
        )
        return pred_t, pred_mean


class ArticulationSkelMLP(ArticulationBaseMLP):
    """Encode a skeleton over time using an MLP

    Args:
        frame_info (FrameInfo): Metadata about the frames in a dataset
        skel_type (str): Skeleton type ("human" or "quad")
        joint_angles: (B, 3) If provided, initial joint angles
        num_se3 (int): Number of bones
        D (int): Number of linear layers
        W (int): Number of hidden units in each MLP layer
        num_freq_t (int): Number of frequencies in time Fourier embedding
        skips (List(int)): List of layers to add skip connections at
        activation (Function): Activation function to use (e.g. nn.ReLU())
    """

    def __init__(
        self,
        frame_info,
        skel_type,
        joint_angles,
        num_inst,
        D=5,
        W=256,
        num_freq_t=6,
        skips=[],
        activation=nn.ReLU(True),
    ):
        self.skel_type = skel_type
        # get skeleton
        rest_joints, edges, symm_idx = get_predefined_skeleton(skel_type)
        num_se3 = len(rest_joints)

        # xyz encoding layers
        super(ArticulationSkelMLP, self).__init__(
            frame_info,
            num_se3,
            D=D,
            W=W,
            num_freq_t=num_freq_t,
            skips=skips,
            activation=activation,
        )

        # register the skeleton
        self.edges, self.symm_idx = edges, symm_idx
        self.register_buffer("rest_joints", rest_joints)  # K, 3

        # output layers
        # self.so3 = nn.Sequential(
        #     nn.Linear(W, W // 2),
        #     activation,
        #     nn.Linear(W // 2, self.num_se3 * 3),
        # )
        self.so3 = SO3Layer(W=W, out_rots=self.num_se3, out_type="mat")

        self.logscale = nn.Parameter(torch.zeros(1))
        self.shift = nn.Parameter(torch.zeros(3))
        self.register_buffer("orient", torch.tensor([1.0, 0.0, 0.0, 0.0]))

        # instance bone length
        self.log_bone_len = CondMLP(
            num_inst,
            in_channels=0,
            D=2,
            W=64,
            out_channels=self.num_se3,
        )

        # initialize with per-frame pose
        if joint_angles is not None:
            self.register_buffer(
                "init_vals",
                torch.tensor(joint_angles, dtype=torch.float32),
                persistent=False,
            )
        else:
            self.init_vals = None

        # override the loss function
        def loss_fn(gt):
            inst_id = self.time_embedding.frame_to_vid
            device = self.parameters().__next__().device

            # loss on canonical shape
            t_embed = self.time_embedding.get_mean_embedding(device)
            so3 = self.forward(t_embed, None, return_so3=True)  # 1, num_channels
            loss_rest = so3.pow(2).mean() * 0.01

            if self.init_vals is not None:
                # # sample frameids
                # nsample = 32
                # frame_id = torch.randint(0, self.time_embedding.num_frames, (nsample,), device=device)
                # frame_id = frame_id * 0 + 40
                # inst_id = inst_id[frame_id]
                # gt = gt[frame_id]
                frame_id = None

                t_embed = self.time_embedding(frame_id=frame_id)
                pred = self.forward(t_embed, inst_id, return_so3=True)
                if gt.shape[0] != pred.shape[0]:
                    pred = pred[-gt.shape[0]:]
                    print("shape mismatch, use last K frames of predictions to match gt")
                loss_t = F.mse_loss(axis_angle_to_matrix(pred), axis_angle_to_matrix(gt))
                # loss_t = F.mse_loss(pred, gt) * 0.02
                loss = loss_t + loss_rest
            else:
                loss = loss_rest
            return loss

        self.loss_fn = loss_fn

    def mlp_init(self):
        """For skeleton fields, initialize bone lengths and rest joint angles
        from an external skeleton
        """
        super().mlp_init()

    def forward(
        self,
        t_embed,
        inst_id,
        return_so3=False,
        override_so3=None,
        override_log_bone_len=None,
        override_local_rest_joints=None,
    ):
        """
        Args:
            t_embed: (M, self.W) Time Fourier embedding
            inst_id: (M,) Instance id. If None, evaluate for the mean instance
            return_so3 (bool): If True, return computed joint angles instead
            override_so3: (M,K,3) If given, override computed joint angles from
                inputs. Used during reanimation
            override_log_bone_len: If given, override computed bone lengths
                from inputs
            override_local_rest_joints: If given, override local rest joints
                from inputs
        Returns:
            out: ((M,B,4), (M,B,4)) Predicted bone-to-object transforms for
                each bone, written as dual quaternions
        """
        # compute so3
        if override_so3 is None:
            t_feat = super(ArticulationSkelMLP, self).forward(t_embed)
            exp = self.so3(t_feat).reshape(
                *t_embed.shape[:-1], self.num_se3, 3, 3
            )  # joint angles, so3 exp
            # *t_embed.shape[:-1], self.num_se3, 3
        else:
            so3 = override_so3
            exp = so3_to_exp_map(so3)
            # exp = so3

        if return_so3:
            so3 = quaternion_to_axis_angle(matrix_to_quaternion(exp))
            return so3

        # get relative joints
        if override_local_rest_joints is None:
            local_rest_joints = self.compute_rel_rest_joints(
                inst_id=inst_id, override_log_bone_len=override_log_bone_len
            )
        else:
            local_rest_joints = override_local_rest_joints

        # run forward kinematics
        out = self.fk_se3(local_rest_joints, exp, self.edges)
        out = self.shift_joints_to_bones(out)
        out = apply_root_offset(out, self.get_shift(), self.orient)
        return out

    def get_shift(self):
        """Get the root shift. Only allow shift along the yz axis.
        Returns:
            shift: (3,) Root shift
        """
        shift = self.shift
        shift = shift * torch.tensor([0.0, 1.0, 1.0], device=shift.device)
        return shift

    def shift_joints_to_bones(self, se3):
        return shift_joints_to_bones_dq(se3, self.edges)

    def compute_rel_rest_joints(self, inst_id=None, override_log_bone_len=None):
        """Compute relative position difference from parent to child bone
        coordinate frames, without scale

        Args:
            inst_id: (M,) instance id. If None, compute for the mean instance
            override_log_bone_len: If provided, override computed bone lengths
                from inputs
        Returns:
            rel_rest_joints: Translations from parent to child joints
        """
        # get relative joints
        rel_rest_joints = self.rest_joints_to_local(self.rest_joints, self.edges)

        # match the shape
        rel_rest_joints = rel_rest_joints[None]
        if inst_id is not None:
            rel_rest_joints = rel_rest_joints.repeat(inst_id.shape[0], 1, 1)

        # update bone length
        empty_feat = torch.zeros_like(rel_rest_joints[..., 0, :0])  # (B, 0)
        if override_log_bone_len is not None:
            log_bone_len_inc = override_log_bone_len
        else:
            log_bone_len_inc = self.log_bone_len(empty_feat, inst_id)
        bone_length = (log_bone_len_inc + self.logscale).exp()
        bone_length = (bone_length + bone_length[..., self.symm_idx]) / 2
        rel_rest_joints = rel_rest_joints * bone_length[..., None]
        return rel_rest_joints

    def fk_se3(self, local_rest_joints, exp, edges):
        """Forward kinematics for a skeleton"""
        return fk_se3(local_rest_joints, exp, edges)

    def rest_joints_to_local(self, rest_joints, edges):
        """Convert rest joints to local coordinates"""
        return rest_joints_to_local(rest_joints, edges)

    def get_vals(self, frame_id=None, return_so3=False, override_so3=None):
        """Compute articulation parameters at the given frames.

        Args:
            frame_id: (M,) Frame id. If None, compute values at all frames
            return_so3 (bool): If True, return computed joint angles instead
            override_so3: (M,K,3) If given, override computed joint angles with
                inputs. Used during reanimation
        Returns:
            pred: ((M,B,4), (M,B,4)) Predicted bone-to-object transforms at each
                frame, written as duql quaternions
        """
        if frame_id is None:
            inst_id = self.time_embedding.frame_to_vid
        else:
            inst_id = self.time_embedding.raw_fid_to_vid[frame_id.long()]
        t_embed = self.time_embedding(frame_id)
        pred = self.forward(
            t_embed, inst_id, return_so3=return_so3, override_so3=override_so3
        )
        return pred

    def get_vals_and_mean(self, frame_id=None):
        """Compute bone-to-object transforms at the given frames, and also for
        the rest shape. Faster than calling forward kinematics separately

        Args:
            frame_id: (M,) Frame id. If None, compute values at all frames
        Returns:
            pred_t: ((M,B,4), (M,B,4)) Predicted bone-to-object transforms at
                each frame, written as dual quaternions
            pred_mean: ((M,B,4), (M,B,4)) Predicted bone-to-object transforms
                for the rest shape
        """
        if frame_id is None:
            inst_id = self.time_embedding.frame_to_vid
        else:
            inst_id = self.time_embedding.raw_fid_to_vid[frame_id]
        bs = inst_id.shape[0]
        # t embedding
        t_embed = self.time_embedding(frame_id)
        # mean embedding
        device = self.parameters().__next__().device
        t_embed_mean = self.time_embedding.get_mean_embedding(device)
        t_embed_mean = t_embed_mean.expand(bs, *t_embed_mean.shape[1:])
        # concat
        t_embed = torch.cat([t_embed, t_embed_mean], dim=0)

        # forward
        rel_rest_joints_c = self.compute_rel_rest_joints()  # canonical skel
        rel_rest_joints_c = rel_rest_joints_c.repeat(bs, 1, 1)
        rel_rest_joints_i = self.compute_rel_rest_joints(inst_id=inst_id)  # inst skel
        rel_rest_joints = torch.cat([rel_rest_joints_i, rel_rest_joints_c], dim=0)
        pred = self.forward(t_embed, None, override_local_rest_joints=rel_rest_joints)
        # split
        pred_t = pred[0][:bs], pred[1][:bs]
        pred_mean = pred[0][bs:], pred[1][bs:]

        # # sanity check
        # pred_tt = self.get_vals(frame_id)
        # pred_mm = self.get_mean_vals()
        # pred_mm = (
        #     pred_mm[0].expand_as(pred_tt[0]).contiguous(),
        #     pred_mm[1].expand_as(pred_tt[1]).contiguous(),
        # )

        # print((pred_t[0] - pred_tt[0]).abs().max())
        # print((pred_t[1] - pred_tt[1]).abs().max())
        # print((pred_mean[0] - pred_mm[0]).abs().max())
        # print((pred_mean[1] - pred_mm[1]).abs().max())

        return pred_t, pred_mean

    def skel_prior_loss(self):
        """Encourage the skeleton rest pose to be near the pose initialization.
        Computes L2 loss on joint axis-angles and bone lengths

        Returns:
            loss: (0,) Skeleton prior loss
        """
        # get rest joint angles increment
        device = self.parameters().__next__().device
        t_embed = self.time_embedding.get_mean_embedding(device)
        so3 = self.forward(t_embed, None, return_so3=True)  # 1, num_channels
        loss_so3 = so3.pow(2).mean()

        # get average log bone length increment
        # inst_id = torch.arange(0, self.time_embedding.num_vids).long().to(device)
        # empty_feat = torch.zeros_like(inst_id[:, None][:, :0])  # (1, 0)
        # log_bone_len_inc = self.log_bone_len(empty_feat, inst_id)
        empty_feat = torch.zeros_like(so3[..., 0, :0])  # (1, 0)
        log_bone_len_inc = self.log_bone_len(empty_feat, None)
        loss_bone = 0.2 * log_bone_len_inc.pow(2).mean()

        loss = loss_so3 + loss_bone

        # # alternative: minimize joint location difference
        # device = self.parameters().__next__().device
        # t_embed = self.time_embedding.get_mean_embedding(device)
        # bones_dq = self.forward(t_embed, None)
        # trans_pred, rot_pred = dual_quaternion_to_quaternion_translation(bones_dq)[1]

        # bones_dq = self.forward(
        #     None,
        #     None,
        #     override_so3=torch.zeros(1, self.num_se3, 3, device=device),
        #     override_log_bone_len=torch.zeros(1, self.num_se3, device=device),
        # )
        # trans_gt, rot_gt = dual_quaternion_to_quaternion_translation(bones_dq)[1]  # B,3

        # loss = (trans_gt - trans_pred).norm(2, -1).mean()
        # trimesh.Trimesh(vertices=bones_pred.detach().cpu()).export("tmp/bones_pred.obj")
        # trimesh.Trimesh(vertices=bones_gt.detach().cpu()).export("tmp/bones_gt.obj")
        return loss


class ArticulationURDFMLP(ArticulationSkelMLP):
    """Encode a skeleton over time using an MLP

    Args:
        frame_info (FrameInfo): Metadata about the frames in a dataset
        skel_type (str): Skeleton type ("human" or "quad")
        joint_angles: (B, 3) If provided, initial joint angles
        num_se3 (int): Number of bones
        D (int): Number of linear layers
        W (int): Number of hidden units in each MLP layer
        num_freq_t (int): Number of frequencies in time Fourier embedding
        skips (List(int)): List of layers to add skip connections at
        activation (Function): Activation function to use (e.g. nn.ReLU())
    """

    def __init__(
        self,
        frame_info,
        skel_type,
        joint_angles,
        num_inst,
        D=5,
        W=256,
        num_freq_t=6,
        skips=[],
        activation=nn.ReLU(True),
    ):
        super().__init__(
            frame_info,
            skel_type,
            joint_angles,
            num_inst,
            D=D,
            W=W,
            num_freq_t=num_freq_t,
            skips=skips,
            activation=activation,
        )

        (
            local_rest_coord,
            scale_factor,
            orient,
            offset,
            bone_centers,
            bone_sizes,
        ) = self.parse_urdf(skel_type)
        self.logscale.data = torch.log(scale_factor)
        self.shift.data = offset  # same scale as object field
        self.orient.data = orient
        self.register_buffer("bone_centers", bone_centers, persistent=False)
        self.register_buffer("bone_sizes", bone_sizes, persistent=False)

        # get local rest rotation matrices, pick the first coordinate in rpy of ball joints
        # by default: transform points from child to parent
        local_rest_coord = torch.tensor(local_rest_coord, dtype=torch.float32)
        self.register_buffer("local_rest_coord", local_rest_coord, persistent=False)
        self.rest_joints = None

    def parse_urdf(self, urdf_name):
        """Load the URDF file for the skeleton"""
        from urdfpy import URDF
        import os

        current_dir = os.path.dirname(os.path.abspath(__file__))
        urdf_path = f"{current_dir}/../../projects/ppr/ppr-diffphys/data/urdf_templates/{urdf_name}.urdf"
        urdf = URDF.load(urdf_path)

        local_rest_coord = np.stack([i.origin for i in urdf.joints], 0)[::3]

        if urdf_name == "human":
            offset = torch.tensor([0.0, 0.0, 0.0])
            orient = torch.tensor([0.0, -1.0, 0.0, 0.0])  # wxyz
            scale_factor = torch.tensor([0.4])
        elif urdf_name == "smpl":
            physical_scale = 0.5 # this has to be the same as init_scale
            offset = torch.tensor([0.0, -0.2 * physical_scale, 0.0])
            orient = torch.tensor([0.0, -1.0, 0.0, 0.0])  # wxyz
            scale_factor = torch.tensor([physical_scale])
        elif urdf_name == "quad":
            offset = torch.tensor([0.0, -0.02, 0.02])
            orient = torch.tensor([1.0, -0.8, 0.0, 0.0])
            scale_factor = torch.tensor([0.1])
        else:
            raise NotImplementedError
        orient = F.normalize(orient, dim=-1)

        # get center/size of each link
        bone_centers = []
        bone_sizes = []
        for link in urdf.links:
            if len(link.visuals) == 0:
                continue
            bone_bounds = link.collision_mesh.bounds
            center = (bone_bounds[1] + bone_bounds[0]) / 2
            size = (bone_bounds[1] - bone_bounds[0]) / 2
            center = torch.tensor(center, dtype=torch.float)
            size = torch.tensor(size, dtype=torch.float)
            bone_centers.append(center)
            bone_sizes.append(size)

        bone_centers = torch.stack(bone_centers, dim=0)[1:]  # skip root
        bone_sizes = torch.stack(bone_sizes, dim=0)[1:]  # skip root
        return local_rest_coord, scale_factor, orient, offset, bone_centers, bone_sizes

    def fk_se3(self, local_rest_joints, exp, edges):
        return fk_se3(
            local_rest_joints,
            exp,
            edges,
            local_rest_coord=self.local_rest_coord.clone(),
        )

    def rest_joints_to_local(self, rest_joints, edges):
        return self.local_rest_coord[:, :3, 3].clone()

    def shift_joints_to_bones(self, bone_to_obj):
        idn_quat = torch.zeros_like(bone_to_obj[0])
        idn_quat[..., 0] = 1.0
        bone_centers = self.bone_centers.expand_as(idn_quat[..., :3])
        bone_centers = bone_centers * self.logscale.exp().clone()
        link_transform = quaternion_translation_to_dual_quaternion(
            idn_quat, bone_centers
        )
        bone_to_obj = dual_quaternion_mul(bone_to_obj, link_transform)
        return bone_to_obj
