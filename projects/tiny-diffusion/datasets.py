import pickle as pkl
import numpy as np
import pdb
import os
import torch
import trimesh
import sys, os

sys.path.insert(0, os.getcwd())
from lab4d.utils.quat_transform import matrix_to_axis_angle


def to_torch(ndarray):
    if type(ndarray).__module__ == "numpy":
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor".format(type(ndarray)))
    return ndarray


def convert_to_gmd(root_se3, joint_so3, use_ego=True):
    # get root
    root_se3 = np.linalg.inv(root_se3)  # coordiante in world space

    if use_ego:
        # # re-orient se3 to the first frame
        # root_se3 = np.linalg.inv(root_se3[0]) @ root_se3
        # root_se3 = np.linalg.inv(root_se3[64]) @ root_se3
        # G(63->t) = G(w->63)^-1 * G(w->t)
        root_se3 = np.linalg.inv(root_se3[63]) @ root_se3

    # to torch
    root_se3 = to_torch(root_se3)

    # # get 4 tuple: rotation along y axis, xyz translation
    # y_rot = geometry.matrix_to_axis_angle(root_se3[:, :3, :3])[:, 1:2]
    # xyz = root_se3[:, :3, 3]
    # root_se3 = torch.cat([y_rot, xyz], axis=1)  # N,4

    # get 6 tuple: rotation along y axis, xyz translation
    rot = matrix_to_axis_angle(root_se3[:, :3, :3])
    xyz = root_se3[:, :3, 3]
    root_se3 = torch.cat([rot, xyz], axis=1)  # N,6

    # get rotations
    joint_so3 = to_torch(joint_so3).view(joint_so3.shape[0], -1)

    ret = torch.cat([root_se3, joint_so3], axis=1)  # N,4+75
    ret = ret.T.unsqueeze(1)  # 81,1,N
    ret = ret.to(torch.float32)
    return ret


class dummy_t2m:
    def __init__(self, se3, pose):
        se3 = np.asarray(se3).reshape(-1, 4, 4)
        pose = np.asarray(pose).reshape(-1, 25, 3)
        ret = convert_to_gmd(se3, pose)  # K,1,N
        self.std = ret.std(axis=-1).view(-1)
        self.mean = ret.mean(axis=-1).view(-1)
        self.mean[:] = 0
        self.std[:] = 1

    def get_std_mean(self):
        return self.std, self.mean

    def inv_transform(self, data, traject_only=None):
        std, mean = self.get_std_mean()
        if traject_only:
            std = std[:6]
            mean = mean[:6]
        return data * std + mean

    def inv_transform_th(self, data, traject_only=False, use_rand_proj=False):
        std, mean = self.get_std_mean()
        if traject_only:
            std = std[:4]
            mean = mean[:4]
        return data * std.to(data.device) + mean.to(data.device)

    def transform_th(self, data, traject_only=False, use_rand_proj=False):
        std, mean = self.get_std_mean()
        if traject_only:
            std = std[:4]
            mean = mean[:4]
        data = (data - mean.to(data.device)) / std.to(data.device)
        return data


class CustomLoader(torch.utils.data.Dataset):
    dataname = "custom"

    def __init__(self, datapath="dataset/Custom", split="train", **kargs):
        self.datapath = datapath

        super().__init__(**kargs)

        pkldatafilepath = os.path.join(datapath, "customposes.pkl")
        data = pkl.load(open(pkldatafilepath, "rb"))

        self._pose = [x for x in data["poses"]]
        self._num_frames_in_video = [p.shape[0] for p in self._pose]
        self._joints = [x for x in data["joints3D"]]
        self._se3 = [x for x in data["se3"]]
        self._actions = [x for x in data["y"]]

        total_num_actions = 12
        self.num_actions = total_num_actions

        self._train = list(range(len(self._pose)))

        keep_actions = np.arange(0, total_num_actions)

        self._action_to_label = {x: i for i, x in enumerate(keep_actions)}
        self._label_to_action = {i: x for i, x in enumerate(keep_actions)}

        mesh_path = "../vid2sim/database/polycam/Oct5at10-49AM-poly/raw_lowres.obj"
        self.mesh = trimesh.load(mesh_path)

        # self._action_classes = humanact12_coarse_action_enumerator

        self.t2m_dataset = dummy_t2m(self._se3, self._pose)

    def _load_joints3D(self, ind, frame_ix):
        return self._joints[ind][frame_ix]

    def _load_rotvec(self, ind, frame_ix):
        pose = self._pose[ind][frame_ix].reshape(-1, 25, 3)
        return pose

    def _load_se3(self, ind, frame_ix):
        return self._se3[ind][frame_ix]

    def get_label(self, ind, frame_ix):
        root_se3 = self._load_se3(ind, frame_ix)
        joint_so3 = self._load_rotvec(ind, frame_ix)

        ret = convert_to_gmd(root_se3, joint_so3)

        return ret

    def get_pose_data(self, data_index, frame_ix):
        pose_all = self._load(data_index, frame_ix)
        pose_world = self._load(data_index, frame_ix, use_ego=False)

        # # re-orient se3 to the first frame
        # root_trans = pose[3:6, 0].T  # N,3
        # root_rot = pose[:3, 0].T
        # import utils.rotation_conversions as geometry

        # root_rot = geometry.axis_angle_to_matrix(root_rot)  # N,3,3
        # root_se3 = torch.zeros((root_rot.shape[0], 4, 4), device=pose.device)
        # root_se3[:, :3, :3] = root_rot
        # root_se3[:, :3, 3] = root_trans

        # root_se3 = np.linalg.inv(root_se3[64]) @ root_se3

        # F, 1, T
        # try to cut it down to two parts
        # 0-63: condition
        # 64-223: generation
        # label = label[..., :64]
        pose = pose_all[..., 64:]
        label = pose_all[..., :64]

        # T(224),F(81)
        label = label[:, 0].T

        # voxel map
        # transform to ego space
        mesh_copy = self.mesh.copy()
        world2obj = self._se3[data_index][frame_ix][63]
        mesh_copy.apply_transform(world2obj)
        # voxelize
        voxel_grid = VoxelGrid(mesh_copy, ego_box=3)
        voxel_data = voxel_grid.data

        return pose, label, voxel_data, pose_world

    def _get_item_data_index(self, data_index):
        ret = super()._get_item_data_index(data_index)
        # rename action to condition
        ret["hint"] = ret.pop("action")
        return ret
