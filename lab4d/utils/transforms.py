# Copyright (c) 2023 Gengshan Yang, Carnegie Mellon University.
from lab4d.utils.quat_transform import (
    dual_quaternion_apply,
    dual_quaternion_inverse,
    dual_quaternion_to_quaternion_translation,
)


def get_bone_coords(xyz, bone2obj):
    """Transform points from object canonical space to bone coordinates

    Args:
        xyz: (..., 3) Points in object canonical space
        bone2obj: ((..., B, 4), (..., B, 4)) Bone-to-object SE(3)
            transforms, written as dual quaternions
    Returns:
        xyz_bone: (..., B, 3) Points in bone space
    """
    # transform xyz to bone space
    obj2bone = dual_quaternion_inverse(bone2obj)

    # reshape
    xyz = xyz[..., None, :].expand(xyz.shape[:-1] + (bone2obj[0].shape[-2], 3)).clone()
    expand_shape = xyz.shape[:-2] + (-1, -1)
    obj2bone = (
        obj2bone[0].expand(expand_shape).clone(),
        obj2bone[1].expand(expand_shape).clone(),
    )
    xyz_bone = dual_quaternion_apply(obj2bone, xyz)
    return xyz_bone


def get_xyz_bone_distance(xyz, bone2obj):
    """Compute squared distances from points to bone centers

    Argss:
        xyz: (M, 3) Points in object canonical space
        bone2obj: ((M, B, 4), (M, B, 4)) Bone-to-object SE(3) transforms, written as dual quaternions

    Returns:
        dist2: (M, B) Squared distance to each bone center
    """
    _, center = dual_quaternion_to_quaternion_translation(bone2obj)
    dist2 = (xyz[..., None, :] - center).pow(2).sum(-1)  # M, K
    return dist2
