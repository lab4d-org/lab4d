import sys, os
import numpy as np
import pdb
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.getcwd())
from lab4d.utils.vis_utils import draw_cams


def quaternion_multiply(q1, q2):
    """Multiply two quaternions."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([w, x, y, z])


class DualQuaternion:
    def __init__(self, real, dual):
        self.real = real
        self.dual = dual

    @classmethod
    def from_se3(cls, R, t):
        """Create a dual quaternion from a rotation matrix and translation vector."""
        # Convert rotation to quaternion
        quat = Rotation.from_matrix(R).as_quat()
        q0 = quat[:3]
        w = quat[3]
        real = np.array([w, *q0])

        # Dual part using translation
        dual = 0.5 * quaternion_multiply(np.array([0, *t]), real)

        return cls(real, dual)

    @classmethod
    def average(cls, dual_quats):
        """Compute the average of a list of dual quaternions."""
        mean_real = np.mean([dq.real for dq in dual_quats], axis=0)
        mean_dual = np.mean([dq.dual for dq in dual_quats], axis=0)
        mean_real /= np.linalg.norm(mean_real)
        return cls(mean_real, mean_dual)

    def to_se3(self):
        """Convert dual quaternion to SE(3) (rotation matrix and translation vector)."""
        w, x, y, z = self.real
        R = Rotation.from_quat([x, y, z, w]).as_matrix()
        t = 2 * quaternion_multiply(self.dual, [w, -x, -y, -z])[1:]
        return R, t


def align_se3_using_dq(A_seq, B_seq, valid_idx):
    """Align two SE(3) sequences using dual quaternions."""
    # Convert to dual quaternions
    A_seq_input = A_seq.copy()
    assert A_seq.shape == B_seq.shape

    # filter
    if valid_idx is not None:
        A_seq = A_seq[valid_idx]
        B_seq = B_seq[valid_idx]

    dq_A = [DualQuaternion.from_se3(A[:3, :3], A[:3, 3]) for A in A_seq]
    dq_B = [DualQuaternion.from_se3(B[:3, :3], B[:3, 3]) for B in B_seq]

    # Compute mean dual quaternions
    mean_dq_A = DualQuaternion.average(dq_A)
    mean_dq_B = DualQuaternion.average(dq_B)

    # Find transformation that aligns the means
    R_mean_A, t_mean_A = mean_dq_A.to_se3()
    R_mean_B, t_mean_B = mean_dq_B.to_se3()
    R_align = np.dot(R_mean_B, R_mean_A.T)
    t_align = t_mean_B - np.dot(R_align, t_mean_A)

    # Apply transformation to A_seq
    aligned_A_seq = []
    for A in A_seq_input:
        Ra, ta = A[:3, :3], A[:3, 3]
        Ra_aligned = np.dot(R_align, Ra)
        ta_aligned = np.dot(R_align, ta) + t_align
        T_aligned = np.eye(4)
        T_aligned[:3, :3] = Ra_aligned
        T_aligned[:3, 3] = ta_aligned
        aligned_A_seq.append(T_aligned)

    return np.array(aligned_A_seq)


def filter_bad_frames(extrinsics, errors):
    """
    find frames whose error is greater than the median error
    then assigne value to the last frame
    """
    thresh_error = np.quantile(errors, 0.50)
    valid_idx = np.where(errors <= thresh_error)[0]

    for i in range(len(extrinsics)):
        if i not in valid_idx:
            # assign the value of the nearest valid frame
            if i < valid_idx[0]:
                extrinsics[i] = extrinsics[valid_idx[0]]
            else:
                # find the nearest valid frame
                best_idx = np.argmin(np.abs(valid_idx - i))
                extrinsics[i] = extrinsics[valid_idx[best_idx]]
    return extrinsics


def transform_bg_cams(seqname):
    src_dir = "tmp/dino/"
    extrinsics_trg = np.load("%s/extrinsics-%s.npy" % (src_dir, seqname))
    errors = np.load("%s/errors-%s.npy" % (src_dir, seqname))

    trg_dir = "database/processed/Cameras/Full-Resolution/%s/" % seqname
    extrinsics_old = np.load("%s/00.npy" % trg_dir)
    # TODO fix it
    extrinsics_old[:, :3, 3] *= 0.48

    # find the best match and align the cameras accordingly
    # get the 25% quatile error
    valid_idx = np.where(errors <= np.percentile(errors, 25))[0]
    # valid_idx = np.where(errors <= np.median(errors))[0]
    # valid_idx = [np.argmin(errors)]
    inv_extrinsics_old = np.linalg.inv(extrinsics_old)
    inv_extrinsics_trg = np.linalg.inv(extrinsics_trg)
    extrinsics_new_inv = align_se3_using_dq(
        inv_extrinsics_old, inv_extrinsics_trg, valid_idx
    )
    # # TODO fix it
    # extrinsics_new_inv[:, 0, 3] += 0.5
    # extrinsics_new_inv[:, 1, 3] -= 0.5
    # extrinsics_new_inv[:, 2, 3] += 1.5
    # rect_rot = np.eye(3)
    # rect_rot[1, 1] = -1
    # rect_rot[2, 2] = -1
    # extrinsics_new_inv[:, :3, :3] = rect_rot

    extrinsics_new = np.linalg.inv(extrinsics_new_inv)
    # extrinsics_new = filter_bad_frames(extrinsics_trg, errors)

    cam_old = draw_cams(extrinsics_old)
    cam_new = draw_cams(extrinsics_new)
    cam_trg = draw_cams(extrinsics_trg)
    cam_old.export("tmp/cameras_old.obj")
    cam_new.export("tmp/cameras_new.obj")
    cam_trg.export("tmp/cameras_trg.obj")
    print("cameras vis exported to tmp/cameras_new.obj")

    np.save("%s/aligned-00.npy" % trg_dir, extrinsics_new)
    print("aligned extrinsics saved to %s/aligned-00.npy" % trg_dir)


if __name__ == "__main__":
    seqname = sys.argv[1]
    transform_bg_cams(seqname)
