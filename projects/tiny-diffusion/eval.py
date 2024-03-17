import pdb
import sys, os
import numpy as np
import torch
import trimesh
from utils import get_lab4d_data

sys.path.insert(0, os.getcwd())
from projects.csim.voxelize import BGField

from utils import get_lab4d_data


def eval_ADE(pred, gt):
    """
    pred: (bs,nsamp, T, K, 3)
    gt: (bs, T, K, 3)
    """
    diff = (pred - gt[:, None]).norm(2, -1)  # bs, nsamp, T, K
    diff = diff.mean(2).mean(2)  # bs, nsamp, average over T and K
    mindiff = diff.min(1).values  # bs
    std = pred.std(1).mean(1).mean(1).mean(1)  # bs
    return mindiff, std


def eval_all(pred_goal_all, gt_goal_all, pred_wp_all, gt_wp_all):
    """
    pred_goal_all: (bs, nsamp, T, 3)
    gt_goal_all: (bs, T, 3)
    pred_wp_all: (bs, nsamp, T, K, 3)
    gt_wp_all: (bs, T, K, 3)
    """

    minDE_goal, std_goal = eval_ADE(pred_goal_all, gt_goal_all)
    minADE_wp, std_wp = eval_ADE(pred_wp_all, gt_wp_all)
    # print("Goal minDE_%d: %.3fm" % (nsamp, minDE_goal.mean()))
    # print("Goal std_%d: %.3fm" % (nsamp, std_goal.mean()))
    # print("WP minADE_%d: %.3fm" % (nsamp, minADE_wp.mean()))
    # print("WP std_%d: %.3fm" % (nsamp, std_wp.mean()))
    # combine into single line
    print(
        "Goal minDE/std | WP minADE/std: & %.3f & %.3f & %.3f & %.3f \\\\"
        % (
            minDE_goal.mean(),
            std_goal.mean(),
            minADE_wp.mean(),
            std_wp.mean(),
        )
    )


if __name__ == "__main__":

    # data
    (
        x0_wp_all,
        past_wp_all,
        cam_all,
        x0_to_world_all,
        x0_joints_all,
        past_joints_all,
        x0_angles_all,
        past_angles_all,
        x0_angles_to_world_all,
        # ) = get_lab4d_data("database/motion/S26-train-L64-S1.pkl")
        # ) = get_lab4d_data("database/motion/S26-train-L80-S1.pkl")
        # ) = get_lab4d_data("database/motion/S26-test-L80-S10.pkl")
    ) = get_lab4d_data("database/motion/S26-test-L64-S10.pkl")
    x0_goal_all = x0_wp_all[:, -1:]

    # VISITATION sampling
    bg_field = BGField()
    pred_goal = bg_field.voxel_grid.sample_from_voxel(64, mode="root_visitation")
    pred_goal = torch.tensor(pred_goal, dtype=torch.float32, device="cuda")
    pred_goal = pred_goal[None, :, None, None]  # 1, bs, 1, 1, 3

    pred_wp = x0_wp_all[:, None]

    eval_all(pred_goal, x0_goal_all, pred_wp, x0_wp_all)
