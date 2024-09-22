# Copyright (c) 2023 Gengshan Yang, Carnegie Mellon University.
import os

from absl import flags

opts = flags.FLAGS


class TrainModelConfig:
    # weights of reconstruction terms
    flags.DEFINE_float("mask_wt", 0.1, "weight for silhouette loss")
    flags.DEFINE_float("rgb_wt", 0.1, "weight for color loss")
    flags.DEFINE_float("depth_wt", 0.0, "weight for depth loss")
    flags.DEFINE_float("normal_wt", 0.0, "weight for normal loss")
    flags.DEFINE_float("flow_wt", 0.5, "weight for flow loss")
    flags.DEFINE_float("vis_wt", 1e-2, "weight for visibility loss")
    flags.DEFINE_float("feature_wt", 1e-2, "weight for feature reconstruction loss")
    flags.DEFINE_float("feat_reproj_wt", 0.05, "weight for feature reprojection loss")

    # weights of regularization terms
    flags.DEFINE_float(
        "reg_visibility_wt", 1e-4, "weight for visibility regularization"
    )
    flags.DEFINE_float("reg_eikonal_wt", 0.01, "weight for eikonal regularization")
    flags.DEFINE_float("reg_density_masked_wt", 0.0, "l1 reg on density w/ dropout 0.2")
    flags.DEFINE_float("reg_eikonal_scale_max", 1, "max scaling for eikonal reg")
    flags.DEFINE_float(
        "reg_deform_cyc_wt", 0.05, "weight for deform cyc regularization"
    )
    flags.DEFINE_float("reg_l2_motion_wt", 0.0, "weight for l2 motion regularization")
    flags.DEFINE_float("reg_delta_skin_wt", 1e-3, "weight for delta skinning reg")
    flags.DEFINE_float("reg_skin_entropy_wt", 0.0, "weight for delta skinning reg")
    flags.DEFINE_float("reg_gauss_skin_wt", 0.02, "weight for gauss density loss in 3D")
    # flags.DEFINE_float("reg_gauss_skin_wt", 0.0, "weight for gauss density loss in 3D")
    flags.DEFINE_float("reg_cam_prior_wt", 0.1, "weight for camera regularization")
    flags.DEFINE_float("reg_pose_prior_wt", 0.0, "weight for pose regularization")
    flags.DEFINE_float("reg_shape_prior_wt", 0.0, "weight for shape regularization")
    flags.DEFINE_float(
        "reg_cam_prior_relative_wt", 0.0, "weight for relative camera regularization"
    )
    flags.DEFINE_float("reg_cam_smooth_wt", 0.0, "scale for camera smoothness reg")
    flags.DEFINE_float("reg_skel_prior_wt", 0.01, "weight for skeleton regularization")
    flags.DEFINE_float(
        "reg_gauss_mask_wt", 0.01, "weight for gauss mask regularization"
    )
    flags.DEFINE_float("reg_soft_deform_wt", 100.0, "weight for soft deformation reg")
    flags.DEFINE_float("reg_diffusion_prior_wt", 0.0, "weight for diffusion prior")

    # model
    flags.DEFINE_string("field_type", "fg", "{bg, fg, comp}")
    flags.DEFINE_string(
        "fg_motion", "rigid", "{rigid, dense, bob, skel-human, skel-quad}"
    )
    flags.DEFINE_bool("single_inst", True, "assume the same morphology over videos")
    flags.DEFINE_bool("use_cc", True, "apply connected components for fg")
    flags.DEFINE_float("beta_prob_final_fg", 0.2, "probability of final morphology beta")
    flags.DEFINE_float("beta_prob_init_fg", 1.0, "probability of initial morphology beta")
    flags.DEFINE_float("beta_prob_final_bg", 0.2, "probability of final morphology beta")
    flags.DEFINE_float("beta_prob_init_bg", 1.0, "probability of initial morphology beta")
    flags.DEFINE_string("scene_type", "share-1", "assume the same scene over videos")
    flags.DEFINE_string("intrinsics_type", "mlp", "{mlp, const}")
    flags.DEFINE_string("extrinsics_type", "mlp", "{mlp, const}")
    flags.DEFINE_integer("feature_channels", 16, "number of channels in features mlp")
    flags.DEFINE_integer("n_depth", 64, "number of depth points along the ray")


class TrainOptConfig:
    # io-related
    flags.DEFINE_string("seqname", "cat", "name of the sequence")
    flags.DEFINE_string("logname", "tmp", "name of the saved log")
    flags.DEFINE_string(
        "data_prefix", "crop", "prefix of the data entries, {crop, full}"
    )
    flags.DEFINE_string("delta_list", "2,4,8", "list of delta values")
    flags.DEFINE_integer("train_res", 256, "size of training images")
    flags.DEFINE_string("logroot", "logdir/", "root directory for log files")
    flags.DEFINE_string("load_suffix", "", "sufix of params, {latest, 0, 10, ...}")
    flags.DEFINE_string("feature_type", "dinov2c16", "{dinov2, cse}")
    flags.DEFINE_string("load_path", "", "path to load pretrained model")
    flags.DEFINE_string("load_path_bg", "", "path to load pretrained model")
    flags.DEFINE_integer("bg_vid", -1, "background video ids")

    # multiview related
    flags.DEFINE_boolean("use_timesync", False, "enforce same pose across all vids")
    flags.DEFINE_float("reg_timesync_cam_wt", 0.0, "regularize root-cam pose by camera")
    flags.DEFINE_boolean("align_root_pose_from_bgcam", False, "use bg cam to align root")

    # optimization-related
    flags.DEFINE_float("learning_rate", 5e-4, "learning rate")
    flags.DEFINE_integer("num_rounds", 20, "number of rounds to train")
    flags.DEFINE_integer("num_rounds_cam_init", 10, "number of rounds for camera init")
    flags.DEFINE_integer("iters_per_round", 200, "number of iterations per round")
    flags.DEFINE_integer("imgs_per_gpu", 128, "images samples per iter, per gpu")
    flags.DEFINE_integer("pixels_per_image", 16, "pixel samples per image")
    # flags.DEFINE_integer("imgs_per_gpu", 1, "size of minibatches per iter")
    # flags.DEFINE_integer("pixels_per_image", 4096, "number of pixel samples per image")
    flags.DEFINE_boolean("use_freq_anneal", True, "whether to use frequency annealing")
    flags.DEFINE_boolean(
        "reset_steps",
        True,
        "reset steps of loss scheduling, set to False if resuming training",
    )
    flags.DEFINE_boolean("mlp_init", True, "whether to initialize mlps")
    flags.DEFINE_boolean("update_geometry_aux", True, "whether to update_geometry_aux")
    flags.DEFINE_boolean("pose_correction", False, "whether to execute pose correction")
    flags.DEFINE_boolean("smpl_init", False, "whether to use smpl initialization routine")
    flags.DEFINE_boolean("load_fg_camera", True, "whether load camera for fg ckpt")
    flags.DEFINE_boolean("freeze_field_bg", False, "whether to freeze bg field")
    flags.DEFINE_boolean("freeze_field_fg", False, "whether to freeze fg field")
    flags.DEFINE_boolean("freeze_field_fgbg", False, "whether to freeze fg+bg field")
    flags.DEFINE_boolean("freeze_camera_bg", False, "whether to freeze bg camera")
    flags.DEFINE_boolean("freeze_camera_fg", False, "whether to freeze fg camera")
    flags.DEFINE_boolean("freeze_scale", True, "whether to freeze scale")
    flags.DEFINE_boolean("alter_flow", False, "alternatve between flow and all terms")
    flags.DEFINE_boolean("freeze_intrinsics", False, "whether to freeze intrinsics")
    flags.DEFINE_boolean("absorb_base", True, "whether to absorb se3 into base")
    flags.DEFINE_float("reset_beta", 0.0, "whether to reset transparency")
    flags.DEFINE_float("init_scale_fg", 0.2, "initial scale for the fg field")
    flags.DEFINE_float("init_scale_bg", 0.05, "initial scale for the bg field")
    flags.DEFINE_float("init_beta", 0.1, "initial transparency for all fields")
    flags.DEFINE_integer("num_freq_xyz", 10, "number of base frequencies for 3D points")
    flags.DEFINE_float("symm_ratio", 0.0, "points to get reflected along x at train")
    flags.DEFINE_integer("num_iter_gtmask_gauss", 4000, "iters use gt mask to learn gauss")

    # efficiency-related
    flags.DEFINE_integer("ngpu", 1, "number of gpus to use")
    flags.DEFINE_integer("num_workers", 2, "Number of workers for dataloading")
    flags.DEFINE_integer("eval_res", 64, "size used for eval visualizations")
    flags.DEFINE_integer("save_freq", 10, "params saving frequency")
    flags.DEFINE_boolean("profile", False, "profile the training loop")


def get_config():
    return opts.flag_values_dict()


def save_config():
    save_dir = os.path.join(opts.logroot, "%s-%s" % (opts.seqname, opts.logname))
    os.makedirs(save_dir, exist_ok=True)
    opts_path = os.path.join(save_dir, "opts.log")
    if os.path.exists(opts_path):
        os.remove(opts_path)
    opts.append_flags_into_file(opts_path)


def load_flags_from_file(filename):
    """Load flags from file and convert to json"""
    opts = {}
    with open(filename, "r", encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            if "=" in line:
                flag = line.strip().split("=")
                if len(flag) == 2:
                    flag_name, flag_value = flag
                else:
                    flag_name = flag
                    flag_value = ""
                flag_name = flag_name.lstrip("--")
                if "." in flag_value and flag_value.replace(".", "").isdigit():
                    flag_value = float(flag_value)
                elif flag_value.isdigit() or flag_value.replace("-", "").isdigit():
                    flag_value = int(flag_value)
            elif line.startswith("--no"):
                flag_name = line.strip().lstrip("--no")
                flag_value = False
            else:
                flag_name = line.strip().lstrip("--")
                flag_value = True
            opts[flag_name] = flag_value
    return opts
