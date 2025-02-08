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
    flags.DEFINE_float("reg_eikonal_scale_max", 1, "max scaling for eikonal reg")
    flags.DEFINE_float(
        "reg_deform_cyc_wt", 0.05, "weight for deform cyc regularization"
    )
    flags.DEFINE_float("reg_delta_skin_wt", 1e-3, "weight for delta skinning reg")
    flags.DEFINE_float("reg_skin_entropy_wt", 0.0, "weight for delta skinning reg")
    flags.DEFINE_float("reg_gauss_skin_wt", 0.02, "weight for gauss density loss in 3D")
    # flags.DEFINE_float("reg_gauss_skin_wt", 0.0, "weight for gauss density loss in 3D")
    flags.DEFINE_float("reg_cam_prior_wt", 0.1, "weight for camera regularization")
    flags.DEFINE_float("reg_skel_prior_wt", 0.01, "weight for skeleton regularization")
    flags.DEFINE_float(
        "reg_gauss_mask_wt", 0.01, "weight for gauss mask regularization"
    )
    flags.DEFINE_float("reg_soft_deform_wt", 100.0, "weight for soft deformation reg")

    # model
    flags.DEFINE_string("field_type", "fg", "{bg, fg, comp}")
    flags.DEFINE_string(
        "fg_motion", "rigid", "{rigid, dense, bob, skel-human, skel-quad}"
    )
    flags.DEFINE_bool("single_inst", True, "assume the same morphology over videos")
    flags.DEFINE_bool("single_scene", True, "assume the same scene over videos")


class TrainOptConfig:
    # io-related
    flags.DEFINE_string("seqname", "cat", "name of the sequence")
    flags.DEFINE_string("logname", "tmp", "name of the saved log")
    flags.DEFINE_string(
        "data_prefix", "crop", "prefix of the data entries, {crop, full}"
    )
    flags.DEFINE_integer("train_res", 256, "size of training images")
    flags.DEFINE_string("logroot", "logdir/", "root directory for log files")
    flags.DEFINE_string("load_suffix", "", "sufix of params, {latest, 0, 10, ...}")
    flags.DEFINE_string("feature_type", "dinov2", "{dinov2, cse}")
    flags.DEFINE_string("load_path", "", "path to load pretrained model")

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
    flags.DEFINE_boolean("pose_correction", False, "whether to execute pose correction")
    flags.DEFINE_boolean("alter_flow", False, "alternatve between flow and all terms")
    flags.DEFINE_boolean("freeze_intrinsics", False, "whether to freeze intrinsics")
    flags.DEFINE_boolean("absorb_base", True, "whether to absorb se3 into base")

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
