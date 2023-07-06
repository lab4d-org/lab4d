# Copyright (c) 2023 Gengshan Yang, Carnegie Mellon University.
import os

from absl import flags

opts = flags.FLAGS


class TrainModelConfig:
    # weights of reconstruction terms
    flags.DEFINE_float("mask_wt", 0.1, "weight for silhouette loss")
    flags.DEFINE_float("rgb_wt", 0.1, "weight for color loss")
    flags.DEFINE_float("depth_wt", 1e-4, "weight for depth loss")
    flags.DEFINE_float("flow_wt", 0.5, "weight for flow loss")
    flags.DEFINE_float("vis_wt", 1e-2, "weight for visibility loss")
    flags.DEFINE_float("feature_wt", 1e-2, "weight for feature reconstruction loss")
    flags.DEFINE_float("feat_reproj_wt", 5e-2, "weight for feature reprojection loss")

    # weights of regularization terms
    flags.DEFINE_float(
        "reg_visibility_wt", 1e-4, "weight for visibility regularization"
    )
    flags.DEFINE_float("reg_eikonal_wt", 1e-3, "weight for eikonal regularization")
    flags.DEFINE_float(
        "reg_deform_cyc_wt", 0.01, "weight for deform cyc regularization"
    )
    flags.DEFINE_float("reg_delta_skin_wt", 5e-3, "weight for delta skinning reg")
    flags.DEFINE_float("reg_skin_entropy_wt", 5e-4, "weight for delta skinning reg")
    flags.DEFINE_float(
        "reg_gauss_skin_wt", 1e-3, "weight for gauss skinning consistency"
    )
    flags.DEFINE_float("reg_cam_prior_wt", 0.1, "weight for camera regularization")
    flags.DEFINE_float("reg_skel_prior_wt", 0.1, "weight for skeleton regularization")
    flags.DEFINE_float(
        "reg_gauss_mask_wt", 0.01, "weight for gauss mask regularization"
    )
    flags.DEFINE_float("reg_soft_deform_wt", 10.0, "weight for soft deformation reg")

    # model
    flags.DEFINE_string("field_type", "fg", "{bg, fg, comp}")
    flags.DEFINE_string(
        "fg_motion", "rigid", "{rigid, dense, bob, skel-human, skel-quad}"
    )
    flags.DEFINE_bool("single_inst", True, "assume the same morphology over objs")


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

    # accuracy-related
    flags.DEFINE_float("learning_rate", 5e-4, "learning rate")
    flags.DEFINE_integer("num_batches", 20, "Number of iterations to train")
    flags.DEFINE_integer("minibatch_iters", 200, "number of minibatches per batch")
    flags.DEFINE_integer("minibatch_size", 128, "size of minibatches per iter, per gpu")
    flags.DEFINE_integer("num_sample_pixels", 16, "number of pixel samples per image")
    # flags.DEFINE_integer("minibatch_size", 1, "size of minibatches per iter")
    # flags.DEFINE_integer("num_sample_pixels", 4096, "number of pixel samples per image")
    flags.DEFINE_boolean(
        "freeze_bone_len", False, "do not change bone length of skeleton"
    )
    flags.DEFINE_boolean(
        "reset_steps",
        True,
        "reset steps of loss scheduling, set to False if resuming training",
    )

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
