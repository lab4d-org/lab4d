# Copyright (c) 2023 Gengshan Yang, Carnegie Mellon University.
import os

from absl import flags

opts = flags.FLAGS


class GSplatConfig:
    # configs related to gaussian splatting
    # flags.DEFINE_float("guidance_sd_wt", 1e-4, "weight for sd loss")
    # flags.DEFINE_float("guidance_zero123_wt", 0.0, "wegiht for zero123 loss")
    flags.DEFINE_float("guidance_sd_wt", 0.0, "weight for sd loss")
    flags.DEFINE_float("guidance_zero123_wt", 1e-4, "wegiht for zero123 loss")
    flags.DEFINE_float("reg_least_deform_wt", 0.0, "weight for least deform loss")
    flags.DEFINE_float("reg_least_action_wt", 0.0, "weight for least action loss")
    flags.DEFINE_float("reg_arap_wt", 0.0, "weight for as-rigid-as-possible loss")
    flags.DEFINE_integer("sh_degree", 0, "spherical harmonics degree")
    flags.DEFINE_integer("num_pts", 5000, "number of points on the mesh")
    flags.DEFINE_float("inc_warmup_ratio", 0.0, "incremental warmup percentage")
