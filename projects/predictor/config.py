# Copyright (c) 2023 Gengshan Yang, Carnegie Mellon University.
import os

from absl import flags

opts = flags.FLAGS


class PredictorConfig:
    # configs related to predictor
    flags.DEFINE_string("poly_1", "", "name of 1st polycam seq, e.g. Feb14at5-55тАпPM-poly")
    flags.DEFINE_string("poly_2", "", "name of 2nd polycam seq")
    flags.DEFINE_bool("inside_out", True, "assume the video is captured with outward facing camera")
    flags.DEFINE_float("trans_wt", 1e-4, "weight of translation regression loss")
    flags.DEFINE_float("rot_wt", 2e-4, "weight of rotation regression loss")
    flags.DEFINE_float("xyz_regress_wt", 0.0, "weight for feature matching loss")
    flags.DEFINE_float("uncertainty_wt", 1.0, "weight of uncertainty regression loss")
    flags.DEFINE_string("model_type", "scene", "model scene or object {scene, object}")

    # diffgs interface
    flags.DEFINE_string("diffgs_path", "", "path to diffgs path e.g., logdir/mouse-1-diffgs-fs-fg-b32-bob-r120-mlp/opts.log")