# Copyright (c) 2023 Gengshan Yang, Carnegie Mellon University.
import os

from absl import flags

opts = flags.FLAGS


class PredictorConfig:
    # configs related to predictor
    flags.DEFINE_string("poly_1", "", "name of 1st polycam seq, e.g. Feb14at5-55тАпPM-poly")
    flags.DEFINE_string("poly_2", "", "name of 2nd polycam seq")
    # pass
