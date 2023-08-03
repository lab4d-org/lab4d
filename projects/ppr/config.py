# Copyright (c) 2023 Gengshan Yang, Carnegie Mellon University.
import os

from absl import flags

opts = flags.FLAGS


class PPRConfig:
    # configs related to ppr
    flags.DEFINE_string("urdf_template", "", "whether to use predefined skeleton")
    flags.DEFINE_float("ratio_phys_cycle", 0.2, "number of iterations per round")
    flags.DEFINE_integer("phys_wdw_len", 24, "length of the physics opt window")
    flags.DEFINE_integer("phys_batch", 20, "number of parallel physics sim")
    flags.DEFINE_string(
        "phys_vid", "0", "whether to optimize selected videos, e.g., 0,1,2"
    )
