# Copyright (c) 2023 Gengshan Yang, Carnegie Mellon University.
import os
import sys
import pdb
from absl import app

sys.path.insert(0, "%s/../../" % os.path.join(os.path.dirname(__file__)))
from lab4d.train import train_ddp

sys.path.insert(0, "%s/../" % os.path.join(os.path.dirname(__file__)))
from gsplat.trainer import GSplatTrainer


def main(_):
    train_ddp(GSplatTrainer)


if __name__ == "__main__":
    app.run(main)