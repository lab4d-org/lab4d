# Copyright (c) 2023 Gengshan Yang, Carnegie Mellon University.
import os
import sys
import pdb
from absl import app

sys.path.insert(0, os.getcwd())
from lab4d.train import train_ddp
from projects.gsplat.trainer import GSplatTrainer


def main(_):
    train_ddp(GSplatTrainer)


if __name__ == "__main__":
    app.run(main)
