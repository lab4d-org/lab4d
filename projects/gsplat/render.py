# Copyright (c) 2023 Gengshan Yang, Carnegie Mellon University.
# python scripts/render.py --seqname --flagfile=logdir/cat-0t10-fg-bob-d0-long/opts.log --load_suffix latest

import os, sys
from absl import app

cwd = os.getcwd()
if cwd not in sys.path:
    sys.path.insert(0, cwd)

from lab4d.render import render, get_config, construct_batch_from_opts
from projects.gsplat.trainer import GSplatTrainer as Trainer


def main(_):
    opts = get_config()
    render(opts, construct_batch_func=construct_batch_from_opts, Trainer=Trainer)


if __name__ == "__main__":
    app.run(main)
