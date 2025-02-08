# Copyright (c) 2023 Gengshan Yang, Carnegie Mellon University.
"""
python projects/ppr/export.py --flagfile=logdir/cat-85-sub-sub-bob-pika-cate-b02/opts.log --load_suffix latest --inst_id 0
"""

import os, sys
from absl import app

cwd = os.getcwd()
if cwd not in sys.path:
    sys.path.insert(0, cwd)

from lab4d.export import export, get_config
import config


def main(_):
    opts = get_config()
    export(opts)


if __name__ == "__main__":
    app.run(main)
