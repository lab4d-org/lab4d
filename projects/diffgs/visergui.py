import os, sys
from absl import app

cwd = os.getcwd()
if cwd not in sys.path:
    sys.path.insert(0, cwd)

from lab4d.render import get_config
from projects.diffgs.trainer import GSplatTrainer as Trainer


def main(_):
    opts = get_config()
    # load model/data
    opts["logroot"] = sys.argv[1].split("=")[1].rsplit("/", 2)[0]
    model, _, _ = Trainer.construct_test_model(opts, return_refs=False)
    while True:
        model.gui.update()


if __name__ == "__main__":
    app.run(main)
