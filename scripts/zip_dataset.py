# Description: Zip the dataset for easy sharing
# Usage: python scripts/zip_dataset.py <vidname>
import configparser
import os
import sys

cwd = os.getcwd()
if cwd not in sys.path:
    sys.path.insert(0, cwd)

from preprocess.libs.io import run_bash_command

vidname = sys.argv[1]

args = []
config = configparser.RawConfigParser()
config.read("database/configs/%s.config" % vidname)
for vidid in range(len(config.sections()) - 1):
    seqname = config.get("data_%d" % vidid, "img_path").strip("/").split("/")[-1]
    run_bash_command(
        f"zip -0 {vidname}.zip -r database/processed/*/Full-Resolution/{seqname}"
    )

run_bash_command(f"zip {vidname}.zip database/configs/{vidname}.config")
