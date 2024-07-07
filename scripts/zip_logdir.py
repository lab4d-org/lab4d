# Description: Zip the logdir for easy sharing
# Usage: python scripts/zip_logdir <dir>
import os
import pdb
import sys

cwd = os.getcwd()
if cwd not in sys.path:
    sys.path.insert(0, cwd)

from preprocess.libs.io import run_bash_command

logpath = sys.argv[1]

logname = logpath.strip("/").split("/")[-1]
print(logname)

run_bash_command(f"zip log-{logname}.zip {logpath}/*")
run_bash_command(f"zip log-{logname}.zip tmp/{logname}.pkl")
run_bash_command(f"zip log-{logname}.zip database/motion/{logname}*.pkl")
