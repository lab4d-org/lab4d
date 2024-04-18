import subprocess
import sys

seqname = sys.argv[1]
logname = "compose-ft"
for i in range(1, 27):
    command = f"python ../lab4d/submit.py lab4d/export.py --flagfile=logdir/{seqname}-{logname}/opts.log --load_suffix latest --inst_id {i} --vis_thresh -20 --grid_size 128 --data_prefix full 0"
    subprocess.run(command, shell=True)