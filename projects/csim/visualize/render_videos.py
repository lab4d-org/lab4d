import subprocess
import configparser
import sys
import pdb

seqname = sys.argv[1]

# get number of videos
config = configparser.ConfigParser()
config.read(f"database/configs/{seqname}.config")
num_seq = len(config.sections()) - 1

if len(sys.argv)<3:
    logname = "compose-ft"
else:
    logname = sys.argv[2]

if len(sys.argv)<4:
    start_idx = 0
    end_idx = num_seq
else:
    start_idx = int(sys.argv[3])
    end_idx = int(sys.argv[4])

for i in range(start_idx, end_idx):
    command = f"python lab4d/export.py --flagfile=logdir/{seqname}-{logname}/opts.log --load_suffix latest --inst_id {i} --vis_thresh -20 --grid_size 128 --data_prefix full 0"
    # command = f"python ../lab4d/submit.py lab4d/export.py --flagfile=logdir/{seqname}-{logname}/opts.log --load_suffix latest --inst_id {i} --vis_thresh -20 --grid_size 128 --data_prefix full 0"
    # command = f"python ../lab4d/submit.py lab4d/render_mesh.py --mode shape --testdir logdir/{seqname}-{logname}/export_{i:04d}/"
    print(command)
    subprocess.run(command, shell=True)