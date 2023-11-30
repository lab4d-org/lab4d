import sys, os

sys.path.insert(0, os.getcwd())
from preprocess.scripts.tsdf_fusion import tsdf_fusion

seqname = "Oct5at10-49AM-poly-0000"
# seqname = "2023-11-11--11-51-53-0000"
tsdf_fusion(seqname, 0)  # , voxel_size=0.01, use_gpu=True)
