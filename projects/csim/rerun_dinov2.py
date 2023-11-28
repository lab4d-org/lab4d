import sys, os

sys.path.insert(0, os.getcwd())
from preprocess.scripts.extract_dinov2 import extract_dinov2

vidname = sys.argv[1]
extract_dinov2(vidname, component_id=0, ndim=-1)
extract_dinov2(vidname, component_id=1, ndim=-1)
