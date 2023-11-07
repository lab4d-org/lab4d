import sys
from preprocess.scripts.extract_dinov2 import extract_dinov2

vidname = sys.argv[1]
extract_dinov2(vidname, 256, component_id=0)
