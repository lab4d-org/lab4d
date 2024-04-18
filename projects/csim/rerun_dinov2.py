import sys, os

sys.path.insert(0, os.getcwd())
from preprocess.scripts.extract_dinov2 import extract_dinov2
from preprocess.scripts.canonical_registration import canonical_registration

vidname = sys.argv[1]
extract_dinov2(vidname, component_id=0)
extract_dinov2(vidname, component_id=1)
# canonical_registration("%s-0000" % vidname, 256, "quad", component_id=1, mode="opt")
