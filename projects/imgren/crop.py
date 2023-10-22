import sys, os

cwd = os.getcwd()
if cwd not in sys.path:
    sys.path.insert(0, cwd)

from preprocess.scripts.crop import extract_crop
from preprocess.third_party.omnivision.normal import extract_normal
from preprocess.scripts.extract_dinov2 import extract_dinov2


vidname = "car-turnaround-2"
seqname = "car-turnaround-2-0000"

# extract_normal(seqname)

res = 32
extract_crop(seqname, res, 0)
extract_crop(seqname, res, 1)

extract_dinov2(vidname, res)
