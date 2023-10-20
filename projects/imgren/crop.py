import sys, os

cwd = os.getcwd()
if cwd not in sys.path:
    sys.path.insert(0, cwd)

from preprocess.scripts.crop import extract_crop
from preprocess.third_party.omnivision.normal import extract_normal


seqname = "car-turnaround-2-0000"

# extract_normal(seqname)

extract_crop(seqname, 64, 0)
extract_crop(seqname, 64, 1)
