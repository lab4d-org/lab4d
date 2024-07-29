import sys, os

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"  # OpenMP threads
os.environ["MKL_NUM_THREADS"] = "1"  # For Intel MKL
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # For NumExpr
os.system("rm -rf ~/.cache")
os.system("ln -s /mnt/home/gengshany/.cache/ ~/.cache")
os.system("rm -rf ~/.torch")
os.system("ln -s /mnt/home/gengshany/.torch/ ~/.torch")

sys.path.insert(0, os.getcwd())
from preprocess.scripts.extract_dinov2 import extract_dinov2
from preprocess.scripts.canonical_registration import canonical_registration

vidname = sys.argv[1]
gpulist = [int(n) for n in sys.argv[2].split(",")]
print("gpulist: ")
print(gpulist)
extract_dinov2(vidname, component_id=0)
extract_dinov2(vidname, component_id=1)
# canonical_registration("%s-0000" % vidname, 256, "quad", component_id=1, mode="opt")
