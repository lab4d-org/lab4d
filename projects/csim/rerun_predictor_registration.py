import os
import configparser

seqname="cat-pikachu-2024-08"
logpath="logdir/predictor-fromft-4x-overfit-randrot-randmaskp-longl-2024-05-07--19-25-33-ft4"

config = configparser.RawConfigParser()
config.read("database/configs/%s.config" % seqname)

imglist_all = []
for vidid in range(1, len(config.sections()) - 1):
    seqname = config.get("data_%d" % vidid, "img_path").strip("/").split("/")[-1]
    # rgb path
    imgdir = "database/processed/JPEGImages/Full-Resolution/%s" % seqname
    # cmd = f'python ../lab4d/submit.py projects/predictor/inference_3dgs.py --flagfile={logpath}/opts.log --load_suffix latest --image_dir {imgdir} 0'
    cmd = f'python ../lab4d/submit.py preprocess/scripts/canonical_registration.py {seqname} 256 quad from_predictor 0'
    print(vidid)
    print(cmd)
    os.system(cmd)