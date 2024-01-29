# python scripts/reorder_config.py home-2023-11
import configparser
import cv2
import glob
import pdb
import sys
from natsort import natsorted

seqname = sys.argv[1]

new_config = configparser.ConfigParser()
new_config["data"] = {
    "dframe": "1",
    "init_frame": "0",
    "end_frame": "-1",
    "can_frame": "-1",
}


def update_config(config_path, new_config, start_idx=0):
    config = configparser.ConfigParser()
    config.read(config_path)
    for it, sec_name in enumerate(natsorted(config.sections())[1:]):
        new_config["data_%d" % (start_idx + it)] = config[sec_name]


if seqname[-1] == "*":
    for config_path in sorted(glob.glob("database/configs/%s.config" % (seqname))):
        if "database/configs/%s.config" % seqname[:-1] == config_path:
            continue
        update_config(config_path, new_config, start_idx=len(new_config.sections()) - 1)
    with open("database/configs/%s.config" % (seqname[:-1]), "w") as configfile:
        new_config.write(configfile)
else:
    update_config("database/configs/%s.config" % seqname, new_config)
    with open("database/configs/%s.config" % (seqname), "w") as configfile:
        new_config.write(configfile)
