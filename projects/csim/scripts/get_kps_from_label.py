import numpy as np
import json

group_path = "database/eval/matches/cat-sync/"
# group_path = "database/eval/matches/cat-vid8/"
group_data = json.load(open("%s/groups.json" % group_path))["groups"]

kp_id1 = group_data[0]["keypoints"][0][0]
kp_id2 = group_data[0]["keypoints"][1][0]
kp_list = []

path_kp1 = "%s/views/view_%d.json" % (group_path, kp_id1)
path_kp2 = "%s/views/view_%d.json" % (group_path, kp_id2)

print(path_kp1)
kp1s = []
for kp1 in json.load(open(path_kp1))["keypoints"]:
    kp1s.append(kp1["pos"])

print(path_kp2)
kp2s = []
for kp2 in json.load(open(path_kp2))["keypoints"]:
    kp2s.append(kp2["pos"])

# canonical => new vid
kps = np.concatenate([np.stack(kp1s,0), np.stack(kp2s, 0)], 1)
print(kps.shape)
np.save("database/eval/matches/kps-cat-sync.npy", kps)