import trimesh
import numpy as np
import glob
import os

rootdir = os.path.dirname(__file__)

for path in glob.glob("%s/_static/meshes/*.obj" % rootdir):
    print(path)
    m = trimesh.load(path, process=False)
    # cv coordinate to gl coordinate
    m.vertices = np.stack(
        [m.vertices[:, 0], -m.vertices[:, 1], -m.vertices[:, 2]], axis=1
    )
    m.export(path.replace(".obj", ".glb"))
