# move all database/processed/Features/Full-Resolution/%s/crop-256-%s.npy to database/processed/Features/Full-Resolution/%s/crop-%s.npy
import os
import glob

for path in glob.glob("database/processed/Features/Full-Resolution/*/*"):
    if "-256-" in path:
        new_path = path.replace("-256-", "-")
        os.rename(path, new_path)
        print(f"move {path} to {new_path}")
