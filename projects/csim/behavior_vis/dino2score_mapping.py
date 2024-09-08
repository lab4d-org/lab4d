import os, sys
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import einops
import matplotlib.cm as cm

sys.path.insert(0, os.getcwd())
from lab4d.nnutils.base import BaseMLP


# load data
data = np.load("tmp/x.npy", allow_pickle=True).item()
x_test = np.load("tmp/x_test.npy", allow_pickle=True).item()["xx"]
x = torch.tensor(data["x"], dtype=torch.float32).cuda()
y = torch.tensor(data["y"][:, None], dtype=torch.float32).cuda()

# remap 0.1 to 1
y = (y * 10).clamp(0, 1)

# get non-zero pairs
x = x[y[:, 0] > 0]
y = y[y[:, 0] > 0]


x_test = torch.tensor(x_test, dtype=torch.float32).cuda()

predictor = nn.Sequential(
    BaseMLP(
        in_channels=384, out_channels=1, D=8, W=256, skips=[1, 2, 3, 4, 5, 6, 7, 8]
    ),
    nn.Sigmoid(),
)
predictor = predictor.cuda()
predictor.train()

num_iter = 10000
batch_size = 4096
optimizer = torch.optim.AdamW(predictor.parameters())
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=2e-4, total_steps=num_iter + 1, pct_start=0.1
)

i = 0
while True:
    sample_idx = np.random.choice(len(x), batch_size)
    optimizer.zero_grad()
    y_pred = predictor(x[sample_idx])
    loss = F.binary_cross_entropy(y_pred, y[sample_idx])

    loss.backward()
    optimizer.step()
    scheduler.step()
    if i % 100 == 0:
        print("iter %d loss %f" % (i, loss.item()))

        # test the model
        with torch.no_grad():
            predictor.eval()
            num_imgs = 9
            idxs = np.linspace(0, len(x_test) - 1, num_imgs).astype(int)
            x_test = x_test[idxs]
            y_test = predictor(x_test)
            y_test = y_test.cpu().numpy()[..., 0]
            # colorize
            y_test = cm.viridis(y_test)[..., :3]
            y_test = einops.rearrange(
                y_test, "(b1 b2) h w c -> (b1 h) (b2 w) c", b1=3, b2=3
            )
            cv2.imwrite("tmp/pred_vis.png", y_test * 255)
            predictor.train()

    i += 1
    if i > num_iter:
        break
