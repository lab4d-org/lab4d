import math
import pdb
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch_dct as dct
from celluloid import Camera
from tqdm.auto import tqdm

import ddpm
from arch import TemporalUnet, TimestepEmbedder


### forward and reverse process animation
num_timesteps = 50
noise_scheduler = ddpm.NoiseScheduler(num_timesteps=num_timesteps).cuda()

# forward
x0 = torch.zeros((100, 2))
x0[:, 0] = torch.linspace(-10, 10, 100)
x0 = x0.cuda()

forward_samples = []
forward_samples.append(x0)
for t in range(len(noise_scheduler)):
    timesteps = torch.tensor(np.repeat(t, len(x0)), dtype=torch.long, device="cuda")
    noise = torch.randn_like(x0, device="cuda")
    sample = noise_scheduler.add_noise(x0, noise, timesteps)
    forward_samples.append(sample)

# reverse
model = TemporalUnet(2, 128).cuda()
time_embedding = TimestepEmbedder(128).cuda()
# model = ddpm.MLP()
# path = "exps/base/model.pth"
# model.load_state_dict(torch.load(path))
model.eval()
eval_batch_size = len(x0)
sample = torch.randn(1, eval_batch_size, 2, device="cuda")
timesteps = list(range(num_timesteps))[::-1]
reverse_samples = []
reverse_samples.append(sample.cpu().numpy())
for i, t in enumerate(tqdm(timesteps)):
    t = torch.tensor(np.repeat(t, 1), dtype=torch.long, device="cuda")
    with torch.no_grad():
        cond = time_embedding(t)[0]
        sample = sample.permute(1, 0, 2)
        residual = model(sample, cond)
        residual = residual.permute(1, 0, 2)
    sample = noise_scheduler.step(residual, t[0], sample)
    reverse_samples.append(sample.cpu().numpy())

# plot
xmin, ymin = x0.min(0)[0]
xmax, ymax = x0.max(0)[0]
xymin = min(xmin, ymin)
xymax = max(xmax, ymax)

fig, ax = plt.subplots()
camera = Camera(fig)

# forward
for i, sample in enumerate(forward_samples):
    plt.scatter(sample[:, 0], sample[:, 1], alpha=0.5, s=15, color="blue")
    ax.text(0.0, 0.95, f"step {i: 4} / {num_timesteps}", transform=ax.transAxes)
    ax.text(0.0, 1.01, "Forward process", transform=ax.transAxes, size=15)
    plt.xlim(xymin, xymax)
    plt.ylim(xymin, xymax)
    plt.axis("off")
    camera.snap()

# reverse
for i, sample in enumerate(reverse_samples):
    plt.scatter(sample[:, 0], sample[:, 1], alpha=0.5, s=15, color="blue")
    ax.text(0.0, 0.95, f"step {i: 4} / {num_timesteps}", transform=ax.transAxes)
    ax.text(0.0, 1.01, "Reverse process", transform=ax.transAxes, size=15)
    plt.xlim(xymin, xymax)
    plt.ylim(xymin, xymax)
    plt.axis("off")
    camera.snap()

animation = camera.animate(blit=True, interval=35)
animation.save("static/animation.mp4")
print("Animation saved to static/animation.mp4")
