import math

import matplotlib.pyplot as plt
import numpy as np
import torch
from celluloid import Camera
from tqdm.auto import tqdm

import ddpm
import datasets

### forward and reverse process animation


num_timesteps = 250
noise_scheduler = ddpm.NoiseScheduler(num_timesteps=num_timesteps)

model = ddpm.MLP()
path = "exps/base/model.pth"
model.load_state_dict(torch.load(path))
model.eval()

dataset = datasets.get_dataset("dino", n=1000)
x0 = dataset.tensors[0]

forward_samples = []
forward_samples.append(x0)
for t in range(len(noise_scheduler)):
    timesteps = np.repeat(t, len(x0))
    noise = torch.randn_like(x0)
    sample = noise_scheduler.add_noise(x0, noise, timesteps)
    forward_samples.append(sample)

eval_batch_size = len(dataset)
sample = torch.randn(eval_batch_size, 2)
timesteps = list(range(num_timesteps))[::-1]
reverse_samples = []
reverse_samples.append(sample.numpy())
for i, t in enumerate(tqdm(timesteps)):
    t = torch.from_numpy(np.repeat(t, eval_batch_size)).long()
    with torch.no_grad():
        residual = model(sample, t)
    sample = noise_scheduler.step(residual, t[0], sample)
    reverse_samples.append(sample.numpy())

xmin, xmax = -3.5, 3.5
ymin, ymax = -4.0, 4.75

fig, ax = plt.subplots()
camera = Camera(fig)

# forward
for i, sample in enumerate(forward_samples):
    plt.scatter(sample[:, 0], sample[:, 1], alpha=0.5, s=15, color="blue")
    ax.text(0.0, 0.95, f"step {i: 4} / {num_timesteps}", transform=ax.transAxes)
    ax.text(0.0, 1.01, "Forward process", transform=ax.transAxes, size=15)
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.axis("off")
    camera.snap()

# reverse
for i, sample in enumerate(reverse_samples):
    plt.scatter(sample[:, 0], sample[:, 1], alpha=0.5, s=15, color="blue")
    ax.text(0.0, 0.95, f"step {i: 4} / {num_timesteps}", transform=ax.transAxes)
    ax.text(0.0, 1.01, "Reverse process", transform=ax.transAxes, size=15)
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.axis("off")
    camera.snap()

animation = camera.animate(blit=True, interval=35)
animation.save("static/animation.mp4")
