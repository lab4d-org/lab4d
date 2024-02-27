import torch
import torch.nn as nn
from torch.nn import functional as F


class NoiseScheduler(nn.Module):
    def __init__(
        self,
        num_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
    ):
        super().__init__()

        self.num_timesteps = num_timesteps
        if beta_schedule == "linear":
            betas = torch.linspace(
                beta_start, beta_end, num_timesteps, dtype=torch.float32
            )
        elif beta_schedule == "quadratic":
            betas = (
                torch.linspace(
                    beta_start**0.5, beta_end**0.5, num_timesteps, dtype=torch.float32
                )
                ** 2
            )

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # required for self.add_noise
        sqrt_alphas_cumprod = alphas_cumprod**0.5
        sqrt_one_minus_alphas_cumprod = (1 - alphas_cumprod) ** 0.5

        # required for reconstruct_x0
        sqrt_inv_alphas_cumprod = torch.sqrt(1 / alphas_cumprod)
        sqrt_inv_alphas_cumprod_minus_one = torch.sqrt(1 / alphas_cumprod - 1)

        # required for q_posterior
        posterior_mean_coef1 = (
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        posterior_mean_coef2 = (
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)
        )

        # convert to buffer
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", sqrt_alphas_cumprod)
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", sqrt_one_minus_alphas_cumprod
        )
        self.register_buffer("sqrt_inv_alphas_cumprod", sqrt_inv_alphas_cumprod)
        self.register_buffer(
            "sqrt_inv_alphas_cumprod_minus_one", sqrt_inv_alphas_cumprod_minus_one
        )
        self.register_buffer("posterior_mean_coef1", posterior_mean_coef1)
        self.register_buffer("posterior_mean_coef2", posterior_mean_coef2)

    def reconstruct_x0(self, x_t, t, noise):
        s1 = self.sqrt_inv_alphas_cumprod[t]
        s2 = self.sqrt_inv_alphas_cumprod_minus_one[t]
        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)
        return s1 * x_t - s2 * noise

    def q_posterior(self, x_0, x_t, t):
        s1 = self.posterior_mean_coef1[t]
        s2 = self.posterior_mean_coef2[t]
        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)
        mu = s1 * x_0 + s2 * x_t
        return mu

    def get_variance(self, t):
        if t == 0:
            return 0

        variance = (
            self.betas[t]
            * (1.0 - self.alphas_cumprod_prev[t])
            / (1.0 - self.alphas_cumprod[t])
        )
        variance = variance.clip(1e-20)
        return variance

    def step(self, model_output, timestep, sample, std):
        t = timestep
        pred_original_sample = self.reconstruct_x0(sample, t, model_output)
        pred_prev_sample = self.q_posterior(pred_original_sample, sample, t)

        variance = 0
        if t > 0:
            noise = torch.randn_like(model_output)
            noise = noise * std
            variance = (self.get_variance(t) ** 0.5) * noise

        pred_prev_sample = pred_prev_sample + variance

        return pred_prev_sample

    def add_noise(self, x_start, x_noise, timesteps):
        s1 = self.sqrt_alphas_cumprod[timesteps]
        s2 = self.sqrt_one_minus_alphas_cumprod[timesteps]

        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)

        return s1 * x_start + s2 * x_noise

    def sample_noise(self, clean, std):
        shape = clean.shape
        noise = torch.randn(shape, device="cuda")
        # dataset std
        noise = noise * std
        timesteps = torch.randint(0, self.num_timesteps, (shape[0],), device="cuda")
        timesteps = timesteps.long()
        noisy = self.add_noise(clean, noise, timesteps)
        t_frac = timesteps[:, None] / self.num_timesteps
        return noise, noisy, t_frac

    def __len__(self):
        return self.num_timesteps
