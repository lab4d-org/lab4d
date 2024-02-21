# taken from guided motion diffusion
import torch
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange
import numpy as np


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    """
    Conv1d --> GroupNorm --> Mish
    """

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8, zero=False):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(
                inp_channels, out_channels, kernel_size, padding=kernel_size // 2
            ),
            # adding the height dimension for group norm
            Rearrange("batch channels horizon -> batch channels 1 horizon"),
            nn.GroupNorm(n_groups, out_channels),
            Rearrange("batch channels 1 horizon -> batch channels horizon"),
            nn.Mish(),
        )

        if zero:
            # zero init the convolution
            nn.init.zeros_(self.block[0].weight)
            nn.init.zeros_(self.block[0].bias)

    def forward(self, x):
        """
        Args:
            x: [n, c, l]
        """
        return self.block(x)


def ada_shift_scale(x, shift, scale):
    return x * (1 + scale) + shift


class Conv1dAdaGNBlock(nn.Module):
    """
    Conv1d --> GroupNorm --> Mish
    """

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv1d(
                inp_channels, out_channels, kernel_size, padding=kernel_size // 2
            ),
            # adding the height dimension for group norm
            Rearrange("batch channels horizon -> batch channels 1 horizon"),
            nn.GroupNorm(n_groups, out_channels),
            Rearrange("batch channels 1 horizon -> batch channels horizon"),
        )
        self.block2 = nn.Mish()

    def forward(self, x, c):
        """
        Args:
            x: [n, nfeat, l]
            c: [n, ncond, 1]
        """
        scale, shift = c.chunk(2, dim=1)
        x = self.block1(x)
        x = ada_shift_scale(x, shift, scale)
        x = self.block2(x)
        return x


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: einops.rearrange(t, "b (h c) d -> b h c d", h=self.heads), qkv
        )
        q = q * self.scale

        k = k.softmax(dim=-1)
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = einops.rearrange(out, "b h c d -> b (h c) d")
        return self.to_out(out)


class ResidualTemporalBlock(nn.Module):
    def __init__(
        self,
        inp_channels,
        out_channels,
        embed_dim,
        kernel_size=5,
        adagn=False,
        zero=False,
    ):
        super().__init__()
        self.adagn = adagn

        self.blocks = nn.ModuleList(
            [
                # adagn only the first conv (following guided-diffusion)
                (
                    Conv1dAdaGNBlock(inp_channels, out_channels, kernel_size)
                    if adagn
                    else Conv1dBlock(inp_channels, out_channels, kernel_size)
                ),
                Conv1dBlock(out_channels, out_channels, kernel_size, zero=zero),
            ]
        )

        self.time_mlp = nn.Sequential(
            nn.Mish(),
            # adagn = scale and shift
            nn.Linear(embed_dim, out_channels * 2 if adagn else out_channels),
            Rearrange("batch t -> batch t 1"),
        )

        if adagn:
            # zero the linear layer in the time_mlp so that the default behaviour is identity
            nn.init.zeros_(self.time_mlp[1].weight)
            nn.init.zeros_(self.time_mlp[1].bias)

        self.residual_conv = (
            nn.Conv1d(inp_channels, out_channels, 1)
            if inp_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, t):
        """
        x : [ batch_size x inp_channels x horizon ]
        t : [ batch_size x embed_dim ]
        returns:
        out : [ batch_size x out_channels x horizon ]
        """
        cond = self.time_mlp(t)
        if self.adagn:
            # using adagn
            out = self.blocks[0](x, cond)
        else:
            # using addition
            out = self.blocks[0](x) + cond
        out = self.blocks[1](out)
        return out + self.residual_conv(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer("pe", pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[: x.shape[0], :]
        return self.dropout(x)


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = PositionalEncoding(latent_dim, dropout=0)

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)


class TemporalUnet(nn.Module):
    def __init__(
        self,
        input_dim,
        cond_dim,
        dim=256,
        dim_mults=(1, 2, 4, 8),
        attention=False,
        adagn=False,
        zero=False,
    ):
        super().__init__()

        dims = [input_dim, *map(lambda m: int(dim * m), dim_mults)]
        print("dims: ", dims, "mults: ", dim_mults)
        in_out = list(zip(dims[:-1], dims[1:]))
        print(f"[ models/temporal ] Channel dimensions: {in_out}")

        time_dim = dim
        self.time_mlp = nn.Sequential(
            nn.Linear(cond_dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim),
        )

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        # print(in_out)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        ResidualTemporalBlock(
                            dim_in, dim_out, embed_dim=time_dim, adagn=adagn, zero=zero
                        ),
                        ResidualTemporalBlock(
                            dim_out, dim_out, embed_dim=time_dim, adagn=adagn, zero=zero
                        ),
                        (
                            Residual(PreNorm(dim_out, LinearAttention(dim_out)))
                            if attention
                            else nn.Identity()
                        ),
                        Downsample1d(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = ResidualTemporalBlock(
            mid_dim, mid_dim, embed_dim=time_dim, adagn=adagn, zero=zero
        )
        self.mid_attn = (
            Residual(PreNorm(mid_dim, LinearAttention(mid_dim)))
            if attention
            else nn.Identity()
        )
        self.mid_block2 = ResidualTemporalBlock(
            mid_dim, mid_dim, embed_dim=time_dim, adagn=adagn, zero=zero
        )

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            # print(dim_out, dim_in)
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        ResidualTemporalBlock(
                            dim_out * 2,
                            dim_in,
                            embed_dim=time_dim,
                            adagn=adagn,
                            zero=zero,
                        ),
                        ResidualTemporalBlock(
                            dim_in, dim_in, embed_dim=time_dim, adagn=adagn, zero=zero
                        ),
                        (
                            Residual(PreNorm(dim_in, LinearAttention(dim_in)))
                            if attention
                            else nn.Identity()
                        ),
                        Upsample1d(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        # use the last dim_in to support the case where the mult doesn't start with 1.
        self.final_conv = nn.Sequential(
            Conv1dBlock(dim_in, dim_in, kernel_size=5),
            nn.Conv1d(dim_in, input_dim, 1),
        )

        if zero:
            # zero the convolution in the final conv
            nn.init.zeros_(self.final_conv[1].weight)
            nn.init.zeros_(self.final_conv[1].bias)

    def forward(self, x, cond):
        """
        x : [ seqlen x batch x dim ]
        cons: [ batch x cond_dim]
        """
        x = einops.rearrange(x, "s b d -> b d s")
        # print('x:', x.shape)

        c = self.time_mlp(cond)
        # print('c:', c.shape)
        h = []

        for resnet, resnet2, attn, downsample in self.downs:
            x = resnet(x, c)
            x = resnet2(x, c)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, c)
        x = self.mid_attn(x)
        x = self.mid_block2(x, c)

        for resnet, resnet2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, c)
            x = resnet2(x, c)
            x = attn(x)
            x = upsample(x)

        x = self.final_conv(x)
        # print('x:', x.shape)

        x = einops.rearrange(x, "b d s -> s b d")
        return x
