# Copyright (c) 2023 Gengshan Yang, Carnegie Mellon University.
import torch
import torch.nn as nn
import torch.nn.functional as F

from lab4d.nnutils.embedding import InstEmbedding
from functorch import vmap, combine_state_for_ensemble


class ScaleLayer(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.register_buffer("scale", torch.FloatTensor([scale]))

    def forward(self, input):
        return input * self.scale


class BaseMLP(nn.Module):
    """Adapted from https://github.com/kwea123/nerf_pl/blob/master/models/nerf.py

    Args:
        D (int): Number of linear layers for density (sigma) encoder
        W (int): Number of hidden units in each MLP layer
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        skips (List(int)): List of layers to add skip connections at
        activation (Function): Activation function to use (e.g. nn.ReLU())
        final_act (bool): If True, apply the activation function to the output
    """

    def __init__(
        self,
        D=8,
        W=256,
        in_channels=63,
        out_channels=3,
        skips=[4],
        activation=nn.ReLU(True),
        final_act=False,
    ):
        super().__init__()
        self.D = D
        self.W = W
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.skips = skips

        if in_channels == 0:
            return

        # linear layers
        for i in range(D):
            if i == 0:
                layer = nn.Linear(in_channels, W)
            elif i in skips:
                layer = nn.Linear(W + in_channels, W)
            else:
                layer = nn.Linear(W, W)
            layer = nn.Sequential(layer, activation)
            setattr(self, f"linear_{i+1}", layer)
        if final_act:
            self.linear_final = nn.Sequential(nn.Linear(W, out_channels), activation)
        else:
            self.linear_final = nn.Linear(W, out_channels)

    def forward(self, x):
        """
        Args:
            x: (..., self.in_channels)
        Returns:
            out: (..., self.out_channels)
        """
        out = x
        for i in range(self.D):
            if i in self.skips:
                out = torch.cat([x, out], -1)
            out = getattr(self, f"linear_{i+1}")(out)
        out = self.linear_final(out)
        return out


class CondMLP(BaseMLP):
    """A MLP that accepts both input `x` and condition `c`

    Args:
        num_inst (int): Number of distinct object instances. If --nosingle_inst
            is passed, this is equal to the number of videos, as we assume each
            video captures a different instance. Otherwise, we assume all videos
            capture the same instance and set this to 1.
        D (int): Number of linear layers for density (sigma) encoder
        W (int): Number of hidden units in each MLP layer
        in_channels (int): Number of channels in input `x`
        inst_channels (int): Number of channels in condition `c`
        out_channels (int): Number of output channels
        skips (List(int)): List of layers to add skip connections at
        activation (Function): Activation function to use (e.g. nn.ReLU())
        final_act (bool): If True, apply the activation function to the output
    """

    def __init__(
        self,
        num_inst,
        D=8,
        W=256,
        in_channels=63,
        inst_channels=32,
        out_channels=3,
        skips=[4],
        activation=nn.ReLU(True),
        final_act=False,
    ):
        super().__init__(
            D=D,
            W=W,
            in_channels=in_channels + inst_channels,
            out_channels=out_channels,
            skips=skips,
            activation=activation,
            final_act=final_act,
        )

        self.inst_embedding = InstEmbedding(num_inst, inst_channels)

    def forward(self, feat, inst_id):
        """
        Args:
            feat: (M, ..., self.in_channels)
            inst_id: (M,) Instance id, or None to use the average instance
        Returns:
            out: (M, ..., self.out_channels)
        """
        if inst_id is None:
            if self.inst_embedding.out_channels > 0:
                inst_code = self.inst_embedding.get_mean_embedding()
                inst_code = inst_code.expand(feat.shape[:-1] + (-1,))
                # print("inst_embedding exists but inst_id is None, using mean inst_code")
            else:
                # empty, falls back to single-instance NeRF
                inst_code = torch.zeros(feat.shape[:-1] + (0,), device=feat.device)
        else:
            inst_code = self.inst_embedding(inst_id)
            inst_code = inst_code.view(
                inst_code.shape[:1] + (1,) * (feat.ndim - 2) + (-1,)
            )
            inst_code = inst_code.expand(feat.shape[:-1] + (-1,))

        feat = torch.cat([feat, inst_code], -1)
        # if both input feature and inst_code are empty, return zeros
        if feat.shape[-1] == 0:
            return feat
        return super().forward(feat)

    @staticmethod
    def get_dim_inst(num_inst, inst_channels):
        if num_inst > 1:
            return inst_channels
        else:
            return 0


# class PosEncArch(nn.Module):
#     def __init__(self, in_channels, N_freqs) -> None:
#         super().__init__()
#         self.pos_embedding = PosEmbedding(in_channels, N_freqs)


class DictMLP(BaseMLP):
    """A MLP that accepts both input `x` and condition `c`

    Args:
        num_inst (int): Number of distinct object instances. If --nosingle_inst
            is passed, this is equal to the number of videos, as we assume each
            video captures a different instance. Otherwise, we assume all videos
            capture the same instance and set this to 1.
        D (int): Number of linear layers for density (sigma) encoder
        W (int): Number of hidden units in each MLP layer
        in_channels (int): Number of channels in input `x`
        inst_channels (int): Number of channels in condition `c`
        out_channels (int): Number of output channels
        skips (List(int)): List of layers to add skip connections at
        activation (Function): Activation function to use (e.g. nn.ReLU())
        final_act (bool): If True, apply the activation function to the output
    """

    def __init__(
        self,
        num_inst,
        D=8,
        W=256,
        in_channels=63,
        inst_channels=32,
        out_channels=3,
        skips=[4],
        activation=nn.ReLU(True),
        final_act=False,
    ):
        super().__init__(
            D=D,
            W=W,
            in_channels=in_channels + inst_channels,
            out_channels=out_channels,
            skips=skips,
            activation=activation,
            final_act=False,
        )

        self.basis = BaseMLP(
            D=D,
            W=W,
            in_channels=in_channels,
            out_channels=out_channels,
            skips=skips,
            activation=activation,
            final_act=final_act,
        )

        self.inst_embedding = InstEmbedding(num_inst, inst_channels)

    def forward(self, feat, inst_id):
        """
        Args:
            feat: (M, ..., self.in_channels)
            inst_id: (M,) Instance id, or None to use the average instance
        Returns:
            out: (M, ..., self.out_channels)
        """
        if inst_id is None:
            if self.inst_embedding.out_channels > 0:
                inst_code = self.inst_embedding.get_mean_embedding()
                inst_code = inst_code.expand(feat.shape[:-1] + (-1,))
                # print("inst_embedding exists but inst_id is None, using mean inst_code")
            else:
                # empty, falls back to single-instance NeRF
                inst_code = torch.zeros(feat.shape[:-1] + (0,), device=feat.device)
        else:
            inst_code = self.inst_embedding(inst_id)
            inst_code = inst_code.view(
                inst_code.shape[:1] + (1,) * (feat.ndim - 2) + (-1,)
            )
            inst_code = inst_code.expand(feat.shape[:-1] + (-1,))

        out = torch.cat([feat, inst_code], -1)
        # if both input feature and inst_code are empty, return zeros
        if out.shape[-1] == 0:
            return out
        coeff = super().forward(out)
        coeff = F.normalize(coeff, dim=-1)
        basis = self.basis(feat)
        out = coeff * basis
        return out

    @staticmethod
    def get_dim_inst(num_inst, inst_channels):
        if num_inst > 1:
            return inst_channels
        else:
            return 0


class MultiMLP(nn.Module):
    """Independent MLP for each instance"""

    def __init__(self, num_inst, inst_channels=32, **kwargs):
        super(MultiMLP, self).__init__()
        self.in_channels = kwargs["in_channels"]
        self.out_channels = kwargs["out_channels"]
        self.num_inst = num_inst
        # ensemble version
        self.nets = []
        for i in range(num_inst):
            self.nets.append(BaseMLP(**kwargs))
        self.nets = nn.ModuleList(self.nets)

    def forward(self, feat, inst_id):
        """
        Args:
            feat: (M, ..., self.in_channels)
            inst_id: (M,) Instance id, or None to use the average instance
        Returns:
            out: (M, ..., self.out_channels)
        """
        # rearrange the batch dimension
        shape = feat.shape[:-1]
        device = feat.device
        inst_id = inst_id.view((-1,) + (1,) * (len(shape) - 1))
        inst_id = inst_id.expand(shape)

        # sequential version: avoid duplicate computation
        out = torch.zeros(shape + (self.out_channels,), device=feat.device)
        empty_input = torch.zeros(1, 1, self.in_channels, device=feat.device)
        for it, net in enumerate(self.nets):
            id_sel = inst_id == it
            if id_sel.sum() == 0:
                out = out + self.nets[it](empty_input).mean() * 0
                continue
            out[id_sel] = net(feat[id_sel])
        return out


class MixMLP(nn.Module):
    """Mixing CondMLP and MultiMLP"""

    def __init__(self, num_inst, inst_channels=32, **kwargs):
        super(MixMLP, self).__init__()
        self.multimlp = MultiMLP(num_inst, inst_channels=inst_channels, **kwargs)
        kwargs["D"] *= 5  # 5
        kwargs["W"] *= 2  # 128
        self.condmlp = CondMLP(num_inst, inst_channels=inst_channels, **kwargs)

    def forward(self, feat, inst_id):
        out1 = self.condmlp(feat, inst_id)
        out2 = self.multimlp(feat, inst_id)
        out = out1 + out2
        return out


# class Triplane(nn.Module):
#     """Triplane"""

#     def __init__(self, num_inst, inst_channels=32, **kwargs) -> None:
#         super(Triplane, self).__init__()
#         init_scale = 0.1
#         resolution = 128
#         num_components = 24
#         self.plane = nn.Parameter(
#             init_scale * torch.randn((3 * resolution * resolution, num_components))
#         )

#     def forward(self, feat, inst_id):
#         """
#         Args:
#             feat: (M, ..., self.in_channels)
#             inst_id: (M,) Instance id, or None to use the average instance
#         Returns:
#             out: (M, ..., self.out_channels)
#         """
#         # rearrange the batch dimension
#         shape = feat.shape[:-1]
#         return out
