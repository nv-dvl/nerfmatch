# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import torch
import torch.nn as nn


class FourierEmbedding(nn.Module):
    def __init__(self, num_freqs, logscale=True, scale=1.0):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        The scale defines how the input to the embedding is rescaled.
        Set scale to pi for a normalized scene to cover the whole range of sin and cos.
        """
        super().__init__()
        self.funcs = [torch.sin, torch.cos]
        self.scale = scale
        self.num_freqs = num_freqs
        self.logscale = logscale
        max_freq = num_freqs - 1

        if logscale:
            self.freqs = 2 ** torch.linspace(0, max_freq, num_freqs)
        else:
            self.freqs = torch.linspace(1, 2**max_freq, num_freqs)

    def get_embedding_dim(self, in_dim):
        return 2 * in_dim * self.num_freqs + in_dim

    def forward(self, x, **kwargs):
        """
        Inputs:
            x: (B, D)
        Outputs:
            out: (B, 2 * D * num_freqs + D)
        """
        out = [x]
        for freq in self.freqs:
            for func in self.funcs:
                out += [func(freq * x * self.scale)]
        return torch.cat(out, dim=-1)

    def __repr__(self):
        return f"FourierEmbedding(num_freqs={self.num_freqs}, logscale={self.logscale}, scale={self.scale})"


class PositionalEncodingMIP(nn.Module):
    def __init__(self, num_freqs, min_deg=0):
        super(PositionalEncodingMIP, self).__init__()
        self.min_deg = min_deg
        self.max_deg = num_freqs
        self.num_freqs = num_freqs
        self.scales = nn.Parameter(
            torch.tensor([2**i for i in range(min_deg, self.max_deg)]),
            requires_grad=False,
        )

    def get_embedding_dim(self, in_dim):
        return 2 * in_dim * self.num_freqs + in_dim

    def forward(self, x, y=None):
        shape = list(x.shape[:-1]) + [-1]
        x_enc = (x[..., None, :] * self.scales[:, None]).reshape(shape)
        x_enc = torch.cat((x_enc, x_enc + 0.5 * torch.pi), -1)
        if y is not None:
            # IPE
            y_enc = (y[..., None, :] * self.scales[:, None] ** 2).reshape(shape)
            y_enc = torch.cat((y_enc, y_enc), -1)
            x_ret = torch.exp(-0.5 * y_enc) * torch.sin(x_enc)
            y_ret = torch.maximum(
                torch.zeros_like(y_enc),
                0.5 * (1 - torch.exp(-2 * y_enc) * torch.cos(2 * x_enc)) - x_ret**2,
            )
            return x_ret, y_ret
        else:
            # PE
            x_ret = torch.sin(x_enc)
            x_ret = torch.cat((x_ret, x), -1)
            return x_ret
