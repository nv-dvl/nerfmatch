# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import torch
from einops import rearrange
from torch import nn

ACTIVIATION_FNS = {
    "relu": nn.ReLU,
    "elu": nn.ELU,
    "gelu": nn.GELU,
    "leaky": nn.LeakyReLU,
}


class MLP(nn.Module):
    def __init__(
        self, layer_dims, relu=False, bias=True, sigmoid=False, last_relu=False
    ):
        super().__init__()

        # Construct layers
        layers = []
        num_layers = len(layer_dims)
        for i in range(num_layers - 1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1], bias=bias))
            if relu and i != (num_layers - 2):
                layers.append(nn.ReLU())
        if sigmoid:
            layers.append(nn.Sigmoid())
        if last_relu:
            layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class FullAttention(nn.Module):
    """Proposed in:
    Attention is All You Need.
    """

    def __init__(self, head_dim):
        super().__init__()
        self.temperature = head_dim**0.5

    def forward(self, query, key, value):
        qk = torch.einsum("blhd, bshd -> blsh", query / self.temperature, key)
        z = torch.softmax(qk, dim=2)
        attended = torch.einsum("blsh, bshd -> blhd", z, value)
        return attended


class LocalitySelfAttention(nn.Module):
    """Proposed in:
        Vision Transformer for Small-Size Datasets
    It is designed for self-attention. If the attention is
    used for cross domain attending, it is not suitable.
    """

    def __init__(self, head_dim):
        super().__init__()
        self.scale = nn.Parameter(torch.log(torch.tensor(head_dim**-0.5)))

    def forward(self, query, key, value):
        qk = torch.einsum("blhd, bshd -> blsh", query, key) * self.scale.exp()

        # Masking diagnoal with negative infinity
        mask = torch.eye(qk.shape[2]).to(qk.device).bool().unsqueeze(-1)
        mask_value = -torch.finfo(qk.dtype).max
        dots = qk.masked_fill(mask, mask_value)

        z = torch.softmax(qk, dim=2)
        attended = torch.einsum("nlsh, nshd -> nlhd", z, value)
        return attended


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        model_dim,
        context_dim=None,
        head_num=8,
        head_dim=64,
        att_type="full",
        dropout=0.0,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.head_num = head_num
        inner_dim = head_dim * head_num
        context_dim = context_dim if context_dim else model_dim

        # Weight matrices
        self.proj_q = nn.Linear(model_dim, inner_dim, bias=False)
        self.proj_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.proj_v = nn.Linear(context_dim, inner_dim, bias=False)

        # Attention
        if att_type == "full":
            self.attend = FullAttention(head_dim)
        elif att_type == "lsa":
            self.attend = LocalitySelfAttention(head_dim)
        else:
            raise TypeError(f"Unexpected att_type={att_type}")

        # Output
        proj_out_list = [nn.Linear(inner_dim, model_dim, bias=False)]
        if dropout > 0:
            self.proj_out_list.append(nn.Dropout(dropout))
        self.proj_out = nn.Sequential(*proj_out_list)

    def forward(self, query, key, value):
        # Project qkv to target dimension
        q = self.proj_q(query)
        k = self.proj_k(key)
        v = self.proj_v(value)
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b n h d", h=self.head_num), (q, k, v)
        )

        # Compute attention
        attended = rearrange(self.attend(q, k, v), "b n h d -> b n (h d)")

        # Project to target dim
        out = self.proj_out(attended)
        return out


class FeedForwardNetwork(nn.Module):
    def __init__(
        self, in_dim, out_dim, hidden_dim=None, act_fn="relu", dropout=0.0, bias=True
    ):
        super().__init__()
        hidden_dim = in_dim if not hidden_dim else hidden_dim

        # Construct layers
        layers = [
            nn.Linear(in_dim, hidden_dim, bias=bias),
            ACTIVIATION_FNS[act_fn](),
            nn.Linear(hidden_dim, out_dim, bias=bias),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class GenericEncoderLayer(nn.Module):
    """A generic transformer encoder layer.
    Args:
        - att_type: full|linear|lsa
        - att_mode: self|cross
    """

    def __init__(
        self,
        model_dim=512,
        context_dim=None,
        head_num=8,
        head_dim=64,
        norm_type="pre",
        act_fn="gelu",
        att_type="full",
        att_mode="self",
        dropout=0.0,
    ):

        super().__init__()
        assert not (
            att_type == "lsa" and att_mode == "cross"
        ), f"LocalSelfAttention is not suitable for cross attention!"
        self.norm_type = norm_type
        self.att_mode = att_mode
        context_dim = context_dim if context_dim else model_dim

        # Multi-head attention
        self.attention = MultiHeadAttention(
            model_dim,
            context_dim=context_dim,
            head_num=head_num,
            head_dim=head_dim,
            att_type=att_type,
            dropout=dropout,
        )

        # Layer norm
        norm1_list = [nn.LayerNorm(model_dim)]
        if self.norm_type == "pre" and self.att_mode == "cross":
            norm1_list.append(nn.LayerNorm(context_dim))
        self.norm1 = nn.Sequential(*norm1_list)

        # Feed-forward network
        self.feedforward = FeedForwardNetwork(
            model_dim, model_dim, act_fn=act_fn, dropout=dropout
        )

        # Layer norm
        self.norm2 = nn.LayerNorm(model_dim)

    def forward_post_norm(self, x, context):
        query, key, value = x, context, context

        # Attention
        out = self.attention(query, key, value)
        out = x + out
        out = self.norm1(out)

        # Feed-forward
        out = self.feedforward(out)
        out = x + out
        out = self.norm2(out)
        return out

    def forward_pre_norm(self, x, context):
        # Pre layer norm
        if self.att_mode == "cross":
            norm_x, norm_c = self.norm1
        else:
            norm_x = norm_c = self.norm1
        x = norm_x(x)
        context = norm_c(context)

        # Attention
        query, key, value = x, context, context
        out = self.attention(query, key, value)
        out = x + out

        # Feed-forward
        out = self.norm2(out)
        out = self.feedforward(out)
        out = x + out
        return out

    def forward(self, x, context=None):
        # x: [B, N, D] context: [B, M, D]

        if self.att_mode == "self":
            assert context is None, f"self attention does not expect extra context"
            context = x

        if self.norm_type == "pre":
            return self.forward_pre_norm(x, context)
        return self.forward_post_norm(x, context)


class SelfAttentionBlock(nn.Module):
    def __init__(
        self,
        layer_num,
        model_dim=256,
        head_num=8,
        head_dim=64,
        norm_type="pre",
        act_fn="gelu",
        att_type="full",
        dropout=0.0,
    ):
        super().__init__()
        layers = [
            GenericEncoderLayer(
                model_dim=model_dim,
                head_num=head_num,
                head_dim=head_dim,
                norm_type=norm_type,
                act_fn=act_fn,
                att_type=att_type,
                att_mode="self",
                dropout=dropout,
            )
            for i in range(layer_num)
        ]
        self.layers = nn.Sequential(*layers)
        print(f"Init self attention: att_type={att_type}")

    def forward(self, x):
        return self.layers(x)


class SelfCrossAttentionBlock(nn.Module):
    def __init__(
        self,
        sa_layer_num=3,
        model_dim=256,
        head_dim=64,
        sa_head_num=8,
        ca_head_num=8,
        norm_type="pre",
        sa_act="relu",
        ca_act="gelu",
        sa_type="lsa",
        ca_type="full",
    ):
        super().__init__()

        self.sa = SelfAttentionBlock(
            sa_layer_num,
            model_dim=model_dim,
            head_num=sa_head_num,
            head_dim=head_dim,
            act_fn=sa_act,
            att_type=sa_type,
        )

        self.ca = GenericEncoderLayer(
            model_dim=model_dim,
            context_dim=model_dim,
            head_num=ca_head_num,
            head_dim=head_dim,
            act_fn=ca_act,
            att_mode="cross",
            att_type=ca_type,
        )

    def forward(self, x1, x2):
        # Self attention
        x1 = self.sa(x1)
        x2 = self.sa(x2)

        # Cross attention
        out1 = self.ca(x1, x2)
        out2 = self.ca(x2, x1)
        return out1, out2
