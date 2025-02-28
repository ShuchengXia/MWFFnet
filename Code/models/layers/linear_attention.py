import torch
import torch.nn as nn
from torch.nn import functional as F

from .conv_layer import ConvLayer

class LinearSelfAttention(nn.Module):
    def __init__(self, embed_dim=144, attn_dropout=0.0, bias=True):
        super(LinearSelfAttention, self).__init__()
        self.qkv_proj = ConvLayer(in_channels=embed_dim,
                                  out_channels=1 + (2 * embed_dim),
                                  kernel_size=1,
                                  use_norm=False,
                                  use_act=False,
                                  bias=bias)
        self.attn_dropout = nn.Dropout(p=attn_dropout)
        self.out_proj = ConvLayer(in_channels=embed_dim,
                                  out_channels=embed_dim,
                                  kernel_size=1,
                                  use_norm=False,
                                  use_act=False,
                                  bias=bias)
        self.embed_dim = embed_dim


    def forward_self_attn(self, x):
        # [B, C, P, N] --> [B, h + 2d, P, N]
        qkv = self.qkv_proj(x)

        # Project x into query, key and value
        # Query --> [B, 1, P, N]
        # value, key --> [B, d, P, N]
        query, key, value = torch.split(
            qkv, split_size_or_sections=[1, self.embed_dim, self.embed_dim], dim=1
        )
        # apply softmax along N dimension
        context_scores = F.softmax(query, dim=-1)
        # Uncomment below line to visualize context scores
        # self.visualize_context_scores(context_scores=context_scores)
        context_scores = self.attn_dropout(context_scores)

        # Compute context vector
        # [B, d, P, N] x [B, 1, P, N] -> [B, d, P, N]
        context_vector = key * context_scores
        # [B, d, P, N] --> [B, d, P, 1]
        context_vector = torch.sum(context_vector, dim=-1, keepdim=True)

        # combine context vector with values
        # [B, d, P, N] * [B, d, P, 1] --> [B, d, P, N]
        out = F.relu(value) * context_vector.expand_as(value)
        out = self.out_proj(out)
        return out

    def forward_cross_attn(self, x, x_prev=None):
        # x --> [B, C, P, N]
        # x_prev = [B, C, P, M]

        batch_size, in_dim, kv_patch_area, kv_num_patches = x.shape

        q_patch_area, q_num_patches = x.shape[-2:]

        assert (
            kv_patch_area == q_patch_area
        ), "The number of pixels in a patch for query and key_value should be the same"

        # compute query, key, and value
        # [B, C, P, M] --> [B, 1 + d, P, M]
        qk = F.conv2d(
            x_prev,
            weight=self.qkv_proj.block.conv.weight[: self.embed_dim + 1, ...],
            bias=self.qkv_proj.block.conv.bias[: self.embed_dim + 1, ...],
        )
        # [B, 1 + d, P, M] --> [B, 1, P, M], [B, d, P, M]
        query, key = torch.split(qk, split_size_or_sections=[1, self.embed_dim], dim=1)
        # [B, C, P, N] --> [B, d, P, N]
        value = F.conv2d(
            x,
            weight=self.qkv_proj.block.conv.weight[self.embed_dim + 1 :, ...],
            bias=self.qkv_proj.block.conv.bias[self.embed_dim + 1 :, ...],
        )

        # apply softmax along M dimension
        context_scores = F.softmax(query, dim=-1)
        context_scores = self.attn_dropout(context_scores)

        # compute context vector
        # [B, d, P, M] * [B, 1, P, M] -> [B, d, P, M]
        context_vector = key * context_scores
        # [B, d, P, M] --> [B, d, P, 1]
        context_vector = torch.sum(context_vector, dim=-1, keepdim=True)

        # combine context vector with values
        # [B, d, P, N] * [B, d, P, 1] --> [B, d, P, N]
        out = F.relu(value) * context_vector.expand_as(value)
        out = self.out_proj(out)
        return out

    def forward(self,x, x_prev=None):
        if x_prev is None:
            out = self.forward_self_attn(x)
        else:
            out = self.forward_cross_attn(x, x_prev)

        return out
