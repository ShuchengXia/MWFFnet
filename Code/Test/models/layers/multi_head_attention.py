

import torch
import math
from torch import nn
from ..modules.architectures import LinearLayer


class MultiHeadAttention(nn.Module):
    '''
            This layer applies a multi-head attention as described in "Attention is all you need" paper
            https://arxiv.org/abs/1706.03762
    '''

    def __init__(self, embed_dim=144, num_heads=8, attn_dropout=0., proj_drop=0., bias=True):
        """
        :param embed_dim: Embedding dimension
        :param num_heads: Number of attention heads
        :param attn_dropout: Attention dropout
        :param bias: Bias
        """
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        head_dim = embed_dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = head_dim ** -0.5
        # self.qkv = nn.Linear(in_features=embed_dim, out_features=3*embed_dim, bias=bias)
        # self.proj = nn.Linear(in_features=embed_dim, out_features=embed_dim, bias=bias)
        self.qkv = LinearLayer(in_features=embed_dim, out_features=3 * embed_dim, bias=bias)
        self.proj = LinearLayer(in_features=embed_dim, out_features=embed_dim, bias=bias)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.softmax(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)

        return x



class EffAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.reduce = nn.Linear(dim, dim // 2, bias=qkv_bias)
        self.qkv = nn.Linear(dim // 2, dim // 2 * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim // 2, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        print('scale', self.scale)
        print(dim)
        print(dim // num_heads)
        # self.proj = nn.Linear(dim, dim)
        # self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        x = self.reduce(x)
        B, N, C = x.shape
        # pdb.set_trace()
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # q = x.reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # k = x.reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # v = x.reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # qkv: 3*16*8*37*96
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        # pdb.set_trace()

        q_all = torch.split(q, math.ceil(N // 4), dim=-2)
        k_all = torch.split(k, math.ceil(N // 4), dim=-2)
        v_all = torch.split(v, math.ceil(N // 4), dim=-2)

        output = []
        for q, k, v in zip(q_all, k_all, v_all):
            attn = (q @ k.transpose(-2, -1)) * self.scale  # 16*8*37*37
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            trans_x = (attn @ v).transpose(1, 2)  # .reshape(B, N, C)

            output.append(trans_x)
        # pdb.set_trace()
        # attn = torch.cat(att, dim=-2)
        # x = (attn @ v).transpose(1, 2).reshape(B, N, C) #16*37*768
        x = torch.cat(output, dim=1)
        x = x.reshape(B, N, C)
        # pdb.set_trace()
        x = self.proj(x)
        # x = self.proj_drop(x)
        return x

