
from torch import nn
import math
import torch
from torch.nn import functional as F
from torch.nn import init as init
from timm.models.layers import trunc_normal_
from einops import rearrange
import numbers
# from .transformer import TransformerEncoder
from .architectures import ConvLayer, CBAMBlock, Res_CBAM, ChannelAttention, SpatialAttention
# from .DRSformer_arch import TransformerBlock

class LFEM(nn.Module):
    """
        LFEM: Low Frequency Enhancement Module
        in_channels: :math:`C_{in}` from an expected input of size :math:`(N, C_{in}, H, W)
        transformer_dim: Input dimension to the transformer unit. Default: 144
        ffn_dim: Dimension of the FFN block. Default: 288
        n_transformer_blocks: Number of transformer blocks. Default: 3
        head_dim: Head dimension in the multi-head attention. Default: 18
        attn_dropout: Dropout in multi-head attention. Default: 0.0
        dropout: Dropout rate. Default: 0.0
        ffn_dropout: Dropout between FFN layers in transformer. Default: 0.0
        atch_h: Patch height for unfolding operation. Default: 2
        patch_w: Patch width for unfolding operation. Default: 2
        transformer_norm_layer: Normalization layer in the transformer block. Default: layer_norm
        conv_ksize: The kernel size of convolution. Default: 3
        The partial network settings were inspired by the paper:
        MobileViT: Light-weight, general-purpose, and mobile-friendly vision transformer
            https://arxiv.org/abs/2110.02178

    """
    def __init__(self, in_channels=96, transformer_dim=144, ffn_dim=288,num_topk=50,
                 n_transformer_blocks=3, head_dim=18, attn_dropout=0.0, dropout=0.0,
                 ffn_dropout=0.0, patch_h=2, patch_w=2,transformer_norm_layer="layer_norm", conv_ksize=3):
        super(LFEM, self).__init__()
        #
        self.local_rep0 = ConvLayer(
            in_channels=in_channels,
            out_channels=transformer_dim,
            kernel_size=conv_ksize,
            stride=1,
            use_norm=False,
            use_act=True
        )
        #
        # self.local_rep1 = ConvLayer(
        #      in_channels=96,
        #      out_channels=transformer_dim,
        #      kernel_size=1,
        #      stride=1,
        #      use_norm=False,
        #      use_act=False
        # )

        # assert transformer_dim % head_dim == 0
        # num_heads = transformer_dim // head_dim
        #
        # ffn_dims = [ffn_dim] * n_transformer_blocks

        # global_rep = [
        #     TransformerEncoder(embed_dim=transformer_dim, ffn_latent_dim=ffn_dims[block_idx], num_heads=num_heads,num_topk=num_topk,
        #                        attn_dropout=attn_dropout, dropout=dropout, ffn_dropout=ffn_dropout)
        #     for block_idx in range(n_transformer_blocks)
        # ]
        # global_rep.append(
        #     nn.LayerNorm(normalized_shape=transformer_dim, elementwise_affine=True)
        # )
        # self.global_rep = nn.Sequential(*global_rep)
        global_rep = [
            TransformerBlock(dim=transformer_dim,
                             num_heads=8,
                             ffn_expansion_factor=2.66,
                             bias=False,
                             LayerNorm_type='WithBias')
            for i in range(n_transformer_blocks)]

        self.global_rep = nn.Sequential(*global_rep)

        self.conv_1x1_out = ConvLayer(
            in_channels=transformer_dim*2,
            out_channels=transformer_dim,
            kernel_size=1,
            stride=1,
            use_norm=False,
            use_act=True
        )

        self.conv_3x3_out = ConvLayer(
            in_channels=transformer_dim,
            out_channels=in_channels,
            kernel_size=conv_ksize,
            use_norm=False,
            use_act=True
        )

        self.n_transformer_blocks = n_transformer_blocks
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.patch_area = self.patch_w * self.patch_h
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()


    # def unfolding(self, feature_map):
    #     patch_w, patch_h = self.patch_w, self.patch_h
    #     patch_area = int(patch_w * patch_h)
    #     batch_size, in_channels, orig_h, orig_w = feature_map.shape
    #
    #     new_h = int(math.ceil(orig_h / self.patch_h) * self.patch_h)
    #     new_w = int(math.ceil(orig_w / self.patch_w) * self.patch_w)
    #
    #     interpolate = False
    #     if new_w != orig_w or new_h != orig_h:
    #         # Note: Padding can be done, but then it needs to be handled in attention function.
    #         feature_map = F.interpolate(feature_map, size=(new_h, new_w), mode="bilinear", align_corners=False)
    #         interpolate = True
    #
    #     # number of patches along width and height
    #     num_patch_w = new_w // patch_w # n_w
    #     num_patch_h = new_h // patch_h # n_h
    #     num_patches = num_patch_h * num_patch_w # N
    #
    #     # [B, C, H, W] --> [B * C * n_h, p_h, n_w, p_w]
    #     reshaped_fm = feature_map.reshape(batch_size * in_channels * num_patch_h, patch_h, num_patch_w, patch_w)
    #     # [B * C * n_h, p_h, n_w, p_w] --> [B * C * n_h, n_w, p_h, p_w]
    #     transposed_fm = reshaped_fm.transpose(1, 2)
    #     # [B * C * n_h, n_w, p_h, p_w] --> [B, C, N, P] where P = p_h * p_w and N = n_h * n_w
    #     reshaped_fm = transposed_fm.reshape(batch_size, in_channels, num_patches, patch_area)
    #     # [B, C, N, P] --> [B, P, N, C]
    #     transposed_fm = reshaped_fm.transpose(1, 3)
    #     # [B, P, N, C] --> [BP, N, C]
    #     patches = transposed_fm.reshape(batch_size * patch_area, num_patches, -1)
    #
    #     info_dict = {
    #         "orig_size": (orig_h, orig_w),
    #         "batch_size": batch_size,
    #         "interpolate": interpolate,
    #         "total_patches": num_patches,
    #         "num_patches_w": num_patch_w,
    #         "num_patches_h": num_patch_h
    #     }
    #
    #     return patches, info_dict
    #
    # def folding(self, patches, info_dict):
    #     n_dim = patches.dim()
    #     assert n_dim == 3, "Tensor should be of shape BPxNxC. Got: {}".format(patches.shape)
    #     # [BP, N, C] --> [B, P, N, C]
    #     patches = patches.contiguous().view(info_dict["batch_size"], self.patch_area, info_dict["total_patches"], -1)
    #
    #     batch_size, pixels, num_patches, channels = patches.size()
    #     num_patch_h = info_dict["num_patches_h"]
    #     num_patch_w = info_dict["num_patches_w"]
    #
    #     # [B, P, N, C] --> [B, C, N, P]
    #     patches = patches.transpose(1, 3)
    #
    #     # [B, C, N, P] --> [B*C*n_h, n_w, p_h, p_w]
    #     feature_map = patches.reshape(batch_size * channels * num_patch_h, num_patch_w, self.patch_h, self.patch_w)
    #     # [B*C*n_h, n_w, p_h, p_w] --> [B*C*n_h, p_h, n_w, p_w]
    #     feature_map = feature_map.transpose(1, 2)
    #     # [B*C*n_h, p_h, n_w, p_w] --> [B, C, H, W]
    #     feature_map = feature_map.reshape(batch_size, channels, num_patch_h * self.patch_h, num_patch_w * self.patch_w)
    #     if info_dict["interpolate"]:
    #         feature_map = F.interpolate(feature_map, size=info_dict["orig_size"], mode="bilinear", align_corners=False)
    #     return feature_map
    #
    # def forward(self, x):
    #
    #     res0 = x
    #     fm = self.local_rep0(x)
    #     fm = self.local_rep1(fm)
    #     res1 = fm
    #     # convert feature map to patches
    #     patches, info_dict = self.unfolding(fm)
    #
    #     # learn global representations
    #     patches = self.global_rep(patches)
    #
    #     # [B x Patch x Patches x C] --> [B x C x Patches x Patch]
    #     fm = self.folding(patches=patches, info_dict=info_dict)
    #
    #     # fm = self.global_rep(fm)
    #
    #     fm = self.conv_1x1_out(torch.cat((res1, fm), dim=1))
    #     # fm = self.conv_3x3_out(fm)
    #     fm = res0 + fm
    #
    #     return fm
    def forward(self, x):

        res1 = x.clone()
        fm = self.local_rep0(x)
        # fm = self.local_rep1(fm)
        # fm = self.local(fm)
        res2 = fm
        # # DRSFormer
        # fm = self.global_rep(x)
        fm = self.global_rep(fm)
        fm = self.conv_1x1_out(torch.cat((res2, fm), dim=1))
        fm = self.conv_3x3_out(fm)
        fm = res1 + fm
        return fm

##########################################################################
# Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1, groups=hidden_features * 2, bias=bias)
        self.act = nn.GELU()
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = self.act(x1) * x2
        x = self.project_out(x)
        return x
###### Dual Gated Feed-Forward Networ
# class FeedForward(nn.Module):
#     def __init__(self, dim, ffn_expansion_factor, bias):
#         super(FeedForward, self).__init__()
#
#         hidden_features = int(dim*ffn_expansion_factor)
#
#         self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)
#
#         self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)
#
#         self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)
#
#     def forward(self, x):
#         x = self.project_in(x)
#         x1, x2 = self.dwconv(x).chunk(2, dim=1)
#         x = F.gelu(x2)*x1 + F.gelu(x1)*x2
#         x = self.project_out(x)
#         return x


##  Mixed-Scale Feed-forward Network (MSFN)
# class FeedForward(nn.Module):
#     def __init__(self, dim, ffn_expansion_factor, bias):
#         super(FeedForward, self).__init__()
#
#         hidden_features = int(dim * ffn_expansion_factor)
#
#         self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
#
#         self.dwconv3x3 = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1, groups=hidden_features * 2, bias=bias)
#         self.dwconv5x5 = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=5, stride=1, padding=2, groups=hidden_features * 2, bias=bias)
#         self.relu3 = nn.ReLU()
#         self.relu5 = nn.ReLU()
#         # self.relu3 = nn.GELU()
#         # self.relu5 = nn.GELU()
#
#         self.dwconv3x3_1 = nn.Conv2d(hidden_features * 2, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features, bias=bias)
#         self.dwconv5x5_1 = nn.Conv2d(hidden_features * 2, hidden_features, kernel_size=5, stride=1, padding=2, groups=hidden_features, bias=bias)
#
#         self.relu3_1 = nn.ReLU()
#         self.relu5_1 = nn.ReLU()
#         # self.relu3_1 = nn.GELU()
#         # self.relu5_1 = nn.GELU()
#
#         self.project_out = nn.Conv2d(hidden_features * 2, dim, kernel_size=1, bias=bias)
#
#     def forward(self, x):
#         x = self.project_in(x)
#         x1_3, x2_3 = self.relu3(self.dwconv3x3(x)).chunk(2, dim=1)
#         x1_5, x2_5 = self.relu5(self.dwconv5x5(x)).chunk(2, dim=1)
#
#         x1 = torch.cat([x1_3, x1_5], dim=1)
#         x2 = torch.cat([x2_3, x2_5], dim=1)
#
#         x1 = self.relu3_1(self.dwconv3x3_1(x1))
#         x2 = self.relu5_1(self.dwconv5x5_1(x2))
#
#         x = torch.cat([x1, x2], dim=1)
#
#         x = self.project_out(x)
#
#         return x

##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Trans_Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Trans_Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature  # @矩阵乘法运算符
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

# ##  Top-K Sparse Attention (TKSA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads

        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.attn1 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn2 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn3 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn4 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)

        # self.dwconv = nn.Sequential(
        #     nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim),
        #     nn.GELU()
        # )
        #
        # self.channel_interaction = nn.Sequential(
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.Conv2d(dim, dim // 8, kernel_size=1),
        #     nn.GELU(),
        #     nn.Conv2d(dim // 8, dim, kernel_size=1),
        # )
        # self.spatial_interaction = nn.Sequential(
        #     nn.Conv2d(dim, dim // 8, kernel_size=1),
        #     nn.GELU(),
        #     nn.Conv2d(dim // 8, 1, kernel_size=1)
        # )

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        # v_dw = v
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        _, _, C, _ = q.shape

        mask1 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        mask2 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        mask3 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        mask4 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)

        attn = (q @ k.transpose(-2, -1)) * self.temperature

        index = torch.topk(attn, k=int(C/2), dim=-1, largest=True)[1]
        mask1.scatter_(-1, index, 1.)
        attn1 = torch.where(mask1 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(C*2/3), dim=-1, largest=True)[1]
        mask2.scatter_(-1, index, 1.)
        attn2 = torch.where(mask2 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(C*3/4), dim=-1, largest=True)[1]
        mask3.scatter_(-1, index, 1.)
        attn3 = torch.where(mask3 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(C*4/5), dim=-1, largest=True)[1]
        mask4.scatter_(-1, index, 1.)
        attn4 = torch.where(mask4 > 0, attn, torch.full_like(attn, float('-inf')))

        attn1 = attn1.softmax(dim=-1)
        attn2 = attn2.softmax(dim=-1)
        attn3 = attn3.softmax(dim=-1)
        attn4 = attn4.softmax(dim=-1)

        out1 = (attn1 @ v)
        out2 = (attn2 @ v)
        out3 = (attn3 @ v)
        out4 = (attn4 @ v)

        out = out1 * self.attn1 + out2 * self.attn2 + out3 * self.attn3 + out4 * self.attn4

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        # v_dw = self.dwconv(v_dw)
        # s_v_dw = self.spatial_interaction(v_dw)
        # c_out = self.channel_interaction(out)
        # out = out * torch.sigmoid(s_v_dw) + v_dw * torch.sigmoid(c_out)

        out = self.project_out(out)
        return out



# 没有TOP-K的cross Attention
class Cross_Attention_1(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Cross_Attention_1, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        dwkernel_size = 3  #默认为3
        paddings = dwkernel_size // 2
        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=dwkernel_size, stride=1, padding=paddings, groups=dim * 2, bias=bias)

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=dwkernel_size , stride=1, padding=paddings, groups=dim, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        b, c, h, w = x.shape

        kv = self.kv_dwconv(self.kv(x))
        k, v = kv.chunk(2, dim=1)
        q = self.q_dwconv(self.q(y))

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, init_values=1e-5, use_layer_scale=True):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):

        # pre-LN
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x

# class Cross_Attention(nn.Module):
#     def __init__(self, dim, num_heads, bias):
#         super(Cross_Attention, self).__init__()
#         self.num_heads = num_heads
#         self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
#
#         self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
#         self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
#
#         self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
#         self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
#
#         self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
#
#     def forward(self, x, y):
#         b, c, h, w = x.shape
#
#         kv = self.kv_dwconv(self.kv(x))
#         k, v = kv.chunk(2, dim=1)
#         q = self.q_dwconv(self.q(y))
#
#         q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
#         k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
#         v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
#
#         q = torch.nn.functional.normalize(q, dim=-1)
#         k = torch.nn.functional.normalize(k, dim=-1)
#
#         attn = (q @ k.transpose(-2, -1)) * self.temperature
#         attn = attn.softmax(dim=-1)
#
#         out = (attn @ v)
#
#         out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
#
#         out = self.project_out(out)
#         return out

# 适用TOP-K的cross Transformer
class Cross_Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Cross_Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        dwkernel_size = 3  #默认为3
        paddings = dwkernel_size // 2
        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=dwkernel_size, stride=1, padding=paddings, groups=dim * 2, bias=bias)

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=dwkernel_size , stride=1, padding=paddings, groups=dim, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.attn1 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn2 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn3 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn4 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)

    def forward(self, x, y):
        b, c, h, w = x.shape

        kv = self.kv_dwconv(self.kv(x))
        k, v = kv.chunk(2, dim=1)
        q = self.q_dwconv(self.q(y))

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        _, _, C, _ = q.shape

        mask1 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        mask2 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        mask3 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        mask4 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)

        attn = (q @ k.transpose(-2, -1)) * self.temperature

        index = torch.topk(attn, k=int(C / 2), dim=-1, largest=True)[1]
        mask1.scatter_(-1, index, 1.)
        attn1 = torch.where(mask1 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(C * 2 / 3), dim=-1, largest=True)[1]
        mask2.scatter_(-1, index, 1.)
        attn2 = torch.where(mask2 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(C * 3 / 4), dim=-1, largest=True)[1]
        mask3.scatter_(-1, index, 1.)
        attn3 = torch.where(mask3 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(C * 4 / 5), dim=-1, largest=True)[1]
        mask4.scatter_(-1, index, 1.)
        attn4 = torch.where(mask4 > 0, attn, torch.full_like(attn, float('-inf')))

        attn1 = attn1.softmax(dim=-1)
        attn2 = attn2.softmax(dim=-1)
        attn3 = attn3.softmax(dim=-1)
        attn4 = attn4.softmax(dim=-1)

        out1 = (attn1 @ v)
        out2 = (attn2 @ v)
        out3 = (attn3 @ v)
        out4 = (attn4 @ v)

        out = out1 * self.attn1 + out2 * self.attn2 + out3 * self.attn3 + out4 * self.attn4

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

class Cross_TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(Cross_TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Cross_Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x, y):
        x1 = self.norm1(x)
        y = self.norm1(y)
        x = x + self.attn(x1, y)
        x = x + self.ffn(self.norm2(x))

        return x

class Pixel_Attention(nn.Module):
    def __init__(self, kernel_size, stride):
        super(Pixel_Attention, self).__init__()
        self.max_pooling = nn.MaxPool2d(
            kernel_size=kernel_size,
            stride=stride
        )
        self.ave_pooling = nn.AvgPool2d(
            kernel_size=kernel_size,
            stride=stride
        )
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, padding=1)
        self.nonlinear = nn.Sigmoid()

    def forward(self, x):

        # (b, w, h, c)
        out = x.transpose(1, 3)
        max_out = self.max_pooling(out)
        ave_out = self.ave_pooling(out)
        max_out = max_out.transpose(1, 3)
        ave_out = ave_out.transpose(1, 3)
        out = torch.cat([max_out, ave_out], dim=1)
        out = self.conv(out)
        out = self.nonlinear(out)
        out = x * out

        return out


# class LayerNorm(nn.Module):
#     r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
#     The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
#     shape (batch_size, height, width, channels) while channels_first corresponds to inputs
#     with shape (batch_size, channels, height, width).
#     """
#
#     def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
#         super().__init__()
#         self.weight = nn.Parameter(torch.ones(normalized_shape))
#         self.bias = nn.Parameter(torch.zeros(normalized_shape))
#         self.eps = eps
#         self.data_format = data_format
#         if self.data_format not in ["channels_last", "channels_first"]:
#             raise NotImplementedError
#         self.normalized_shape = (normalized_shape,)
#
#     def forward(self, x):
#         if self.data_format == "channels_last":
#             return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
#         elif self.data_format == "channels_first":
#             u = x.mean(1, keepdim=True)
#             s = (x - u).pow(2).mean(1, keepdim=True)
#             x = (x - u) / torch.sqrt(s + self.eps)
#             x = self.weight[:, None, None] * x + self.bias[:, None, None]
#             return x

def get_dwconv(dim, kernel, bias):
    return nn.Conv2d(dim, dim, kernel_size=kernel, padding=(kernel - 1) // 2, bias=bias, groups=dim)


class GlobalLocalFilter(nn.Module):
    def __init__(self, dim, h=14, w=8):
        super().__init__()
        self.dw = nn.Conv2d(dim // 2, dim // 2, kernel_size=3, padding=1, bias=False, groups=dim // 2)
        self.complex_weight = nn.Parameter(torch.randn(dim // 2, h, w, 2, dtype=torch.float32) * 0.02)
        trunc_normal_(self.complex_weight, std=.02)
        self.pre_norm = LayerNorm(dim, eps=1e-6, data_format='channels_first')
        self.post_norm = LayerNorm(dim, eps=1e-6, data_format='channels_first')

    def forward(self, x):
        x = self.pre_norm(x)
        x1, x2 = torch.chunk(x, 2, dim=1)
        x1 = self.dw(x1)

        x2 = x2.to(torch.float32)
        B, C, a, b = x2.shape
        x2 = torch.fft.rfft2(x2, dim=(2, 3), norm='ortho')

        weight = self.complex_weight
        if not weight.shape[1:3] == x2.shape[2:4]:
            weight = F.interpolate(weight.permute(3, 0, 1, 2), size=x2.shape[2:4], mode='bilinear', align_corners=True).permute(1, 2, 3, 0)

        weight = torch.view_as_complex(weight.contiguous())

        x2 = x2 * weight
        x2 = torch.fft.irfft2(x2, s=(a, b), dim=(2, 3), norm='ortho')

        x = torch.cat([x1.unsqueeze(2), x2.unsqueeze(2)], dim=2).reshape(B, 2 * C, a, b)
        x = self.post_norm(x)
        return x


class gnconv(nn.Module):
    def __init__(self, dim, order=5, gflayer=None, h=14, w=8, s=1.0):
        super().__init__()
        self.order = order
        self.dims = [dim // 2 ** i for i in range(order)]
        self.dims.reverse()
        self.proj_in = nn.Conv2d(dim, 2 * dim, 1)

        if gflayer is None:
            self.dwconv = get_dwconv(sum(self.dims), 7, True)
        else:
            self.dwconv = gflayer(sum(self.dims), h=h, w=w)

        self.proj_out = nn.Conv2d(dim, dim, 1)

        self.pws = nn.ModuleList(
            [nn.Conv2d(self.dims[i], self.dims[i + 1], 1) for i in range(order - 1)]
        )

        self.scale = s
        self.softmax = nn.Softmax(dim=-1)
        print('[gnconv]', order, 'order with dims=', self.dims, 'scale=%.4f' % self.scale)

    def forward(self, x, mask=None, dummy=False):
        B, C, H, W = x.shape

        fused_x = self.proj_in(x)
        pwa, abc = torch.split(fused_x, (self.dims[0], sum(self.dims)), dim=1)

        dw_abc = self.dwconv(abc) * self.scale

        dw_list = torch.split(dw_abc, self.dims, dim=1)
        x = pwa * dw_list[0]

        for i in range(self.order - 1):
            if i < self.order - 2:
                x = self.pws[i](x) * dw_list[i + 1]
                x = self.softmax(x)
            else:
                x = self.pws[i](x) * dw_list[i + 1]

        x = self.proj_out(x)

        return x


class Block(nn.Module):
    r""" HorNet block
    """

    def __init__(self, dim, layer_scale_init_value=1e-6, gnconv=gnconv):
        super().__init__()

        self.norm1 = LayerNorm(dim, eps=1e-6, data_format='channels_first')
        self.gnconv = gnconv(dim)  # depthwise conv
        self.norm2 = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

        self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones(dim),
                                   requires_grad=True) if layer_scale_init_value > 0 else None

        self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                   requires_grad=True) if layer_scale_init_value > 0 else None


    def forward(self, x):
        B, C, H, W = x.shape
        if self.gamma1 is not None:
            gamma1 = self.gamma1.view(C, 1, 1)
        else:
            gamma1 = 1
        x = x + gamma1 * self.gnconv(self.norm1(x))

        input = x
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm2(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma2 is not None:
            x = self.gamma2 * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + x
        return x

class Dual_TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(Dual_TransformerBlock, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.attn1 = Attention(dim, num_heads, bias)

        self.norm3 = LayerNorm(dim, LayerNorm_type)
        self.attn2 = Cross_Attention(dim, num_heads, bias)
        self.norm4 = LayerNorm(dim, LayerNorm_type)
        self.ffn1 = FeedForward(dim, ffn_expansion_factor, bias)

        self.attn3 = Cross_Attention(dim, num_heads, bias)

        self.norm5 = LayerNorm(dim, LayerNorm_type)
        self.ffn2 = FeedForward(dim, ffn_expansion_factor, bias)


    def forward(self, x):
        x1 = self.norm1(x[0])

        x2 = self.attn1(self.norm2(x[1])) + x[1]
        x2 = self.attn2(x1, self.norm3(x2)) + x2
        x2 = self.ffn1(self.norm4(x2)) + x2

        x1 = self.attn3(x2, x1) + x[0]
        x1 = self.ffn2(self.norm5(x1)) + x1


        return x1

