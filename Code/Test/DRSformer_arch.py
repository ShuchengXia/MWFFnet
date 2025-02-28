# 模型的主结构，在之前懒得改名称了，后续自己更改

import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers
import math

from einops import rearrange
from models.modules.architectures import ChannelAttention, InvertedResidualBlock, SFT, \
    Res_block, ConvLayer, DWT_2D, IDWT_2D, RDB
from models.modules.mobilevit_block import Cross_Attention, Attention, FeedForward, TransformerBlock


class Image_Downconv(nn.Module):
    def __init__(self, in_channels):
        super(Image_Downconv, self).__init__()
        self.conv = nn.Sequential(
            ConvLayer(in_channels=3, out_channels=in_channels // 4, kernel_size=3, stride=1, use_act=False),
            ConvLayer(in_channels=in_channels // 4, out_channels=in_channels // 2, kernel_size=3, stride=1, use_act=False),
            ConvLayer(in_channels=in_channels // 2, out_channels=in_channels, kernel_size=3, stride=1, use_act=False)
        )

    def forward(self, x):

        return self.conv(x)


class Image_UPconv(nn.Module):
    def __init__(self, in_channels):
        super(Image_UPconv, self).__init__()

        self.conv = nn.Sequential(
            ConvLayer(in_channels=in_channels, out_channels=in_channels // 2, kernel_size=3, stride=1, use_act=False),
            ConvLayer(in_channels=in_channels // 2, out_channels=32, kernel_size=3, stride=1, use_act=False)
        )

    def forward(self, x):

        return self.conv(x)

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x

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
        return x / torch.sqrt(sigma + 1e-5) * self.weight


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
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)
class Self_TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, init_values=1e-5, use_layer_scale=True):
        super(Self_TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn1 = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)

        self.attn2 = Cross_Attention(dim, num_heads, bias)
        # self.attn2 = Attention(dim, num_heads, bias)
        self.norm3 = LayerNorm(dim, LayerNorm_type)
        self.ffn1 = FeedForward(dim, ffn_expansion_factor, bias)
        self.norm4 = LayerNorm(dim, LayerNorm_type)

    def forward(self, x):

        x2 = x + self.attn1(self.norm1(x))
        # x2 = x2 + self.attn2(x, self.norm2(x2))  # 以浅一层的特征x作KV, 更深层次的特征x2作Q默认
        # x2 = x + self.attn2(x2, self.norm2(x))  # 以浅一层的特征作Q, 更深层次的特征作KV
        x2 = x2 + self.attn2(self.norm4(x), self.norm2(x2))  # 以浅一层的特征x作KV, 更深层次的特征x2作Q
        # x2 = x + self.attn2(self.norm2(x2), self.norm4(x))  # 以深层次的特征x2作KV, 以浅一层的特征x作Q
        # x2 = x2 + self.attn2(self.norm2(x2))  # 取消自Transformer支路
        x2 = x2 + self.ffn1(self.norm3(x2))

        return x2

class Dual_TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, init_values=1e-5, use_layer_scale=True):
        super(Dual_TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.Self_attn = nn.Sequential(*[
            Self_TransformerBlock(dim=dim,
                                  num_heads=num_heads,
                                  ffn_expansion_factor=ffn_expansion_factor,
                                  bias=bias,
                                  LayerNorm_type=LayerNorm_type
                                  )
            for _ in range(1)
        ])
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.attn2 = Cross_Attention(dim, num_heads, bias)
        self.norm3 = LayerNorm(dim, LayerNorm_type)
        self.ffn2 = FeedForward(dim, ffn_expansion_factor, bias)
    def forward(self, x):

        x2 = self.Self_attn(x[1])
        # x1 = x[0] + self.attn2(x2, self.norm1(x[0]))    #以浅一层的特征作Q, 更深层次的特征作KV 默认
        x1 = x2 + self.attn2(self.norm1(x[0]),  self.norm2(x2))  # 以浅一层的特征x[0]作KV, 更深层次的x2作Q
        # x1 = x[0] + self.attn2(self.norm2(x2), self.norm1(x[0]))  # 以深层次的特征x2作KV, 更浅一层的特征x[0]作Q
        x1 = x1 + self.ffn2(self.norm3(x1))
        return x1



## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


class Mix_Block(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias, LayerNorm_type, num_heads=8, num_blocks=2,use_layer_scale=False):
        super(Mix_Block, self).__init__()
        #
        self.Global = nn.Sequential(*[
            TransformerBlock(dim=dim,
                             num_heads=num_heads,
                             ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias,
                             LayerNorm_type=LayerNorm_type,
                             use_layer_scale=use_layer_scale)
            for i in range(num_blocks)]
                                    )

    def forward(self, x):
        x = self.Global(x)
        return x


class DRSformer(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=32,
                 num_blocks=[2, 2, 2, 2],
                 heads=[8, 8, 8, 8],
                 ffn_expansion_factor=2.66,
                 bias=False,                  # 默认为True
                 LayerNorm_type='WithBias',  ## Other option 'BiasFree' 默认用'WithBias'
                 use_layer_scale=False,
                 use_act=False,
                 conv_bias=False,
                 ):
        super(DRSformer, self).__init__()

        self.patch_embed = ConvLayer(in_channels=inp_channels,
                                         out_channels=dim,
                                         kernel_size=3,
                                         stride=1,
                                         use_act=use_act,
                                         bias=conv_bias)

        # self.encoder_level0 = Mix_Block(
        #     dim=dim,
        #     num_heads=heads[0],
        #     num_blocks=num_blocks[0],
        #     ffn_expansion_factor=ffn_expansion_factor,
        #     bias=bias,
        #     LayerNorm_type=LayerNorm_type
        # )
        self.encoder_level1 = nn.Sequential(*[
            Res_block(in_channels=dim)
            for i in range(6)
        ])
        self.dwt = DWT_2D(wave='haar')
        self.idwt = IDWT_2D(wave='haar')
        self.reduce_channel2 = ConvLayer(in_channels=dim,
                                         out_channels=dim // 2,
                                         kernel_size=3,
                                         stride=1,
                                         use_act=use_act,
                                         bias=conv_bias)
        # self.expand_encoder_channel2 = ConvLayer(in_channels=(dim * 2 ** 0) * 4,
        #                                          out_channels=(dim * 2 ** 0) * 2,
        #                                          kernel_size=3,
        #                                          stride=1,
        #                                          use_act=False)

        self.encoder_level2 = Self_TransformerBlock(
            dim=int(dim * 2 ** 1),
            num_heads=heads[1],
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            LayerNorm_type=LayerNorm_type,
            use_layer_scale=use_layer_scale
        )
        self.reduce_channel3 = ConvLayer(in_channels=int(dim * 2 ** 1),
                                         out_channels=dim,
                                         kernel_size=3,
                                         stride=1,
                                         use_act=use_act,
                                         bias=conv_bias)
        # self.expand_encoder_channel3 = ConvLayer(in_channels=(dim * 2 ** 1) * 4,
        #                                          out_channels=(dim * 2 ** 1) * 2,
        #                                          kernel_size=3,
        #                                          stride=1,
        #                                          use_act=False)
        self.encoder_level3 =  Self_TransformerBlock(
            dim=int(dim * 2 ** 2),
            num_heads=heads[2],
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            LayerNorm_type=LayerNorm_type,
            use_layer_scale=use_layer_scale
        )

        self.reduce_channel4 = ConvLayer(in_channels=int(dim * 2 ** 2),
                                         out_channels=dim * 2,
                                         kernel_size=3,
                                         stride=1,
                                         use_act=use_act,
                                         bias=conv_bias)
        # self.expand_encoder_channel4 = ConvLayer(in_channels=(dim * 2 ** 2) * 4,
        #                                          out_channels=(dim * 2 ** 2) * 2,
        #                                          kernel_size=3,
        #                                          stride=1,
        #                                          use_act=False)
        self.latent = Self_TransformerBlock(
            dim=int(dim * 2 ** 3),
            num_heads=heads[3],
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            LayerNorm_type=LayerNorm_type,
            use_layer_scale=use_layer_scale
        )
        ## From Level 4 to Level 3

        self.expand_channel4 = ConvLayer(in_channels=int(dim * 2 ** 3),
                                         out_channels=int(dim * 2 ** 4),
                                         kernel_size=3,
                                         stride=1,
                                         use_act=use_act,
                                         bias=conv_bias)
        self.decoder_level3 = nn.Sequential(*[
            Dual_TransformerBlock(dim=int(dim * 2 ** 2),
                                  num_heads=heads[2],
                                  ffn_expansion_factor=ffn_expansion_factor,
                                  bias=bias,
                                  LayerNorm_type=LayerNorm_type,
                                  use_layer_scale=use_layer_scale)
            for i in range(1)]
                                    )

        self.expand_channel3 = ConvLayer(in_channels=int(dim * 2 ** 2),
                                         out_channels=int(dim * 2 ** 3),
                                         kernel_size=3,
                                         stride=1,
                                         use_act=use_act,
                                         bias=conv_bias)

        self.decoder_level2 = nn.Sequential(*[
            Dual_TransformerBlock(dim=int(dim * 2 ** 1),
                                  num_heads=heads[1],
                                  ffn_expansion_factor=ffn_expansion_factor,
                                  bias=bias,
                                  LayerNorm_type=LayerNorm_type,
                                  use_layer_scale=use_layer_scale)
            for i in range(1)]
                                    )

        self.decoder_level0 = nn.Sequential(*[
            Dual_TransformerBlock(dim=int(dim * 2 ** 0),
                                  num_heads=heads[0],
                                  ffn_expansion_factor=ffn_expansion_factor,
                                  bias=bias,
                                  LayerNorm_type=LayerNorm_type)
            for i in range(1)]
                                    )

        # self.decoder_level1 = nn.Sequential(*[
        #     Res_block(in_channels=dim)
        #     for i in range(6)
        # ])
        self.expand_channel2 = ConvLayer(in_channels=int(dim * 2 ** 1),
                                         out_channels=int(dim * 2 ** 2),
                                         kernel_size=3,
                                         stride=1,
                                         use_act=use_act,
                                         bias=conv_bias)

        self.output = ConvLayer(in_channels=(dim * 2 ** 0),
                                out_channels=3,
                                kernel_size=3,
                                stride=1,
                                use_act=False,
                                bias=conv_bias)

    def forward(self, inp_img):

        inp_enc_level1 = self.patch_embed(inp_img)
        # inp_enc_level1 = self.encoder_level0(inp_enc_level1)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.dwt(self.reduce_channel2(out_enc_level1))
        # inp_enc_level2 = self.expand_encoder_channel2(self.dwt(out_enc_level1))
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.dwt(self.reduce_channel3(out_enc_level2))
        # inp_enc_level3 = self.expand_encoder_channel3(self.dwt(out_enc_level2))
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.dwt(self.reduce_channel4(out_enc_level3))
        # inp_enc_level4 = self.expand_encoder_channel4(self.dwt(out_enc_level3))
        latent = self.latent(inp_enc_level4)

        ############################################################
        # U-net up
        inp_dec_level3 = self.idwt(self.expand_channel4(latent))
        out_dec_level3 = self.decoder_level3((out_enc_level3, inp_dec_level3))

        inp_dec_level2 = self.idwt(self.expand_channel3(out_dec_level3))
        out_dec_level2 = self.decoder_level2((out_enc_level2, inp_dec_level2))

        inp_dec_level1 = self.idwt(self.expand_channel2(out_dec_level2))
        out_dec_level1 = self.decoder_level0((out_enc_level1, inp_dec_level1))
        # out_dec_level1 = self.decoder_level1(out_dec_level1)
        # out_dec_level1 = self.decoder_level1(out_enc_level1+inp_dec_level1)
        out_dec_level1 = self.output(out_dec_level1) + inp_img

        return out_dec_level1


if __name__ == '__main__':
    import ptflops
    from thop import profile

    input = torch.rand(1, 3, 256, 256)
    model = DRSformer(dim=32)
    # print(model)
    # output = model(input)
    #
     # from fvcore.nn import FlopCountAnalysis, parameter_count_table
     #
     # flops = FlopCountAnalysis(model, input)
     # print("FLOPs: ", flops.total() / 1000 ** 3)

    macs, params_total = ptflops.get_model_complexity_info(model, (3, 256, 256), as_strings=False,
                                                           print_per_layer_stat=True, verbose=False)
    inputs = torch.randn(1, 3, 512, 512)
    flops, params = profile(model, (inputs,))
    print('flops: ', flops / 1000 ** 3, 'params: ', params / 1000 ** 2)

    # print(macs / 1000 ** 3)
    # print('{:<30}  {:<8}'.format('params_total M: ', params_total / (1000. ** 2)))
