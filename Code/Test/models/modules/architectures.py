import torch.nn as nn
import torch.nn.functional as F
import torch
import pywt
import math
from torch.autograd import Function
# from ..layers.conv_layer import ConvLayer


class Layer_norm(nn.Module):
    def __init__(self, num_features):
        super(Layer_norm, self).__init__()
        self.eps = 1e-5
        self.gamma = nn.Parameter(torch.Tensor(1, num_features, 1, 1))
        self.beta = nn.Parameter(torch.Tensor(1, num_features, 1, 1))
        self.gamma.data.fill_(1.0)
        self.beta.data.fill_(0.0)

    def forward(self, input):
        ln_mean, ln_var = torch.mean(input, dim=[1, 2, 3], keepdim=True), torch.var(input, dim=[1, 2, 3], keepdim=True)
        out = (input - ln_mean) / torch.sqrt(ln_var + self.eps)
        out = out * self.gamma.expand(input.shape[0], -1, -1, -1) + self.beta.expand(input.shape[0], -1, -1, -1)
        return out

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, use_act=True, use_norm=False, bias=False):
        super(ConvLayer, self).__init__()
        self.use_act = use_act
        self.use_norm = use_norm
        if self.use_norm:
            self.norm = Layer_norm()
            bias = False
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=kernel_size//2,
                              bias=bias
                              )
        if self.use_act:
            # self.act = nn.GELU()
            self.act = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.use_norm:
            x = self.norm(x)
        if self.use_act:
            x = self.act(x)
        return x

class Res_block(nn.Module):
    def __init__(self, in_channels=32, init_values=1e-5, use_layer_scale=True):
        super(Res_block, self).__init__()

        self.conv0 = ConvLayer(in_channels=in_channels, out_channels=in_channels,
                               kernel_size=3, stride=1, use_act=True, use_norm=False)

        self.conv1 = ConvLayer(in_channels=in_channels, out_channels=in_channels,
                               kernel_size=3, stride=1, use_act=False, use_norm=False)

        # self.gamma_1 = nn.Parameter(init_values * torch.ones((in_channels)), requires_grad=True) if use_layer_scale else 1

    def forward(self, x):

        res = self.conv0(x)
        res = self.conv1(res)
        # res = x + self.gamma_1.view(1, -1, 1, 1) * res
        res = x + res

        return res


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False), nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout) * x

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=5):
        super().__init__()
        assert kernel_size in (3, 5, 7), "kernel size must be 3 or 5 or 7"
        self.conv = ConvLayer(in_channels=2,
                              out_channels=1,
                              kernel_size=kernel_size,
                              use_act=False,
                              bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avgout, maxout], dim=1)
        attention = self.conv(attention)
        return self.sigmoid(attention) * x

class CBAMBlock(nn.Module):
    def __init__(self, channel):
        super(CBAMBlock, self).__init__()
        self.ca = ChannelAttention(in_planes=channel)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x

class Res_CBAM(nn.Module):
    def __init__(self, in_channels=3,
                 out_channels=32,
                 conv_ksize=3):
        super(Res_CBAM, self).__init__()

        self.conv0 = ConvLayer(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=conv_ksize,
                               stride=1,
                               use_act=True,
                               use_norm=False)

        self.conv1 = ConvLayer(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=conv_ksize,
                               stride=1,
                               use_act=False,
                               use_norm=False)

        self.CBAM = CBAMBlock(channel=out_channels)

    def forward(self, x):
        res = self.conv0(x)
        res = self.conv1(res)
        res = self.CBAM(res)
        res = res + x
        return res

class SFT(nn.Module):
    """
        SFT: Affine transformation
        in_channels: :math:`C_{in}` from an expected input of size :math:`(N, C_{in}, H, W)
        mid_channels: :math:`C_{in}` from an expected input of size :math:`(N, C_{in}, H, W)
        conv_ksize: The kernel size of convolution. Default: 3

    """
    def __init__(self,in_channels=3,
                 out_channels=32,
                 conv_ksize=3):
        super(SFT, self).__init__()

        self.conv0 = ConvLayer(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=conv_ksize,
                               stride=1,
                               use_act=False,
                               use_norm=False)

        self.conv1 = ConvLayer(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=conv_ksize,
                               stride=1,
                               use_act=False,
                               use_norm=False)

    def forward(self, x):

        alpha = self.conv0(x[1])
        beta = self.conv1(x[1])
        sft = x[0] * alpha + beta
        # sft = x[0] * alpha + beta + x[0]
        return sft

class SAM(nn.Module):
  """Supervised attention module for multi-stage training.

  Introduced by MPRNet [CVPR2021]: https://github.com/swz30/MPRNet
  """
  def __init__(self, in_channels, kernel_size=1, bias=False):
      super(SAM, self).__init__()

      self.conv1 = ConvLayer(in_channels=in_channels,
                             out_channels=in_channels,
                             kernel_size=kernel_size,
                             stride=1,
                             use_act=False,
                             use_norm=False,
                             bias=bias
                             )
      self.conv2 = ConvLayer(in_channels=3,
                             out_channels=in_channels,
                             kernel_size=kernel_size,
                             stride=1,
                             use_act=False,
                             use_norm=False,
                             bias=bias
                             )
  def forward(self, x, img):
      x1 = self.conv1(x)
      x2 = torch.sigmoid(self.conv2(img))
      x1 = x1 * x2
      x1 = x1 + x

      return x1

##---------- Resizing Modules ----------
class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=0.5):
        super(DownSample, self).__init__()
        self.down = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor,
                        mode='bilinear',
                        align_corners=False),
            ConvLayer(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=1,
                      stride=1,
                      use_act=False,
                      use_norm=False,
                      bias=False
                      )
        )

    def forward(self, x):
        x = self.down(x)
        return x

class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(UpSample, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor,
                        mode='bilinear',
                        align_corners=False),
            ConvLayer(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=1,
                      stride=1,
                      use_act=False,
                      use_norm=False,
                      bias=False
                      )
        )


    def forward(self, x):
        x = self.up(x)
        return x

class InvertedResidualBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=2, expand_ratio=6, activation=nn.ReLU6):
        super(InvertedResidualBlock, self).__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        # assert stride in [1, 2]
        hidden_dim = int(in_channels * expand_ratio)
        self.is_residual = self.stride == 1 and in_channels == out_channels
        self.conv = nn.Sequential(
            # pw Point-wise
            nn.Conv2d(in_channels, hidden_dim, 1, 1, 0, bias=False),
            activation(inplace=True),
            # dw  Depth-wise
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, kernel_size//2, groups=hidden_dim, bias=False),
            activation(inplace=True),
            # pw-linear, Point-wise linear
            nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False),

        )
    def forward(self, x):
        if self.stride == 1 and self.in_channels == self.out_channels:
            res = self.conv(x)
            x = x + res
        else:
            x = self.conv(x)
        return x

class Res_CA(nn.Module):
    def __init__(self, in_channels=3,
                 mid_channels=32,
                 conv_ksize=3
                 ):
        super(Res_CA, self).__init__()
        self.conv0 = ConvLayer(in_channels=in_channels,
                               out_channels=mid_channels,
                               kernel_size=conv_ksize,
                               stride=1,
                               use_act=True,
                               use_norm=False)     # default norm =False
        self.conv1 = ConvLayer(in_channels=mid_channels,
                               out_channels=mid_channels,
                               kernel_size=conv_ksize,
                               stride=1,
                               use_act=False,
                               use_norm=False)
        self.CA = ChannelAttention(in_planes=mid_channels)

    def forward(self, x):
        res = self.conv0(x)
        res = self.conv1(res)
        res = self.CA(res)
        res = res + x
        return res

def same_padding(images, ksizes, strides, rates):

    assert len(images.size()) == 4
    batch_size, channel, rows, cols = images.size()
    out_rows = (rows + strides[0] - 1) // strides[0]
    out_cols = (cols + strides[1] - 1) // strides[1]
    effective_k_row = (ksizes[0] - 1) * rates[0] + 1
    effective_k_col = (ksizes[1] - 1) * rates[1] + 1
    padding_rows = max(0, (out_rows-1)*strides[0]+effective_k_row-rows)
    padding_cols = max(0, (out_cols-1)*strides[1]+effective_k_col-cols)
    # Pad the input
    padding_top = int(padding_rows / 2.)
    padding_left = int(padding_cols / 2.)
    padding_bottom = padding_rows - padding_top
    padding_right = padding_cols - padding_left
    paddings = (padding_left, padding_right, padding_top, padding_bottom)
    images = torch.nn.ZeroPad2d(paddings)(images)
    return images

def extract_image_patches(images, ksizes, strides, rates, padding='same'):
    """
    Extract patches from images and put them in the C output dimension.
    :param padding:
    :param images: [batch, channels, in_rows, in_cols]. A 4-D Tensor with shape
    :param ksizes: [ksize_rows, ksize_cols]. The size of the sliding window for
     each dimension of images
    :param strides: [stride_rows, stride_cols]
    :param rates: [dilation_rows, dilation_cols]
    :return: A Tensor
    """
    assert len(images.size()) == 4
    assert padding in ['same', 'valid']
    batch_size, channel, height, width = images.size()

    if padding == 'same':
        images = same_padding(images, ksizes, strides, rates)
    elif padding == 'valid':
        pass
    else:
        raise NotImplementedError('Unsupported padding type: {}.\
                Only "same" or "valid" are supported.'.format(padding))

    unfold = torch.nn.Unfold(kernel_size=ksizes,
                             dilation=rates,
                             padding=0,
                             stride=strides)
    patches = unfold(images)
    return patches  # [N, C*k*k, L], L is the total number of such blocks

def reverse_patches(images, out_size, ksizes, strides, padding):
    """
    Extract patches from images and put them in the C output dimension.
    :param padding:
    :param images: [batch, channels, in_rows, in_cols]. A 4-D Tensor with shape
    :param ksizes: [ksize_rows, ksize_cols]. The size of the sliding window for
     each dimension of images
    :param strides: [stride_rows, stride_cols]
    :param rates: [dilation_rows, dilation_cols]
    :return: A Tensor
    """
    unfold = torch.nn.Fold(output_size = out_size,
                            kernel_size=ksizes,
                            dilation=1,
                            padding=padding,
                            stride=strides)
    patches = unfold(images)
    return patches  # [N, C*k*k, L], L is the total number of such blocks
def reduce_mean(x, axis=None, keepdim=False):
    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = torch.mean(x, dim=i, keepdim=keepdim)
    return x


class DWT_Function(Function):
    @staticmethod
    def forward(ctx, x, w_ll, w_lh, w_hl, w_hh):
        x = x.contiguous()
        ctx.save_for_backward(w_ll, w_lh, w_hl, w_hh)
        ctx.shape = x.shape

        dim = x.shape[1]
        x_ll = torch.nn.functional.conv2d(x, w_ll.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_lh = torch.nn.functional.conv2d(x, w_lh.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_hl = torch.nn.functional.conv2d(x, w_hl.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_hh = torch.nn.functional.conv2d(x, w_hh.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
        return x
        # H = torch.cat([x_lh, x_hl, x_hh], dim=1)
        # return (x_ll, H)
        # return x_ll, x_lh, x_hl, x_hh
    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            w_ll, w_lh, w_hl, w_hh = ctx.saved_tensors
            B, C, H, W = ctx.shape
            dx = dx.view(B, 4, -1, H // 2, W // 2)

            dx = dx.transpose(1, 2).reshape(B, -1, H // 2, W // 2)
            filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0).repeat(C, 1, 1, 1)
            dx = torch.nn.functional.conv_transpose2d(dx, filters, stride=2, groups=C)

        return dx, None, None, None, None


class IDWT_Function(Function):
    @staticmethod
    def forward(ctx, x, filters):
        ctx.save_for_backward(filters)
        ctx.shape = x.shape

        B, _, H, W = x.shape
        x = x.view(B, 4, -1, H, W).transpose(1, 2)
        C = x.shape[1]
        x = x.reshape(B, -1, H, W)
        filters = filters.repeat(C, 1, 1, 1)
        x = torch.nn.functional.conv_transpose2d(x, filters, stride=2, groups=C)
        return x

    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            filters = ctx.saved_tensors
            filters = filters[0]
            B, C, H, W = ctx.shape
            C = C // 4
            dx = dx.contiguous()

            w_ll, w_lh, w_hl, w_hh = torch.unbind(filters, dim=0)
            x_ll = torch.nn.functional.conv2d(dx, w_ll.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            x_lh = torch.nn.functional.conv2d(dx, w_lh.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            x_hl = torch.nn.functional.conv2d(dx, w_hl.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            x_hh = torch.nn.functional.conv2d(dx, w_hh.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            dx = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
        return dx, None


class IDWT_2D(nn.Module):
    def __init__(self, wave):
        super(IDWT_2D, self).__init__()
        w = pywt.Wavelet(wave)
        rec_hi = torch.Tensor(w.rec_hi)
        rec_lo = torch.Tensor(w.rec_lo)

        w_ll = rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1)
        w_lh = rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1)
        w_hl = rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1)
        w_hh = rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)

        w_ll = w_ll.unsqueeze(0).unsqueeze(1)
        w_lh = w_lh.unsqueeze(0).unsqueeze(1)
        w_hl = w_hl.unsqueeze(0).unsqueeze(1)
        w_hh = w_hh.unsqueeze(0).unsqueeze(1)
        filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0)
        self.register_buffer('filters', filters)
        self.filters = self.filters.to(dtype=torch.float32)

    def forward(self, x):
        return IDWT_Function.apply(x, self.filters)


class DWT_2D(nn.Module):
    def __init__(self, wave):
        super(DWT_2D, self).__init__()
        w = pywt.Wavelet(wave)
        dec_hi = torch.Tensor(w.dec_hi[::-1])
        dec_lo = torch.Tensor(w.dec_lo[::-1])

        w_ll = dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1)
        w_lh = dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1)
        w_hl = dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1)
        w_hh = dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)

        self.register_buffer('w_ll', w_ll.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_lh', w_lh.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_hl', w_hl.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_hh', w_hh.unsqueeze(0).unsqueeze(0))

        self.w_ll = self.w_ll.to(dtype=torch.float32)
        self.w_lh = self.w_lh.to(dtype=torch.float32)
        self.w_hl = self.w_hl.to(dtype=torch.float32)
        self.w_hh = self.w_hh.to(dtype=torch.float32)

    def forward(self, x):
        return DWT_Function.apply(x, self.w_ll, self.w_lh, self.w_hl, self.w_hh)
        # (L, H) = DWT_Function.apply(x, self.w_ll, self.w_lh, self.w_hl, self.w_hh)
        # return (L, H)
class LinearLayer(nn.Module):
    def __init__(self, in_features=32, out_features=64, bias=True):
        """
            Applies a linear transformation to the input data

            :param in_features: size of each input sample
            :param out_features:  size of each output sample
            :param bias: Add bias (learnable) or not
        """
        super(LinearLayer, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        self.in_features = in_features
        self.out_features = out_features
        self.reset_params()


    def reset_params(self):
        if self.weight is not None:
            torch.nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            torch.nn.init.constant_(self.bias, 0)

    def forward(self, x):
        if self.bias is not None and x.dim() == 2:
            x = torch.addmm(self.bias, x, self.weight.t())
        else:
            x = x.matmul(self.weight.t())
            if self.bias is not None:
                x += self.bias
        return x


def Unfolding(feature_map, patch_h, patch_w):
    patch_area = int(patch_w * patch_h)
    batch_size, in_channels, orig_h, orig_w = feature_map.shape

    new_h = int(math.ceil(orig_h / patch_h) * patch_h)
    new_w = int(math.ceil(orig_w / patch_w) * patch_w)

    interpolate = False
    if new_w != orig_w or new_h != orig_h:
        # Note: Padding can be done, but then it needs to be handled in attention function.
        feature_map = F.interpolate(feature_map, size=(new_h, new_w), mode="bilinear", align_corners=False)
        interpolate = True

    # number of patches along width and height
    num_patch_w = new_w // patch_w # n_w
    num_patch_h = new_h // patch_h # n_h
    num_patches = num_patch_h * num_patch_w # N

    # [B, C, H, W] --> [B * C * n_h, p_h, n_w, p_w]
    reshaped_fm = feature_map.reshape(batch_size * in_channels * num_patch_h, patch_h, num_patch_w, patch_w)
    # [B * C * n_h, p_h, n_w, p_w] --> [B * C * n_h, n_w, p_h, p_w]
    transposed_fm = reshaped_fm.transpose(1, 2)
    # [B * C * n_h, n_w, p_h, p_w] --> [B, C, N, P] where P = p_h * p_w and N = n_h * n_w
    reshaped_fm = transposed_fm.reshape(batch_size, in_channels, num_patches, patch_area)
    # [B, C, N, P] --> [B, P, N, C]
    transposed_fm = reshaped_fm.transpose(1, 3)
    # [B, P, N, C] --> [BP, N, C]
    patches = transposed_fm.reshape(batch_size * patch_area, num_patches, -1)

    info_dict = {
        "orig_size": (orig_h, orig_w),
        "batch_size": batch_size,
        "interpolate": interpolate,
        "total_patches": num_patches,
        "num_patches_w": num_patch_w,
        "num_patches_h": num_patch_h
    }

    return patches, info_dict

def Folding(patches, info_dict, patch_h, patch_w):
    n_dim = patches.dim()
    patch_area = patch_h * patch_w
    assert n_dim == 3, "Tensor should be of shape BPxNxC. Got: {}".format(patches.shape)
    # [BP, N, C] --> [B, P, N, C]
    patches = patches.contiguous().view(info_dict["batch_size"], patch_area, info_dict["total_patches"], -1)

    batch_size, pixels, num_patches, channels = patches.size()
    num_patch_h = info_dict["num_patches_h"]
    num_patch_w = info_dict["num_patches_w"]

    # [B, P, N, C] --> [B, C, N, P]
    patches = patches.transpose(1, 3)

    # [B, C, N, P] --> [B*C*n_h, n_w, p_h, p_w]
    feature_map = patches.reshape(batch_size * channels * num_patch_h, num_patch_w, patch_h, patch_w)
    # [B*C*n_h, n_w, p_h, p_w] --> [B*C*n_h, p_h, n_w, p_w]
    feature_map = feature_map.transpose(1, 2)
    # [B*C*n_h, p_h, n_w, p_w] --> [B, C, H, W]
    feature_map = feature_map.reshape(batch_size, channels, num_patch_h * patch_h, num_patch_w * patch_w)
    if info_dict["interpolate"]:
        feature_map = F.interpolate(feature_map, size=info_dict["orig_size"], mode="bilinear", align_corners=False)
    return feature_map

class Conv2d_cd(nn.Module):
    def __init__(self, in_channels, out_channles, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False):
        super(Conv2d_cd, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channles, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        #self.theta = theta

    def forward(self, x):
        #print('x.shape:', x.shape)
        out_normal = self.conv(x)
        #print('out_normal.shape:', out_normal.shape)
        # if math.fabs(self.theta - 0.) < 1e-8:
        #     return out_normal
        # else:
        # [C_out, C_in, kernel_size, kernel_size] = self.conv.weight.shape
        kernel_diff = self.conv.weight.sum(dim=[2, 3], keepdim=True)
        out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride,
                            padding=0, groups=self.conv.groups)
        #print('out_diff.shape', out_diff.shape)
        return out_normal - out_diff

class HDCB(nn.Module):
    def __init__(self, dilation_rates, embedding_dim=32):
        super(HDCB, self).__init__()
        self.dilation_1 = nn.Sequential(
            Conv2d_cd(embedding_dim, embedding_dim, padding=dilation_rates[0], dilation=dilation_rates[0],
                      groups=embedding_dim),
            nn.PReLU()
        )
        self.d1_1x1 = nn.Sequential(
            nn.Conv2d(embedding_dim * 2, embedding_dim, kernel_size=1),
            nn.LeakyReLU()
        )

        self.dilation_2 = nn.Sequential(
            Conv2d_cd(embedding_dim, embedding_dim, padding=dilation_rates[1], dilation=dilation_rates[1],
                      groups=embedding_dim),
            nn.PReLU()
        )
        self.d2_1x1 = nn.Sequential(
            nn.Conv2d(embedding_dim * 3, embedding_dim, kernel_size=1),
            nn.LeakyReLU()
        )

        self.dilation_3 = nn.Sequential(
            Conv2d_cd(embedding_dim, embedding_dim, padding=dilation_rates[2], dilation=dilation_rates[2],
                      groups=embedding_dim),
            nn.PReLU()
        )
        self.d3_1x1 = nn.Sequential(
            nn.Conv2d(embedding_dim * 4, embedding_dim, kernel_size=1),
            nn.LeakyReLU()
        )

    def forward(self, x):
        d1 = self.dilation_1(x)
        d2 = self.dilation_2(self.d1_1x1(torch.cat([x, d1], dim=1)))
        d3 = self.dilation_3(self.d2_1x1(torch.cat([x, d1, d2], dim=1)))
        out = self.d3_1x1(torch.cat([x, d1, d2, d3], dim=1))

        return out
class Fuse_HF(nn.Module):
    def __init__(self, hf_dim=32, dilation_rates=[3, 2, 1], num_hdcb=3):
        super(Fuse_HF, self).__init__()
        self.conv_in_1 = ConvLayer(in_channels=3,
                                    out_channels=hf_dim,
                                    kernel_size=3,
                                    stride=1,
                                    use_act=True,
                                    use_norm=False)
        self.hf_hd_block = nn.Sequential(*[
            HDCB(dilation_rates=dilation_rates, embedding_dim=hf_dim)
            for i in range(num_hdcb)
        ])

    def forward(self, x):

        x = self.conv_in_1(x)
        x = x + self.hf_hd_block(x)

        return x

class make_dense(nn.Module):
  def __init__(self, nChannels, growthRate, kernel_size=3):
    super(make_dense, self).__init__()
    # self.conv = nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=(kernel_size-1)//2, bias=False)
    self.conv = ConvLayer(in_channels=nChannels,
                                         out_channels=growthRate,
                                         kernel_size=kernel_size,
                                         stride=1,
                                         use_act=True,
                                         bias=False)
  def forward(self, x):
    out = self.conv(x)
    out = torch.cat((x, out), 1)
    return out

# Residual dense block (RDB) architecture
class RDB(nn.Module):
  def __init__(self, nChannels, nDenselayer, growthRate):
    super(RDB, self).__init__()
    nChannels_ = nChannels
    modules = []
    for i in range(nDenselayer):
        modules.append(make_dense(nChannels_, growthRate))
        nChannels_ += growthRate
    self.dense_layers = nn.Sequential(*modules)
    self.conv_1x1 = ConvLayer(in_channels=nChannels_,
                              out_channels=nChannels,
                              kernel_size=1,
                              stride=1,
                              use_act=False,
                              bias=False)
  def forward(self, x):
    out = self.dense_layers(x)
    out = self.conv_1x1(out)
    out = out + x
    return out


