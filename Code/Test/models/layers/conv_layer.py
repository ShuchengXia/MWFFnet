
import torch
from torch import nn
from torch.nn.parameter import Parameter
import logging
from .non_linear_layers import get_activation_fn

class Aff_channel(nn.Module):
    def __init__(self, dim, channel_first=False):
        super().__init__()
        # learnable
        self.alpha = nn.Parameter(torch.ones([1, 1, 1, dim]))
        self.beta = nn.Parameter(torch.zeros([1, 1, 1, dim]))
        self.color = nn.Parameter(torch.eye(dim))
        self.channel_first = channel_first

    def forward(self, x):
        input = x.permute(0, 2, 3, 1)

        if self.channel_first:
            x1 = torch.tensordot(input, self.color, dims=[[-1], [-1]])
            x2 = x1 * self.alpha + self.beta
        else:
            x1 = input * self.alpha + self.beta
            x2 = torch.tensordot(x1, self.color, dims=[[-1], [-1]])
        out = x2.permute(0, 3, 1, 2)
        return out


class normalization_layer(nn.Module):
    def __init__(self, num_features, norm_type, eps=1e-5):
        super(normalization_layer, self).__init__()
        self.norm_type = norm_type
        momentum = 0.1
        if self.norm_type == 'layer_norm':
            self.eps = eps
            self.gamma = Parameter(torch.Tensor(1, num_features, 1, 1))
            self.beta = Parameter(torch.Tensor(1, num_features, 1, 1))
            self.gamma.data.fill_(1.0)
            self.beta.data.fill_(0.0)
        elif self.norm_type == 'instance_norm':
            self.norm = nn.InstanceNorm2d(num_features=num_features, momentum=momentum)
        elif self.norm_type == 'batch_norm':
            self.norm = nn.BatchNorm2d(num_features=num_features, momentum=momentum)
        elif self.norm_type == 'color_norm':
            self.Aff_channel = Aff_channel(num_features)
        else:
            logging.info(
                'Not supported normalization layer arguments:{}'.format(norm_type))
    def forward(self, input):
        if self.norm_type == 'layer_norm':
            ln_mean, ln_var = torch.mean(input, dim=[1, 2, 3], keepdim=True), torch.var(input, dim=[1, 2, 3], keepdim=True)
            out = (input - ln_mean) / torch.sqrt(ln_var + self.eps)
            out = out * self.gamma.expand(input.shape[0], -1, -1, -1) + self.beta.expand(input.shape[0], -1, -1, -1)
        elif self.norm_type == 'instance_norm':
            out = self.norm(input)
        elif self.norm_type == 'batch_norm':
            out = self.norm(input)
        elif self.norm_type == 'color_norm':
            out = self.Aff_channel(input)
        return out

class ConvLayer(nn.Module):
    def __init__(self, in_channels=3, out_channels=32, kernel_size=3,
                 stride=1, dilation=1, groups=1, bias=False, padding_mode='zeros',
                 act_type='gelu', norm_type='layer_norm', use_norm=False, use_act=True,
                 ):
        """
            Applies a 2D convolution over an input signal composed of several input planes.
            :param opts: arguments
            :param in_channels: number of input channels
            :param out_channels: number of output channels
            :param kernel_size: kernel size
            :param stride: move the kernel by this amount during convolution operation
            :param dilation: Add zeros between kernel elements to increase the effective receptive field of the kernel.
            :param groups: Number of groups. If groups=in_channels=out_channels, then it is a depth-wise convolution
            :param bias: Add bias or not
            :param padding_mode: Padding mode. Default is zeros
            :param use_norm: Use normalization layer after convolution layer or not. Default is True.
            :param use_act: Use activation layer after convolution layer/convolution layer followed by batch
            normalization or not. Default is True.
        """
        super(ConvLayer, self).__init__()

        if use_norm:
            assert not bias, 'Do not use bias when using normalization layers.'

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        if isinstance(stride, int):
            stride = (stride, stride)

        if isinstance(dilation, int):
            dilation = (dilation, dilation)

        assert isinstance(kernel_size, (tuple, list))
        assert isinstance(stride, (tuple, list))
        assert isinstance(dilation, (tuple, list))

        padding = (int((kernel_size[0] - 1) / 2) * dilation[0], int((kernel_size[1] - 1) / 2) * dilation[1])

        if in_channels % groups != 0:
            logging.info('Input channels are not divisible by groups. {}%{} != 0 '.format(in_channels, groups))
        if out_channels % groups != 0:
            logging.info('Output channels are not divisible by groups. {}%{} != 0 '.format(out_channels, groups))

        block = nn.Sequential()
        conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias,
                               padding_mode=padding_mode)

        block.add_module(name="conv", module=conv_layer)

        self.norm_name = None
        self.use_norm = use_norm
        if self.use_norm and norm_type is not None:
            norm_layer = normalization_layer(num_features=out_channels, norm_type=norm_type)
            block.add_module(name="norm", module=norm_layer)


        self.act_name = None

        self.use_act = use_act
        if act_type is not None and self.use_act:
            neg_slope = 0.1
            inplace = True
            act_layer = get_activation_fn(act_type=act_type,
                                          inplace=inplace,
                                          negative_slope=neg_slope,
                                          num_parameters=out_channels)
            block.add_module(name="act", module=act_layer)

        self.block = block

    def forward(self, x):
        x = self.block(x)
        return x

