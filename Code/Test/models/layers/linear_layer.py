
import torch
from torch import nn

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


