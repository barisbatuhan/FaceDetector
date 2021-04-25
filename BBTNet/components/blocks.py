import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import BatchNorm2d

class ConvBnRelu(torch.nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        kernel_size, 
        padding=0,
        stride=1,
        dilation=1,
        groups=1,
        bias=False,
        momentum=0.1,
        eps=1e-5,
        leaky=0,
        ):
        super(ConvBnRelu, self).__init__()

        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, dilation=dilation,padding=padding,
            groups=groups, bias=bias
            )
        self.bn = BatchNorm2d(out_channels,eps=eps,momentum=momentum)
        
        if leaky == -1:
            self.activ = None
        elif leaky == 0:
            self.activ = nn.ReLU()
        else:
            self.activ = nn.LeakyReLU(negative_slope=leaky)


    def forward(self, x):
        x_val = self.bn(self.conv(x))
        if self.activ != None:
            return self.activ(x_val)
        return x_val