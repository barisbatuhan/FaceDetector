# External Libraries
import torch
import torch.nn as nn
# Internal Files
from BBTNet.components.blocks import ConvBnRelu

class SSH(torch.nn.Module):
    def __init__(self, in_size=256, out_size=256):
        super(SSH, self).__init__()
        out_size4 = in_size//4; out_size2 = in_size//2
        self.conv128 = ConvBnRelu(in_size, out_size2, 3, bias=False, padding=1, leaky=-1)
        self.conv64_1_1 = ConvBnRelu(in_size, out_size4, 3, bias=False, padding=1)
        self.conv64_1_2 = ConvBnRelu(out_size4, out_size4, 3, bias=False, padding=1, leaky=-1)
        self.conv64_2_1 = ConvBnRelu(out_size4, out_size4, 3, bias=False, padding=1)
        self.conv64_2_2 = ConvBnRelu(out_size4, out_size4, 3, bias=False, padding=1, leaky=-1)
        self.activ = nn.ReLU()

    def forward(self, x):
        o1 = self.conv128(x)
        o2_1 = self.conv64_1_1(x)
        o2_2 = self.conv64_1_2(o2_1)
        o3_1 = self.conv64_2_1(o2_1)
        o3_2 = self.conv64_2_2(o3_1)
        return self.activ(torch.cat([o1, o2_2, o3_2], dim=1))