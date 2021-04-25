# External Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
# Internal Files
from BBTNet.components.blocks import ConvBnRelu

class FeaturePyramid(torch.nn.Module):
    def __init__(self, in_sizes=[256, 512, 1024, 2048], out_size=256):
        super(FeaturePyramid, self).__init__()
        
        self.lateral_ins = nn.ModuleList()
        # getting layers for lateral connections
        for ins in in_sizes:
            self.lateral_ins.append(ConvBnRelu(ins, out_size, 1, bias=False))
        
        self.lateral_ins.append(
            ConvBnRelu(in_sizes[-1], out_size, 3, stride=2, padding=1, bias=False)
        )
        self.extra = True
        self.lateral_outs = nn.ModuleList()
        for _ in range(len(self.lateral_ins)-1):
            self.lateral_outs.append(
                ConvBnRelu(out_size, out_size, 3, padding=1, bias=False)
            )

    def forward(self, xs):
        outs = []
        outs.append(self.lateral_ins[-1](xs[-1])) # P6

        inter_outs = []
        for idx in range(4):
            inter_outs.append(self.lateral_ins[idx](xs[idx]))  
        outs.append(inter_outs[-1]) # P5

        for idx in range(3, 0, -1):
            conn = F.interpolate(
                inter_outs[idx], 
                size=[inter_outs[idx-1].size(2), inter_outs[idx-1].size(3)], 
                mode="nearest"
            )
            outs.append(self.lateral_outs[idx-1](conn + inter_outs[idx-1])) # P4, P3, P2
        
        outs.reverse()
        return outs