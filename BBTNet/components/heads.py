# External Libraries
import torch
import torch.nn as nn
# Internal Files
from configs import *

class HeadGetter(torch.nn.Module):
    def __init__(self, input_dims, task_len, lateral_conn=5, num_anchors=3):
        super(HeadGetter, self).__init__()
        self.layers = nn.ModuleList()
        self.task_len = task_len
        for _ in range(lateral_conn):
            self.layers.append(
                nn.Conv2d(input_dims, num_anchors*task_len, 1, bias=True)
            )

    def forward(self, xs):
        proposals = []
        T = self.task_len
        for i, x in enumerate(xs):
            proposal = self.layers[i](x)
            N, C, H, W = proposal.size()
            A = int(W*H*(C/T))
            proposal = proposal.permute(0,2,3,1).contiguous()
            proposal = proposal.view(proposal.shape[0], -1, T)
            proposals.append(proposal)
        return torch.cat(proposals, dim=1)