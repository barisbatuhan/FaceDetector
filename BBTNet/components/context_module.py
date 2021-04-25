# External Libraries
import torch
import torch.nn as nn
# Internal Files
from BBTNet.components.ssh import SSH

class ContextModule(torch.nn.Module):
    def __init__(self, in_size=256, out_size=256):
        super(ContextModule, self).__init__()
        self.ssh_list = nn.ModuleList()
        for i in range(5):
            self.ssh_list.append(SSH(in_size=in_size, out_size=out_size))

    def forward(self, xs):
        outs = []
        for i in range(len(xs)):
            outs.append(self.ssh_list[i](xs[i]))
        return outs