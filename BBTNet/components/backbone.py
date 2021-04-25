# External Libraries
import torch
import torch.nn as nn
import torchvision.models as models

class Backbone(torch.nn.Module):
    def __init__(self, backbone="resnet50"):
        super(Backbone, self).__init__()
        
        if backbone == "resnet50" or backbone == "resnet152":
            if backbone == "resnet50":
                backbone = models.resnet50(pretrained=True)
            elif backbone == "resnet152":
                backbone = models.resnet152(pretrained=True)
            self.in_sizes = [256, 512, 1024, 2048]
            self.init_layer = nn.Sequential(
                backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool
            )
            self.layer1 = backbone.layer1
            self.layer2 = backbone.layer2
            self.layer3 = backbone.layer3
            self.layer4 = backbone.layer4
        
        elif backbone == "mobilenetv2":
            backbone = models.mobilenet_v2(pretrained=True)
            self.in_sizes = [24, 32, 96, 160]
            self.init_layer = backbone.features[0]
            self.layer1 = backbone.features[1:4]
            self.layer2 = backbone.features[4:7]
            self.layer3 = backbone.features[7:14]
            self.layer4 = backbone.features[14:19] # may be 18 also      
    
    def forward(self, x):
        c2 = self.layer1(self.init_layer(x))
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)   
        return [c2, c3, c4, c5]
        
