"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import torch.nn as nn
import geffnet


class EfficientNet(nn.Module):
    def __init__(self, ):
        super(EfficientNet, self).__init__()
        self.model = geffnet.tf_efficientnet_b0(pretrained=True, drop_rate=0.25, drop_connect_rate=0.2)

    def forward(self, x):
        x = self.model.conv_stem(x)
        x = self.model.bn1(x)
        x = self.model.act1(x)
        feature_maps = []
        for block in self.model.blocks:
            x = block(x)
            if block[0].conv_dw.stride == (2, 2):
                feature_maps.append(x)
        # _, c3, c4, c5 = feature_maps
        # print ("AA")
        # print (c3.shape)
        # print (c4.shape)
        # print (c5.shape)
        # return c3, c4, c5
        return feature_maps[1:]

if __name__ == '__main__':
    import torch
    model = EfficientNet()
    dummy = torch.rand(2,3,512,512)
    output = model(dummy)
    a,b,c = output
    print (a.shape)
    print (b.shape)
    print (c.shape)