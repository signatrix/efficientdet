"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import torch
import torch.nn as nn
import geffnet
from efficientnet_pytorch import EfficientNet as EffNet


class EfficientNet(nn.Module):
    def __init__(self, ):
        super(EfficientNet, self).__init__()
        model = geffnet.tf_efficientnet_b0(pretrained=True, drop_rate=0.25, drop_connect_rate=0.2)
        del model.conv_head
        del model.bn2
        del model.act2
        del model.global_pool
        del model.classifier
        self.model = model

    def forward(self, x):
        x = self.model.conv_stem(x)
        x = self.model.bn1(x)
        x = self.model.act1(x)
        feature_maps = []
        for block in self.model.blocks:
            x = block(x)
            if block[0].conv_dw.stride == (2, 2):
                feature_maps.append(x)

        return feature_maps[1:]

# class EfficientNet(nn.Module):
#     def __init__(self, ):
#         super(EfficientNet, self).__init__()
#         model = EffNet.from_pretrained('efficientnet-b0')
#         del model._conv_head
#         del model._bn1
#         del model._avg_pooling
#         del model._dropout
#         del model._fc
#         self.model = model
#
#     def forward(self, x):
#         x = self.model._swish(self.model._bn0(self.model._conv_stem(x)))
#         feature_maps = []
#         for idx, block in enumerate(self.model._blocks):
#             drop_connect_rate = self.model._global_params.drop_connect_rate
#             if drop_connect_rate:
#                 drop_connect_rate *= float(idx) / len(self.model._blocks)
#             x = block(x, drop_connect_rate=drop_connect_rate)
#             if block._depthwise_conv.stride == [2, 2]:
#                 feature_maps.append(x)
#
#         return feature_maps[1:]

