"""
@author: Viet Nguyen <vn@signatrix.com>
"""
import torch.nn as nn
import torch
import torch.nn.functional as F
from src.bifpn import BiFPN


class EfficientDet(nn.Module):
    def __init__(self, backbone_net, num_anchors=9, num_classes=20, compound_coef=0):
        super(EfficientDet, self).__init__()
        self.compound_coef = compound_coef
        self.backbone_net = backbone_net
        self.out_channels = [64, 88, 112, 160, 224, 288, 384, 384][self.compound_coef]
        self.conv6 = nn.Conv2d(192, self.out_channels, kernel_size=3, stride=2, padding=1)
        self.conv7 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=2, padding=1)
        self.lateral_conn5 = nn.Conv2d(192, self.out_channels, kernel_size=1, stride=1, padding=0)
        self.lateral_conn4 = nn.Conv2d(80, self.out_channels, kernel_size=1, stride=1, padding=0)
        self.lateral_conn3 = nn.Conv2d(40, self.out_channels, kernel_size=1, stride=1, padding=0)
        self.conv5 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1)

        self.feature_maps = nn.Sequential(*[BiFPN(self.out_channels) for _ in range(min(2 + self.compound_coef, 8))])

        self.num_classes = num_classes
        self.classifier = self._sub_network(self.num_classes * num_anchors)
        self.regressor = self._sub_network(4 * num_anchors)

    def _sub_network(self, out_channels):
        layers = []
        for _ in range(3 + self.compound_coef // 3):
            layers.append(nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU(True))
        layers.append(nn.Conv2d(self.out_channels, out_channels, kernel_size=3, stride=1, padding=1))
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        _, _, h, w = y.size()
        x_upsampled = F.upsample(x, [h, w], mode='bilinear', align_corners=True)
        return x_upsampled + y

    def forward(self, x):
        bs = x.size(0)
        c3, c4, c5 = self.backbone_net(x)
        p6 = self.conv6(c5)
        p7 = self.conv7(F.relu(p6))
        p5 = self.lateral_conn5(c5)
        p4 = self._upsample_add(p5, self.lateral_conn4(c4))
        p3 = self._upsample_add(p4, self.lateral_conn3(c3))
        p5 = self.conv5(p5)
        x = [p3, p4, p5, p6, p7]
        x = self.feature_maps(x)
        pred_cls = []
        pred_loc = []
        for feature in x:
            cls_pred = self.classifier(feature)
            loc_pred = self.regressor(feature)
            cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous()  # [b, c, h, w] => [b, h, w, c]
            cls_pred = cls_pred.view(bs, -1, self.num_classes)  # [b, h, w, c] => [b, h * w * anchors, num_classes]
            loc_pred = loc_pred.permute(0, 2, 3, 1).contiguous()
            loc_pred = loc_pred.view(bs, -1, 4)
            pred_cls.append(cls_pred)
            pred_loc.append(loc_pred)

        return torch.cat(pred_loc, 1), torch.cat(pred_cls, 1)
