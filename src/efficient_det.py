"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import torch.nn as nn
import torch


class EfficientDet(nn.Module):
    def __init__(self, backbone_net, feature_net, num_anchors=9, num_classes=20, compound_coef=0):
        super(EfficientDet, self).__init__()
        self.compound_coef = compound_coef
        self.backbone_net = backbone_net
        self.feature_maps = nn.ModuleList([feature_net(is_first=True) if idx == 0 else feature_net(is_first=False) for idx in
                             range(max(2 + self.compound_coef, 8))])

        self.feature_size = [64, 88, 112, 160, 224, 288, 384, 384][self.compound_coef]
        self.num_classes = num_classes
        self.classifier = self._sub_network(self.num_classes * num_anchors)
        self.regressor = self._sub_network(4 * num_anchors)

    def _sub_network(self, out_channels):
        layers = []
        for _ in range(3 + self.compound_coef // 3):
            layers.append(nn.Conv2d(self.feature_size, self.feature_size, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU(True))
        layers.append(nn.Conv2d(self.feature_size, out_channels, kernel_size=3, stride=1, padding=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        bs = x.size(0)
        x = self.backbone_net(x)
        for feature_map in self.feature_maps:
            x = feature_map(x)
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
