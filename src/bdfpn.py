"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BDFPN(nn.Module):
    def __init__(self, size=[40, 80, 192], feature_size=64, epsilon=0.0001, is_first=False):
        super(BDFPN, self).__init__()
        self.epsilon = epsilon
        self.is_first = is_first

        self.p3 = nn.Conv2d(size[0], feature_size, kernel_size=1, stride=1, padding=0)
        self.p4 = nn.Conv2d(size[1], feature_size, kernel_size=1, stride=1, padding=0)
        self.p5 = nn.Conv2d(size[2], feature_size, kernel_size=1, stride=1, padding=0)

        self.p6 = nn.Conv2d(size[2], feature_size, kernel_size=3, stride=2, padding=1)

        self.p7 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)
        )

        self.p3_td = nn.Sequential(nn.Conv2d(feature_size, feature_size, kernel_size=1),
                                   nn.BatchNorm2d(feature_size))
        self.p4_td = nn.Sequential(nn.Conv2d(feature_size, feature_size, kernel_size=1),
                                   nn.BatchNorm2d(feature_size))
        self.p5_td = nn.Sequential(nn.Conv2d(feature_size, feature_size, kernel_size=1),
                                   nn.BatchNorm2d(feature_size))
        self.p6_td = nn.Sequential(nn.Conv2d(feature_size, feature_size, kernel_size=1),
                                   nn.BatchNorm2d(feature_size))
        self.p7_td = nn.Sequential(nn.Conv2d(feature_size, feature_size, kernel_size=1),
                                   nn.BatchNorm2d(feature_size))

        self.p3_out = nn.Sequential(nn.Conv2d(feature_size, feature_size, kernel_size=1),
                                    nn.BatchNorm2d(feature_size))
        self.p4_out = nn.Sequential(nn.Conv2d(feature_size, feature_size, kernel_size=1),
                                    nn.BatchNorm2d(feature_size))
        self.p5_out = nn.Sequential(nn.Conv2d(feature_size, feature_size, kernel_size=1),
                                    nn.BatchNorm2d(feature_size))
        self.p6_out = nn.Sequential(nn.Conv2d(feature_size, feature_size, kernel_size=1),
                                    nn.BatchNorm2d(feature_size))
        self.p7_out = nn.Sequential(nn.Conv2d(feature_size, feature_size, kernel_size=1),
                                    nn.BatchNorm2d(feature_size))


        self.w1 = nn.Parameter(torch.Tensor(2))
        self.w1_relu = nn.ReLU()
        self.w2 = nn.Parameter(torch.Tensor(3))
        self.w2_relu = nn.ReLU()

    def forward(self, x):
        if self.is_first:
            c3, c4, c5 = x
            p3_x = self.p3(c3)
            p4_x = self.p4(c4)
            p5_x = self.p5(c5)
            p6_x = self.p6(c5)
            p7_x = self.p7(p6_x)
        else:
            p3_x, p4_x, p5_x, p6_x, p7_x = x

        w1 = self.w1_relu(self.w1)
        w1 /= torch.sum(w1, dim=0) + self.epsilon
        w2 = self.w2_relu(self.w2)
        w2 /= torch.sum(w2, dim=0) + self.epsilon

        p7_td = p7_x
        p6_td = self.p6_td(w1[0] * p6_x + w1[1] * F.interpolate(p7_x, scale_factor=2, mode='nearest'))
        p5_td = self.p5_td(w1[0] * p5_x + w1[1] * F.interpolate(p6_x, scale_factor=2, mode='nearest'))
        p4_td = self.p4_td(w1[0] * p4_x + w1[1] * F.interpolate(p5_x, scale_factor=2, mode='nearest'))
        p3_td = self.p3_td(w1[0] * p3_x + w1[1] * F.interpolate(p4_x, scale_factor=2, mode='nearest'))

        p7_out = self.p7_out(
            w2[0] * p7_x + w2[1] * p7_td + w2[2] * F.interpolate(p6_td, scale_factor=0.5, mode='nearest'))
        p6_out = self.p6_out(
            w2[0] * p6_x + w2[1] * p6_td + w2[2] * F.interpolate(p5_td, scale_factor=0.5, mode='nearest'))
        p5_out = self.p5_out(
            w2[0] * p5_x + w2[1] * p5_td + w2[2] * F.interpolate(p4_td, scale_factor=0.5, mode='nearest'))
        p4_out = self.p4_out(
            w2[0] * p4_x + w2[1] * p4_td + w2[2] * F.interpolate(p3_td, scale_factor=0.5, mode='nearest'))
        p3_out = p3_td

        return [p3_out, p4_out, p5_out, p6_out, p7_out]
