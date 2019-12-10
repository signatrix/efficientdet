"""
@author: Viet Nguyen <vn@signatrix.com>
"""
import torch
import torch.nn as nn
import numpy as np


class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma, num_classes):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes

    def forward(self, pred, target):
        p = torch.eye(self.num_classes+1)[target][:, 1:]
        if torch.cuda.is_available():
            p = p.cuda()
        pt = pred * (2 * p - 1)
        pt = (2 * pt + 1).sigmoid()
        w = self.alpha * p + (1 - self.alpha) * (1 - p)
        loss = -w * pt.log() / 2
        loss[loss == -np.inf] = 0
        return loss.sum()



