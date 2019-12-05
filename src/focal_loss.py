"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma, num_classes):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes


    # # Standard focal loss
    # def forward(self, pred, target):
    #     t = torch.eye(self.num_classes + 1)[target][:, 1:]
    #     if torch.cuda.is_available():
    #         t = t.cuda()
    #
    #     p = pred.sigmoid()
    #     pt = p * t + (1 - p) * (1 - t)  # pt = p if t > 0 else 1-p
    #     w = self.alpha * t + (1 - self.alpha) * (1 - t)  # w = alpha if t > 0 else 1-alpha
    #     w = w * (1 - pt).pow(self.gamma)
    #
    #     return F.binary_cross_entropy_with_logits(pred, t, w, size_average=False)

    # Alternative focal loss
    def forward(self, pred, target):
        p = torch.eye(self.num_classes+1)[target][:, 1:]
        if torch.cuda.is_available():
            p = p.cuda()
        pt = pred * (2 * p - 1)  # Eq.2 in the paper
        pt = (2 * pt + 1).sigmoid()
        w = self.alpha * p + (1 - self.alpha) * (1 - p)
        loss = -w * pt.log() / 2
        return loss.sum()



