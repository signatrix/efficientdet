"""
@author: Viet Nguyen <vn@signatrix.com>
"""
import numpy as np
from math import sqrt
import torch.nn as nn
import torch
from torch.utils.data.dataloader import default_collate
from src.efficient_det import EfficientDet
from src.efficient_net import EfficientNet
from math import log


def generate_anchor_base(anchor_sizes, aspect_ratios, scale_ratios):
    anchor_wh = []
    for size in anchor_sizes:
        for ar in aspect_ratios:
            h = sqrt(size / ar)
            w = ar * h
            for sr in scale_ratios:
                anchor_w = w * sr
                anchor_h = h * sr
                anchor_wh.append([anchor_w, anchor_h])
    fms_num = len(anchor_sizes)

    return np.array(anchor_wh).reshape((fms_num, -1, 2))


def generate_anchors(anchor_sizes, anchor_wh, input_size):
    boxes = []
    input_size = np.array(input_size)
    fms_num = len(anchor_sizes)
    fms_size = [np.ceil(input_size / pow(2.0, i + 3)) for i in range(fms_num)]
    for i in range(fms_num):
        fm_size = fms_size[i]
        fm_w, fm_h = int(fm_size[0]), int(fm_size[1])
        scale = input_size / fm_size
        xy = makegrid(fm_w, fm_h) * scale + 0.5
        xy = np.repeat(xy.reshape((fm_h, fm_w, 1, 2)), 9, 2)
        wh = np.repeat(np.repeat(anchor_wh[i][None, None, :, :], fm_h, 0), fm_w, 1)
        boxes.append(np.concatenate((np.floor(xy), np.floor(wh)), axis=3).reshape((-1, 4)))
    boxes = np.concatenate(boxes, axis=0)
    return boxes


def makegrid(width, height):
    col_1 = np.tile(np.arange(0, width), height)
    col_2 = np.repeat(np.arange(0, height), width)
    result = np.stack((col_1, col_2), axis=-1)
    return result


def tlbr_to_xywh(boxes):
    tl = boxes[:, :2]
    br = boxes[:, 2:]
    return np.concatenate(((tl + br) / 2, br - tl), 1)

def xywh_to_tlbr(boxes):
    tl = boxes[:, :2]
    br = boxes[:, 2:]
    return np.concatenate((tl - br / 2, tl + br / 2), 1)

def box_iou(bbox_1, bbox_2):
    tl = np.maximum(bbox_1[:, None, :2], bbox_2[:, :2])
    br = np.minimum(bbox_1[:, None, 2:], bbox_2[:, 2:])
    area_i = np.prod(br - tl, axis=2) * (tl < br).all(axis=2)
    area_a = np.prod(bbox_1[:, 2:] - bbox_1[:, :2], axis=1)
    area_b = np.prod(bbox_2[:, 2:] - bbox_2[:, :2], axis=1)

    return area_i / (area_a[:, None] + area_b - area_i)

def box_nms(boxes, scores, threshold=0.5):

    xmin = boxes[:,0]
    ymin = boxes[:,1]
    xmax = boxes[:,2]
    ymax = boxes[:,3]

    areas = (xmax-xmin) * (ymax-ymin)
    _, order = torch.sort(scores, 0, True)

    keep = []
    prob = []
    while torch.numel(order) > 0:
        if torch.numel(order) > 1:
            i = order[0]
        else:
            i = order.item()
        keep.append(i)
        prob.append(scores[i])
        if torch.numel(order) == 1:
            break

        curr_xmin = xmin[order[1:]].clamp(min=xmin[i])
        curr_ymin = ymin[order[1:]].clamp(min=ymin[i])
        curr_xmax = xmax[order[1:]].clamp(max=xmax[i])
        curr_ymax = ymax[order[1:]].clamp(max=ymax[i])

        w = (curr_xmax-curr_xmin).clamp(min=0)
        h = (curr_ymax-curr_ymin).clamp(min=0)
        intersect = w*h

        iou = intersect / (areas[i] + areas[order[1:]] - intersect)

        indices = torch.squeeze(torch.nonzero(iou<=threshold))
        if torch.numel(indices) == 0:
            break
        order = order[indices+1]
    return torch.Tensor(keep).long(), torch.Tensor(prob)

def custom_collate_fn(batch):
    items = list(zip(*batch))
    items[0] = default_collate(items[0])
    items[1] = default_collate(items[1])
    items[2] = default_collate(items[2])
    return items


def get_efficientdet(efficient_backbone, num_classes):
    efficientnet = EfficientNet()
    efficientdet = EfficientDet(backbone_net=efficientnet, num_classes=num_classes)

    for m in efficientdet.classifier.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, mean=0, std=0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    for m in efficientdet.regressor.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, mean=0, std=0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    pi = 0.01
    nn.init.constant_(efficientdet.classifier[-1].bias, -log((1-pi)/pi))

    return efficientdet



if __name__ == '__main__':
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    model = get_efficientdet(True, 20)
    print (count_parameters(model))
    # dummy = torch.rand(2,3,512,512)
    # a,b = model(dummy)
    # print (a.shape)
    # print (b.shape)
