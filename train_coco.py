"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from src.coco_dataset import COCODataset
from src.utils import get_efficientdet
from src.focal_loss import FocalLoss
from tensorboardX import SummaryWriter
import shutil


def get_args():
    parser = argparse.ArgumentParser("EfficientDet: Scalable and Efficient Object Detection implementation by Signatrix GmbH")
    parser.add_argument("--image_size", type=int, default=512, help="The common width and height for all images")
    parser.add_argument("--batch_size", type=int, default=5, help="The number of images per batch")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--decay", type=float, default=1e-4)
    parser.add_argument('--alpha', type=float, default=0.25)
    parser.add_argument('--gamma', type=float, default=2.0)
    parser.add_argument("--num_epoches", type=int, default=100)
    parser.add_argument("--test_interval", type=int, default=1, help="Number of epoches between testing phases")
    parser.add_argument("--es_min_delta", type=float, default=0.0,
                        help="Early stopping's parameter: minimum change loss to qualify as an improvement")
    parser.add_argument("--es_patience", type=int, default=0,
                        help="Early stopping's parameter: number of epochs with no improvement after which training will be stopped. Set to 0 to disable this technique.")
    parser.add_argument("--year", type=str, default="2014", choices=["2014", "2017"],
                        help="The year of dataset (2014 or 2017)")
    parser.add_argument("--data_path", type=str, default="data/COCO", help="the root folder of dataset")
    parser.add_argument("--log_path", type=str, default="tensorboard/signatrix_efficientdet_coco")
    parser.add_argument("--saved_path", type=str, default="trained_models")

    args = parser.parse_args()
    return args


def train(opt):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)

    learning_rate_schedule = {"0": opt.lr, "5": opt.lr / 10,
                              "10": opt.lr / 100}
    
    training_params = {"batch_size": opt.batch_size,
                       "shuffle": True,
                       "drop_last": True,
                       "num_workers": 12}

    test_params = {"batch_size": opt.batch_size,
                   "shuffle": False,
                   "drop_last": False,
                   "num_workers": 12}

    if opt.year == "2014":
        training_set_1 = COCODataset(opt.data_path, opt.year, "train", opt.image_size)
        training_set_2 = COCODataset(opt.data_path, opt.year, "valminusminival", opt.image_size)
        training_set = ConcatDataset([training_set_1, training_set_2])
        training_set.num_classes = training_set_1.num_classes
        training_generator = DataLoader(training_set, **training_params)

        test_set = COCODataset(opt.data_path, opt.year, "minival", opt.image_size, is_training=False)
        test_generator = DataLoader(test_set, **test_params)
    else:  # 2017
        training_set = COCODataset(opt.data_path, opt.year, "train", opt.image_size)
        training_generator = DataLoader(training_set, **training_params)

        test_set = COCODataset(opt.data_path, opt.year, "val", opt.image_size, is_training=False)
        test_generator = DataLoader(test_set, **test_params)

    model = get_efficientdet(efficient_backbone="b0", num_classes=training_set.num_classes)

    log_path = os.path.join(opt.log_path, "{}".format(opt.year))
    if os.path.isdir(log_path):
        shutil.rmtree(log_path)
    os.makedirs(log_path)
    writer = SummaryWriter(log_path)
    if torch.cuda.is_available():
        writer.add_graph(model.cpu(), torch.rand(opt.batch_size, 3, opt.image_size, opt.image_size))
        model = nn.DataParallel(model)
        model.cuda()
    else:
        writer.add_graph(model, torch.rand(opt.batch_size, 3, opt.image_size, opt.image_size))

    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), opt.lr, momentum=opt.momentum,
                                weight_decay=opt.decay)
    criterion = FocalLoss(opt.alpha, opt.gamma, training_set.num_classes)
    best_loss = 1e5
    best_epoch = 0
    model.train()
    num_iter_per_epoch = len(training_generator)
    for epoch in range(opt.num_epoches):
        if str(epoch) in learning_rate_schedule.keys():
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate_schedule[str(epoch)]
        for iter, batch in enumerate(training_generator):
            images, gt_loc, gt_cls = batch
            if torch.cuda.is_available():
                images = images.cuda()
                gt_loc = gt_loc.cuda()
                gt_cls = gt_cls.cuda()

            pos_boxes = gt_cls > 0
            num_pos_boxes = torch.sum(pos_boxes).float()
            if num_pos_boxes == 0:
                print("There are no positive boxes")
                continue
            optimizer.zero_grad()
            pred_loc, pred_cls = model(images)
            mask = pos_boxes.unsqueeze(2).expand_as(pred_loc)
            pred_loc = pred_loc[mask].view(-1, 4)
            gt_loc = gt_loc[mask].view(-1, 4)
            loc_loss = F.smooth_l1_loss(pred_loc, gt_loc.float(), size_average=False) / num_pos_boxes

            boxes = gt_cls > -1
            mask = boxes.unsqueeze(2).expand_as(pred_cls)
            pred_cls = pred_cls[mask].view(-1, training_set.num_classes)
            focal_loss = criterion(pred_cls, gt_cls[boxes]) / num_pos_boxes
            loss = loc_loss + focal_loss
            loss.backward()
            optimizer.step()
            print(
                "Epoch: {}/{}, Iteration: {}/{}, Lr: {}, Loss:{} (Loc_loss:{} Cls_loss (focal loss):{})".format(
                    epoch + 1,
                    opt.num_epoches,
                    iter + 1,
                    num_iter_per_epoch,
                    optimizer.param_groups[0]['lr'],
                    loss,
                    loc_loss,
                    focal_loss))
            writer.add_scalar('Train/Total_loss', loss, epoch * num_iter_per_epoch + iter)
            writer.add_scalar('Train/Loc_loss', loc_loss, epoch * num_iter_per_epoch + iter)
            writer.add_scalar('Train/Cls_loss (focal loss)', focal_loss, epoch * num_iter_per_epoch + iter)
        if epoch % opt.test_interval == 0:
            model.eval()
            loss_ls = []
            loss_loc_ls = []
            loss_cls_ls = []
            for te_iter, te_batch in enumerate(test_generator):
                te_images, te_gt_loc, te_gt_cls = te_batch
                num_sample = te_gt_loc.size()[0]
                if torch.cuda.is_available():
                    te_images = te_images.cuda()
                    te_gt_loc = te_gt_loc.cuda()
                    te_gt_cls = te_gt_cls.cuda()
                te_pos_boxes = te_gt_cls > 0
                num_te_pos_boxes = torch.sum(te_pos_boxes).float()
                if num_te_pos_boxes == 0:
                    continue
                with torch.no_grad():
                    te_pred_loc, te_pred_cls = model(te_images)
                    te_mask = te_pos_boxes.unsqueeze(2).expand_as(te_pred_loc)
                    te_pred_loc = te_pred_loc[te_mask].view(-1, 4)
                    te_gt_loc = te_gt_loc[te_mask].view(-1, 4)
                    te_loc_loss = F.smooth_l1_loss(te_pred_loc, te_gt_loc.float(),
                                                   size_average=False) / num_te_pos_boxes

                    te_boxes = te_gt_cls > -1
                    te_mask = te_boxes.unsqueeze(2).expand_as(te_pred_cls)
                    te_pred_cls = te_pred_cls[te_mask].view(-1, training_set.num_classes)
                    te_focal_loss = criterion(te_pred_cls, te_gt_cls[te_boxes]) / num_te_pos_boxes
                    te_loss = te_loc_loss + te_focal_loss

                loss_ls.append(te_loss * num_sample)
                loss_loc_ls.append(te_loc_loss * num_sample)
                loss_cls_ls.append(te_focal_loss * num_sample)

            loss_ls = sum(loss_ls) / test_set.__len__()
            loss_loc_ls = sum(loss_loc_ls) / test_set.__len__()
            loss_cls_ls = sum(loss_cls_ls) / test_set.__len__()

            print("Epoch: {}/{}, Lr: {}, Loss:{} (Loc_loss:{} Cls_loss (focal loss):{})".format(
                epoch + 1,
                opt.num_epoches,
                optimizer.param_groups[0]['lr'],
                loss_ls,
                loss_loc_ls,
                loss_cls_ls))
            writer.add_scalar('Test/Total_loss', loss_ls, epoch)
            writer.add_scalar('Test/Loc_loss', loss_loc_ls, epoch)
            writer.add_scalar('Test/Cls_loss (focal loss)', loss_cls_ls, epoch)
            model.train()
            if loss_ls + opt.es_min_delta < best_loss:
                best_loss = loss_ls
                best_epoch = epoch
                torch.save(model, opt.saved_path + os.sep + "signatrix_efficientdet_coco.pth")
            # Early stopping
            if epoch - best_epoch > opt.es_patience > 0:
                print("Stop training at epoch {}. The lowest loss achieved is {}".format(epoch, loss_ls))
                break
    writer.close()


if __name__ == "__main__":
    opt = get_args()
    train(opt)
