"""
@author: Viet Nguyen <vn@signatrix.com>
"""
import os
from torch.utils.data import Dataset
from src.data_augmentation import *
from src.utils import generate_anchor_base, generate_anchors, xywh_to_tlbr, box_iou
from src.config import *
import pickle
import copy


class COCODataset(Dataset):
    def __init__(self, root_path="data/COCO", year="2014", mode="train", image_size=600, is_training=True):
        # if mode in ["train", "val"] and year in ["2014", "2015", "2017"]:
        if mode in ["valminusminival", "minival"]:
            self.image_path = os.path.join(root_path, "images","val2014")
        else:
            self.image_path = os.path.join(root_path, "images", "{}{}".format(mode, year))
        anno_path = os.path.join(root_path, "anno_pickle", "COCO_{}{}.pkl".format(mode, year))
        id_list_path = pickle.load(open(anno_path, "rb"))
        id_list_path = list(id_list_path.values())
        self.id_list_path = [item for item in id_list_path if item["objects"] != []]
        self.classes = COCO_CLASSES
        self.class_ids = COCO_CLASS_IDS
        self.image_size = image_size
        self.num_classes = len(self.classes)
        self.num_images = len(self.id_list_path)
        self.anchor_sizes = ANCHOR_SIZES
        self.aspect_ratios = ASPECT_RATIOS
        self.scale_ratios = SCALE_RATIOS
        anchor_wh = generate_anchor_base(self.anchor_sizes, self.aspect_ratios, self.scale_ratios)
        self.xy_wh_anchors = generate_anchors(self.anchor_sizes, anchor_wh, [self.image_size, self.image_size])
        self.tl_br_anchors = xywh_to_tlbr(self.xy_wh_anchors)
        self.is_training = is_training

    def __len__(self):
        return self.num_images

    def __getitem__(self, item):
        image_path = os.path.join(self.image_path, self.id_list_path[item]["file_name"])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255
        image[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
        image[:, :, 1] = (image[:, :, 1] - 0.456) / 0.224
        image[:, :, 2] = (image[:, :, 2] - 0.406) / 0.225
        objects = copy.deepcopy(self.id_list_path[item]["objects"])

        for idx in range(len(objects)):
            objects[idx][4] = self.class_ids.index(objects[idx][4])
        if self.is_training:
            transformations = Compose([VerticalFlip(), Crop(), Resize(self.image_size)])
        else:
            transformations = Compose([Resize(self.image_size)])
        image, tl_br_boxes, xy_wh_boxes, labels = transformations((image, objects))

        ious = box_iou(self.tl_br_anchors, tl_br_boxes)
        max_ious = np.max(ious, axis=1)
        max_ids = np.argmax(ious, axis=1)
        xy_wh_boxes = xy_wh_boxes[max_ids]
        loc_xy = (xy_wh_boxes[:, :2] - self.xy_wh_anchors[:, :2]) / self.xy_wh_anchors[:, 2:]
        loc_wh = np.log(xy_wh_boxes[:, 2:] / self.xy_wh_anchors[:, 2:])
        loc_wh = np.nan_to_num(loc_wh)
        gt_loc = np.concatenate((loc_xy, loc_wh), axis=1)
        gt_cls = labels[max_ids] + 1
        gt_cls[max_ious < 0.5] = 0
        ignore = (max_ious < 0.5) & (max_ious >= 0.4)
        gt_cls[ignore] = -1
        return np.transpose(image, (2, 0, 1)), gt_loc.astype(np.float32), gt_cls
