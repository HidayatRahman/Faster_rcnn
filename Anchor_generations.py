import numpy as np
import cv2
import os
import math
import tensorflow as tf
from utils import parse_xmlfile, bbox_overlaps, bbox2loc, bbox_iou, _get_inside_index, _unmap, generate_base_anchors



# resize_width = 800
# resize_height = 800

class AnchorTargetCreator(object):
    """Assign the ground truth bounding boxes to anchors.
    Assigns the ground truth bounding boxes to anchors for training Region
    Proposal Networks introduced in Faster R-CNN [#]_.
    Offsets and scales to match anchors to the ground truth are
    calculated using the encoding scheme of
    :func:`model.utils.bbox_tools.bbox2loc`.
    .. [#] Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun. \
    Faster R-CNN: Towards Real-Time Object Detection with \
    Region Proposal Networks. NIPS 2015.
    Args:
        n_sample (int): The number of regions to produce.
        pos_iou_thresh (float): Anchors with IoU above this
            threshold will be assigned as positive.
        neg_iou_thresh (float): Anchors with IoU below this
            threshold will be assigned as negative.
        pos_ratio (float): Ratio of positive regions in the
            sampled regions.
    """

    def __init__(self,
                 n_sample=256,
                 pos_iou_thresh=0.7, neg_iou_thresh=0.3,
                 pos_ratio=0.5):
        self.n_sample = n_sample
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh = neg_iou_thresh
        self.pos_ratio = pos_ratio

    def __call__(self, base_anchors, bbox , Img_height, img_width):


        """Assign ground truth supervision to sampled subset of anchors.
        Types of input arrays and output arrays are same.
        Here are notations.
        * :math:`S` is the number of anchors.
        * :math:`R` is the number of bounding boxes.
        Args:
            bbox (array): Coordinates of bounding boxes. Its shape is
                :math:`(R, 4)`.
            anchor (array): Coordinates of anchors. Its shape is
                :math:`(S, 4)`.
            img_size (tuple of ints): A tuple :obj:`H, W`, which
                is a tuple of height and width of an image.
        Returns:
            (array, array):
            #NOTE: it's scale not only  offset
            * **loc**: Offsets and scales to match the anchors to \
                the ground truth bounding boxes. Its shape is :math:`(S, 4)`.
            * **label**: Labels of anchors with values \
                :obj:`(1=positive, 0=negative, -1=ignore)`. Its shape \
                is :math:`(S,)`.
        """

        anchor = base_anchors
        num_anchors = len(anchor)
        inside_index = _get_inside_index(anchor, Img_height, img_width)
        anchor = anchor[inside_index]
        argmax_ious, label = self._create_label(inside_index, anchor, bbox)
        loc = bbox2loc(anchor, bbox[argmax_ious])
        label = _unmap(label, num_anchors, inside_index, fill= -1)
        loc = _unmap(loc, num_anchors, inside_index, fill= 0)
        label =np.reshape(label, [-1,1])
        
        return loc, label


    def _create_label(self, inside_index, anchor, bbox):
        # label: 1 is positive, 0 is negative, -1 is dont care
        label = np.empty((len(inside_index),), dtype=np.int32)
        label.fill(-1)

        argmax_ious, max_ious, gt_argmax_ious = \
            self._calc_ious(anchor, bbox, inside_index)

        # assign negative labels first so that positive labels can clobber them
        label[max_ious < self.neg_iou_thresh] = 0

        # positive label: for each gt, anchor with highest iou
        label[gt_argmax_ious] = 1

        # positive label: above threshold IOU
        label[max_ious >= self.pos_iou_thresh] = 1

        # subsample positive labels if we have too many
        n_pos = int(self.pos_ratio * self.n_sample)
        pos_index = np.where(label == 1)[0]
        if len(pos_index) > n_pos:
            disable_index = np.random.choice(
                pos_index, size=(len(pos_index) - n_pos), replace=False)
            label[disable_index] = -1

        # subsample negative labels if we have too many
        n_neg = self.n_sample - np.sum(label == 1)
        neg_index = np.where(label == 0)[0]
        if len(neg_index) > n_neg:
            disable_index = np.random.choice(
                neg_index, size=(len(neg_index) - n_neg), replace=False)
            label[disable_index] = -1

        return argmax_ious, label

    def _calc_ious(self, anchor, bbox, inside_index):
        # ious between the anchors and the gt boxes
        ious = bbox_iou(anchor, bbox)
        argmax_ious = ious.argmax(axis=1)
        max_ious = ious[np.arange(len(inside_index)), argmax_ious]
        gt_argmax_ious = ious.argmax(axis=0)
        gt_max_ious = ious[gt_argmax_ious, np.arange(ious.shape[1])]
        gt_argmax_ious = np.where(ious == gt_max_ious)[0]

        return argmax_ious, max_ious, gt_argmax_ious
    

def gen_data(data_path, resize_width, resize_height, type):
    if not os.path.exists(data_path):
        print("dataset path {0} doesn't exist".format(data_path))
        exit(-1)
    Annotations_path = os.path.join(data_path, "Annotations")
    Images_path = os.path.join(data_path, "JPEGImages")
    train_path= os.path.join(data_path, "ImageSets","Main",type+".txt")
    file_ = open(train_path)
    categories =[]
    imgs = []
    gt_bbox = []
    for line in file_.readlines():
        line = line.strip()
        category, bbox = parse_xmlfile(os.path.join(Annotations_path,line+'.xml'), h_scale=resize_height, w_scale= resize_width)
        img_path = os.path.join(Images_path, line+'.jpg')
        if len(bbox) != 0 :
            categories.append(category)
            imgs.append(img_path)
            gt_bbox.append(bbox)
    return categories, gt_bbox, imgs


def process_image(img_path, img_width, Img_height):
    x_img= cv2.imread(img_path)
    x_img = cv2.resize(x_img, (img_width, Img_height), interpolation=cv2.INTER_CUBIC)
    x_img = x_img[:,:, (2, 1, 0)]  # BGR -> RGB
    x_img = x_img.astype(np.float32)
    x_img[:, :, 0] -= 103.939
    x_img[:, :, 1] -= 116.779
    x_img[:, :, 2] -= 123.68
    x_img = np.transpose(x_img, (2, 0, 1))
    x_img = np.expand_dims(x_img, axis=0)
    x_img = np.transpose(x_img, (0, 2, 3, 1))
    return x_img


def gen_datapipeline(base_anchors, labels, bbox, img_path, img_width, Img_height):
    Anc =AnchorTargetCreator()
    while True:
        for img,box in zip(img_path,bbox):
            anchor_bbox, anchor_label =Anc(base_anchors, box , Img_height, img_width)
            img = process_image(img ,img_width,Img_height)
            yield anchor_bbox, anchor_label, img