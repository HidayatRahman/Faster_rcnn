import os
import sys
import numpy as np
import xml.etree.ElementTree as ET
import tensorflow as tf

def parse_xmlfile(xml_file,h_scale=1,w_scale=1):
    try:
        if os.path.isfile(xml_file):
            tree = ET.parse(xml_file)
        else:
            print( "File does not Exist\n")
    except Exception:
        print('Failed to parse: ' + xml_file, file=sys.stderr)
        return None
    category=[]
    xmin=[]
    ymin=[]
    xmax=[]
    ymax=[]

    element = tree.getroot()
    element_objs = element.findall('object')
    element_filename = element.find('filename').text
    element_width = int(element.find('size').find('width').text)
    element_height = int(element.find('size').find('height').text)
    h_scale = h_scale/element_height
    w_scale = w_scale/element_width
    for element_obj in element_objs:
        category.append(element_obj.find('name').text)
        obj_bbox = element_obj.find('bndbox')
        xmin.append(float(round(float(obj_bbox.find('xmin').text)))*w_scale)
        ymin.append(float(round(float(obj_bbox.find('ymin').text)))*h_scale)
        xmax.append(float(round(float(obj_bbox.find('xmax').text)))*w_scale)
        ymax.append(float(round(float(obj_bbox.find('ymax').text)))*h_scale)
    
    gt_boxes=[list(box) for box in zip(ymin,xmin,ymax,xmax)]
    return category, np.asarray(gt_boxes, np.float)

def bbox_overlaps(valid_anchors, bbox):

    ious = np.zeros((len(valid_anchors), len(bbox)), dtype=np.float32)
    # ious.fill(0)
    # print(bbox)
    for num1, i in enumerate(valid_anchors):
        ya1, xa1, ya2, xa2 = i  
        anchor_area = (ya2 - ya1) * (xa2 - xa1)
        for num2, j in enumerate(bbox):
            yb1, xb1, yb2, xb2 = j
            box_area = (yb2- yb1) * (xb2 - xb1)
            inter_x1 = max([xb1, xa1])
            inter_y1 = max([yb1, ya1])
            inter_x2 = min([xb2, xa2])
            inter_y2 = min([yb2, ya2])
            if (inter_x1 < inter_x2) and (inter_y1 < inter_y2):
                iter_area = (inter_y2 - inter_y1) * (inter_x2 - inter_x1)
                iou = iter_area / (anchor_area+ box_area - iter_area)            
            else:
                iou = 0.

            ious[num1, num2] = iou
    return ious

def loc2bbox(src_bbox, loc):
    """Decode bounding boxes from bounding box offsets and scales.
    Given bounding box offsets and scales computed by
    :meth:`bbox2loc`, this function decodes the representation to
    coordinates in 2D image coordinates.
    Given scales and offsets :math:`t_y, t_x, t_h, t_w` and a bounding
    box whose center is :math:`(y, x) = p_y, p_x` and size :math:`p_h, p_w`,
    the decoded bounding box's center :math:`\\hat{g}_y`, :math:`\\hat{g}_x`
    and size :math:`\\hat{g}_h`, :math:`\\hat{g}_w` are calculated
    by the following formulas.
    * :math:`\\hat{g}_y = p_h t_y + p_y`
    * :math:`\\hat{g}_x = p_w t_x + p_x`
    * :math:`\\hat{g}_h = p_h \\exp(t_h)`
    * :math:`\\hat{g}_w = p_w \\exp(t_w)`
    The decoding formulas are used in works such as R-CNN [#]_.
    The output is same type as the type of the inputs.
    .. [#] Ross Girshick, Jeff Donahue, Trevor Darrell, Jitendra Malik. \
    Rich feature hierarchies for accurate object detection and semantic \
    segmentation. CVPR 2014.
    Args:
        src_bbox (array): A coordinates of bounding boxes.
            Its shape is :math:`(R, 4)`. These coordinates are
            :math:`p_{ymin}, p_{xmin}, p_{ymax}, p_{xmax}`.
        loc (array): An array with offsets and scales.
            The shapes of :obj:`src_bbox` and :obj:`loc` should be same.
            This contains values :math:`t_y, t_x, t_h, t_w`.
    Returns:
        array:
        Decoded bounding box coordinates. Its shape is :math:`(R, 4)`. \
        The second axis contains four values \
        :math:`\\hat{g}_{ymin}, \\hat{g}_{xmin},
        \\hat{g}_{ymax}, \\hat{g}_{xmax}`.
    """

    if src_bbox.shape[0] == 0:
        return xp.zeros((0, 4), dtype=loc.dtype)

    src_bbox = src_bbox.astype(src_bbox.dtype, copy=False)

    src_height = src_bbox[:, 2] - src_bbox[:, 0]
    src_width = src_bbox[:, 3] - src_bbox[:, 1]
    src_ctr_y = src_bbox[:, 0] + 0.5 * src_height
    src_ctr_x = src_bbox[:, 1] + 0.5 * src_width

    dy = loc[:, 0::4]
    dx = loc[:, 1::4]
    dh = loc[:, 2::4]
    dw = loc[:, 3::4]

    ctr_y = dy * src_height[:, xp.newaxis] + src_ctr_y[:, xp.newaxis]
    ctr_x = dx * src_width[:, xp.newaxis] + src_ctr_x[:, xp.newaxis]
    h = xp.exp(dh) * src_height[:, xp.newaxis]
    w = xp.exp(dw) * src_width[:, xp.newaxis]

    dst_bbox = xp.zeros(loc.shape, dtype=loc.dtype)
    dst_bbox[:, 0::4] = ctr_y - 0.5 * h
    dst_bbox[:, 1::4] = ctr_x - 0.5 * w
    dst_bbox[:, 2::4] = ctr_y + 0.5 * h
    dst_bbox[:, 3::4] = ctr_x + 0.5 * w

    return dst_bbox

def bbox2loc(src_bbox, dst_bbox):
    """Encodes the source and the destination bounding boxes to "loc".
    Given bounding boxes, this function computes offsets and scales
    to match the source bounding boxes to the target bounding boxes.
    Mathematcially, given a bounding box whose center is
    :math:`(y, x) = p_y, p_x` and
    size :math:`p_h, p_w` and the target bounding box whose center is
    :math:`g_y, g_x` and size :math:`g_h, g_w`, the offsets and scales
    :math:`t_y, t_x, t_h, t_w` can be computed by the following formulas.
    * :math:`t_y = \\frac{(g_y - p_y)} {p_h}`
    * :math:`t_x = \\frac{(g_x - p_x)} {p_w}`
    * :math:`t_h = \\log(\\frac{g_h} {p_h})`
    * :math:`t_w = \\log(\\frac{g_w} {p_w})`
    The output is same type as the type of the inputs.
    The encoding formulas are used in works such as R-CNN [#]_.
    .. [#] Ross Girshick, Jeff Donahue, Trevor Darrell, Jitendra Malik. \
    Rich feature hierarchies for accurate object detection and semantic \
    segmentation. CVPR 2014.
    Args:
        src_bbox (array): An image coordinate array whose shape is
            :math:`(R, 4)`. :math:`R` is the number of bounding boxes.
            These coordinates are
            :math:`p_{ymin}, p_{xmin}, p_{ymax}, p_{xmax}`.
        dst_bbox (array): An image coordinate array whose shape is
            :math:`(R, 4)`.
            These coordinates are
            :math:`g_{ymin}, g_{xmin}, g_{ymax}, g_{xmax}`.
    Returns:
        array:
        Bounding box offsets and scales from :obj:`src_bbox` \
        to :obj:`dst_bbox`. \
        This has shape :math:`(R, 4)`.
        The second axis contains four values :math:`t_y, t_x, t_h, t_w`.
    """

    height = src_bbox[:, 2] - src_bbox[:, 0]
    width = src_bbox[:, 3] - src_bbox[:, 1]
    ctr_y = src_bbox[:, 0] + 0.5 * height
    ctr_x = src_bbox[:, 1] + 0.5 * width

    base_height = dst_bbox[:, 2] - dst_bbox[:, 0]
    base_width = dst_bbox[:, 3] - dst_bbox[:, 1]
    base_ctr_y = dst_bbox[:, 0] + 0.5 * base_height
    base_ctr_x = dst_bbox[:, 1] + 0.5 * base_width

    eps = np.finfo(height.dtype).eps
    height = np.maximum(height, eps)
    width = np.maximum(width, eps)

    dy = (base_ctr_y - ctr_y) / height
    dx = (base_ctr_x - ctr_x) / width
    dh = np.log(base_height / height)
    dw = np.log(base_width / width)

    loc = np.vstack((dy, dx, dh, dw)).transpose()
    return loc

def bbox_iou(bbox_a, bbox_b):
    """Calculate the Intersection of Unions (IoUs) between bounding boxes.
    IoU is calculated as a ratio of area of the intersection
    and area of the union.
    This function accepts both :obj:`numpy.ndarray` and :obj:`cupy.ndarray` as
    inputs. Please note that both :obj:`bbox_a` and :obj:`bbox_b` need to be
    same type.
    The output is same type as the type of the inputs.
    Args:
        bbox_a (array): An array whose shape is :math:`(N, 4)`.
            :math:`N` is the number of bounding boxes.
            The dtype should be :obj:`numpy.float32`.
        bbox_b (array): An array similar to :obj:`bbox_a`,
            whose shape is :math:`(K, 4)`.
            The dtype should be :obj:`numpy.float32`.
    Returns:
        array:
        An array whose shape is :math:`(N, K)`. \
        An element at index :math:`(n, k)` contains IoUs between \
        :math:`n` th bounding box in :obj:`bbox_a` and :math:`k` th bounding \
        box in :obj:`bbox_b`.
    """
    if bbox_a.shape[1] != 4 or bbox_b.shape[1] != 4:
        raise IndexError

    # top left
    tl = np.maximum(bbox_a[:, None, :2], bbox_b[:, :2])
    # bottom right
    br = np.minimum(bbox_a[:, None, 2:], bbox_b[:, 2:])

    area_i = np.prod(br - tl, axis=2) * (tl < br).all(axis=2)
    area_a = np.prod(bbox_a[:, 2:] - bbox_a[:, :2], axis=1)
    area_b = np.prod(bbox_b[:, 2:] - bbox_b[:, :2], axis=1)
    return area_i / (area_a[:, None] + area_b - area_i)

def _unmap(data, count, index, fill=0):
    # Unmap a subset of item (data) back to the original set of items (of
    # size count)

    if len(data.shape) == 1:
        ret = np.empty((count,), dtype=data.dtype)
        ret.fill(fill)
        ret[index] = data
    else:
        ret = np.empty((count,) + data.shape[1:], dtype=data.dtype)
        ret.fill(fill)
        ret[index, :] = data
    return ret

def _get_inside_index(anchor, H, W):
    # Calc indicies of anchors which are located completely inside of the image
    # whose size is speficied.
    index_inside = np.where(
        (anchor[:, 0] >= 0) &
        (anchor[:, 1] >= 0) &
        (anchor[:, 2] <= H) &
        (anchor[:, 3] <= W)
    )[0]
    return index_inside

def generate_base_anchors(ratios, scales):
    Stride =16 # Stride this will remain fixed
    Img_size = 800 # Image size should be 800 X 800
    feature_size = round(800/16)

    # Generating Centers 
    index = 0
    ctr_x = np.arange(Stride, (feature_size+1) * Stride, Stride)
    ctr_y = np.arange(Stride, (feature_size+1) * Stride, Stride)
    ctr = np.zeros((len(ctr_x) * len(ctr_y), 2), dtype=np.float)
    for x in range(len(ctr_x)):
        for y in range(len(ctr_y)):
            ctr[index, 1] = ctr_x[x] - Stride/2
            ctr[index, 0] = ctr_y[y] - Stride/2
            index +=1
    # bbox format should be y1, x1, y2, x2
    num_anchors =len(ratios)*len(scales)
    anchors =np.zeros(((feature_size * feature_size* num_anchors),4), dtype =np.float)

    # Anchor Generation at each center 
    index = 0
    for c in ctr:
        center_x,center_y = c
        for i in range(len(ratios)):
            for j in range( len(scales)):
                h = Stride*scales[j]*np.sqrt(ratios[i])
                w = Stride*scales[j]*np.sqrt(1./ratios[i])
                
                anchors[index, 0] = center_y - h/2
                anchors[index, 1] = center_x - w/2
                anchors[index, 2] = center_y + h/2
                anchors[index, 3] = center_x + w/2
                index+=1
    return anchors


def gen_data(data_path, type, batdh_size):
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
        category, bbox = parse_xmlfile(os.path.join(Annotations_path,line+'.xml'))
        img_path = os.path.join(Images_path, line+'.jpg')
        categories.append(category)
        imgs.append(img_path)
        gt_bbox.append(bbox)
        
    # return categories, gt_bbox, imgs
    dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(batch_size).repeat()