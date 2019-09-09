import numpy as np
from tensorflow.contrib.keras.api.keras import layers
from tensorflow.contrib.keras.api.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from tensorflow.contrib.keras.api.keras.models import Model, load_model
from tensorflow.contrib.keras.api.keras.preprocessing import image
from resnet_utils import *
from tensorflow.contrib.keras.api.keras.initializers import glorot_uniform
import tensorflow.contrib.keras.api.keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)



def rpn(base_layers, num_anchors):

    '''
    Region proposal network(RPN) is normally a three layer network, the first layer is the feature mat which is generated 
    from the feature extraction(In our case it's Resnet 50) The other two are classification and regression layer
    '''
    
    x = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='rpn_cls_score')(base_layers)
    class_layer = Conv2D(num_anchors, (1, 1), activation='sigmoid', kernel_initializer='uniform', name='rpn_cls_score')(x)
    bbox_layer = Conv2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero', name='rpn_bbox_pred')(x)

    return [class_layer, bbox_layer, base_layers]


def get_rpn_loss (y_true, y_pred, rpn_label, rpn_out):
    """Implements Smooth-L1 loss.
    y_true and y_pred are typically: [N, 4], but could be any shape.
    """

     # Rpn Classifier Loss

    rpn_cls_score = tf.reshape(rpn_out,[-1,1])
    rpn_labels = tf.reshape(tf.convert_to_tensor(rpn_label, tf.float32),[-1,1])
    rpn_select_indexes = tf.where(tf.not_equal(rpn_labels, -1))[:,0]
    logit_score = tf.reshape(tf.gather(rpn_cls_score, rpn_select_indexes),[-1 ,1])
    ground_truth_labels = tf.gather(rpn_labels, rpn_select_indexes)
    rpn_cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels= ground_truth_labels, logits= logit_score ))
    Acc = tf.equal(tf.equal(logit_score,1), tf.equal(ground_truth_labels,1))
    accuracy = tf.reduce_mean(tf.cast(Acc, "float"))

    # Rpn Regressor loss
    losses ={}
    y_pred = tf.reshape(y_pred,[-1, 4])
    y_true = tf.reshape(tf.convert_to_tensor(y_true, tf.float32), [-1,4])
    y_pred = tf.reshape(tf.gather(y_pred, rpn_select_indexes), [-1,4])
    y_true = tf.reshape(tf.gather(y_true, rpn_select_indexes), [-1,4])

    diff = K.abs(y_true - y_pred)
    less_than_one = K.cast(K.less(diff, 1.0), "float32")
    loss = tf.reduce_mean((less_than_one * 0.5 * diff**2) + (1 - less_than_one) * (diff - 0.5))

    losses["cls_loss"] = rpn_cross_entropy
    losses["cls_accuracy"] = accuracy
    losses["cls_pred"] = logit_score
    losses["cls_labels"] = ground_truth_labels
    losses["reg_loss"]=loss
    losses["reg_pred"] = y_pred
    losses["reg_groudtruth"]= y_true

    return losses


  






    