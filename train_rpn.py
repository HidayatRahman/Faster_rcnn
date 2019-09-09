import os
import sys
import math
import numpy as np
import cv2
path = os.path.join(os.getcwd() , "script")
sys.path.append(path)
import tensorflow as tf
from RPN import rpn, get_rpn_loss
from resnet import ResNet50
from Anchor_generations import AnchorTargetCreator, gen_datapipeline , gen_data
from utils import generate_base_anchors
import tensorflow.contrib.keras.api.keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--r",dest= "ratios", type=list, help = "ratios for anchor generation ", default=[.5,1,2])
parser.add_argument("--s", dest= "scales", type= list, help = "different scales for generating anchors", default=[8,16,32])
parser.add_argument("--anc", dest= "num_anchors", type= float, help=" pos iou threshold" , default=9)
parser.add_argument("--H", dest= "img_height", type= int, help = " Img height" , default= 800)
parser.add_argument("--W", dest= "img_width", type= int, help= "Img Width", default= 800)
parser.add_argument("--Bath_sz", dest="batch_size", type= int, help= " Bach Size", default= 4)
parser.add_argument("--data_path", dest= "data_path", type= str, help= " Dataset Path", default= "/pascalVoc/devkit/")

parser.add_argument("--model_path", dest= "model_path", type= str,  help= "Path to store Model", default= "./Checkpoint/rpn")

args = parser.parse_args()
Anchors = AnchorTargetCreator()
num_anchors = args.num_anchors
logfile = open("log.txt",'w')

train_cls, train_gt_boxes, train_imgs_path = gen_data(args.data_path,args.img_width, args.img_height , 'train')
val_cls, val_gt_boxes, val_imgs_path = gen_data(args.data_path, args.img_width, args.img_height, 'val')
base_anchors = generate_base_anchors(args.ratios, args.scales)
num_epochs = 120000

train_gen = gen_datapipeline(base_anchors, train_cls, train_gt_boxes, train_imgs_path, args.img_width, args.img_height)
val_gen = gen_datapipeline(base_anchors, val_cls, val_gt_boxes, val_imgs_path, args.img_width, args.img_height)

tf.reset_default_graph()

with tf.Session() as sess:
    anchor_boxes = tf.placeholder(dtype= tf.float32, shape= (None, 4))
    labels = tf.placeholder(dtype = tf.float32, shape= (None, 1))
    input_img = tf.placeholder(dtype =tf.float32, shape=(None,None,3))
    input1 , base_layers = ResNet50(input_img.shape)
    rpn_out = rpn(base_layers , num_anchors)
    losses = get_rpn_loss(anchor_boxes, rpn_out[1], labels, rpn_out[0])
    cls_loss= losses["cls_loss"]
    cls_accuracy = losses["cls_accuracy"]
    cls_pred = losses["cls_pred"]
    cls_labels = losses["cls_labels"]
    reg_loss = losses["reg_loss"]
    reg_pred = losses["reg_pred"]
    reg_labels = losses["reg_groudtruth"]

    # rpn_cls_loss, Acc = rpn_cls_loss(labels , rpn_out[0])
    # rpn_bbox_loss= smooth_l1_loss(anchor_boxes , rpn_out[1])

    loss = cls_loss + reg_loss
    optimizer = tf.train.AdamOptimizer(learning_rate=0.009).minimize(loss)
    # cls_op = tf.train.AdamOptimizer(learning_rate=0.009).minimize(cls_loss)
    # reg_op = tf.train.AdamOptimizer(learning_rate=0.009).minimize(reg_loss)

    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    total_loss =0
    Accuracy =0
    classifier_loss =0
    regressor_loss=0
    for i in range(1, num_epochs):
        bbox, anchor_labels, img = next(train_gen)
        _, current_loss, reg_trainloss, cls_trainloss, train_acc = sess.run([optimizer, loss, reg_loss, cls_loss, cls_accuracy], \
            feed_dict={anchor_boxes:bbox, labels: anchor_labels, input1:img, K.learning_phase(): 1})
        total_loss += current_loss
        classifier_loss+= cls_trainloss
        Accuracy +=train_acc
        regressor_loss += reg_trainloss
        if i%100==0:
            print("Total loss: {0}, rpn_cls_loss: {1}, rpn_reg_loss: {2}, Accuracy {3}".format(total_loss/100, classifier_loss/100, regressor_loss/100, Accuracy/100))
            logfile.write("Total loss: {0}, rpn_cls_loss: {1}, rpn_reg_loss: {2}, Accuracy {3}\n".format(total_loss/100, classifier_loss/100, regressor_loss/100, Accuracy/100)
)
            classifier_loss = 0
            regressor_loss = 0
            Accuracy = 0
            total_loss = 0
            if i>1 and i% 10000 == 0:
                for j in range(1000):
                    bbox, anchor_labels, img = next(val_gen)
                    val_accuracy, val_clspred, val_clslabels, val_regpred, val_reglabels = sess.run(cls_accuracy, cls_pred, cls_labels, reg_pred, reg_labels, feed_dict={anchor_boxes:bbox, labels: anchor_labels, input1:img, K.learning_phase(): 0})
                    Accuracy +=val_accuracy
                    
                print("Validation Accuracy: {0}\n".format(Accuracy/1000))
                logfile.write("Validation Accuracy: {0}\n".format(Accuracy/1000))
                classifier_loss = 0
                regressor_loss = 0
                Accuracy = 0
                total_loss = 0

        if i>1 and i% 20000 == 0:
            saver.save(sess,os.path.join(args.model_path,"rpn") ,global_step=i)
            print("checkpoint saved to {0}".format(args.model_path))
            logfile.write("checkpoint saved to {0}\n".format(args.model_path))
            
logfile.close()

        