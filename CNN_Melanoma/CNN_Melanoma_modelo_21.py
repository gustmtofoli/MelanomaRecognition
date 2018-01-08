# from mistune import preprocessing

from sklearn import metrics as sk
from sklearn.metrics import cohen_kappa_score
import cv2
import os
import matplotlib.pyplot as plt
import math
import InfoMessages as Info
import ImageUtils as imageUtils
# from Preprocessing import load_images


import Preprocessing as preprocessing


import tkinter as tk
from tkinter import filedialog
import numpy as np
import tensorflow as tf
import pandas as pd
from keras.utils import np_utils
import CSVUtils as csvUtils
import time
import Front as front
# import datumio as dtd
# import datagen as dtd
# # import datumio.datagen as dtd
# import datumio.datumio.datagen as dtd

path_Train = front.chooseFolder("TRAIN") # choose train folder
X_Train = imageUtils.loadImages(path_Train, "TRAIN") # load train images
X_Train = preprocessing.removeBackgrounds(X_Train)

path_Test = front.chooseFolder("TEST") # choose test folder
X_Test = imageUtils.loadImages(path_Test, "TEST") # load test images
X_Test = preprocessing.removeBackgrounds(X_Test)

path_Train_Ground_Truth = "/home/gustavo/Documentos/DATA_BASE/ISIC-2017_Training_Part3_GroundTruth.csv"

path_Test_Ground_Truth = "/home/gustavo/Documentos/DATA_BASE/ISIC-2017_Test_v2_Part3_GroundTruth.csv"

y_Train = csvUtils.getGroundTruthCSV(path_Train_Ground_Truth, n = 2000)

y_Test = csvUtils.getGroundTruthCSV(path_Test_Ground_Truth, n = 600)


def center_normalize(x):
    return (x-np.mean(x))/np.std(x)


X_Train = center_normalize(X_Train)
X_Test = center_normalize(X_Test)

sess = tf.Session()
img_size = 64 # 64
num_channels = 3
num_classes = 2
img_size_flat = img_size*img_size*num_channels

with tf.name_scope("Input"):
    x = tf.placeholder(tf.float32, shape = [None,img_size_flat], name='x')
    x = tf.reshape(x, [-1, img_size, img_size, num_channels])

y_true=tf.placeholder(tf.float32,shape=[None,num_classes],name='y_true')
y_true_cls=tf.argmax(y_true,dimension=1)
dropout=0.75
with tf.name_scope('dropout'):
    keep_prob=tf.placeholder(tf.float32)
    tf.summary.scalar('dropout_keep_probability', keep_prob)

filter_size1 = 3 # 3
num_filters1 = 32 # 16
filter_size2 = 3 # 3
num_filters2 = 32 # 16
filter_size3 = 3
num_filters3 = 32
filter_size4 = 3
num_filters4 = 32
filter_size5 = 3
num_filters5 = 32
# fc_size= 64 # 32

def weights_initilization(shape):
    return tf.Variable(tf.random_normal(shape,stddev=0.05))


def biases_initilization(length):
    return tf.Variable(tf.random_normal(shape=[length]))

def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)



def conv_layer(input,input_channels,filter_size,num_filters,layer_name,use_pooling=True):
    with tf.name_scope(layer_name):
        print("inside conv_layer")
        shape=[filter_size,filter_size,input_channels,num_filters]
        with tf.name_scope('Weights'):
            weights=weights_initilization(shape)
            variable_summaries(weights)
        with tf.name_scope("Biases"):
            biases=biases_initilization(length=num_filters)
            variable_summaries(biases)
        conv_layer=tf.nn.conv2d(input=input,filter=weights,strides=[1,1,1,1],padding='SAME')
        print("\tlayer conv: ", conv_layer.shape)
        conv_layer+=biases

        if use_pooling:
            pooling_layer=tf.nn.max_pool(value=conv_layer,ksize=[1,2,2,1],
                                         strides=[1,2,2,1],padding='SAME')
            print("\tlayer after pooling: ", pooling_layer.shape)

            Activation_layer=tf.nn.relu(pooling_layer)
            print("\tlayer after relu: ", Activation_layer.shape)

    return Activation_layer,weights

def fc_layer(input,num_inputs,num_outputs,layer_name,keep_prob,use_relu=True):
    print("inside fc_layer")
    with tf.name_scope(layer_name):
        with tf.name_scope("Weights_FC"):
            weights=weights_initilization(shape=[num_inputs,num_outputs])
            variable_summaries(weights)
        with tf.name_scope("biases_FC"):
            biases=biases_initilization(length=num_outputs)
            variable_summaries(biases)
        with tf.name_scope("Wx_plus_b"):
            layer=tf.matmul(input,weights)+biases
            print("\tlayer after matmul: ", layer.shape)
            tf.summary.histogram('Preactivation',layer)
        with tf.name_scope('Activation'):
            if use_relu:
                layer=tf.nn.relu(layer)
                print("\tlayer after relu: ", layer.shape)
                tf.summary.histogram('activation',layer)
        with tf.name_scope('dropout'):
            layer=tf.nn.dropout(layer,keep_prob)
            print("\tkeeo_prob: ", keep_prob)
            print("\tlayer fc: ", layer.shape)
    return layer,weights

def flatten_layer(layer,layer_name):
    print("inside faltten_layer")
    with tf.name_scope(layer_name):
        layer_shape=layer.get_shape()
        print("\tlayer: ", layer_shape)
        num_features=layer_shape[1:4].num_elements()
        print("\tnum_features: ", num_features)
        layer_flat=tf.reshape(layer,[-1,num_features])
        print("\tlayer flat: ", layer_flat.shape)
    return layer_flat,num_features

layer_conv1,weights_conv1=conv_layer(input=x,input_channels=num_channels,filter_size=filter_size1,
                                     num_filters=num_filters1,layer_name='layer1',use_pooling=True)
layer_conv2,weights_conv2=conv_layer(input=layer_conv1,input_channels=num_filters2,filter_size=filter_size2,
                                     num_filters=num_filters2,use_pooling=True,layer_name='layer2')
layer_conv3,weights_conv3=conv_layer(input=layer_conv2,input_channels=num_filters3,filter_size=filter_size3,
                                     num_filters=num_filters3,use_pooling=True,layer_name='layer3')
# layer_conv4,weights_conv4=conv_layer(input=layer_conv3,input_channels=num_filters4,filter_size=filter_size4,
#                                      num_filters=num_filters4,use_pooling=True,layer_name='layer4')
# layer_conv5,weights_conv5=conv_layer(input=layer_conv4,input_channels=num_filters2,filter_size=filter_size3,
#                                      num_filters=num_filters2,use_pooling=True,layer_name='layer5')
# layer_conv6,weights_conv6=conv_layer(input=layer_conv5,input_channels=num_filters2,filter_size=filter_size3,
#                                      num_filters=num_filters2,use_pooling=True,layer_name='layer6')
# layer_conv7,weights_conv7=conv_layer(input=layer_conv6,input_channels=num_filters2,filter_size=filter_size3,
#                                      num_filters=num_filters2,use_pooling=True,layer_name='layer7')

# layer_conv4,weights_conv4=conv_layer(input=layer_conv3,input_channels=num_filters3,filter_size=filter_size2,
#                                      num_filters=num_filters2,use_pooling=True,layer_name='layer3')
layer_flat,num_features=flatten_layer(layer_conv3,'layer8')

layer_fc1,weights_FC1=fc_layer(input=layer_flat,num_inputs=num_features,num_outputs=num_classes,keep_prob=keep_prob,layer_name='layer9',use_relu=True)


y_pred=tf.nn.softmax(layer_fc1)
print("y_pred: ", y_pred.shape)
y_pred_cls=tf.argmax(y_pred,dimension=1)
print("y_pred_cls: ", y_pred_cls.shape)
with tf.name_scope('cross_entropy'):
    cross_entropy=tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc1,labels=y_true)
with tf.name_scope("total"):
    cost=tf.reduce_mean(cross_entropy)
tf.summary.scalar('cost',cost)


with tf.name_scope('train'):
    optimizer=tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cross_entropy)
with tf.name_scope("Accuracy"):
    with tf.name_scope('correct_prediction'):
        correct_prediction=tf.equal(y_pred_cls,y_true_cls)
    with tf.name_scope('accuracy'):
        accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
tf.summary.scalar('accuracy', accuracy)
#tf.summary.scalar('correct_prediction',correct_prediction)


batch_size=100
merged=tf.summary.merge_all()
# train_writer=tf.summary.FileWriter(logs_path+'/train',sess.graph)
# test_writer=tf.summary.FileWriter(logdir=logs_path+"/test")
sess.run(tf.global_variables_initializer())
with sess.as_default():
    tf.global_variables_initializer().run()
train_batch_size=batch_size


def print_progress(epoch, feed_dict_train, feed_dict_test, test_loss, info_acc_train):
    acc = sess.run(accuracy, feed_dict=feed_dict_train)
    test_acc = sess.run(accuracy, feed_dict=feed_dict_test)

    info_acc_train.write(str(acc))
    info_acc_train.write("\n")
    msg = "Epoch {0} --- Training Accuracy:{1:>6.1%},Testing Accuracy:{2:>6.1%}, Test Loss:{3:.3f}"
    print(msg.format(epoch + 1, acc, test_acc, test_loss))

# def print_progress_train(epoch, feed_dict_train):
#     acc = sess.run(accuracy, feed_dict=feed_dict_train)
#     # test_acc = sess.run(accuracy, feed_dict=feed_dict_test)
#     # train_acc = sess.run(acc)
#     msg = "Epoch {0} --- Training Accuracy:{1:>6.1%},Testing Accuracy:{2:>6.1%}, Test Loss:{3:.3f}"
#     print(msg.format(epoch + 1, acc))












total_iterations = 0
batch_size = 100
img_size_flat = img_size * img_size * num_channels


def optimize(num_iterations):
    train_size = 100
    succ_count = 0
    res = []
    info_time = open("info_time.txt","w")
    info_succ_percent = open("info_succ_percent.txt", "w")
    info_acc_train = open("info_acc_train.txt", "w")
    info_feed_dict_Train = open("info_feed_dict_Train", "w")
    global total_iterations
    best_test_loss = float("inf")
    # summary_writer = tf.summary.FileWriter(logs_path, graph=sess.graph)
    for i in range(total_iterations, total_iterations + num_iterations): # for i in range(total_iterations, total_iterations + num_iterations):
        start_time = time.time()
        train_idx = np.random.randint(X_Train.shape[0], size=train_size)

        batch_x_Train = X_Train[train_idx, :]
        # batch_x_Train = X_Train
        batch_y_Train = y_Train[train_idx]
        # batch_y_Train = y_Train

        test_idx = np.random.randint(X_Test.shape[0], size=1)
        batch_x_Test = X_Test[test_idx, :]
        # batch_x_Test = X_Test
        batch_y_Test = y_Test[test_idx]
        # batch_y_Test = y_Test
        train_size += 1
        # batch_x_Test=batch_x_Test.reshape(batch_size,img_size_flat)
        feed_dict_train = {x: batch_x_Train,
                           y_true: batch_y_Train, keep_prob: dropout}

        info_feed_dict_Train.write(str(info_feed_dict_Train))

        feed_dict_test = {x: batch_x_Test,
                          y_true: batch_y_Test, keep_prob: 1.0}

        test_acc_ = sess.run(accuracy, feed_dict=feed_dict_test)
        # print(test_acc_)
        if test_acc_ == 1.0:
            res.append(1)
            succ_count += 1
        else:
            res.append(0)

        _, c, summary = sess.run([optimizer, cost, merged], feed_dict=feed_dict_train)
        if i % int(X_Train.shape[0] / batch_size) == 0:
            test_loss = sess.run(cost, feed_dict=feed_dict_test)
            epoch = int(i / (X_Train.shape[0] / batch_size))
            # summary_writer.add_summary(summary, epoch * batch_size + i)
            # print_progress_train(epoch, feed_dict_train)
            # print("training number images: ", train_idx)
            # print("testing number images: ", test_idx)
            succ_percent = (succ_count / len(res)) * 100
            print(" => succ percent", succ_percent)
            info_succ_percent.write(str(succ_percent))
            info_succ_percent.write("\n")
            print(' => Read and process test data time: {} seconds'.format(round(time.time() - start_time, 2)))
            info_time.write(str(time.time() - start_time))
            info_time.write("\n")
            print_progress(epoch, feed_dict_train, feed_dict_test, test_loss, info_acc_train)
        total_iterations += num_iterations
    info_time.close()
    info_succ_percent.close()
    info_acc_train.close()
    info_feed_dict_Train.close()

optimize(num_iterations=1000)
