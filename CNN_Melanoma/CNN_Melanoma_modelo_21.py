from sklearn import metrics as sk
from sklearn.metrics import cohen_kappa_score
# import datumio.datagen as dtd
import datumio.datumio.datagen as dtd

# import datumio as dtd

# import datagen as dtd

import cv2
import os
import matplotlib.pyplot as plt
import math
from Preprocessing import load_images
import InfoMessages as Info
import tkinter as tk
from tkinter import filedialog
import numpy as np
import tensorflow as tf
import pandas as pd
from keras.utils import np_utils
import tensorflow.contrib.slim as slim

tk.Tk().withdraw()
path_Train=filedialog.askdirectory()
print(path_Train)

X_Train=load_images(path_Train)

type(X_Train)

X_Train=np.array(X_Train)
X_Train.shape

tk.Tk().withdraw()
path_Test=filedialog.askdirectory()
print(path_Test)

X_Test=load_images(path_Test)
X_Test=np.array(X_Test)

data_Train=pd.read_csv("/home/gustavo/Documentos/DATA_BASE/ISIC-2017_Training_Part3_GroundTruth.csv")
data_Test=pd.read_csv("/home/gustavo/Documentos/DATA_BASE/ISIC-2017_Test_v2_Part3_GroundTruth.csv")

data_Train=data_Train.iloc[0:2000,1]
data_Test=data_Test.iloc[0:600,1]
print(data_Train.shape)
print(data_Test.shape)

y_Train=data_Train
y_Test=data_Test

y_Train=np.array(y_Train)
y_Test=np.array(y_Test)

y_Train = np_utils.to_categorical(data_Train)
y_Test = np_utils.to_categorical(data_Test)
print(y_Train.shape)
print(y_Test.shape)

tk.Tk().withdraw()
path_Validation=filedialog.askdirectory()
print(path_Validation)
X_Val=load_images(path_Validation)
X_Val=np.array(X_Val)
X_Val.shape

y_Val=pd.read_csv("/home/gustavo/Documentos/DATA_BASE/ISIC-2017_Validation_Part3_GroundTruth.csv")
Data_Validation=y_Val.iloc[0:150,1]
print(y_Val.shape)
print("\nData_Validation:\n", Data_Validation)
y_Val = np_utils.to_categorical(Data_Validation)
print(y_Val.shape)

def center_normalize(x):
    return (x-np.mean(x))/np.std(x)


X_Train=center_normalize(X_Train)
X_Test=center_normalize(X_Test)

sess=tf.Session()
img_size=64
num_channels=3
num_classes=2
img_size_flat=img_size*img_size*num_channels
with tf.name_scope("Input"):
    x=tf.placeholder(tf.float32,shape=[None,img_size_flat],name='x')
    x=tf.reshape(x,[-1,img_size,img_size,num_channels])
logs_path = '/tmp/tensorflow_logs/example'

y_true=tf.placeholder(tf.float32,shape=[None,num_classes],name='y_true')
y_true_cls=tf.argmax(y_true,dimension=1)
dropout=0.75
with tf.name_scope('dropout'):
    keep_prob=tf.placeholder(tf.float32)
    tf.summary.scalar('dropout_keep_probability', keep_prob)

filter_size1=3
num_filters1=16
filter_size2=3
num_filters2=16
fc_size=32

def weights_initilization(shape):
    return tf.Variable(tf.random_normal(shape,stddev=0.05))
def biases_initilization(length):
    return tf.Variable(tf.random_normal(shape=[length]))


def conv_layer(input,input_channels,filter_size,num_filters,layer_name,use_pooling=True):
    with tf.name_scope(layer_name):
        shape=[filter_size,filter_size,input_channels,num_filters]
        with tf.name_scope('Weights'):
            weights=weights_initilization(shape)
            variable_summaries(weights)
        with tf.name_scope("Biases"):
            biases=biases_initilization(length=num_filters)
            variable_summaries(biases)
        conv_layer=tf.nn.conv2d(input=input,filter=weights,strides=[1,1,1,1],padding='SAME')
        conv_layer+=biases

        if use_pooling:
            pooling_layer=tf.nn.max_pool(value=conv_layer,ksize=[1,2,2,1],
                                        strides=[1,2,2,1],padding='SAME')
        Activation_layer=tf.nn.relu(pooling_layer)
    return Activation_layer,weights

def flatten_layer(layer,layer_name):
    with tf.name_scope(layer_name):
        layer_shape=layer.get_shape()
        num_features=layer_shape[1:4].num_elements()
        layer_flat=tf.reshape(layer,[-1,num_features])
    return layer_flat,num_features

def fc_layer(input,num_inputs,num_outputs,layer_name,keep_prob,use_relu=True):
    with tf.name_scope(layer_name):
        with tf.name_scope("Weights_FC"):
            weights=weights_initilization(shape=[num_inputs,num_outputs])
            variable_summaries(weights)
        with tf.name_scope("biases_FC"):
            biases=biases_initilization(length=num_outputs)
            variable_summaries(biases)
        with tf.name_scope("Wx_plus_b"):
            layer=tf.matmul(input,weights)+biases
            tf.summary.histogram('Preactivation',layer)
        with tf.name_scope('Activation'):
            if use_relu:
                layer=tf.nn.relu(layer)
                tf.summary.histogram('activation',layer)
        with tf.name_scope('dropout'):
            layer=tf.nn.dropout(layer,keep_prob)
    return layer,weights

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


layer_conv1,weights_conv1=conv_layer(input=x,input_channels=num_channels,filter_size=filter_size1,
                                     num_filters=num_filters1,layer_name='layer1',use_pooling=True)
layer_conv2,weights_conv2=conv_layer(input=layer_conv1,input_channels=num_filters1,filter_size=filter_size2,
                                     num_filters=num_filters2,use_pooling=True,layer_name='layer2')
layer_conv3,weights_conv3=conv_layer(input=layer_conv1,input_channels=num_filters1,filter_size=filter_size2,
                                     num_filters=num_filters2,use_pooling=True,layer_name='layer3')
layer_flat,num_features=flatten_layer(layer_conv3,'layer4')
layer_fc1,weights_FC1=fc_layer(input=layer_flat,num_inputs=num_features,num_outputs=num_classes,keep_prob=keep_prob,layer_name='layer5',use_relu=True)

y_pred=tf.nn.softmax(layer_fc1)
y_pred_cls=tf.argmax(y_pred,dimension=1)
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
train_writer=tf.summary.FileWriter(logs_path+'/train',sess.graph)
test_writer=tf.summary.FileWriter(logdir=logs_path+"/test")
sess.run(tf.global_variables_initializer())
with sess.as_default():
    tf.global_variables_initializer().run()
train_batch_size=batch_size


def print_progress(epoch, feed_dict_train, feed_dict_test, test_loss, feed_dict_val):
    acc = sess.run(accuracy, feed_dict=feed_dict_train)
    test_acc = sess.run(accuracy, feed_dict=feed_dict_test)
    val_acc = sess.run(accuracy, feed_dict=feed_dict_val)

    msg = "Epoch {0} --- Training Accuracy:{1:>6.1%},Testing Accuracy:{2:>6.1%}, Test Loss:{3:.3f}, Validation Accuracy:{4:>6.1%}"
    print(msg.format(epoch + 1, acc, test_acc, test_loss, val_acc))


total_iterations = 0
batch_size = 100
img_size_flat = img_size * img_size * num_channels


def optimize(num_iterations):
    info_optimize = open("info_Optimize","w")
    info_output = open("info_Output", "w")
    global total_iterations
    best_test_loss = float("inf")
    summary_writer = tf.summary.FileWriter(logs_path, graph=sess.graph)
    for i in range(total_iterations, total_iterations + num_iterations):
        info_optimize.write("start FOR\n")
        info_optimize.write("total_iterations: "+str(total_iterations))
        info_optimize.write("\n")
        info_optimize.write("num_iterations: "+str(num_iterations))
        info_optimize.write("\n")
        info_optimize.write("i: "+str(i))
        info_optimize.write("\n")

        train_idx = np.random.randint(X_Train.shape[0], size=100)
        info_optimize.write("train_idx: "+str(train_idx))
        # info_output.write("\n")

        batch_x_Train = X_Train[train_idx, :]
        batch_y_Train = y_Train[train_idx]
        # info_output.write("batch_x_Train:")
        # info_output.write("\n")
        # info_output.write(str(batch_x_Train))
        # info_output.write("\n")
        # info_output.write("\n")
        # info_output.write("batch_y_Train:")
        # info_output.write("\n")
        # info_output.write(str(batch_y_Train))

        # batch_x_Train=batch_x_Train.reshape(batch_size,img_size_flat)
        train_idx = np.random.randint(X_Test.shape[0], size=100)
        batch_x_Test = X_Test[train_idx, :]
        batch_y_Test = y_Test[train_idx]
        # info_output.write("batch_x_Test:")
        # info_output.write("\n")
        # info_output.write(str(batch_x_Test))
        # info_output.write("\n")
        # info_output.write("\n")
        # info_output.write("batch_y_Test:")
        # info_output.write("\n")
        # info_output.write(str(batch_y_Test))




        train_idx = np.random.randint(X_Val.shape[0], size=100)
        batch_x_Val = X_Val[train_idx, :]
        batch_y_Val = y_Val[train_idx]
        # info_output.write("batch_x_Validation:")
        # info_output.write("\n")
        # info_output.write(str(batch_x_Val))
        # info_output.write("\n")
        # info_output.write("\n")
        # info_output.write("batch_y_Validation:")
        # info_output.write("\n")
        # info_output.write(str(batch_y_Val))





        # batch_x_Test=batch_x_Test.reshape(batch_size,img_size_flat)
        feed_dict_train = {x: batch_x_Train,
                           y_true: batch_y_Train, keep_prob: 0.5}

        feed_dict_test = {x: batch_x_Test,
                          y_true: batch_y_Test, keep_prob: 1.0}

        feed_dict_val = {x: batch_x_Val,
                          y_true: batch_y_Val, keep_prob: 1.0}

        _, c, summary = sess.run([optimizer, cost, merged], feed_dict=feed_dict_train)
        if i % int(X_Train.shape[0] / batch_size) == 0:
            test_loss = sess.run(cost, feed_dict=feed_dict_test)
            epoch = int(i / (X_Train.shape[0] / batch_size))
            summary_writer.add_summary(summary, epoch * batch_size + i)
            print_progress(epoch, feed_dict_train, feed_dict_test, test_loss, feed_dict_val)
        total_iterations += num_iterations
        info_optimize.write("epoch: "+str(epoch))
        info_optimize.write("\n\n")
        # info_output.write("\n\n")
    info_optimize.close()
    # info_output.close()

optimize(num_iterations=1000)